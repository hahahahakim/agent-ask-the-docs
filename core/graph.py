"""
LangGraph StateGraph for the 0G Labs Documentation Agent.

Architecture:
    START → agent_node ⇄ tools_node  (standard ReAct loop)
                │ (no tool calls)        │
                ▼                        │ (tool_calls_made >= MAX_TOOL_CALLS)
               END                       ▼
                               final_agent_node → END
                               (LLM without tools — forces synthesis)

final_agent_node guarantees a text answer when the tool-call cap is hit
while the model still wants more tools. Without tools bound, the model must
synthesise from whatever context has been accumulated.

Note: reflection was removed because zai-org/GLM-5.1-FP8 does not reliably
produce structured JSON output, which caused OUTPUT_PARSING_FAILURE errors.
Re-add when the model or endpoint supports structured output.
"""

import json
import re
import uuid
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# Hard cap on tool calls per query — prevents runaway fetching and recursion
# limit errors. Each tool call = 2 graph steps (agent → tools → agent), so
# 4 tool calls × 2 steps + 1 initial = 9 steps maximum, well under the
# recursion_limit of 50 set in config.
MAX_TOOL_CALLS = 4

# ---------------------------------------------------------------------------
# Text-format tool-call parser
#
# Some OpenAI-compatible endpoints (GLM, Qwen, etc.) occasionally return tool
# calls as <tool_call>…</tool_call> text content instead of structured
# tool_calls.  LangGraph sees tool_calls=[] on the AIMessage and routes
# straight to END, so the tool is never executed.
#
# _parse_text_tool_calls() extracts these and returns a list of structured
# dicts that can be set on a replacement AIMessage, allowing the ReAct loop
# to continue normally.
# ---------------------------------------------------------------------------

_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>(.*?)(?:</tool_call>|$)",
    re.DOTALL,
)
# Python-style:  func_name(key="value", key2=value2)
_PY_CALL_RE = re.compile(r"^(\w+)\((.*)\)\s*$", re.DOTALL)
# key=value pair (value may or may not be quoted)
_KV_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\)]+))')
# Untagged tool calls the model may emit without <tool_call> wrappers.
# Flexible enough to catch all observed variants regardless of underscores:
#   fetch_pages_parallel, pages_parallel, pagesparallel, fetchpagesparallel,
#   fetch_page, fetchpage, page
_UNTAGGED_TOOL_RE = re.compile(
    r'\b((?:fetch_?)?(?:pages?_?parallel|page))\s*\(([^)]*)\)',
    re.DOTALL,
)


def _normalize_tool_name(raw: str) -> str:
    """Map any variant of the tool name to its canonical form."""
    s = raw.lower().replace("_", "")
    if "parallel" in s:
        return "fetch_pages_parallel"
    return "fetch_page"


def _parse_untagged_tool_calls(content: str) -> list:
    """Extract tool calls from plain Python-style calls with no <tool_call> wrapper.

    Some model responses emit e.g. ``pagesparallel(urls="...")`` as raw text
    without any tags.  Without this parser, agent_node returns the raw text as
    the final answer instead of executing the tool.
    """
    calls = []
    for match in _UNTAGGED_TOOL_RE.finditer(content):
        func_name = _normalize_tool_name(match.group(1))
        args: dict = {}
        for kv in _KV_RE.finditer(match.group(2)):
            key = kv.group(1)
            val = kv.group(2) or kv.group(3) or (kv.group(4) or "").strip()
            args[key] = val
        if args:
            calls.append({
                "id": f"tc_{uuid.uuid4().hex[:8]}",
                "name": func_name,
                "args": args,
            })
    return calls


def _parse_text_tool_calls(content: str) -> list:
    """Return structured tool-call dicts extracted from <tool_call> text blocks."""
    calls = []
    for match in _TOOL_CALL_TAG_RE.finditer(content):
        raw = match.group(1).strip()
        if not raw:
            continue

        # --- JSON format: {"name": "func", "arguments": {...}} ---
        try:
            data = json.loads(raw)
            name = (
                data.get("name")
                or data.get("function", {}).get("name", "")
            )
            args = data.get("arguments") or data.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name and isinstance(args, dict):
                calls.append({
                    "id": f"tc_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "args": args,
                })
                continue
        except Exception:
            pass

        # --- Python-style: func_name(key="value", …) ---
        py = _PY_CALL_RE.match(raw)
        if py:
            func_name = py.group(1)
            args: dict = {}
            for kv in _KV_RE.finditer(py.group(2)):
                key = kv.group(1)
                # quoted groups 2/3, unquoted group 4
                val = kv.group(2) or kv.group(3) or (kv.group(4) or "").strip()
                args[key] = val
            if func_name:
                calls.append({
                    "id": f"tc_{uuid.uuid4().hex[:8]}",
                    "name": func_name,
                    "args": args,
                })

    return calls


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_made: int  # incremented by tools_node; stops loop at MAX_TOOL_CALLS


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_call_tools(state: AgentState) -> str:
    """Route after agent_node.

    - tool calls present AND under cap → "tools"
    - tool calls present AND cap hit → "final_agent" (force synthesis without tools)
    - no tool calls → END
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END
    if state.get("tool_calls_made", 0) >= MAX_TOOL_CALLS:
        return "final_agent"
    return "tools"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(llm, tools: list, system_prompt: str, checkpointer, debug: bool = False) -> CompiledGraph:
    """Build and compile the documentation agent graph."""

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def _trim_history(state: AgentState) -> list:
        """Return a trimmed message list for the LLM.

        Always keeps the first HumanMessage (the user's query) so the model
        never loses track of what was asked.  Appends up to 7 of the most
        recent messages after that, keeping total context ≤ 8 messages.
        """
        history = list(state["messages"])
        if len(history) <= 8:
            return history

        # Locate the first HumanMessage (the original user query)
        first_human_idx = next(
            (i for i, m in enumerate(history) if isinstance(m, HumanMessage)),
            None,
        )
        if first_human_idx is None:
            # No HumanMessage at all — just take the tail
            return history[-8:]

        human_msg = history[first_human_idx]
        # Everything after the HumanMessage, capped at 7 most-recent
        recent = history[first_human_idx + 1:][-7:]
        return [human_msg] + recent

    async def agent_node(state: AgentState) -> dict:
        """Call the LLM with tools bound, prepending the system prompt.

        After the call, check for text-format <tool_call> blocks.  Some
        endpoints emit these as plain content instead of structured tool_calls.
        Convert them so the ReAct routing logic sees real tool calls.
        """
        messages = [SystemMessage(content=system_prompt)] + _trim_history(state)
        response = await llm_with_tools.ainvoke(messages)

        content = response.content if isinstance(response.content, str) else ""

        # Convert text-format <tool_call> blocks to structured tool calls
        if content and "<tool_call>" in content and not response.tool_calls:
            parsed = _parse_text_tool_calls(content)
            if parsed:
                response = AIMessage(content="", tool_calls=parsed)

        # Also handle untagged Python-style calls (model forgot the <tool_call> wrapper)
        elif content and not response.tool_calls:
            parsed = _parse_untagged_tool_calls(content)
            if parsed:
                response = AIMessage(content="", tool_calls=parsed)

        return {"messages": [response]}

    async def final_agent_node(state: AgentState) -> dict:
        """Force synthesis by calling the LLM WITHOUT tools.

        Runs when the tool-call cap is hit and the model still wants more tools.
        An explicit suffix is appended to the system prompt to break models
        (GLM, Qwen) out of "tool-calling mode" so they produce prose instead
        of more <tool_call> text blocks.

        If the model response is empty after stripping tool-call artifacts, a
        canned "couldn't find" message is returned so callers always get text.
        """
        # Collect all ToolMessage content accumulated during this query.
        # This is the raw documentation that was retrieved — use it to build
        # the synthesis prompt directly rather than relying on the model to
        # recall it from its context window (GLM/Qwen models lose track of
        # accumulated context when stuck in tool-calling mode).
        history = _trim_history(state)
        tool_context = "\n\n".join(
            msg.content
            for msg in history
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str)
        )

        forced_suffix = (
            "\n\nIMPORTANT — FINAL ANSWER REQUIRED:\n"
            "Stop all tool calls immediately. Do NOT output any <tool_call> tags.\n"
            "Write your complete answer in plain prose, using ONLY the documentation "
            "excerpts provided below. If the excerpts do not fully answer the question, "
            "state what was found and recommend the user visit https://docs.0g.ai/ or "
            "the 0G Discord / Telegram.\n\n"
            "--- RETRIEVED DOCUMENTATION ---\n"
            f"{tool_context or '(no documentation retrieved)'}\n"
            "--- END OF DOCUMENTATION ---"
        )
        # Pass only the original user question — no tool-call history that could
        # re-trigger the model's tool-calling instinct.
        user_msg = next(
            (msg for msg in history if isinstance(msg, HumanMessage)), None
        )
        messages = [
            SystemMessage(content=system_prompt + forced_suffix),
            *(([user_msg]) if user_msg else []),
        ]
        response = await llm.ainvoke(messages)  # plain llm — no tools bound

        content = response.content if isinstance(response.content, str) else ""

        # Strip any residual <tool_call> blocks the model may still emit
        if "<tool_call>" in content:
            content = re.sub(
                r"<tool_call>.*?(?:</tool_call>|$)", "", content, flags=re.DOTALL
            ).strip()

        # Absolute fallback — model returned nothing useful
        if not content:
            content = (
                "I searched the 0G documentation but could not produce a complete answer "
                "from the retrieved pages. Please visit https://docs.0g.ai/ directly, "
                "or ask in the 0G Discord / Telegram for further help."
            )

        return {"messages": [AIMessage(content=content)]}

    async def tools_node_with_counter(state: AgentState) -> dict:
        """Execute tools (async) and increment the call counter."""
        result = await tool_node.ainvoke(state)
        current = state.get("tool_calls_made", 0)
        result["tool_calls_made"] = current + 1
        return result

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node_with_counter)
    graph.add_node("final_agent", final_agent_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_call_tools,
        {"tools": "tools", "final_agent": "final_agent", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("final_agent", END)

    return graph.compile(
        checkpointer=checkpointer,
        debug=debug,
    )
