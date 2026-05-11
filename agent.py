"""
0G Labs Documentation Agent — CLI entry point.

Usage:
    python agent.py                  # interactive loop
    python agent.py "your question"  # single query, then exit
    python agent.py --verbose        # show LangGraph debug traces
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import re
import sys
import uuid
from pathlib import Path

# LangChain and LangGraph manage their own warning filters via
# surface_langchain_deprecation_warnings(), which re-inserts 'default' filters
# on every transitive import and cannot be overridden via filterwarnings().
# Redirecting stderr during the noisy imports is the only reliable fix.
import contextlib, io

with contextlib.redirect_stderr(io.StringIO()):
    from dotenv import load_dotenv
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule

load_dotenv()

console = Console()
SKILL_DIR = Path(__file__).parent / "skills" / "0g-docs-search"
RECURSION_LIMIT = 50

_clear_page_cache = None  # set by build_agent() at startup
_warm_cache_fn = None     # set by build_agent() at startup
_prefetch_fn = None       # set by build_agent() at startup

# ---------------------------------------------------------------------------
# Keyword router — maps query terms to likely documentation URLs.
# Matched URLs are pre-fetched BEFORE the agent runs so the first LLM call
# is the synthesis call, not a planning call.  Add entries as new topics emerge.
# ---------------------------------------------------------------------------

_ROUTE_MAP: list = [
    # Storage SDK / CLI — includes turbo/standard endpoint queries
    (
        [
            "storage sdk", "storage cli", "upload", "download", "0g storage", "zgs",
            "file storage", "turbo", "storage endpoint", "storage node",
            "indexer", "flow contract",
        ],
        [
            "https://docs.0g.ai/developer-hub/building-on-0g/storage/sdk",
            "https://docs.0g.ai/developer-hub/building-on-0g/storage/storage-cli",
            "https://docs.0g.ai/developer-hub/network-info",
        ],
    ),
    # Data Availability
    (
        ["data availability", " da ", "da node", "0g da", "blob submission", "submit blob", "blob to da", "submit a blob", "da blob", "da integration"],
        [
            "https://docs.0g.ai/developer-hub/building-on-0g/da-integration",
        ],
    ),
    # Compute Network / Inference (developer)
    (
        [
            "compute network", "inference api", "model api", "ai model", "llm endpoint", "compute sdk",
            "list model", "list the model", "available model", "models on", "models available",
            "which model", "what model", "supported model", "inference model", "inference endpoint",
            "mainnet model", "testnet model", "model name", "deepseek", "qwen", "whisper", "glm",
        ],
        [
            "https://pc.0g.ai/playground",
            "https://docs.0g.ai/developer-hub/building-on-0g/compute-network/inference",
            "https://build.0g.ai/compute",
        ],
    ),
    # 0G AI overview / general AI ecosystem questions
    (
        ["ai context", "0g ai", "what is 0g", "overview", "ai infrastructure",
         "decentralized ai", "ai network", "0g network", "how does 0g work"],
        [
            "https://docs.0g.ai/ai-context",
            "https://docs.0g.ai/",
        ],
    ),
    # Private Computer  (different product from Compute Network)
    (
        ["private computer", "pc.0g", "verifiable ai", "verifiable inference", "pc api", "pc credit"],
        [
            "https://pc.0g.ai/",
            "https://pc.0g.ai/playground",
        ],
    ),
    # Network endpoints / chain info
    (
        ["rpc endpoint", "network info", "chain id", "contract address", "rpc url", "network endpoint"],
        [
            "https://docs.0g.ai/developer-hub/network-info",
        ],
    ),
    # Chain / staking / validators / EVM development
    (
        [
            "delegate", "undelegate", "staking", "stake", "validator", "node operator",
            "delegation", "evm", "solidity", "hardhat", "foundry", "deploy contract",
            "smart contract", "chain info", "build.0g.ai/chain",
        ],
        [
            "https://docs.0g.ai/developer-hub/building-on-0g/contracts-on-0g/staking-interfaces",
            "https://build.0g.ai/chain",
            "https://app.0g.ai/",
        ],
    ),
    # Blog / product announcements
    (
        ["blog", "announcement", "news", "0g pay", "go to market", "gtm", "ecosystem", "partner"],
        [
            "https://0g.ai/sitemap.xml",
        ],
    ),
]


def _route_query(query: str) -> list:
    """Return deduplicated URLs likely relevant to *query* based on keywords."""
    # Pad with spaces so short terms like " da " don't match "data"
    q = f" {query.lower()} "
    urls: list = []
    for keywords, candidate_urls in _ROUTE_MAP:
        if any(kw in q for kw in keywords):
            for u in candidate_urls:
                if u not in urls:
                    urls.append(u)
    return urls


# ---------------------------------------------------------------------------
# Skill loader
# ---------------------------------------------------------------------------

def _load_skill() -> tuple:
    """Return (system_prompt, tools, clear_page_cache, warm_cache, prefetch_urls)."""
    skill_md = (SKILL_DIR / "SKILL.md").read_text()
    prompt = re.sub(r"^---\n.*?\n---\n*", "", skill_md, flags=re.DOTALL).strip()

    spec = importlib.util.spec_from_file_location("skill_script", SKILL_DIR / "script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return prompt, [module.fetch_pages_parallel, module.fetch_page], module.clear_page_cache, module.warm_cache, module.prefetch_urls


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

async def warm_cache() -> None:
    """Pre-fetch all known documentation pages into the SQLite cache."""
    if _warm_cache_fn is not None:
        await _warm_cache_fn()


def build_agent(verbose: bool = False):
    global _clear_page_cache, _warm_cache_fn, _prefetch_fn
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("MODEL_NAME", "gpt-4o")

    if not api_key:
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY is not set in .env")
        sys.exit(1)

    with contextlib.redirect_stderr(io.StringIO()):
        from core.persistence import get_checkpointer
        from core.graph import build_graph

    system_prompt, all_tools, _clear_page_cache, _warm_cache_fn, _prefetch_fn = _load_skill()

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0,
        timeout=60,        # fail after 60s instead of hanging indefinitely
        max_retries=1,     # one retry on transient errors (503, 429, network blip)
    )

    return build_graph(llm, all_tools, system_prompt, get_checkpointer(), debug=verbose)


# ---------------------------------------------------------------------------
# Query runner — streams tool-call feedback and final response
# ---------------------------------------------------------------------------

def _url_label(url: str) -> str:
    """Return a short human-readable label for a documentation URL."""
    u = url.lower()
    if "sitemap" in u:
        return "0G sitemap"
    if "0g.ai/blog" in u:
        return "0G Blog"
    if "docs.0g.ai" in u:
        return "0G Docs"
    if "build.0g.ai" in u:
        return "0G Builder Hub"
    if "pc.0g.ai" in u:
        return "0G Private Computer"
    if "app.0g.ai" in u:
        return "0G App"
    if "0g.ai" in u:
        return "0G Website"
    return url


def _describe_fetch(tool_name: str, tool_input: dict) -> str:
    """Build a 'Searching …' label from a tool call's input."""
    if tool_name == "fetch_page":
        return f"Searching {_url_label(tool_input.get('url', ''))}"
    if tool_name == "fetch_pages_parallel":
        urls = [u.strip() for u in tool_input.get("urls", "").split(",") if u.strip()]
        labels = list(dict.fromkeys(_url_label(u) for u in urls))  # deduplicate, keep order
        if not labels:
            return "Fetching pages"
        if len(labels) == 1:
            return f"Searching {labels[0]}"
        if len(labels) == 2:
            return f"Searching {labels[0]} and {labels[1]}"
        return "Searching " + ", ".join(labels[:-1]) + f", and {labels[-1]}"
    return f"Running {tool_name}"


def _extract_text(content) -> str:
    """Extract plain text from a str or list-of-blocks content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)
# Matches Python-style function calls the model may emit as plain text instead of
# structured tool calls: fetch_pages_parallel(urls="...") or fetch_page(url="...")
_TOOL_FN_CALL_RE = re.compile(r"\bfetch_(?:pages_parallel|page)\s*\([^)]*\)?", re.DOTALL)


def _clean_text(text: str) -> str:
    """Strip tool-call markup from text. Returns empty string if nothing remains."""
    text = _TOOL_CALL_BLOCK_RE.sub("", text)
    text = _TOOL_FN_CALL_RE.sub("", text)
    return text.strip()


async def _build_initial_input(query: str, pre_urls: list | None = None) -> dict:
    """Build the LangGraph input dict for *query*.

    Three paths (evaluated in order):
      A. Blog/dynamic keywords → live fetch (blog changes daily, never indexed)
      B. RAG hit (distance < threshold) → inject index chunks, skip HTTP fetch
      C. RAG miss or no route match → live fetch fallback

    *pre_urls* may be supplied by the caller to avoid a second _route_query call.
    Always resets tool_calls_made so the counter never bleeds across turns.
    """
    if pre_urls is None:
        pre_urls = _route_query(query)

    # Path A — blog / sitemap / playground queries always bypass RAG (content changes frequently)
    is_dynamic = any("0g.ai/blog" in u or "sitemap" in u or "pc.0g.ai/playground" in u for u in pre_urls)

    if not is_dynamic:
        # Path B — try the RAG index first
        try:
            from core.rag import query_index, DISTANCE_THRESHOLD  # noqa: PLC0415
            rag_content, min_distance = await query_index(query)
            if rag_content and min_distance < DISTANCE_THRESHOLD:
                tool_call_id = f"pre_{uuid.uuid4().hex[:8]}"
                source_urls = pre_urls or ["https://docs.0g.ai/"]
                return {
                    "messages": [
                        HumanMessage(content=query),
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "id": tool_call_id,
                                "name": "fetch_pages_parallel",
                                "args": {"urls": ",".join(source_urls)},
                            }],
                        ),
                        ToolMessage(content=rag_content, tool_call_id=tool_call_id),
                    ],
                    "tool_calls_made": 1,
                }
        except (ImportError, OSError):
            pass  # RAG not installed or data dir missing — fall through to live fetch

    # Path C (and Path A) — live fetch
    if pre_urls and _prefetch_fn is not None:
        content = await _prefetch_fn(pre_urls)
        if content and len(content) > 200:
            tool_call_id = f"pre_{uuid.uuid4().hex[:8]}"
            return {
                "messages": [
                    HumanMessage(content=query),
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "id": tool_call_id,
                            "name": "fetch_pages_parallel",
                            "args": {"urls": ",".join(pre_urls)},
                        }],
                    ),
                    ToolMessage(content=content, tool_call_id=tool_call_id),
                ],
                "tool_calls_made": 1,
            }

    return {
        "messages": [HumanMessage(content=query)],
        "tool_calls_made": 0,
    }


async def run_query(agent, query: str, config: dict) -> str:
    final_ai_content = None   # fallback: last AI message from graph output
    _tool_start_times: dict = {}  # run_id → start time for cache detection
    _turn_buf: list[str] = []     # tokens buffered for the current model invocation
    _turn_has_tool_call = False   # True if the current invocation issued a tool call

    initial_input = await _build_initial_input(query)

    async for event in agent.astream_events(
        initial_input,
        config=config,
        version="v2",
    ):
        kind = event["event"]
        run_id = event.get("run_id", "")

        if kind == "on_tool_start":
            _tool_start_times[run_id] = asyncio.get_event_loop().time()
            label = _describe_fetch(event["name"], event.get("data", {}).get("input", {}))
            console.print(f"  [dim]→ {label}...[/dim]")

        elif kind == "on_tool_end":
            elapsed = asyncio.get_event_loop().time() - _tool_start_times.pop(run_id, 0)
            if elapsed < 0.3:
                console.print(f"  [dim]    (from cache, {elapsed * 1000:.0f}ms)[/dim]")

        elif kind == "on_tool_error":
            _tool_start_times.pop(run_id, None)
            console.print(f"  [bold red]✗ {event['name']} failed:[/bold red] {event['data'].get('error', '')}")

        elif kind == "on_chat_model_start":
            _turn_buf.clear()
            _turn_has_tool_call = False

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                _turn_has_tool_call = True
                _turn_buf.clear()
                continue
            if not _turn_has_tool_call:
                text = _extract_text(chunk.content)
                if text and "<tool_call>" not in text:
                    _turn_buf.append(text)

        elif kind == "on_chat_model_end":
            # Discard tool-calling turns; the buffer for the final answer turn
            # is read after the loop ends.
            if _turn_has_tool_call:
                _turn_buf.clear()
            _turn_has_tool_call = False

        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            # Keep the final graph output as a fallback in case streaming
            # captured nothing (e.g. model returns content only on finish).
            # Skip ToolMessages — they contain raw fetched page content, not answers.
            output = event.get("data", {}).get("output") or {}
            messages = output.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, (HumanMessage, ToolMessage)):
                    break
                text = _clean_text(_extract_text(getattr(msg, "content", "")))
                if text and not getattr(msg, "tool_calls", None):
                    final_ai_content = text
                    break

    # _turn_buf holds tokens from the last non-tool-call invocation (the final answer)
    if _turn_buf:
        cleaned = _clean_text("".join(_turn_buf))
        if cleaned:
            return cleaned

    # Streaming produced nothing — use the final graph state instead
    if final_ai_content:
        return final_ai_content

    # Last resort: read the saved state directly from the checkpointer.
    # This catches models that don't stream tokens and endpoints where
    # on_chain_end fires before the state is fully written.
    try:
        state = await agent.aget_state(config)
        for msg in reversed(state.values.get("messages", [])):
            if isinstance(msg, (HumanMessage, ToolMessage)):
                break
            text = _clean_text(_extract_text(getattr(msg, "content", "")))
            if text and not getattr(msg, "tool_calls", None):
                return text
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# CLI streaming — streams tokens to the terminal as they arrive
# ---------------------------------------------------------------------------

async def cli_stream_response(agent, query: str, config: dict) -> None:
    """Stream the agent's response to the terminal with live Markdown rendering.

    Phase 1 (fetching): spinner label updates as each tool call runs.
    Phase 2 (answering): spinner stops, a Rich Live panel renders the growing
    Markdown in-place so formatting is correct from the very first token.
    Falls back to a single Markdown render if the endpoint doesn't stream.
    """
    response_parts: list[str] = []
    answer_started = False
    final_ai_content = None
    _tool_start_times: dict = {}
    _turn_buf: list[str] = []
    _turn_has_tool_call = False
    live: Live | None = None

    status = console.status("[dim]Thinking...[/dim]", spinner="dots")
    status.start()

    # Compute route once — used for both the spinner label and _build_initial_input.
    pre_urls = _route_query(query)
    if pre_urls:
        labels = list(dict.fromkeys(_url_label(u) for u in pre_urls))
        label_str = " & ".join(labels[:2]) + (" & ..." if len(labels) > 2 else "")
        status.update(f"[dim]→ Fetching {label_str}...[/dim]")

    initial_input = await _build_initial_input(query, pre_urls=pre_urls)

    if pre_urls:
        status.update("[dim]Thinking...[/dim]")

    try:
        async for event in agent.astream_events(
            initial_input,
            config=config,
            version="v2",
        ):
            kind = event["event"]
            run_id = event.get("run_id", "")

            if kind == "on_tool_start":
                _tool_start_times[run_id] = asyncio.get_event_loop().time()
                label = _describe_fetch(event["name"], event.get("data", {}).get("input", {}))
                status.update(f"[dim]→ {label}...[/dim]")

            elif kind == "on_tool_end":
                elapsed = asyncio.get_event_loop().time() - _tool_start_times.pop(run_id, 0)
                if elapsed < 0.3:
                    status.update("[dim]→ (from cache)[/dim]")
                status.update("[dim]Thinking...[/dim]")

            elif kind == "on_tool_error":
                _tool_start_times.pop(run_id, None)
                status.update("[dim]Thinking...[/dim]")

            elif kind == "on_chat_model_start":
                _turn_buf.clear()
                _turn_has_tool_call = False

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    _turn_has_tool_call = True
                    _turn_buf.clear()
                    continue
                if not _turn_has_tool_call:
                    text = _extract_text(chunk.content)
                    if text and "<tool_call>" not in text:
                        _turn_buf.append(text)

            elif kind == "on_chat_model_end":
                # Flush buffered tokens only for turns that didn't issue tool calls
                if not _turn_has_tool_call and _turn_buf:
                    if not answer_started:
                        answer_started = True
                        status.stop()
                        console.print(Rule(style="dim"))
                        console.print("[bold green]Agent:[/bold green]")
                        live = Live(
                            console=console,
                            refresh_per_second=15,
                            transient=False,
                        )
                        live.start()
                    response_parts.extend(_turn_buf)
                    if live:
                        live.update(Markdown("".join(response_parts)))
                _turn_buf.clear()
                _turn_has_tool_call = False

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output") or {}
                messages = output.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, (HumanMessage, ToolMessage)):
                        break
                    text = _clean_text(_extract_text(getattr(msg, "content", "")))
                    if text and not getattr(msg, "tool_calls", None):
                        final_ai_content = text
                        break
    finally:
        status.stop()
        if live:
            live.stop()

    if response_parts:
        return  # Live already rendered the final Markdown in-place

    # Fallback: endpoint didn't stream — render once as Markdown.
    # Skip ToolMessages — they hold raw fetched page content, not synthesized answers.
    answer = final_ai_content
    if not answer:
        try:
            state = await agent.aget_state(config)
            for msg in reversed(state.values.get("messages", [])):
                if isinstance(msg, (HumanMessage, ToolMessage)):
                    break
                text = _clean_text(_extract_text(getattr(msg, "content", "")))
                if text and not getattr(msg, "tool_calls", None):
                    answer = text
                    break
        except Exception:
            pass

    console.print(Rule(style="dim"))
    console.print("[bold green]Agent:[/bold green]")
    if answer:
        console.print(Markdown(answer))
    else:
        console.print("[dim]The agent didn't return a response. Please try your question again.[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def interactive_loop(agent) -> None:
    console.print(Panel(
        "[bold cyan]0G Labs Documentation Agent[/bold cyan]\n"
        "[dim]Live search · docs.0g.ai · build.0g.ai · pc.0g.ai · app.0g.ai · blog[/dim]\n"
        "[dim][bold]exit[/bold] to quit · [bold]clear[/bold] to reset history · [bold]recache[/bold] to refresh docs cache[/dim]",
        border_style="cyan",
    ))

    thread_id = str(uuid.uuid4())

    # Pre-warm the page cache in the background so the first query is fast.
    asyncio.create_task(warm_cache())

    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if query.lower() == "clear":
            thread_id = str(uuid.uuid4())   # new thread = fresh memory
            console.print("[dim]Conversation history cleared.[/dim]")
            continue
        if query.lower() == "recache" or query.lower().startswith("recache "):
            # parse optional URL argument
            parts = query.split(maxsplit=1)
            url_arg = parts[1].strip() if len(parts) > 1 else None
            if _clear_page_cache is not None:
                msg = _clear_page_cache(url_arg)
                console.print(f"[dim]{msg}[/dim]")
                if url_arg:
                    try:
                        from core.rag import drop_url_chunks  # noqa: PLC0415
                        await asyncio.to_thread(drop_url_chunks, url_arg)
                        console.print(f"[dim]RAG index cleared for: {url_arg}[/dim]")
                    except Exception:
                        pass
            else:
                console.print("[dim]Cache not available.[/dim]")
            continue

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        t0 = asyncio.get_event_loop().time()
        try:
            await cli_stream_response(agent, query, config)
        except Exception as e:
            console.print(f"[bold red]Agent error:[/bold red] {e}")
        elapsed = asyncio.get_event_loop().time() - t0
        console.print(f"[dim]({elapsed:.1f}s)[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def single_query(agent, query: str) -> None:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}
    t0 = asyncio.get_event_loop().time()
    await cli_stream_response(agent, query, config)
    elapsed = asyncio.get_event_loop().time() - t0
    console.print(f"[dim]({elapsed:.1f}s)[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="0G Labs documentation agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="?", help="Query to answer (omit for interactive mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show LangGraph debug traces")
    args = parser.parse_args()

    agent = build_agent(verbose=args.verbose)

    if args.query:
        asyncio.run(single_query(agent, args.query))
    else:
        asyncio.run(interactive_loop(agent))


if __name__ == "__main__":
    main()
