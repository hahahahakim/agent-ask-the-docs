"""Query decomposer — detects compound multi-topic queries and splits them.

A compound query spans two or more distinct documentation topics in a single
question. Examples:
  - "How do I upload to Storage AND what inference models are available?"
  - "What is 0G DA and how is it different from 0G Storage?"

For single-topic queries (the majority), decompose() returns the original
query unchanged in a single-element list.

Detection is heuristic (no LLM call) — uses conjunctive patterns to keep
latency near-zero for simple queries.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Conjunctive patterns that signal a compound query
# ---------------------------------------------------------------------------

# Each pattern is a tuple of (compiled_regex, split_pattern_string).
# The split_pattern_string is used with re.split() to divide the query;
# we keep the first two segments to form the two sub-queries.
_COMPOUND_PATTERNS: list[tuple] = [
    # "X, how/what/where/when/who/can/do/is/are Y" — comma followed by question word
    (
        re.compile(r"(?<!\w),\s+(?=how|what|where|when|who|can|do|is|are)", re.IGNORECASE),
        r"(?<!\w),\s+(?=how|what|where|when|who|can|do|is|are)",
    ),
    # "X as well as Y"
    (
        re.compile(r"\bas\s+well\s+as\b", re.IGNORECASE),
        r"\bas\s+well\s+as\b",
    ),
    # "X while Y"
    (
        re.compile(r"\bwhile\b", re.IGNORECASE),
        r"\bwhile\b",
    ),
    # "X plus Y"
    (
        re.compile(r"\bplus\b", re.IGNORECASE),
        r"\bplus\b",
    ),
    # "X also Y"
    (
        re.compile(r"\balso\b", re.IGNORECASE),
        r"\balso\b",
    ),
    # "X and Y" — last, most generic; checked after the more specific ones
    (
        re.compile(r"\band\b", re.IGNORECASE),
        r"\band\b",
    ),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose(query: str) -> list:
    """Return a list of atomic sub-queries for *query*.

    Returns [query] unchanged for single-topic queries.
    Returns [sub1, sub2, ...] for compound queries split on conjunctive
    boundaries.

    Never raises — returns [query] on any error.
    """
    try:
        return _decompose(query)
    except Exception:
        return [query]


def is_compound(query: str) -> bool:
    """Return True if *query* appears to be a compound multi-topic question."""
    return len(decompose(query)) > 1


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _decompose(query: str) -> list:
    """Internal decompose logic (may raise; caller wraps in try/except)."""
    q = query.strip()

    # Short queries are never compound (< 8 words can't meaningfully span two topics)
    if len(q.split()) < 8:
        return [q]

    # Try each pattern in priority order.  Use the first one that produces a
    # valid split (both parts >= 5 words).
    for compiled_re, split_pattern in _COMPOUND_PATTERNS:
        if not compiled_re.search(q):
            continue

        # Split on the first occurrence only — re.split with maxsplit=1 gives
        # us [left, right].  The connector itself is consumed (not kept).
        parts = re.split(split_pattern, q, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) < 2:
            continue

        left = parts[0].strip().rstrip(",").strip()
        right_raw = parts[1].strip() if len(parts) > 1 else ""

        # Reject the split if either raw part is too short (aggressive split).
        # Check raw lengths BEFORE _ensure_complete so that bare fragments like
        # "RPC URL?" (2 words) are caught — they indicate a list ("chain ID and
        # RPC URL") rather than two independent questions.
        # Allow 4-word parts — "What is 0G DA" is a valid sub-query.
        if len(left.split()) < 4 or len(right_raw.split()) < 4:
            continue

        # Ensure right part reads as a complete thought.
        right = _ensure_complete(right_raw, left)

        return [left, right]

    return [q]


def _ensure_complete(fragment: str, context: str) -> str:
    """Make *fragment* a complete question if it is just a bare noun phrase.

    Most right-hand fragments after splitting on "and" / "as well as" / etc.
    are already complete questions ("how is it different from …").  The only
    case that needs repair is a bare noun phrase like "the RPC endpoints"
    (from "What are the RPC endpoints and how do I deploy a contract?").
    In that situation the real informational split is on the conjunction that
    follows, so we leave the fragment as-is — the word-count gate in
    _decompose() will catch truly degenerate splits.
    """
    if not fragment:
        return fragment

    # Already looks like a question (starts with an interrogative or "how/what/…")
    first_word = fragment.split()[0].lower()
    question_starters = {
        "how", "what", "where", "when", "who", "which", "why", "can", "do",
        "does", "is", "are", "will", "would", "should", "could",
    }
    if first_word in question_starters:
        return fragment

    # Bare fragment — prepend a generic question starter so it reads naturally
    return "What is " + fragment
