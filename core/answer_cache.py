"""
Answer-level response cache for the docs agent.

Caches LLM answers keyed by a hash of the normalised query string.
Uses the same SQLite file as the page cache (data/page_cache.db) but
a separate table (answer_cache) so the two caches can be cleared
independently.

TTL: 86 400 s (24 h) — same as the page cache.
"""

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

_DB_PATH = Path(os.getenv("DATA_DIR", "./data")) / "page_cache.db"
_TTL = 86_400  # 24 hours in seconds


def _ensure_table() -> None:
    """Create the answer_cache table if it doesn't exist. Idempotent."""
    os.makedirs(_DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=5)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_cache (
                key        TEXT PRIMARY KEY,
                answer     TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _make_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def answer_cache_get(query: str) -> Optional[str]:
    """Return the cached answer for *query* if it exists and has not expired.

    Returns None when the entry is absent, expired, or the query is "recache".
    """
    if query.strip().lower() == "recache":
        return None
    _ensure_table()
    key = _make_key(query)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=5)
    try:
        cursor = conn.execute(
            "SELECT answer, created_at FROM answer_cache WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        answer, created_at = row
        if (time.time() - created_at) > _TTL:
            return None
        return answer
    finally:
        conn.close()


_FALLBACK_PREFIXES = (
    "I searched the 0G documentation but could not produce a complete answer",
    "I was unable to find an answer",
)


def answer_cache_set(query: str, answer: str) -> None:
    """Store *answer* for *query*. Uses INSERT OR REPLACE.

    Skips empty answers, "recache" queries, and known fallback/error responses
    so degraded answers are never served from cache on a future request.
    """
    if not answer:
        return
    if query.strip().lower() == "recache":
        return
    if any(answer.startswith(prefix) for prefix in _FALLBACK_PREFIXES):
        return
    _ensure_table()
    key = _make_key(query)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=5)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO answer_cache (key, answer, created_at) VALUES (?, ?, ?)",
            (key, answer, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def answer_cache_clear() -> str:
    """Delete all rows from answer_cache.

    Returns a message of the form "Answer cache cleared (N entries)."
    """
    _ensure_table()
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=5)
    try:
        cursor = conn.execute("DELETE FROM answer_cache")
        conn.commit()
        return f"Answer cache cleared ({cursor.rowcount} entries)."
    finally:
        conn.close()
