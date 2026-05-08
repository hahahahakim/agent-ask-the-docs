"""
Persistence utilities for the 0G Labs docs agent.

Provides:
  get_checkpointer()          — MemorySaver for LangGraph conversation history
  ThreadTracker               — lazy TTL expiry for conversation threads
"""

import os
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_checkpointer():
    """Return a MemorySaver for LangGraph conversation history.

    Conversation history is kept for the lifetime of the process.
    SQLite-backed persistence will be added once langgraph-checkpoint-sqlite
    resolves its aiosqlite compatibility issue.
    """
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


class ThreadTracker:
    """Lazy TTL expiry for conversation threads.

    Tracks the last-active timestamp for each thread_id.  On every request,
    call ``maybe_expire()`` before running the agent — if the thread has been
    inactive longer than the TTL, its checkpoints are wiped from the
    MemorySaver and the conversation starts fresh.

    Note: this is in-process state (like MemorySaver itself).  It is
    consistent within a single worker process.  With multiple workers each
    tracker would be independent — another reason to keep --workers 1.
    """

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._last_active: Dict[str, float] = {}

    def touch(self, thread_id: str) -> None:
        """Record that thread_id was just active."""
        self._last_active[thread_id] = time.time()

    def is_expired(self, thread_id: str) -> bool:
        """Return True if the thread exists and has been inactive past TTL."""
        ts = self._last_active.get(thread_id)
        if ts is None:
            return False  # new thread — not expired
        return (time.time() - ts) > self.ttl_seconds

    def _wipe(self, thread_id: str, checkpointer) -> None:
        """Delete all MemorySaver checkpoints for thread_id."""
        # MemorySaver.storage keys are (thread_id, checkpoint_ns, checkpoint_id)
        if hasattr(checkpointer, "storage"):
            stale = [k for k in list(checkpointer.storage) if k[0] == thread_id]
            for k in stale:
                del checkpointer.storage[k]
        # MemorySaver also tracks pending writes
        if hasattr(checkpointer, "writes"):
            stale = [k for k in list(checkpointer.writes) if k[0] == thread_id]
            for k in stale:
                del checkpointer.writes[k]
        self._last_active.pop(thread_id, None)

    def maybe_expire(self, thread_id: str, checkpointer) -> bool:
        """Wipe thread_id if it has expired.  Returns True if it was wiped."""
        if self.is_expired(thread_id):
            self._wipe(thread_id, checkpointer)
            return True
        return False


def get_thread_tracker(ttl_hours: int = 24) -> ThreadTracker:
    return ThreadTracker(ttl_seconds=ttl_hours * 3600)


