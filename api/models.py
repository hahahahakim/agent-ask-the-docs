"""
Pydantic request / response models for the chat API.

Input validation is the first line of defence against malformed or
malicious payloads. Every field is validated before the request reaches
any business logic.
"""

import re
import uuid

from pydantic import BaseModel, Field, field_validator

# Matches UUIDs (e.g. from uuid4) and Telegram IDs (tg-12345678)
_THREAD_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("query")
    @classmethod
    def _strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank after stripping whitespace")
        return v

    @field_validator("thread_id")
    @classmethod
    def _validate_thread_id(cls, v: str) -> str:
        """
        Enforce a strict allowlist on thread_id to prevent path traversal or
        injection attacks against the MemorySaver key space.
        """
        if not _THREAD_ID_RE.match(v):
            raise ValueError(
                "thread_id must be 1–128 characters: letters, digits, hyphens, underscores only"
            )
        return v


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    model: str
    duration_s: float
