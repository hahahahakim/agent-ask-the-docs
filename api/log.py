"""Structured log formatter, logging configuration, and per-request log helpers."""

import logging
from datetime import datetime, timezone
from typing import FrozenSet, Optional

from fastapi import Request


class StructuredFormatter(logging.Formatter):
    """Header line + indented key=value pairs for every extra= kwarg."""

    _COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET = "\033[0m"

    # All attributes present on every LogRecord — never treat these as extras.
    _STANDARD_ATTRS: FrozenSet[str] = frozenset({
        "args", "asctime", "color_message", "created", "exc_info", "exc_text",
        "filename", "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        ts = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d} +0000"

        color = self._COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname}{self._RESET}" if color else record.levelname

        lines = [f"[{ts}] {level}: {record.message}"]
        for key, val in record.__dict__.items():
            if key not in self._STANDARD_ATTRS:
                lines.append(f'    {key}: "{val}"')

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            lines.append(record.exc_text)

        return "\n".join(lines)


def configure_logging() -> None:
    """Apply StructuredFormatter to uvicorn and api loggers; suppress the access log."""
    fmt = StructuredFormatter()

    for name in ("uvicorn", "uvicorn.error"):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(fmt)

    # Suppress uvicorn's built-in access log — route handlers emit structured logs instead.
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = []
    access_logger.propagate = False

    api_logger = logging.getLogger("api")
    api_logger.handlers.clear()   # prevent duplicate handlers if called more than once
    api_logger.propagate = False  # prevent double-logging via the root logger
    api_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    api_logger.addHandler(handler)


def log_request(
    logger: logging.Logger,
    request: Request,
    *,
    thread_id: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Log an incoming request (method, path, ip, optional threadId)."""
    extra: dict = {
        "method": request.method,
        "path": request.url.path,
        "ip": request.client.host if request.client else "-",
    }
    if thread_id is not None:
        extra["threadId"] = thread_id
    logger.log(level, "Incoming request", extra=extra)


def log_completed(logger: logging.Logger, thread_id: str, duration: float) -> None:
    """Log a completed request (threadId, duration)."""
    logger.info("Request completed", extra={
        "threadId": thread_id,
        "duration": f"{duration:.2f}s",
    })
