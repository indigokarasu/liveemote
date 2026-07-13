from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Per-request/per-task trace id, propagated via contextvars across the
# perception -> affect -> behavior -> rendering pipeline. Using contextvars
# (rather than threading locals) keeps the value correct under asyncio where
# a single OS thread serves many concurrent requests, each in its own context.
TRACE_ID_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_trace_id", default=None
)


def get_trace_id() -> Optional[str]:
    """Return the trace id for the current execution context, if any."""
    return TRACE_ID_VAR.get()


def set_trace_id(value: Optional[str]) -> None:
    """Set the trace id for the current execution context."""
    TRACE_ID_VAR.set(value)


def new_trace_id() -> str:
    """Generate a fresh, globally-unique trace id."""
    return uuid.uuid4().hex


def ensure_trace_id() -> str:
    """Return the current trace id, generating and storing one if absent.

    Useful as the entry point for a request: call it once and the same id
    will be visible in every downstream log line and the affect runtime tick
    that runs within this context.
    """
    existing = TRACE_ID_VAR.get()
    if existing:
        return existing
    fresh = new_trace_id()
    TRACE_ID_VAR.set(fresh)
    return fresh


class trace_span:
    """Lightweight context manager that guarantees a trace id is present.

    It does not create a new id if one already exists in the context, so it
    composes cleanly with the request middleware (which sets the id first).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.trace_id: Optional[str] = None

    def __enter__(self) -> "trace_span":
        self.trace_id = ensure_trace_id()
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def traced_log(level: int, message: str, **fields: object) -> None:
    """Emit a log line that always carries the active trace id in structured form.

    Falls back to a sentinel when no trace id is set so that logs from
    background tasks / non-request contexts remain correlatable as "no trace".
    """
    tid = get_trace_id() or "none"
    logger.log(
        level,
        "%s trace_id=%s",
        message,
        tid,
        extra={"trace_id": tid, "audit": {"event": message, **fields}},
    )
