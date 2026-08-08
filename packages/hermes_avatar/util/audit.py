"""Canonical structured audit-event helper used by every resilience-aware adapter.

The LiveEmote codebase already had a half-built convention - ``logger.warning(...,
extra={"audit": {"event": "..."}})`` - scattered across ``apps/demo_server/main.py``
and ``apps/demo_server/routes.py`` and inside the ``CircuitBreaker`` itself. This
module promotes that convention to a single helper so:

* every audit-emitting site produces a uniform JSON-shaped payload
  (``{"audit": {"event": "<name>.<kind>", "name": "<name>", "kind": "<kind>",
  ...fields}}``) - identical shape for prom / JSON-log / future consumers,
* a process-wide (thread-safe) per-name counter cache is bumped on every event
  so ``/api/health`` and tests can read the rolling state without re-parsing
  the log stream,
* audit-emitting code stays a one-liner: ``audit_event("breaker.luxtts",
  "trip", level=logging.WARNING, failure_count=3)`` instead of a hand-rolled
  ``logger.warning(... extra={"audit": {...}})`` every time.

The counter cache is deliberately a tiny in-memory snapshot rather than a
prom counter: it's the source for :func:`snapshot()`, which is consumed by
``/api/health`` and by the new resilience tests. A prom counter sits alongside
it via the existing ``prometheus_client`` Counter primitives in the adapters
and is unaffected by this helper.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Canonical event-kind taxonomy. Centralising the strings means tests and
# downstream consumers (runbook, log filters, prom-rename maps) reference a
# single source of truth instead of grepping for ad-hoc event names.
KIND_TRIP = "trip"
KIND_RECOVER = "recover"
KIND_HALF_OPEN = "half_open"
KIND_VENDOR_FALLBACK = "vendor_fallback"
KIND_COST_CAP_EXCEEDED = "cost_cap_exceeded"
KIND_RETRY_EXHAUSTED = "retry_exhausted"
KIND_RETRY_SCHEDULED = "retry_scheduled"
KIND_STARTUP_DEGRADED = "startup_degraded"
KIND_HTTP_REQUEST = "http_request"

# Thread-safe per-name counter cache. Each entry has the shape:
#   {
#       "events_total":        int,
#       "last_event_at":       float | None,   # time.time() of the latest event
#       "last_event_kind":     str | None,
#       "last_event_error":    str | None,    # str of last `error=` field, if any
#       "last_event_extra":    dict[str, Any],# other fields from the latest event
#   }
_LOCK = threading.Lock()
_EVENTS: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "events_total": 0,
        "last_event_at": None,
        "last_event_kind": None,
        "last_event_error": None,
        "last_event_extra": {},
    }
)


_DEFAULTS = {
    "events_total": 0,
    "last_event_at": None,
    "last_event_kind": None,
    "last_event_error": None,
}


def _snapshot_locked(name: str) -> dict[str, Any]:
    """Return a sanitized view of one counter record.

    A name that has never seen an event must still report ``events_total=0``
    + ``None`` timestamps so callers can compose uniform snapshots -- tests
    assert ``events_total == 0`` after a reset, so this default matters.
    """
    record = _EVENTS.get(name, _DEFAULTS)
    out = {k: record.get(k, _DEFAULTS[k]) for k in _DEFAULTS}
    out["last_event_extra"] = dict(record.get("last_event_extra") or {})
    return out


def audit_event(
    name: str,
    kind: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one canonical structured audit log line and bump the per-name counter.

    Parameters
    ----------
    name:
        The emitting subsystem (``"breaker.luxtts"``, ``"voice.luxtts"``,
        ``"protocol.openai"``, ``"startup"``, ...). One process can carry
        counters for many subsystems; the dict is keyed by ``name``.
    kind:
        A short verb describing what happened (``"trip"``, ``"recover"``,
        ``"vendor_fallback"``, ``"cost_cap_exceeded"``, ...). Use one of the
        ``KIND_*`` constants above so dashboards stay aligned.
    level:
        Standard ``logging`` level. Default ``INFO``; the breaker emits
        ``WARNING`` on trips, ``openai`` emits ``WARNING`` on retry exhaustion.
    **fields:
        Per-event extras (e.g. ``failure_count=3``, ``error="...timeout..."``,
        ``latency_ms=412.7``). They are folded into the structured payload
        (``extra={"audit": {...}}``) AND mirrored onto the counter cache
        snapshot so /api/health can read them.

    Notes
    -----
    The log record's ``extra={"audit": ...}`` payload matches the shape that
    existed in the codebase before this helper (so existing log filters keep
    working); the helper's only NEW contribution is the per-name counter cache
    + the standardisation of the event string (``f"{name}.{kind}"``).
    """
    payload: dict[str, Any] = {"event": f"{name}.{kind}", "kind": kind, "name": name}
    for k, v in fields.items():
        # Avoid shadowing the two reserved keys so an errant `event=...`
        # call cannot drop the canonical event string.
        if k in ("event",):
            continue
        payload[k] = v
    logger.log(level, "audit %s.%s", name, kind, extra={"audit": payload})

    with _LOCK:
        rec = _EVENTS[name]
        rec["events_total"] += 1
        rec["last_event_at"] = time.time()
        rec["last_event_kind"] = kind
        if "error" in fields:
            rec["last_event_error"] = str(fields["error"])
        rec["last_event_extra"] = {k: v for k, v in fields.items() if k != "error"}


def snapshot(name: str | None = None) -> dict[str, Any]:
    """Return a copy of the per-name counter cache for one name (or all of them).

    The returned dict is independent of the internal cache, so callers can
    mutate it freely without affecting subsequent ``audit_event`` calls.
    Pass ``name=None`` to retrieve every counters record keyed by name.
    """
    with _LOCK:
        if name is None:
            return {n: _snapshot_locked(n) for n in list(_EVENTS.keys())}
        return _snapshot_locked(name)


def reset(*names: str | None) -> None:
    """Clear the per-name counter cache; intended for test isolation.

    Accepts zero or more names; no args means clear every counter. Passing a
    single ``None`` is treated the same as no args (clear-all).
    """
    with _LOCK:
        if not names or names == (None,):
            _EVENTS.clear()
        else:
            for n in names:
                if n is not None:
                    _EVENTS.pop(n, None)


# Backward-compatible name used by the health endpoint and older integrations.
# Keep ``snapshot`` as the canonical implementation while allowing callers to
# use the more explicit audit-specific spelling.
audit_snapshot = snapshot


def consume_recent(name: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return an OK-level log-stream snapshot stub.

    Currently returns the latest ``events_total`` count + last_event_* fields as
    a single-element list. Reserved for a future upgrade that tails the log
    stream; for now it gives operators a uniform read-side API surface so
    downstream code doesn't have to special-case ``snapshot()`` vs a future
    stream tail.
    """
    s = snapshot(name)
    if not s.get("events_total"):
        return []
    return [
        {
            "name": name,
            "kind": s.get("last_event_kind"),
            "at": s.get("last_event_at"),
            "error": s.get("last_event_error"),
            "extra": s.get("last_event_extra") or {},
        }
    ][: max(0, int(limit))]
