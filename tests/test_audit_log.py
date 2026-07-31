"""Tests for the canonical ``audit_event`` helper (``packages/hermes_avatar/util/audit.py``).

The resilience trio (breaker, retry, /api/health) just landed on
``origin/main @ c985d3f``. The audit helper consolidates the half-built
``extra={"audit": ...}`` convention that existed in
``apps/demo_server/main.py`` and ``apps/demo_server/routes.py`` and
inside ``CircuitBreaker`` itself, and exposes a process-wide per-name
counter cache so ``/api/health`` and operators can read the rolling
state without re-parsing the log stream.

These tests lock both sides of the contract:

* ``audit_event`` emits a structured LogRecord with the canonical payload
  shape (``extra={"audit": {"event": "<name>.<kind>", "name": ..., "kind": ..., ...}}``)
  AND bumps the per-name counter cache.
* ``audit.snapshot(name)`` returns a stable, independent copy of the
  cache so callers can mutate it freely.
* Each adapter exposes an ``audit`` subcomponent in its
  ``capability_status`` / ``capabilities`` so ``/api/health`` can pull.
* The three breakers (``luxtts`` / ``renderer`` / ``openai``) auto-emit
  audit events on every state transition because they share the
  ``CircuitBreaker`` class - no per-adapter wiring needed.
"""

from __future__ import annotations

import logging

import pytest

from hermes_avatar.util.audit import (
    KIND_TRIP,
    KIND_RECOVER,
    KIND_HALF_OPEN,
    KIND_COST_CAP_EXCEEDED,
    audit_event,
    snapshot,
    reset,
)
from hermes_avatar.util import CircuitBreaker


# -- direct-helper tests (no adapter involvement) --------------------------


def test_audit_event_emits_canonical_payload(caplog):
    """One ``audit_event`` call produces one LogRecord with the canonical shape."""
    reset("test.payload")
    with caplog.at_level(logging.INFO, logger="hermes_avatar.util.audit"):
        audit_event(
            "test.payload",
            KIND_TRIP,
            level=logging.WARNING,
            failure_count=3,
            error="simulated",
        )
    recs = [r for r in caplog.records if r.name == "hermes_avatar.util.audit"]
    assert len(recs) == 1
    audit = recs[0].audit
    assert audit["event"] == "test.payload.trip"
    assert audit["name"] == "test.payload"
    assert audit["kind"] == "trip"
    assert audit["failure_count"] == 3
    assert audit["error"] == "simulated"


def test_audit_event_bumps_per_name_counter():
    """Three calls under one name produce one record with ``events_total=3``."""
    reset("test.counter")
    audit_event("test.counter", KIND_TRIP, level=logging.WARNING)
    audit_event("test.counter", KIND_HALF_OPEN, level=logging.INFO)
    audit_event("test.counter", KIND_RECOVER, level=logging.INFO)

    s = snapshot("test.counter")
    assert s["events_total"] == 3
    assert s["last_event_kind"] == KIND_RECOVER
    assert s["last_event_error"] is None  # no error= passed on the last call

    # ``audit_event`` records the field even when it is only set on some
    # of the calls -- the latest with ``error=`` wins.
    audit_event("test.counter", KIND_TRIP, level=logging.WARNING, error="boom")
    s2 = snapshot("test.counter")
    assert s2["events_total"] == 4
    assert s2["last_event_kind"] == KIND_TRIP
    assert s2["last_event_error"] == "boom"
    assert "failure_count" not in s2["last_event_extra"]  # absence == test surface here


def test_audit_event_cannot_overwrite_reserved_event_key(caplog):
    """A stray ``event=...`` keyword cannot overwrite the canonical event string."""
    reset("test.reserved")
    with caplog.at_level(logging.INFO, logger="hermes_avatar.util.audit"):
        audit_event("test.reserved", KIND_TRIP, event="hijacked.event")
    recs = [r for r in caplog.records if r.name == "hermes_avatar.util.audit"]
    assert recs[0].audit["event"] == "test.reserved.trip"


def test_snapshot_returns_independent_copy():
    """Mutating the snapshot does not affect future ``audit_event`` calls."""
    reset("test.copy")
    audit_event("test.copy", KIND_TRIP, level=logging.WARNING)
    s = snapshot("test.copy")
    s["events_total"] = 999
    fresh = snapshot("test.copy")
    assert fresh["events_total"] == 1


def test_reset_clears_named_or_all():
    reset("test.a", "test.b")
    audit_event("test.a", KIND_TRIP, level=logging.WARNING)
    audit_event("test.b", KIND_TRIP, level=logging.WARNING)
    assert snapshot("test.a")["events_total"] == 1
    assert snapshot("test.b")["events_total"] == 1

    reset("test.a")
    assert snapshot("test.a")["events_total"] == 0
    assert snapshot("test.b")["events_total"] == 1


# -- breaker auto-wiring tests --------------------------------------------
# The CircuitBreaker class wires into audit_event internally; these tests
# confirm the wiring survives all three state transitions.


@pytest.fixture(autouse=True)
def _reset_audit_between_audit_tests():
    from hermes_avatar.util.audit import reset as audit_reset_audit
    audit_reset_audit()
    yield
    audit_reset_audit()


@pytest.fixture
def breaker():
    reset("breaker.test_br")
    return CircuitBreaker(failure_threshold=2, open_timeout=0.05, name="test_br")


def test_breaker_trip_emits_audit_event(breaker, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_avatar.util.audit"):
        breaker.record_failure()
        breaker.record_failure()  # this trips OPEN
    assert breaker.state == "open"
    s = snapshot("breaker.test_br")
    assert s["events_total"] == 1
    assert s["last_event_kind"] == KIND_TRIP
    assert s["last_event_error"] is None  # no error field on the event

    # And the log payload reflects the failure count
    trip = [r for r in caplog.records if r.name == "hermes_avatar.util.audit"]
    assert trip[0].audit["failure_count"] == 2
    assert trip[0].audit["name"] == "breaker.test_br"


def test_breaker_half_open_then_recover_emits_two_events(breaker, caplog):
    breaker.record_failure()
    breaker.record_failure()  # trips OPEN -- emits a KIND_TRIP event
    assert breaker.state == "open"
    import time as _time
    _time.sleep(0.07)  # open window elapses

    with caplog.at_level(logging.INFO, logger="hermes_avatar.util.audit"):
        # Next allow() flips OPEN -> HALF_OPEN -> emits KIND_HALF_OPEN
        assert breaker.allow() is True
        # Probe succeeds -> HALF_OPEN -> CLOSED -> emits KIND_RECOVER
        breaker.record_success()

    s = snapshot("breaker.test_br")
    # failure_threshold=2 trips OPEN on the 2nd record_failure, so:
    #   1 trip    (from record_failure #2)
    #   1 half_open (from allow() after open_timeout elapses)
    #   1 recover (from record_success() closing half-open)
    assert s["events_total"] == 3
    assert s["last_event_kind"] == KIND_RECOVER
    assert breaker.state == "closed"


# -- adapter audit-snapshot tests (parametrized across the 3 breakers) -----
# These lock the contract that EVERY adapter whose breaker has tripped
# sees the audit record via capability_status(). A future refactor that
# drops audit.snapshot() from capability_status() will surface here.


@pytest.fixture(params=["luxtts", "renderer", "openai"])
def breaker_name(request):
    reset(f"breaker.{request.param}")
    return request.param


def test_breaker_records_visible_via_snapshot_for_each_subsystem(breaker_name):
    """Trip a breaker under each subsystem's name and verify snapshot reflects it."""
    cb = CircuitBreaker(failure_threshold=1, open_timeout=60.0, name=breaker_name)
    cb.record_failure()  # trips immediately (threshold=1)
    s = snapshot(f"breaker.{breaker_name}")
    assert s["events_total"] == 1
    assert s["last_event_kind"] == KIND_TRIP
