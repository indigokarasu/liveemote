"""Resilience tests for the OpenAI-compatible adapter circuit breaker.

These tests lock down the protection against cascading vendor outages
on the LLM-side of the LiveEmote pipeline, mirroring the LuxTTS
circuit-breaker pattern. Without a breaker, every ``/api/event``
Hermes-response call would re-spawn a fresh HTTP POST to whatever
chat-completions endpoint is configured (OpenAI, SambaNova,
llama.cpp, vLLM), burning API credits and 20-second block-and-
timeout cycles during a provider outage.

With the breaker, the 3rd consecutive failure (any combination of
``HTTPStatusError`` / network error / JSON parse failure) trips
``OPEN`` for 30s. During that window the adapter fast-fails into the
same offline ``AgentResponse`` shape the existing ``except Exception``
arm emits -- the avatar's reflect-only mirror-mode behavior is
unchanged from the user view. The single recovery probe after 30s
closes the breaker again on a 200 OK.

Test surface:

* success keeps the breaker CLOSED,
* three consecutive 503 responses trip OPEN,
* an OPEN breaker short-circuits ``httpx.AsyncClient`` entirely --
  zero outbound HTTP POSTs,
* after the open window elapses, a single successful probe closes
  the breaker again (HALF_OPEN -> CLOSED),
* ``capability_status()`` exposes the breaker snapshot so the demo
  server's ``/api/health`` surface can report it alongside the LuxTTS
  and renderer breakers.

Mirrors ``tests/test_luxtts_resilience.py`` in structure; the
mocking shape differs because ``httpx.AsyncClient`` is instantiated
locally inside ``generate_response`` (no transport injection point).
We patch ``httpx.AsyncClient`` to a ``MagicMock`` whose ``__aenter__``
returns itself and whose ``.post`` returns a pre-built ``httpx.Response``.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hermes_avatar.affect.state import UserAffectState
from hermes_avatar.protocol.openai_adapter import (
    AdapterConfig,
    OpenAICompatibleAdapter,
)
from hermes_avatar.util import CircuitBreaker, OPEN, CLOSED


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


def _ok_response(body: dict | None = None) -> httpx.Response:
    """A valid 200 OK OpenAI-shaped response wrapping a JSON MODE response.

    The ``request`` instance is REQUIRED on every mock response --
    ``httpx.Response.raise_for_status()`` aborts with a confusing
    ``Request instance has not been set`` message otherwise, regardless
    of status code. The same fix lets HTTPStatusError fire correctly on
    4xx/5xx so the adapter's ``except httpx.HTTPStatusError`` branch
    actually runs in tests.
    """
    payload = body or {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"text": "hi", "tags": {"voice": {}}}
                    ),
                },
            },
        ],
    }
    return httpx.Response(
        200,
        request=_REQUEST,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _error_response(code: int, body: str = "Service Unavailable") -> httpx.Response:
    return httpx.Response(
        code,
        request=_REQUEST,
        content=body.encode("utf-8"),
        headers={"content-type": "text/plain"},
    )


_REQUEST = httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def _stub_httpx_post_response(response: httpx.Response):
    """Return a ``patch`` that swaps ``httpx.AsyncClient`` for a context-manager shim.

    The shim's ``__aenter__`` returns itself, ``__aexit__`` returns False
    (does not swallow the exception), and ``post`` returns the supplied
    ``response``. Use the returned ``patch`` as a context manager and
    check ``<mc>.call_count`` for "was AsyncClient() ever instantiated?"
    and ``<mc>.return_value.post.call_count`` for "was .post invoked?".
    """
    instance = MagicMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    instance.post = AsyncMock(return_value=response)
    return patch("httpx.AsyncClient", return_value=instance)


def _stub_httpx_post_events(events: list):
    """Return a ``patch`` where ``.post`` serves ``events`` FIFO.

    Each event is either:
      - a real ``httpx.Response`` (returned to the adapter)
      - an ``Exception`` instance (raised on the adapter's try-block)

    The shim records every ``(url, kwargs)`` call on ``instance.calls``
    so a test can assert ``len(<mc>.return_value.calls) == N`` for the
    exact number of outbound HTTP attempts the retry loop issued.
    Asserting here is the cleanest possible lock on the retry-loop math.
    """
    queue = list(events)
    instance = MagicMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    instance.calls: list = []

    async def post(url, **kwargs):
        instance.calls.append((url, kwargs))
        if not queue:
            raise RuntimeError(
                f"unexpected extra post() call "
                f"(already served {len(instance.calls)} of {len(events)})"
            )
        ev = queue.pop(0)
        if isinstance(ev, BaseException):
            raise ev
        return ev

    instance.post = post
    return patch("httpx.AsyncClient", return_value=instance)


def _configured_user() -> UserAffectState:
    return UserAffectState(
        face_detected=True,
        attention=0.7,
        valence=0.3,
        arousal=0.4,
        dominant_expression="happy",
    )


@pytest.fixture
def adapter() -> OpenAICompatibleAdapter:
    """An adapter wired to a real-looking base URL + a short-window test breaker.

    ``open_timeout`` is set to ``0.05`` so a single ``time.sleep(0.08)``
    advances the open window in the HALF_OPEN recovery test. ``retry_base_delay``
    is set to ``0.001`` so retry-loop tests don't sleep for the production
    default of ``0.5s`` per attempt. Production uses ``open_timeout=30.0``
    and ``retry_base_delay=0.5``.
    """
    a = OpenAICompatibleAdapter(
        AdapterConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            # 1ms so tests' retry loop is effectively instant.
            retry_base_delay=0.001,
        )
    )
    a.cb = CircuitBreaker(
        failure_threshold=3,
        open_timeout=0.05,
        name="openai-test",
    )
    return a


def test_success_keeps_breaker_closed(adapter):
    with _stub_httpx_post_response(_ok_response()) as mc:
        out = _run(adapter.generate_response("hello", _configured_user()))
    assert mc.call_count == 1
    assert mc.return_value.post.call_count == 1
    assert adapter.cb.state == CLOSED
    assert adapter.cb.snapshot()["failure_count"] == 0
    assert adapter.last_error is None
    assert out.text == "hi"
    assert out.source == "openai_compatible"


def test_three_failures_trip_breaker_open(adapter):
    err = _error_response(503, "Service Unavailable")
    with _stub_httpx_post_response(err) as mc:
        # Three distinct user_text inputs -- no payload cache, three user-requests.
        for i in range(3):
            out = _run(adapter.generate_response(f"hi-{i}", _configured_user()))
            assert out.text == ""
            assert out.source == "offline"
    # Each user-request retries up to 3 times (1 initial + 2 retries) on a
    # retryable 503. So 3 user-requests * 3 attempts each = 9 HTTP calls in
    # total before the breaker is OPEN. Critical assertion below locks the
    # "ONE breaker failure per user-request" contract: 3 user-requests produce
    # exactly 3 breaker events -- NOT 9. So failure_count is 3 (= threshold).
    assert mc.call_count == 9
    assert adapter.cb.state == OPEN
    snap = adapter.cb.snapshot()
    assert snap["failure_count"] == 3
    assert adapter.last_error is not None
    assert "503" in adapter.last_error


def test_open_short_circuits_httpx(adapter):
    # Pre-trip manually so we exercise ONLY the breaker-gate branch -- no
    # background httpx noise from the trip itself.
    adapter.cb.record_failure()
    adapter.cb.record_failure()
    adapter.cb.record_failure()
    assert adapter.cb.state == OPEN
    adapter.last_error = None

    with _stub_httpx_post_response(_ok_response()) as mc:
        out = _run(adapter.generate_response("hi", _configured_user()))
    # The breaker gate fired before ANY httpx instantiation.
    mc.assert_not_called()
    mc.return_value.post.assert_not_called()
    assert out.text == ""
    assert out.source == "offline"
    # And the reason for short-circuit is captured in last_error for ops.
    assert adapter.last_error is not None
    assert "breaker" in adapter.last_error.lower()


def test_half_open_recovery_closes_breaker(adapter):
    # Trip via real 503 responses so the failure_count is realistic.
    err = _error_response(503, "Service Unavailable")
    with _stub_httpx_post_response(err) as trip_mc:
        for i in range(3):
            _run(adapter.generate_response(f"trip-{i}", _configured_user()))
    # 3 user-requests * 3 attempts each = 9 outbound HTTP calls during the trip phase.
    assert trip_mc.call_count == 9
    assert adapter.cb.state == OPEN
    assert adapter.cb.snapshot()["failure_count"] == 3

    # Open window elapses (``open_timeout=0.05`` in the fixture).
    time.sleep(0.08)

    # Probe -- single 200 OK should close the breaker on the first attempt
    # (no retry needed for a 200).
    with _stub_httpx_post_response(_ok_response()) as probe_mc:
        out = _run(adapter.generate_response("probe", _configured_user()))
    assert probe_mc.call_count == 1
    assert probe_mc.return_value.post.call_count == 1
    assert out.text == "hi"
    assert out.source == "openai_compatible"
    assert adapter.cb.state == CLOSED
    assert adapter.cb.snapshot()["failure_count"] == 0
    assert adapter.last_error is None


def test_capability_status_includes_breaker_snapshot(adapter):
    status = adapter.capability_status()
    assert "circuit_breaker" in status
    snap = status["circuit_breaker"]
    assert snap["name"] == "openai-test"
    # Fresh adapter: CLOSED
    assert snap["state"] == CLOSED

    # One failure -- count reflects it WITHOUT tripping (threshold=3).
    adapter.cb.record_failure()
    tripped = adapter.capability_status()
    assert tripped["circuit_breaker"]["failure_count"] >= 1
    assert tripped["circuit_breaker"]["state"] == CLOSED


# ---------------------------------------------------------------------------
# Retry-loop tests (IMPROVEMENTS 2.2). These lock the contract that the
# retry-with-jitter path runs INSIDE the breaker gate, retries up to
# ``max_retries+1`` times total (1 initial + 2 retries), ONLY retries on
# transient errors (5xx / 429 / network), and records ONE breaker failure
# per user-request even when ALL retry attempts fail.
# ---------------------------------------------------------------------------


def test_retry_on_transient_503_then_success(adapter):
    """Two transient 503s, then a 200 -- retry-loop succeeds on the 3rd attempt."""
    events = [
        _error_response(503, "Service Unavailable"),
        _error_response(503, "Service Unavailable"),
        _ok_response(),
    ]
    with _stub_httpx_post_events(events) as mc:
        out = _run(adapter.generate_response("hello", _configured_user()))
    # 3 HTTP attempts total: 1 initial + 2 retries. Lock the exact retry math.
    assert len(mc.return_value.calls) == 3
    # Final attempt succeeded; breaker recorded one success so count resets.
    assert adapter.cb.state == CLOSED
    assert adapter.cb.snapshot()["failure_count"] == 0
    assert adapter.last_error is None
    assert out.text == "hi"
    assert out.source == "openai_compatible"


def test_retry_exhaustion_records_single_breaker_failure(adapter):
    """All 3 attempts return 503 -- one breaker failure, not three.

    This is the critical user-spec contract: ``record one breaker outcome
    per user-request not per attempt``. With 3 user-requests at threshold=3
    we reach OPEN (see test_three_failures_trip_breaker_open); with 1
    user-request we stay CLOSED but still record exactly one breaker
    failure, regardless of how many attempts the retry loop issued.
    """
    events = [
        _error_response(503, "Service Unavailable"),
        _error_response(503, "Service Unavailable"),
        _error_response(503, "Service Unavailable"),
    ]
    with _stub_httpx_post_events(events) as mc:
        out = _run(adapter.generate_response("hello", _configured_user()))
    assert len(mc.return_value.calls) == 3
    # The breaker contract: ONE failure for the whole failed user-request,
    # not 3 (one per attempt). Threshold=3 so we stay CLOSED.
    assert adapter.cb.state == CLOSED
    assert adapter.cb.snapshot()["failure_count"] == 1
    assert out.text == ""
    assert out.source == "offline"
    assert "503" in (adapter.last_error or "")


def test_non_retryable_4xx_no_retry(adapter):
    """400 Bad Request -- not retryable, no backoff, breaker records 1 failure."""
    events = [_error_response(400, "Bad Request")]
    with _stub_httpx_post_events(events) as mc:
        out = _run(adapter.generate_response("hello", _configured_user()))
    # 400 -> ``is_retryable_error()`` returns False -> exactly ONE outbound call.
    assert len(mc.return_value.calls) == 1
    assert adapter.cb.state == CLOSED
    assert adapter.cb.snapshot()["failure_count"] == 1
    assert out.text == ""
    assert out.source == "offline"
    assert "400" in (adapter.last_error or "")

