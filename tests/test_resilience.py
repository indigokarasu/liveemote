from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from hermes_avatar.util import (
    CLOSED,
    OPEN,
    HALF_OPEN,
    CircuitBreaker,
    compute_backoff_delay,
    is_retryable_error,
)


def test_breaker_starts_closed():
    cb = CircuitBreaker()
    assert cb.state == CLOSED
    assert cb.allow() is True


def test_breaker_trips_open_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, open_timeout=60.0)
    for _ in range(3):
        assert cb.allow() is True
        cb.record_failure()
    assert cb.state == OPEN
    # While open (window not elapsed) allow() refuses.
    assert cb.allow() is False


def test_breaker_half_open_after_timeout_then_recovers():
    cb = CircuitBreaker(failure_threshold=2, open_timeout=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == OPEN
    # Window not elapsed yet.
    assert cb.allow() is False
    time.sleep(0.08)
    # Now transitions to half-open and allows the probe.
    assert cb.allow() is True
    assert cb.state == HALF_OPEN
    cb.record_success()
    assert cb.state == CLOSED
    assert cb.snapshot()["failure_count"] == 0


def test_breaker_half_open_probe_failure_reopens():
    cb = CircuitBreaker(failure_threshold=1, open_timeout=0.05)
    cb.record_failure()
    assert cb.state == OPEN
    time.sleep(0.08)
    assert cb.allow() is True  # half-open probe allowed
    cb.record_failure()
    assert cb.state == OPEN


def test_breaker_reset_on_success_in_closed():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    assert cb.snapshot()["failure_count"] == 1
    cb.record_success()  # closed success resets count
    assert cb.snapshot()["failure_count"] == 0
    assert cb.state == CLOSED


def test_compute_backoff_grows_exponentially_and_caps():
    rng = MagicMock()
    rng.random.return_value = 0.5  # -> jitter term = 0
    d0 = compute_backoff_delay(0, base_delay=0.5, max_delay=4.0, jitter_factor=0.1, rng=rng)
    d1 = compute_backoff_delay(1, base_delay=0.5, max_delay=4.0, jitter_factor=0.1, rng=rng)
    d3 = compute_backoff_delay(3, base_delay=0.5, max_delay=4.0, jitter_factor=0.1, rng=rng)
    assert d0 == 0.5
    assert d1 == 1.0
    # attempt 3 would be 0.5*8=4.0, capped at 4.0
    assert d3 == 4.0
    # monotonic non-decreasing up to the cap
    assert d0 <= d1 <= d3


def test_compute_backoff_jitter_is_bounded():
    rng = MagicMock()
    base_delay, max_delay, jf = 1.0, 10.0, 0.1
    # rng.random()==0 -> jitter = -jf; ==1 -> +jf
    for attempt in range(5):
        rng.random.return_value = 0.0
        lo = compute_backoff_delay(attempt, base_delay, max_delay, jf, rng=rng)
        rng.random.return_value = 1.0
        hi = compute_backoff_delay(attempt, base_delay, max_delay, jf, rng=rng)
        expected = min(base_delay * (2 ** attempt), max_delay)
        assert lo == pytest.approx(expected * (1 - jf))
        assert hi == pytest.approx(expected * (1 + jf))
        assert lo >= 0


def test_is_retryable_http_status_codes():
    class FakeResp:
        def __init__(self, code):
            self.status_code = code

    class FakeExc(Exception):
        pass

    for code in (500, 502, 503, 429):
        exc = FakeExc()
        exc.response = FakeResp(code)
        assert is_retryable_error(exc) is True
    ok = FakeExc()
    ok.response = FakeResp(200)
    assert is_retryable_error(ok) is False
    assert is_retryable_error(FakeExc("boom")) is False


def test_is_retryable_network_keywords():
    for msg in (
        "ConnectionError: connection refused",
        "ReadTimeout: timeout",
        "Service Unavailable",
        "502 Bad Gateway",
        "Gateway Timeout",
    ):
        assert is_retryable_error(RuntimeError(msg)) is True
    assert is_retryable_error(RuntimeError("validation failed")) is False
