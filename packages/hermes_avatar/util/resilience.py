"""Resilience primitives shared across external-service adapters.

This module centralises two cross-cutting concerns that used to be
copy-pasted (and subtly diverged) inside individual adapters:

* :class:`CircuitBreaker` -- a thread-safe closed/open/half-open state
  machine that stops hammering a dead dependency and probes it once
  it may have recovered.
* :func:`compute_backoff_delay` and :func:`is_retryable_error` --
  the exponential-backoff-with-jitter math and transient-error
  classification used when retrying.

Adapters that talk to networks (the LiveTalking renderer, the
ElevenLabs voice backend) import from here so behaviour is consistent
and unit-testable in isolation, without touching the network.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half-open"


class CircuitBreaker:
    """Thread-safe closed/open/half-open breaker for external calls.

    The breaker starts ``CLOSED``. After ``failure_threshold``
    consecutive failures it trips to ``OPEN`` and refuses work (callers
    should fail fast or degrade gracefully). Once ``open_timeout``
    seconds have elapsed it moves to ``HALF_OPEN`` and allows a single
    probe; a success closes it again, a failure re-opens it.

    All mutating operations are guarded by a lock so the breaker is safe
    to share across the concurrent request threads a demo server spawns.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        open_timeout: float = 60.0,
        name: str = "breaker",
    ) -> None:
        self._name = name
        self._failure_threshold = max(1, int(failure_threshold))
        self._open_timeout = float(open_timeout)
        self._lock = threading.Lock()
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = CLOSED

    # -- state access (always under lock) -----------------------------------
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def snapshot(self) -> dict[str, Any]:
        """Structured view for ``capabilities()`` / health probes."""
        with self._lock:
            return {
                "state": self._state,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time,
                "name": self._name,
            }

    # -- decision points -----------------------------------------------------
    def allow(self) -> bool:
        """Return ``True`` if a request may proceed right now.

        Transitions ``OPEN`` -> ``HALF_OPEN`` once the open window has
        elapsed so the next call becomes a single recovery probe.
        """
        with self._lock:
            if self._state == CLOSED:
                return True
            if self._state == HALF_OPEN:
                return True
            # OPEN
            if (
                self._last_failure_time is not None
                and (time.time() - self._last_failure_time) > self._open_timeout
            ):
                self._state = HALF_OPEN
                logger.info(
                    "circuit breaker probing (half-open)",
                    extra={"audit": {"event": "cb.half_open", "name": self._name}},
                )
                return True
            return False

    def record_success(self) -> None:
        """Close the breaker (from half-open) or reset the failure count."""
        with self._lock:
            if self._state == HALF_OPEN:
                self._state = CLOSED
                self._failure_count = 0
                self._last_failure_time = None
                logger.info(
                    "circuit breaker recovered (closed)",
                    extra={"audit": {"event": "cb.recovered", "name": self._name}},
                )
            else:
                # CLOSED: clear any accumulated count.
                self._failure_count = 0
                self._last_failure_time = None

    def record_failure(self) -> None:
        """Record a failure and trip ``OPEN`` once the threshold is hit."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = OPEN
                logger.warning(
                    "circuit breaker tripped (open)",
                    extra={
                        "audit": {
                            "event": "cb.open",
                            "name": self._name,
                            "failure_count": self._failure_count,
                        }
                    },
                )


def is_retryable_error(exc: Exception) -> bool:
    """Classify an exception as a transient, retry-worthy failure.

    Retries on 5xx server errors and 429 rate-limiting, plus common
    network / transient conditions detected by substring. Non-retryable
    errors (auth, validation, 4xx client errors other than 429) are
    surfaced immediately so we don't burn retries on a doomed call.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and (500 <= status_code < 600 or status_code == 429):
            return True

    error_str = str(exc).lower()
    retryable_keywords = (
        "connection",
        "timeout",
        "temporarily",
        "temporary",
        "unavailable",
        "service unavailable",
        "gateway timeout",
        "bad gateway",
        "network",
    )
    return any(k in error_str for k in retryable_keywords)


def compute_backoff_delay(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    jitter_factor: float = 0.1,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff capped at ``max_delay`` with +/- jitter.

    ``attempt`` is the zero-based retry index (``0`` == first retry).
    The base delay doubles each attempt; a small proportional dither
    avoids synchronised retry storms across concurrent callers.
    """
    rng = rng or random
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * jitter_factor * (2 * rng.random() - 1)  # +/- jitter_factor
    return max(0.0, delay + jitter)
