from __future__ import annotations

import time

import pytest

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.smoothing import ema
from hermes_avatar.config.schema import load_config


@pytest.mark.performance
def test_affect_runtime_tick_latency_bounded():
    """1000 ticks of the real AffectRuntime must complete quickly and not
    grow unboundedly (no linear accumulation in the hot path)."""
    # Skip gracefully if the real config cannot be built (needs defaults.yaml).
    try:
        cfg = load_config()
        rt = AffectRuntime(config=cfg)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"AffectRuntime unavailable: {exc}")

    N = 1000
    THRESHOLD_S = 2.0

    # Warm up one tick so module import / first-call cost is excluded.
    rt.tick(0)

    # First half baseline latency.
    t0 = time.perf_counter()
    for i in range(N // 2):
        rt.tick(i)
    first_half = time.perf_counter() - t0

    # Second half — must not be dramatically slower (no unbounded growth).
    t1 = time.perf_counter()
    for i in range(N // 2, N):
        rt.tick(i)
    second_half = time.perf_counter() - t1

    total = first_half + second_half
    assert total < THRESHOLD_S, f"{total:.3f}s for {N} ticks"
    # Guard against latency blow-up: second half within 5x of first.
    if first_half > 0:
        assert second_half < first_half * 5 + 0.5


@pytest.mark.performance
def test_smoothing_ema_latency_bounded():
    """1000 EMA smoothing updates must complete well under 1s."""
    N = 1000
    THRESHOLD_S = 1.0
    t0 = time.perf_counter()
    prev = 0.0
    for i in range(N):
        prev = ema(prev, (i % 100) / 100.0, 0.3)
    elapsed = time.perf_counter() - t0
    assert elapsed < THRESHOLD_S, f"{elapsed:.3f}s for {N} ema calls"


@pytest.mark.performance
def test_event_consume_throughput():
    """Drive the runtime via consume() for many events under a bound."""
    cfg = load_config()
    rt = AffectRuntime(config=cfg)
    N = 500
    t0 = time.perf_counter()
    for i in range(N):
        rt.consume({
            "type": "perception.frame",
            "face_detected": True,
            "expression": {"smile": 0.5, "eye_open": 0.6},
            "timestamp_ms": i * 16,
        })
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"{elapsed:.3f}s for {N} consume() calls"
