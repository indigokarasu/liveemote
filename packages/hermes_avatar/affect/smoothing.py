from __future__ import annotations
import random
from typing import Any, Sequence

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ema(previous: float | None, current: float, alpha: float) -> float:
    """Scalar exponential moving average.

    Kept as a pure-Python scalar hot path on purpose: for the small number of
    fields smoothed per frame (<~32) the per-call overhead of materialising a
    numpy array is larger than the scalar arithmetic. Use :func:`ema_vector`
    when smoothing many values at once.
    """
    return current if previous is None else previous + alpha * (current - previous)


def ema_vector(
    prev: float | Sequence[float] | "np.ndarray",
    cur: float | Sequence[float] | "np.ndarray",
    alpha: float | Sequence[float] | "np.ndarray",
) -> "np.ndarray":
    """Vectorised exponential moving average over numpy arrays.

    Returns ``prev + alpha * (cur - prev)`` elementwise. Broadcasting follows
    numpy rules, so a scalar ``alpha`` applies to every element. For vectors of
    any real size this is several times faster than a Python loop of
    :func:`ema` (see ``tests/test_perf_caching_pooling.py``).
    """
    prev_a = np.asarray(prev, dtype=float)
    cur_a = np.asarray(cur, dtype=float)
    alpha_a = np.asarray(alpha, dtype=float)
    return prev_a + alpha_a * (cur_a - prev_a)


def clamp_vector(
    value: float | Sequence[float] | "np.ndarray",
    low: float | Sequence[float] | "np.ndarray",
    high: float | Sequence[float] | "np.ndarray",
) -> "np.ndarray":
    """Vectorised clamp: ``max(low, min(high, value))`` elementwise."""
    value_a = np.asarray(value, dtype=float)
    low_a = np.asarray(low, dtype=float)
    high_a = np.asarray(high, dtype=float)
    return np.maximum(low_a, np.minimum(high_a, value_a))


class ExpressionLatch:
    def __init__(self, threshold: float = 0.18, dwell_ms: int = 1200) -> None:
        self.threshold = threshold
        self.dwell_ms = dwell_ms
        self.value = "neutral"
        self.changed_ms = 0

    def update(self, candidate: str, confidence: float, now_ms: int) -> str:
        if candidate != self.value and confidence >= self.threshold and now_ms - self.changed_ms >= self.dwell_ms:
            self.value = candidate
            self.changed_ms = now_ms
        return self.value


def reaction_delay(mode: str, delays: Any) -> int:
    """Randomised reaction delay (ms) for the given behavior mode.

    ``delays`` is the ``ReactionDelayConfig`` object (``cfg.affect.reaction_delay_ms``).
    """
    if mode == "mirror":
        return random.randint(delays.mirror_min, delays.mirror_max)
    return random.randint(delays.reflect_min, delays.reflect_max)
