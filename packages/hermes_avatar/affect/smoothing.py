from __future__ import annotations
import random

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def ema(previous: float, current: float, alpha: float) -> float:
    return current if previous is None else previous + alpha * (current - previous)

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

def reaction_delay(mode: str, cfg) -> int:
    delays = cfg.affect.reaction_delay_ms
    if mode == "mirror":
        return random.randint(delays.mirror_min, delays.mirror_max)
    return random.randint(delays.reflect_min, delays.reflect_max)
