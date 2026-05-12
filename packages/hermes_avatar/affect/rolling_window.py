from __future__ import annotations
from collections import deque

class RollingWindow:
    def __init__(self, window_ms: int = 3000) -> None:
        self.window_ms = window_ms
        self.items: deque[tuple[int, dict]] = deque()

    def add(self, timestamp_ms: int, value: dict) -> None:
        self.items.append((timestamp_ms, value))
        self.prune(timestamp_ms)

    def prune(self, now_ms: int) -> None:
        while self.items and now_ms - self.items[0][0] > self.window_ms:
            self.items.popleft()

    def latest(self) -> dict | None:
        return self.items[-1][1] if self.items else None
