from __future__ import annotations

import audioop
import time


class AudioVAD:
    def __init__(
        self, sample_rate: int = 16000, energy_threshold: float = 0.02
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.last_speaking_ms = 0

    def frame_event(self, pcm16: bytes) -> dict:
        rms = audioop.rms(pcm16, 2) / 32768.0 if pcm16 else 0.0
        speaking = rms >= self.energy_threshold
        now = int(time.time() * 1000)
        if speaking:
            self.last_speaking_ms = now
        return {
            "type": "audio.vad",
            "timestamp_ms": now,
            "speaking": speaking,
            "energy": rms,
            "speech_rate": min(1.0, rms * 4),
        }
