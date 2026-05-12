from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class VoiceStyle:
    pace: float = 0.44
    warmth: float = 0.62
    intensity: float = 0.35
    extra: dict = field(default_factory=dict)

@dataclass
class SynthesizedSpeech:
    text: str
    audio_path: str
    sample_rate: int = 48000
    duration_ms: int | None = None
    backend: str = "unknown"
    latency_ms: int | None = None
    engine: str | None = None

class VoiceBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None) -> SynthesizedSpeech:
        raise NotImplementedError
