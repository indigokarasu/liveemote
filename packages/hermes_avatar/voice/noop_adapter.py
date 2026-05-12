from __future__ import annotations

from .base import SynthesizedSpeech, VoiceBackend, VoiceStyle


class NoopVoiceAdapter(VoiceBackend):
    """Silent voice backend for running without TTS or an installed LLM."""

    def capability_status(self) -> dict:
        return {
            "backend": "none",
            "available": False,
            "speaking_enabled": False,
            "reason": "voice synthesis disabled",
        }

    def synthesize(self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None) -> SynthesizedSpeech:
        return SynthesizedSpeech(text=text, audio_path="", sample_rate=0, duration_ms=0, backend="none", latency_ms=0, engine="disabled")
