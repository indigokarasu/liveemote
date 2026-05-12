from __future__ import annotations
from .base import VoiceBackend, VoiceStyle, SynthesizedSpeech

class MossTTSAdapter(VoiceBackend):
    def __init__(self, *_, **__) -> None:
        pass
    def synthesize(self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None) -> SynthesizedSpeech:
        raise NotImplementedError("MOSS-TTS is experimental and enabled only with --voice-backend moss after installing vendor/MOSS-TTS dependencies.")
