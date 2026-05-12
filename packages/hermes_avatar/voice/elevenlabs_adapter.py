from __future__ import annotations
import os
import httpx
from .base import VoiceBackend, VoiceStyle, SynthesizedSpeech
from .voice_cache import VoiceCache


class ElevenLabsAdapter(VoiceBackend):
    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        cache_dir: str = "cache/voice",
    ) -> None:
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        self.cache = VoiceCache(cache_dir)

    def synthesize(
        self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None
    ) -> SynthesizedSpeech:
        if not self.api_key or not self.voice_id:
            raise RuntimeError(
                "ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are required"
            )
        path = self.cache.path_for(text, "elevenlabs")
        if not path.exists():
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": voice_style.warmth,
                },
            }
            headers = {
                "xi-api-key": self.api_key,
                "accept": "audio/wav",
                "content-type": "application/json",
            }
            with httpx.Client(timeout=60) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                path.write_bytes(r.content)
        return SynthesizedSpeech(text=text, audio_path=str(path), backend="elevenlabs")
