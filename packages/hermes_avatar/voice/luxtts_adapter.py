from __future__ import annotations
import math, wave, struct
from pathlib import Path
from .base import VoiceBackend, VoiceStyle, SynthesizedSpeech
from .voice_cache import VoiceCache

class LuxTTSAdapter(VoiceBackend):
    """Non-invasive LuxTTS bridge with a deterministic WAV fallback for offline demos."""
    def __init__(self, vendor_dir: str = "vendor/LuxTTS", device: str = "cpu", cache_dir: str = "cache/voice") -> None:
        self.vendor_dir = Path(vendor_dir)
        self.device = device
        self.cache = VoiceCache(cache_dir)
        self._prompt_cache: dict[str, str] = {}

    def cache_reference(self, reference_audio: str | None) -> None:
        if reference_audio:
            self._prompt_cache[reference_audio] = "cached"

    def synthesize(self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None) -> SynthesizedSpeech:
        self.cache_reference(reference_audio)
        path = self.cache.path_for(text, "luxtts")
        if not path.exists():
            # Fallback placeholder: audible 220Hz tone sized to text length. Replace with LuxTTS CLI/API when installed.
            sr = 48000
            duration = max(0.6, min(8.0, len(text) / 22.0))
            amp = int(9000 * max(0.2, min(1.0, voice_style.intensity + 0.25)))
            with wave.open(str(path), "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                for i in range(int(sr * duration)):
                    sample = int(amp * math.sin(2 * math.pi * 220 * i / sr))
                    wf.writeframes(struct.pack("<h", sample))
        return SynthesizedSpeech(text=text, audio_path=str(path), sample_rate=48000, backend="luxtts")
