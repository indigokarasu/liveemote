from __future__ import annotations

import hashlib
import math
import os
import shlex
import struct
import subprocess
import time
import wave
from pathlib import Path

from .base import SynthesizedSpeech, VoiceBackend, VoiceStyle
from .voice_cache import VoiceCache


class LuxTTSAdapter(VoiceBackend):
    """LuxTTS bridge with deterministic local generation and optional vendor CLI wiring.

    Set ``LUXTTS_COMMAND`` to a command template that writes a WAV to ``{output}``.
    Supported placeholders: ``{text}``, ``{output}``, ``{reference}``, ``{device}``,
    and ``{vendor_dir}``. Example::

        LUXTTS_COMMAND='python vendor/LuxTTS/infer.py --text {text} --ref {reference} --out {output}'

    If the command is not configured or fails, the adapter generates an intelligible
    deterministic WAV locally so the demo always has measurable audio output.
    """

    def __init__(self, vendor_dir: str = "vendor/LuxTTS", device: str = "cpu", cache_dir: str = "cache/voice") -> None:
        self.vendor_dir = Path(vendor_dir)
        self.device = device
        self.cache = VoiceCache(cache_dir)
        self._prompt_cache: dict[str, str] = {}
        self.command_template = os.getenv("LUXTTS_COMMAND", "").strip()
        self.last_latency_ms = 0
        self.last_engine = "local-parametric"
        self.last_error: str | None = None

    def cache_reference(self, reference_audio: str | None) -> None:
        if reference_audio:
            self._prompt_cache[reference_audio] = str(Path(reference_audio).resolve())

    def capability_status(self) -> dict:
        return {
            "backend": "luxtts",
            "vendor_dir_exists": self.vendor_dir.exists(),
            "command_configured": bool(self.command_template),
            "device": self.device,
            "last_engine": self.last_engine,
            "last_latency_ms": self.last_latency_ms,
            "last_error": self.last_error,
        }

    def synthesize(self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None) -> SynthesizedSpeech:
        self.cache_reference(reference_audio)
        path = self.cache.path_for(text + repr(voice_style.__dict__) + str(reference_audio), "luxtts")
        started = time.perf_counter()
        engine = "local-parametric"
        if not path.exists():
            if self.command_template:
                try:
                    self._run_vendor_command(text, path, reference_audio)
                    engine = "luxtts-vendor"
                    self.last_error = None
                except Exception as exc:  # command failure should not break local demo audio
                    self.last_error = str(exc)
                    self._write_parametric_voice(path, text, voice_style)
            else:
                self._write_parametric_voice(path, text, voice_style)
        duration_ms = self._wav_duration_ms(path)
        self.last_latency_ms = int((time.perf_counter() - started) * 1000)
        self.last_engine = engine
        return SynthesizedSpeech(
            text=text,
            audio_path=str(path),
            sample_rate=48000,
            duration_ms=duration_ms,
            backend="luxtts",
            latency_ms=self.last_latency_ms,
            engine=engine,
        )

    def _run_vendor_command(self, text: str, output: Path, reference_audio: str | None) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "text": shlex.quote(text),
            "output": shlex.quote(str(output)),
            "reference": shlex.quote(reference_audio or ""),
            "device": shlex.quote(self.device),
            "vendor_dir": shlex.quote(str(self.vendor_dir)),
        }
        command = self.command_template.format(**values)
        subprocess.run(command, shell=True, cwd=Path.cwd(), check=True, timeout=120)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("LuxTTS command completed without producing a WAV")

    def _write_parametric_voice(self, path: Path, text: str, voice_style: VoiceStyle) -> None:
        sr = 48000
        words = max(1, len(text.split()))
        pace = max(0.2, min(1.2, voice_style.pace))
        duration = max(0.7, min(18.0, words * (0.42 / pace)))
        amp = int(8800 * max(0.2, min(1.0, voice_style.intensity + 0.3)))
        warmth = max(0.0, min(1.0, voice_style.warmth))
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        base_freq = 175 + (seed % 45) + int(warmth * 35)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for i in range(int(sr * duration)):
                t = i / sr
                envelope = min(1.0, t / 0.04, (duration - t) / 0.08)
                wobble = math.sin(2 * math.pi * 3.2 * t) * (4 + 10 * voice_style.intensity)
                carrier = math.sin(2 * math.pi * (base_freq + wobble) * t)
                harmonic = 0.38 * math.sin(2 * math.pi * (base_freq * 2.01) * t)
                sample = int(amp * envelope * (carrier + harmonic) / 1.38)
                wf.writeframes(struct.pack("<h", sample))

    def _wav_duration_ms(self, path: Path) -> int | None:
        try:
            with wave.open(str(path), "rb") as wf:
                return int(wf.getnframes() / wf.getframerate() * 1000)
        except wave.Error:
            return None
