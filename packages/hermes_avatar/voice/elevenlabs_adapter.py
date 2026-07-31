from __future__ import annotations

import os
import time

import httpx
from hermes_avatar.util import (
    CircuitBreaker,
    compute_backoff_delay,
    is_retryable_error,
)
from hermes_avatar.util.audit import (
    audit_event,
    snapshot as audit_snapshot,
    KIND_TRIP,
    KIND_RECOVER,
    KIND_VENDOR_FALLBACK,
)
from .base import VoiceBackend, VoiceStyle, SynthesizedSpeech
from .voice_cache import VoiceCache


class ElevenLabsAdapter(VoiceBackend):
    """ElevenLabs TTS bridge with circuit-breaker-guarded synthesis.

    Network calls to the ElevenLabs API are wrapped in the shared
    :class:`~hermes_avatar.util.CircuitBreaker` plus exponential
    backoff with jitter, so a transient 5xx / 429 / network blip is
    retried a few times instead of failing the whole utterance, and a
    sustained outage trips the breaker so we stop hammering a dead
    dependency. All failures are surfaced as ``RuntimeError`` (the
    orchestrator's voice layer decides how to degrade), but the breaker
    state is exposed via :meth:`capability_status` for health probes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        cache_dir: str = "cache/voice",
    ) -> None:
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        self.cache = VoiceCache(cache_dir)
        # Shared, thread-safe resilience primitives (mirrors the renderer).
        self.cb = CircuitBreaker(failure_threshold=5, open_timeout=60.0, name="elevenlabs")
        self.max_retries = 3
        self.base_delay = 0.5
        self.max_delay = 4.0
        self.jitter_factor = 0.1

    def capability_status(self) -> dict:
        return {
            "backend": "elevenlabs",
            "configured": bool(self.api_key and self.voice_id),
            "circuit_breaker": self.cb.snapshot(),
            "audit": audit_snapshot("voice.elevenlabs"),
        }

    def synthesize(
        self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None
    ) -> SynthesizedSpeech:
        if not self.api_key or not self.voice_id:
            raise RuntimeError(
                "ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are required"
            )
        path = self.cache.path_for(text, "elevenlabs")
        if not path.exists():
            if not self.cb.allow():
                # Breaker is open and the open window has not elapsed: do not
                # hit the network; surface as a clear, retryable failure.
                raise RuntimeError("ElevenLabs circuit breaker is open")
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
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    with httpx.Client(timeout=60) as client:
                        r = client.post(url, headers=headers, json=payload)
                        r.raise_for_status()
                        path.write_bytes(r.content)
                    self.cb.record_success()
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt == self.max_retries:
                        break
                    if not is_retryable_error(exc):
                        break
                    delay = compute_backoff_delay(
                        attempt, self.base_delay, self.max_delay, self.jitter_factor
                    )
                    time.sleep(delay)
            if last_exc is not None:
                self.cb.record_failure()
                audit_event(
                    "voice.elevenlabs",
                    KIND_VENDOR_FALLBACK,
                    level=logging.WARNING,
                    error=str(last_exc),
                )
                raise last_exc
        return SynthesizedSpeech(text=text, audio_path=str(path), backend="elevenlabs")
