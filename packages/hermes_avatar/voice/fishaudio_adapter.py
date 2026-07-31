from __future__ import annotations

import os
import time
import logging

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


class FishAudioAdapter(VoiceBackend):
    """Fish Audio TTS bridge with circuit-breaker-guarded synthesis.

    Calls the Fish Audio REST API (``https://api.fish.audio/v1/tts``) with
    bearer-token auth.  Supports instant voice cloning (``reference_audio``
    per call), persistent voice models (``reference_id``), and streaming
    (deferred to a future ``stream=True`` code path).

    Network calls are wrapped in the shared
    :class:`~hermes_avatar.util.CircuitBreaker` plus exponential backoff with
    jitter — the same pattern used by the ElevenLabs and LuxTTS adapters — so
    a transient 5xx / 429 / timeout is retried a few times before failing the
    utterance, and a sustained outage trips the breaker.

    Environment
    ----------
    ``FISH_API_KEY``
        Bearer token for ``Authorization: Bearer <token>``.
        Generate at https://fish.audio/app/api-keys/.
    ``FISH_REFERENCE_ID`` (optional)
        Persistent voice-model id.  When set the adapter passes it as
        ``reference_id`` on every request; callers can still override with
        per-call ``reference_audio``.
    """

    BASE_URL = "https://api.fish.audio"

    def __init__(
        self,
        api_key: str | None = None,
        reference_id: str | None = None,
        model: str = "s2.1-pro",
        cache_dir: str = "cache/voice",
        latency_model: str | None = None,
        streaming: bool = False,
        chunk_length: int = 200,
    ) -> None:
        self.api_key = api_key or os.getenv("FISH_API_KEY")
        self.reference_id = reference_id or os.getenv("FISH_REFERENCE_ID")
        self.model = model
        self.cache = VoiceCache(cache_dir)
        self.latency_model = latency_model or model
        self.streaming = streaming
        self.chunk_length = chunk_length

        # Shared, thread-safe resilience primitives (mirrors ElevenLabs).
        self.cb = CircuitBreaker(
            failure_threshold=5, open_timeout=60.0, name="fishaudio"
        )
        self.max_retries = 3
        self.base_delay = 0.5
        self.max_delay = 4.0
        self.jitter_factor = 0.1

    # -- VoiceBackend contract ------------------------------------------------

    def capability_status(self) -> dict:
        return {
            "backend": "fishaudio",
            "configured": bool(self.api_key),
            "model": self.model,
            "reference_id": self.reference_id,
            "circuit_breaker": self.cb.snapshot(),
            "audit": audit_snapshot("voice.fishaudio"),
        }

    def synthesize(
        self,
        text: str,
        voice_style: VoiceStyle,
        reference_audio: str | None = None,
    ) -> SynthesizedSpeech:
        if not self.api_key:
            raise RuntimeError("FISH_API_KEY is required for Fish Audio TTS")

        path = self.cache.path_for(text, "fishaudio")
        latency_ms: int | None = None
        if not path.exists():
            if not self.cb.allow():
                # Breaker is open and the open window hasn't elapsed.
                raise RuntimeError("Fish Audio circuit breaker is open")

            url = f"{self.BASE_URL}/v1/tts"
            payload: dict = {
                "text": text,
                "model_id": self.model,
                "latency": "normal",
                "format": "wav",
                "streaming": self.streaming,
                "chunk_length": self.chunk_length,
            }

            # Voice cloning: per-call reference_audio (instant clone)
            # takes priority over persistent reference_id.
            if reference_audio and os.path.isfile(reference_audio):
                import base64
                with open(reference_audio, "rb") as fh:
                    audio_bytes = fh.read()
                payload["references"] = [
                    {
                        "audio": base64.b64encode(audio_bytes).decode(),
                        "text": "",  # Fish Audio accepts empty transcript for auto
                    }
                ]
            elif self.reference_id:
                payload["reference_id"] = self.reference_id

            # Voice-style mapping: Fish Audio supports speed + emotion
            if voice_style.pace:
                payload["speed"] = max(0.5, min(2.0, 1.0 + (voice_style.pace - 0.44) * 1.5))
            if voice_style.intensity:
                payload["emotion_strength"] = max(0.0, min(1.0, voice_style.intensity))

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "accept": "audio/wav",
                "content-type": "application/json",
            }

            last_exc: Exception | None = None
            t0 = time.perf_counter()
            for attempt in range(self.max_retries + 1):
                try:
                    with httpx.Client(timeout=90) as client:
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
                    "voice.fishaudio",
                    KIND_VENDOR_FALLBACK,
                    level=logging.WARNING,
                    error=str(last_exc),
                )
                raise last_exc

        return SynthesizedSpeech(
            text=text,
            audio_path=str(path),
            backend="fishaudio",
            latency_ms=latency_ms,
            engine=self.model,
        )
