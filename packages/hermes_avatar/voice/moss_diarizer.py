"""``MossDiarizer`` — client for the MOSS-Transcribe-Diarize sidecar daemon.

MOSS-Transcribe-Diarize produces transcription + speaker diarization +
timestamps in a single pass and must run in its own Python 3.12 venv (see
``sidecar/moss_daemon.py``). This client lives in the main LiveEmote server
and mirrors the ``FaceFusionSidecarDaemon`` pattern:

* ``MOSS__SIDECAR__URL`` (default ``http://127.0.0.1:8899``) selects the daemon.
* :meth:`diarize` POSTs audio bytes and parses the compact
  ``[start][Sxx]text[end]`` transcript into structured segments.
* :meth:`capability_status` is a cached, non-raising probe so ``/api/status``
  and ``/api/health`` can report the diarization surface without hammering it.
* Every failure mode returns ``available=False`` with a reason — the avatar's
  meeting/transcription features degrade, never crash.

``parse_moss_transcript`` is self-contained (regex-based) so the main process
can parse MOSS output without importing the 3.12-only package.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SIDECAR_URL = "http://127.0.0.1:8899"
HEALTH_TIMEOUT_S = 1.5
PROBE_CACHE_TTL_S = 3.0

_SEGMENT_HEADER_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\]")
_SEGMENT_TAIL_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*$")


@dataclass
class DiarizationSegment:
    start: float
    end: float | None
    speaker: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "speaker": self.speaker, "text": self.text}


@dataclass
class DiarizationResult:
    available: bool = False
    segments: list[DiarizationSegment] = field(default_factory=list)
    text: str = ""
    model: str | None = None
    elapsed_sec: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "segments": [s.to_dict() for s in self.segments],
            "text": self.text,
            "model": self.model,
            "elapsed_sec": self.elapsed_sec,
            "reason": self.reason,
        }


def parse_moss_transcript(text: str) -> list[DiarizationSegment]:
    """Parse the compact MOSS transcript ``[start][Sxx]text[end]`` stream.

    Segments are delimited by the next ``[start][Sxx]`` header; each body may
    end with an ``[end]`` timestamp (or be the last segment with none).
    """
    matches = list(_SEGMENT_HEADER_RE.finditer(text or ""))
    segments: list[DiarizationSegment] = []
    for i, m in enumerate(matches):
        start = float(m.group(1))
        speaker = m.group(2)
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end_pos]
        end: float | None = None
        tail = _SEGMENT_TAIL_RE.search(body)
        if tail:
            end = float(tail.group(1))
            body = body[: tail.start()]
        body = body.strip()
        if body:
            segments.append(DiarizationSegment(start=start, end=end, speaker=speaker, text=body))
    return segments


class MossDiarizer:
    """HTTP client for the MOSS-Transcribe-Diarize sidecar."""

    def __init__(
        self,
        *,
        sidecar_url: str | None = None,
        api_key: str | None = None,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
    ) -> None:
        self.sidecar_url = sidecar_url or os.getenv("MOSS__SIDECAR__URL", DEFAULT_SIDECAR_URL)
        self.api_key = api_key or os.getenv("MOSS__SIDECAR__API_KEY") or None
        self.health_timeout_s = health_timeout_s
        self._headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.last_error: str | None = None
        self._probe_lock = threading.Lock()
        self._probe_ts = 0.0
        self._probe_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------ health
    async def health(self) -> dict[str, Any]:
        """Probe the daemon. Never raises; returns a status dict."""
        try:
            async with httpx.AsyncClient(timeout=self.health_timeout_s) as client:
                resp = await client.get(f"{self.sidecar_url}/health", headers=self._headers)
            if resp.status_code == 200:
                return {"reachable": True, "degraded": False, "detail": resp.json()}
            return {"reachable": True, "degraded": True, "detail": f"HTTP {resp.status_code}"}
        except Exception as exc:
            self.last_error = str(exc)
            return {"reachable": False, "degraded": True, "reason": str(exc)}

    def capability_status(self) -> dict[str, Any]:
        """Synchronous, cached, non-raising probe for /api/status + /api/health.

        Avoids ``asyncio.run`` when an event loop is already running (e.g.
        under ``pytest-asyncio``) by returning a fresh degraded marker.
        """
        now = time.monotonic()
        with self._probe_lock:
            if self._probe_cache is not None and (now - self._probe_ts) <= PROBE_CACHE_TTL_S:
                return dict(self._probe_cache)
            try:
                asyncio.get_running_loop()
                return {"reachable": False, "degraded": True, "reason": "probe skipped (loop running)"}
            except RuntimeError:
                self._probe_cache = asyncio.run(self.health())
                self._probe_ts = now
            return dict(self._probe_cache)

    # ----------------------------------------------------------------- diarize
    async def diarize_async(
        self, audio_bytes: bytes, *, filename: str = "audio.wav", max_new_tokens: int = 2048
    ) -> DiarizationResult:
        """POST audio to the daemon and parse the response. Never raises."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.sidecar_url}/api/v1/transcribe",
                    headers=self._headers,
                    files={"audio": (filename, audio_bytes, "application/octet-stream")},
                    data={"max_new_tokens": str(max_new_tokens)},
                )
            if resp.status_code != 200:
                return DiarizationResult(available=False, reason=f"sidecar HTTP {resp.status_code}")
            data = resp.json()
            segments = [
                DiarizationSegment(
                    start=float(s["start"]),
                    end=float(s["end"]) if s.get("end") is not None else None,
                    speaker=s.get("speaker", ""),
                    text=s.get("text", ""),
                )
                for s in data.get("segments", [])
            ]
            return DiarizationResult(
                available=bool(data.get("available", True)),
                segments=segments,
                text=data.get("text", ""),
                model=data.get("model"),
                elapsed_sec=data.get("elapsed_sec"),
            )
        except Exception as exc:
            self.last_error = str(exc)
            return DiarizationResult(available=False, reason=f"diarize failed: {exc}")

    def diarize(
        self, audio_bytes: bytes, *, filename: str = "audio.wav", max_new_tokens: int = 2048
    ) -> DiarizationResult:
        """Synchronous convenience wrapper around :meth:`diarize_async`."""
        return asyncio.run(self.diarize_async(audio_bytes, filename=filename, max_new_tokens=max_new_tokens))
