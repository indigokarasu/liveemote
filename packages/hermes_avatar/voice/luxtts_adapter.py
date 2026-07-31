from __future__ import annotations

import hashlib
import logging
import math
import os
import shlex
import struct
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Callable

from .base import SynthesizedSpeech, VoiceBackend, VoiceStyle
from .voice_cache import VoiceCache
from hermes_avatar.util import CircuitBreaker
from hermes_avatar.util.audit import (
    audit_event,
    snapshot as audit_snapshot,
    KIND_TRIP,
    KIND_RECOVER,
    KIND_HALF_OPEN,
    KIND_VENDOR_FALLBACK,
    KIND_COST_CAP_EXCEEDED,
)

logger = logging.getLogger(__name__)


# -- rolling per-second subprocess budget -----------------------------------
#
# The LuxTTS vendor subprocess can take ~30-120 seconds per call (blocking
# ``subprocess.run`` with ``timeout=120``). The breaker (commit ``ff83c2f``)
# already protects against per-call FAILURE storms; the cost-cap protects
# against per-call RESOURCE pressure during a thundering herd of
# /api/speak calls when OpenAI + ElevenLabs + LiveTalking are all
# unavailable.
#
# The two protections cooperate:
#   breaker CLOSED + budget room -> subprocess runs.
#   breaker CLOSED + budget exhausted -> fallback (no subprocess), audit
#     event ``voice.luxtts.cost_cap_exceeded``, breaker NOT incremented.
#   breaker OPEN -> fallback (existing policy), orthogonal to the budget.
#   timeout / vendor exception -> budget STILL consumes the elapsed
#     seconds (a hung subprocess is real resource pressure, not a
#     hypothetical one).
class CostWindow:
    """Rolling subprocess-second budget over a sliding time window."""

    HARD_FLOOR_SECONDS = 30.0  # misconfigurations are loud instead of silent

    def __init__(
        self,
        cap_seconds: float = 240.0,
        window_seconds: float = 60.0,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.cap_seconds: float = float(max(cap_seconds, self.HARD_FLOOR_SECONDS))
        self.window_seconds: float = float(max(window_seconds, 1.0))
        self._now_fn = now_fn
        self._samples: deque[tuple[float, float]] = deque()
        self.calls_blocked: int = 0
        self._lock = threading.Lock()

    def would_block(self, sample_seconds: float = 0.0) -> bool:
        """Return True if a hypothetical ``sample_seconds`` call would exceed cap."""
        with self._lock:
            self._prune_locked(self._now_fn())
            return self._used_locked() + max(0.0, sample_seconds) > self.cap_seconds

    def record(self, seconds_used: float, now: float | None = None) -> None:
        """Account for one completed call's actual seconds (even on timeout)."""
        if seconds_used <= 0:
            return
        ts = self._now_fn() if now is None else now
        with self._lock:
            self._prune_locked(ts)
            # Clamp to window_seconds so a single record cannot exceed the cap.
            self._samples.append((ts, min(seconds_used, self.window_seconds)))

    def snapshot(self) -> dict:
        with self._lock:
            self._prune_locked(self._now_fn())
            used = self._used_locked()
            return {
                "cap_seconds": self.cap_seconds,
                "window_seconds": self.window_seconds,
                "used_seconds": used,
                "remaining_seconds": max(0.0, self.cap_seconds - used),
                "calls_blocked": self.calls_blocked,
            }

    def _used_locked(self) -> float:
        return sum(s for _, s in self._samples)

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()


def _default_cost_cap_seconds() -> float:
    """The default budget is 240 subprocess-seconds per minute.

    240 = 2 in-flight calls at the 120s ceiling, OR ~8 short calls at 30s
    each.  Configurable via env var ``LUXTTS_COST_CAP_SECONDS_PER_MINUTE``.
    """
    raw = os.getenv("LUXTTS_COST_CAP_SECONDS_PER_MINUTE", "240").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 240.0


class LuxTTSAdapter(VoiceBackend):
    """LuxTTS bridge with deterministic local generation and optional vendor CLI wiring.

    Set ``LUXTTS_COMMAND`` to a command template that writes a WAV to ``{output}``.
    Supported placeholders: ``{text}``, ``{output}``, ``{reference}``, ``{device}``,
    and ``{vendor_dir}``. Example::

        LUXTTS_COMMAND='python vendor/LuxTTS/infer.py --text {text} --ref {reference} --out {output}'

    If the command is not configured or fails, the adapter generates an intelligible
    deterministic WAV locally so the demo always has measurable audio output.

    Two resilience protections are wired into ``synthesize()``:

    * **Circuit breaker** (commit ``ff83c2f``) -- trips OPEN after 2 failures
      and refuses work for 60 s; the synthesizer falls back to parametric
      audio instead of silently dropping the user's utterance.
    * **Subprocess-second cost cap** (this commit, also surfaced in
      ``apps/demo_server/RUNBOOK.md``) -- per-minute subprocess budget so a
      thundering herd of /api/speak calls during a multi-vendor outage
      can't melt the box even when the breaker is closed.  Default 240
      s/min; configurable via ``LUXTTS_COST_CAP_SECONDS_PER_MINUTE``.
    """

    def __init__(
        self,
        vendor_dir: str = "vendor/LuxTTS",
        device: str = "cpu",
        cache_dir: str = "cache/voice",
        cost_cap_seconds: float | None = None,
        cost_cap_window: float = 60.0,
    ) -> None:
        self.vendor_dir = Path(vendor_dir)
        self.device = device
        self.cache = VoiceCache(cache_dir)
        self._prompt_cache: dict[str, str] = {}
        self.command_template = os.getenv("LUXTTS_COMMAND", "").strip()
        self.last_latency_ms = 0
        self.last_engine = "local-parametric"
        self.last_error: str | None = None
        # Circuit breaker protects /api/speak from cascading 120s vendor hangs.
        # 2 failures (4 minutes of probe time) is highly tolerant for a local
        # subprocess; the 60s open window gives heavy model recovery time
        # before the next probe attempt while keeping the user unblocked
        # with parametric fallback audio in between.
        self.cb = CircuitBreaker(failure_threshold=2, open_timeout=60.0, name="luxtts")
        # Cost cap (IMPROVEMENTS_TODO item 2.3). 240 s/min default
        # (configurable via LUXTTS_COST_CAP_SECONDS_PER_MINUTE). The 30 s
        # HARD_FLOOR inside CostWindow keeps misconfigurations loud.
        self.cost_cap = CostWindow(
            cap_seconds=cost_cap_seconds if cost_cap_seconds is not None else _default_cost_cap_seconds(),
            window_seconds=cost_cap_window,
        )
        # Sample seconds for the would-block preflight check; matches the
        # subprocess timeout in ``_run_vendor_command``.
        self._vendor_max_seconds = 120

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
            "circuit_breaker": self.cb.snapshot(),
            "cost_cap": self.cost_cap.snapshot(),
            # Aggregated audit counter cache so /api/health can introspect
            # what this subsystem has emitted (breaker trips + cost-cap hits
            # + vendor fallbacks) without re-parsing the log stream.
            "audit": audit_snapshot("voice.luxtts"),
        }

    def synthesize(self, text: str, voice_style: VoiceStyle, reference_audio: str | None = None) -> SynthesizedSpeech:
        self.cache_reference(reference_audio)
        path = self.cache.path_for(text + repr(voice_style.__dict__) + str(reference_audio), "luxtts")
        started = time.perf_counter()
        engine = "local-parametric"
        if not path.exists():
            if self.command_template:
                # Gate the vendor call on the breaker FIRST. Unlike the
                # ElevenLabs/LiveTalking adapters (which raise on open),
                # LuxTTS is the deterministic-local fallback path so we
                # must swallow the refusal and emit parametric audio
                # instead -- otherwise the user gets silence.
                if not self.cb.allow():
                    self.last_error = "luxtts circuit breaker open"
                    audit_event(
                        "voice.luxtts",
                        KIND_VENDOR_FALLBACK,
                        level=logging.WARNING,
                        reason="circuit_breaker_open",
                    )
                    self._write_parametric_voice(path, text, voice_style)
                elif self.cost_cap.would_block(sample_seconds=self._vendor_max_seconds):
                    # Cost cap is the orthogonal protection to the breaker:
                    # throttles when the per-minute subprocess budget is
                    # saturated. We do NOT increment the breaker (the
                    # breaker measures vendor HEALTH, not resource pressure)
                    # and we emit a single canonical audit event so
                    # /api/health surfaces it next to the breaker snapshot.
                    self.cost_cap.calls_blocked += 1
                    self.last_error = "luxtts cost cap exceeded"
                    audit_event(
                        "voice.luxtts",
                        KIND_COST_CAP_EXCEEDED,
                        level=logging.WARNING,
                        cap_seconds=self.cost_cap.snapshot()["cap_seconds"],
                        used_seconds=self.cost_cap.snapshot()["used_seconds"],
                    )
                    self._write_parametric_voice(path, text, voice_style)
                else:
                    t0 = time.perf_counter()
                    try:
                        self._run_vendor_command(text, path, reference_audio)
                        # CRITICAL: even on success the seconds are real cost.
                        self.cost_cap.record(time.perf_counter() - t0)
                        self.cb.record_success()
                        engine = "luxtts-vendor"
                        self.last_error = None
                    except Exception as exc:  # command failure should not break local demo audio
                        # Hung subprocess still consumed wall-clock seconds
                        # even on timeout -- record them so the budget is
                        # honest.
                        self.cost_cap.record(time.perf_counter() - t0)
                        self.cb.record_failure()
                        self.last_error = str(exc)
                        audit_event(
                            "voice.luxtts",
                            KIND_VENDOR_FALLBACK,
                            level=logging.WARNING,
                            error=str(exc),
                        )
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
        subprocess.run(shlex.split(command), cwd=Path.cwd(), check=True, timeout=120)
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
