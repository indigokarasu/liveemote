"""Wav2Lip lip-sync adapter — subprocess-based audio→video mouth synthesis.

Architecture
------------
This module wraps the `Wav2Lip <https://github.com/Rudrabha/Wav2Lip>`_
inference pipeline (or its UHQ variant ``sd-wav2lip-uhq``) behind the same
resilience primitives used by the TTS adapters — circuit breaker, exponential
backoff with jitter, and structured audit events.

The adapter is **not** a full ``Renderer`` subclass; it's a standalone
post-TTS step that the orchestrator calls after voice synthesis but before
the renderer's ``speak()``.  Conceptually::

    TTS (LuxTTS / Fish Audio / ElevenLabs)  →  Wav2Lip  →  LiveTalking

When Wav2Lip models or its Python environment are absent the adapter degrades
to a transparent passthrough (returns the original face image), matching the
graceful-degradation contract used by ``FaceSwapAdapter``.

Lifecycle
---------
1. ``__init__`` probes ``vendor_dir`` for a ``run_wav2lip.py`` entrypoint
   (supplied by the operator).  If missing → *passthrough mode*.
2. ``synthesize_lip_sync(audio_path, face_source)``:
   a. Circuit breaker ``allow()`` gate.
   b. Launch ``python run_wav2lip.py --audio <path> --face <path> --out <dir>``
      as a subprocess with a configurable timeout.
   c. On success → return ``{"video_path": str, "latency_ms": int}``.
   d. On failure → retry (transient errors) or trip breaker (sustained).
3. ``capability_status()`` → breakout for ``/api/health``.

Environment
-----------
``WAV2LIP_ENABLED`` (bool, default ``false``)
    Master on/off.  When ``false`` the orchestrator skips lip-sync entirely.
``WAV2LIP_VENDOR_DIR`` (path, default ``vendor/Wav2Lip``)
    Directory containing ``run_wav2lip.py`` + model checkpoints.
``WAV2LIP_MODEL_PATH`` (path, default ``<vendor_dir>/checkpoints/wav2lip_gan.pth``)
    Path to the Wav2Lip checkpoint.
``WAV2LIP_DEVICE`` (``cpu`` | ``cuda``, default ``cpu``)
``WAV2LIP_FACE_DETECTION_BATCH_SIZE`` (int, default 16)
``WAV2LIP_TIMEOUT_SECONDS`` (float, default 120.0)
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

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
    KIND_RETRY_EXHAUSTED,
)

logger = logging.getLogger(__name__)


class Wav2LipAdapter:
    """Subprocess-based Wav2Lip lip-sync with resilience guards.

    Parameters
    ----------
    vendor_dir:
        Directory containing Wav2Lip's ``run_wav2lip.py`` entrypoint +
        model checkpoints.  Pass ``None`` to skip the filesystem probe
        (the adapter starts in passthrough mode).
    model_path:
        Full path to the Wav2Lip checkpoint (``.pth``).
    device:
        ``"cpu"`` or ``"cuda"``.
    enabled:
        When ``False`` the adapter returns passthrough results for every
        call — no subprocess is launched and no breaker state is touched.
    """

    def __init__(
        self,
        vendor_dir: str | None = None,
        model_path: str | None = None,
        device: str = "cpu",
        enabled: bool = False,
        timeout_seconds: float = 120.0,
        face_detection_batch_size: int = 16,
        pads: tuple[int, int, int, int] = (0, 10, 0, 0),
        nosmooth: bool = True,
        resize_factor: int = 1,
        box: tuple[int, int, int, int] | None = None,
    ) -> None:
        # Resolve from env if not explicitly passed.
        _vendor = vendor_dir or os.getenv("WAV2LIP_VENDOR_DIR", "vendor/Wav2Lip")
        self.vendor_dir = Path(_vendor) if _vendor else None
        self.entrypoint: Path | None = None
        self.passthrough = True  # flipped to False once the probe succeeds

        self.device = device or os.getenv("WAV2LIP_DEVICE", "cpu")
        self.timeout = timeout_seconds or float(
            os.getenv("WAV2LIP_TIMEOUT_SECONDS", "120")
        )
        self.face_detection_batch_size = (
            face_detection_batch_size
            if face_detection_batch_size != 16
            else int(os.getenv("WAV2LIP_FACE_DETECTION_BATCH_SIZE", "16"))
        )
        self.pads = pads
        self.nosmooth = nosmooth
        self.resize_factor = resize_factor
        self.box = box
        self.enabled = enabled if enabled else _env_bool("WAV2LIP_ENABLED", False)

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = (
                self.vendor_dir / "checkpoints" / "wav2lip_gan.pth"
                if self.vendor_dir
                else None
            )
            env_model = os.getenv("WAV2LIP_MODEL_PATH")
            if env_model:
                self.model_path = Path(env_model)

        # Probe the filesystem so we can report passthrough vs ready in
        # capability_status() and avoid failing at inference time.
        # Only probe when the adapter is actually enabled — a disabled
        # adapter stays in passthrough regardless of what's on disk.
        if not self.enabled:
            self.passthrough = True
        elif self.vendor_dir and self.vendor_dir.is_dir():
            candidate = self.vendor_dir / "run_wav2lip.py"
            if candidate.is_file():
                self.entrypoint = candidate
                self.passthrough = False

        # Circuit breaker + retry (shared primitives).
        self.cb = CircuitBreaker(
            failure_threshold=5, open_timeout=120.0, name="wav2lip"
        )
        self.max_retries = 3
        self.base_delay = 0.5
        self.max_delay = 4.0
        self.jitter_factor = 0.1

        # Operational counters.
        self.last_latency_ms: int | None = None
        self.total_calls = 0
        self.passthrough_calls = 0

    # -- Public surface -------------------------------------------------------

    def capability_status(self) -> dict[str, Any]:
        """Breakout consumed by ``/api/health`` and the orchestrator's caps."""
        return {
            "backend": "wav2lip",
            "enabled": self.enabled,
            "passthrough": self.passthrough,
            "entrypoint": str(self.entrypoint) if self.entrypoint else None,
            "model_path": str(self.model_path) if self.model_path else None,
            "device": self.device,
            "total_calls": self.total_calls,
            "passthrough_calls": self.passthrough_calls,
            "last_latency_ms": self.last_latency_ms,
            "circuit_breaker": self.cb.snapshot(),
            "audit": audit_snapshot("renderer.wav2lip"),
        }

    def synthesize_lip_sync(
        self,
        audio_path: str,
        face_source: str,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Run Wav2Lip inference (or return passthrough).

        Parameters
        ----------
        audio_path:
            Path to the synthesized TTS ``.wav`` file.
        face_source:
            Path to the avatar's current face image or video frame.
        output_dir:
            Directory for the output video.  Defaults to a temp dir inside
            the voice cache.

        Returns
        -------
        dict with keys ``video_path``, ``latency_ms``, ``passthrough``.
        """
        self.total_calls += 1

        if not self.enabled or self.passthrough:
            self.passthrough_calls += 1
            return {
                "video_path": face_source,
                "latency_ms": 0,
                "passthrough": True,
            }

        if not self.entrypoint or not self.model_path:
            self.passthrough_calls += 1
            return {
                "video_path": face_source,
                "latency_ms": 0,
                "passthrough": True,
                "reason": "entrypoint or model not found",
            }

        if not self.cb.allow():
            self.passthrough_calls += 1
            audit_event(
                "renderer.wav2lip",
                KIND_VENDOR_FALLBACK,
                level=logging.WARNING,
                error="circuit breaker open",
            )
            return {
                "video_path": face_source,
                "latency_ms": 0,
                "passthrough": True,
                "reason": "circuit breaker open",
            }

        out_dir = Path(output_dir) if output_dir else Path(face_source).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"wav2lip_{int(time.time() * 1000)}.mp4"

        # Build the subprocess command.
        cmd: list[str] = [
            sys.executable,
            str(self.entrypoint),
            "--face", str(face_source),
            "--audio", str(audio_path),
            "--outfile", str(out_video),
            "--checkpoint_path", str(self.model_path),
            "--device", self.device,
            "--pads", *[str(p) for p in self.pads],
            "--face_detection_batch_size", str(self.face_detection_batch_size),
            "--resize_factor", str(self.resize_factor),
        ]
        if self.nosmooth:
            cmd.append("--nosmooth")
        if self.box:
            cmd.extend(["--box", *[str(b) for b in self.box]])

        last_exc: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(self.vendor_dir),
                )
                if proc.returncode != 0:
                    stderr_tail = (
                        proc.stderr.strip()[-300:] if proc.stderr else "<no stderr>"
                    )
                    raise RuntimeError(
                        f"Wav2Lip exited {proc.returncode}: {stderr_tail}"
                    )
                elapsed = int((time.perf_counter() - t0) * 1000)
                self.last_latency_ms = elapsed
                self.cb.record_success()
                return {
                    "video_path": str(out_video),
                    "latency_ms": elapsed,
                    "passthrough": False,
                }
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                if not is_retryable_error(exc):
                    break
                delay = compute_backoff_delay(
                    attempt, self.base_delay, self.max_delay, self.jitter_factor
                )
                logger.warning(
                    "wav2lip retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

        # All attempts exhausted or non-retryable error.
        self.cb.record_failure()
        audit_event(
            "renderer.wav2lip",
            KIND_RETRY_EXHAUSTED,
            level=logging.WARNING,
            error=str(last_exc) if last_exc else "unknown",
        )
        # Passthrough: return the original face so the avatar still renders.
        self.passthrough_calls += 1
        return {
            "video_path": face_source,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "passthrough": True,
            "reason": "inference failed",
            "error": str(last_exc) if last_exc else None,
        }


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default
