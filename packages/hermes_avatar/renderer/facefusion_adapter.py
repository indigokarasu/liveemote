from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time
import importlib
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import (
    BackgroundSpec,
    CharacterIndex,
    TrainingReference,
    VisualStyle,
)
from hermes_avatar.config.schema import FaceSwapConfig
from .base import Renderer

logger = logging.getLogger(__name__)

# Default model filenames we look for to decide whether a backend can actually
# run. Missing models => degraded passthrough (never a crash).
DEFAULT_MODELS: dict[str, tuple[str, ...]] = {
    "facefusion": ("inswapper_128_fp16.onnx", "GFPGANv1.4.onnx"),
    "deeplivecam": ("inswapper_128_fp16.onnx", "GFPGANv1.4.onnx"),
}


# ---------------------------------------------------------------------------
# Prometheus metrics (guarded import — never hard-fail if absent)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge

    FACESWAP_SWAPS_TOTAL = Counter(
        "faceswap_swaps_total",
        "Total face-swap frames processed by the renderer adapter",
        ["backend", "mode"],
    )
    FACESWAP_BACKEND_ERRORS = Counter(
        "faceswap_backend_errors_total",
        "Face-swap backend invocation errors",
        ["backend"],
    )
    FACESWAP_DEGRADED = Gauge(
        "faceswap_degraded",
        "1 when the face-swap backend is degraded / running in passthrough mode",
    )
    FACESWAP_SWAP_FPS = Gauge(
        "faceswap_swap_fps",
        "Observed face-swap pipeline frame rate (best effort)",
    )
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - prometheus_client optional in some envs
    _PROM_AVAILABLE = False
    FACESWAP_SWAPS_TOTAL = FACESWAP_BACKEND_ERRORS = FACESWAP_DEGRADED = FACESWAP_SWAP_FPS = None


def record_swap(backend: str, mode: str = "swap") -> None:
    if _PROM_AVAILABLE:
        FACESWAP_SWAPS_TOTAL.labels(backend=backend, mode=mode).inc()


def record_backend_error(backend: str) -> None:
    if _PROM_AVAILABLE:
        FACESWAP_BACKEND_ERRORS.labels(backend=backend).inc()


def set_degraded(backend: str, degraded: bool) -> None:
    if _PROM_AVAILABLE:
        FACESWAP_DEGRADED.set(1 if degraded else 0)


def set_swap_fps(fps: float) -> None:
    if _PROM_AVAILABLE:
        FACESWAP_SWAP_FPS.set(fps)


# ---------------------------------------------------------------------------
# GPU detection (non-fatal)
# ---------------------------------------------------------------------------
def _detect_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend lifecycle manager
# ---------------------------------------------------------------------------
class BackendManager:
    """Owns the face-swap backend process and performs per-frame swaps.

    Every detection / startup step is non-fatal. If the backend binary, models,
    or (optionally) the GPU are unavailable, the manager enters a *degraded*
    state and :meth:`swap` becomes a transparent copy (passthrough). This is the
    expected CI / headless behaviour and must never raise.

    The manager is dependency-injectable: tests pass a ``swap_callable`` and/or a
    fake ``process`` so the pipeline can be exercised without spawning a real
    backend.
    """

    def __init__(
        self,
        config: FaceSwapConfig,
        swap_callable: Callable[[Any], Any] | None = None,
    ) -> None:
        self.config = config
        self.swap_callable = swap_callable
        self.process: subprocess.Popen | None = None
        self.binary: list[str] | None = None
        self.available = False
        self.model_present = False
        self.gpu_present = False
        self.degraded = True
        self.passthrough = True
        self.error: str | None = None
        self.last_startup_error: str | None = None
        self.started_at: float | None = None
        self._lock = threading.Lock()

    # ---- detection ------------------------------------------------------
    def _detect_binary(self) -> list[str] | None:
        cfg = self.config
        if cfg.backend_binary:
            if shutil.which(cfg.backend_binary) or Path(cfg.backend_binary).exists():
                return [cfg.backend_binary]
            return None
        vendor = Path(cfg.vendor_dir)
        if cfg.backend == "facefusion":
            cli = shutil.which("facefusion")
            if cli:
                return [cli]
            for entry in ("facefusion.py", "run.py"):
                if (vendor / entry).exists():
                    return [sys.executable, str(vendor / entry)]
            return None
        if cfg.backend == "deeplivecam":
            for entry in ("run.py", "main.py"):
                if (vendor / entry).exists():
                    return [sys.executable, str(vendor / entry)]
            return None
        return None

    def _models_present(self) -> bool:
        cfg = self.config
        explicit = cfg.model_paths or {}
        if explicit:
            return all(Path(p).exists() for p in explicit.values())
        models_dir = Path(cfg.vendor_dir) / cfg.models_dir
        if not models_dir.is_dir():
            return False
        defaults = DEFAULT_MODELS.get(cfg.backend, ())
        return any((models_dir / name).exists() for name in defaults)

    def _mark_unavailable(self, error: str, model_present: bool | None = None) -> None:
        self.available = False
        self.degraded = True
        self.passthrough = True
        self.error = error
        if model_present is not None:
            self.model_present = model_present
        logger.warning(
            "faceswap backend unavailable",
            extra={
                "audit": {
                    "event": "faceswap.unavailable",
                    "backend": self.config.backend,
                    "error": error,
                }
            },
        )

    def _probe_backend_importable(self) -> str | None:
        """Return an error string if the backend's Python package cannot be
        imported, else ``None``. A backend with models on disk but no importable
        runtime (missing cv2 / onnxruntime / insightface / torch) cannot actually
        run, so we treat that as unavailable rather than spawning a doomed
        process."""
        cfg = self.config
        if cfg.backend == "deeplivecam":
            vendor = str(Path(cfg.vendor_dir).resolve())
            if vendor not in sys.path:
                sys.path.insert(0, vendor)
            try:
                importlib.import_module("modules")
                return None
            except Exception as exc:
                return f"Deep-Live-Cam 'modules' package not importable: {exc}"
        if cfg.backend == "facefusion":
            try:
                importlib.import_module("facefusion")
                return None
            except Exception as exc:
                return f"facefusion package not importable: {exc}"
        return None

    def detect(self) -> None:
        """Detect whether the backend can actually run. Non-fatal."""
        with self._lock:
            self.error = None
            binary = self._detect_binary()
            if not binary:
                self._mark_unavailable(
                    f"Backend '{self.config.backend}' not found in PATH or "
                    f"{self.config.vendor_dir}"
                )
                return
            self.binary = binary

            vendor = Path(self.config.vendor_dir)
            if not vendor.exists():
                self._mark_unavailable(f"Vendor directory missing: {self.config.vendor_dir}")
                return

            self.model_present = self._models_present()
            if not self.model_present:
                self._mark_unavailable(
                    "Required face-swap models not found; running degraded (passthrough)",
                    model_present=False,
                )
                return

            import_err = self._probe_backend_importable()
            if import_err is not None:
                self._mark_unavailable(import_err)
                return

            self.gpu_present = _detect_gpu()
            if self.config.require_gpu and not self.gpu_present:
                self._mark_unavailable("GPU required but not available")
                return

            self.available = True
            self.degraded = False
            self.passthrough = False
            self.error = None

    # ---- lifecycle -------------------------------------------------------
    def _build_command(self, source_face: str) -> list[str]:
        if not self.binary:
            raise RuntimeError("backend binary not detected")
        cfg = self.config
        out = cfg.output_virtual_cam or cfg.output_stream_url or "output.mp4"
        if cfg.backend == "facefusion":
            return [
                *self.binary,
                "headless-run",
                "--source-paths",
                source_face,
                "--target-path",
                cfg.input_source,
                "--output-path",
                out,
                "--execution-providers",
                "cuda" if cfg.device == "cuda" else "cpu",
                "--frame-processors",
                "face_swapper",
                "face_enhancer",
            ]
        # deeplivecam
        return [
            *self.binary,
            "--source",
            source_face,
            "--target",
            cfg.input_source,
            "--output",
            out,
            "--execution-provider",
            "cuda" if cfg.device == "cuda" else "cpu",
            "--keep-fps",
            "--many-faces",
        ]

    def _wait_for_startup(self, source_face: str) -> None:
        """Best-effort readiness probe. Raises if the process died on launch."""
        deadline = time.time() + min(self.config.process_timeout, 5.0)
        while time.time() < deadline:
            if self.process is None:
                raise RuntimeError("backend process not spawned")
            if self.process.poll() is not None:
                stderr = ""
                try:
                    if self.process.stderr is not None:
                        stderr = self.process.stderr.read()[-2000:]
                except Exception:
                    pass
                raise RuntimeError(f"backend exited on startup: {stderr}")
            time.sleep(0.25)

    def start(self, source_face: str | None = None) -> None:
        """Detect, then (if usable) spawn the backend. Never raises."""
        self.detect()
        if not (self.config.enabled and self.available and not self.degraded):
            if self.config.enabled and self.degraded:
                logger.warning(
                    "faceswap starting in passthrough (degraded) mode",
                    extra={
                        "audit": {
                            "event": "faceswap.passthrough_start",
                            "backend": self.config.backend,
                            "reason": self.error,
                        }
                    },
                )
            return
        try:
            cmd = self._build_command(source_face or "")
            logger.info(
                "faceswap backend starting",
                extra={
                    "audit": {
                        "event": "faceswap.start",
                        "backend": self.config.backend,
                        "cmd": " ".join(cmd),
                    }
                },
            )
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._wait_for_startup(source_face or "")
            self.started_at = time.time()
        except Exception as exc:  # pragma: no cover - only exercised with a real backend
            self.last_startup_error = str(exc)
            logger.error(
                "faceswap backend failed to start",
                extra={"audit": {"event": "faceswap.start_failed", "error": str(exc)}},
            )
            self._mark_unavailable(f"backend start failed: {exc}")
            self._cleanup_process()

    def stop(self) -> None:
        self._cleanup_process()
        self.started_at = None

    def _cleanup_process(self) -> None:
        if self.process is not None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception:
                pass
            self.process = None

    # ---- swapping --------------------------------------------------------
    def swap(self, frame: Any) -> Any:
        if not self.available or self.passthrough or self.process is None:
            return frame  # passthrough
        try:
            if self.swap_callable is not None:
                return self.swap_callable(frame)
            return self._swap_via_backend(frame)
        except Exception as exc:  # pragma: no cover - exercised only with a real backend
            record_backend_error(self.config.backend)
            logger.error(
                "faceswap swap failed; falling back to passthrough",
                extra={"audit": {"event": "faceswap.swap_failed", "error": str(exc)}},
            )
            return frame

    def _swap_via_backend(self, frame: Any) -> Any:
        """Call the backend's Python API for a single frame.

        Heavy imports are performed lazily and guarded so a missing backend /
        model degrades to passthrough via :meth:`swap`. In a real GPU
        environment this is where the vendored pipeline would be invoked.
        """
        if self.config.backend == "deeplivecam":
            return self._swap_via_deeplivecam(frame)
        return self._swap_via_facefusion(frame)

    def _swap_via_deeplivecam(self, frame: Any) -> Any:  # pragma: no cover
        vendor = str(Path(self.config.vendor_dir).resolve())
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        try:
            from modules.face_analyser import get_face_analyser  # type: ignore
            from modules.processors.frame.face_swapper import get_face_swapper  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Deep-Live-Cam modules unavailable: {exc}") from exc
        # A real swap would run face detection + inswapper inference on `frame`.
        # We deliberately do not fabricate an output; if the API surface differs
        # from what we expect this raises and swap() falls back to passthrough.
        get_face_analyser()
        get_face_swapper()
        raise NotImplementedError("Deep-Live-Cam in-process swap requires GPU + models")

    def _swap_via_facefusion(self, frame: Any) -> Any:  # pragma: no cover
        try:
            import facefusion  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"facefusion module unavailable: {exc}") from exc
        raise NotImplementedError("facefusion in-process swap requires GPU + models")

    def is_healthy(self) -> bool:
        if self.passthrough:
            return True  # degraded but still serving (passthrough)
        if self.process is None:
            return False
        return self.process.poll() is None

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend,
            "available": self.available,
            "model_present": self.model_present,
            "gpu_present": self.gpu_present,
            "degraded": self.degraded,
            "passthrough": self.passthrough,
            "process_running": self.process is not None and self.process.poll() is None,
            "binary": " ".join(self.binary) if self.binary else None,
            "error": self.error,
            "last_startup_error": self.last_startup_error,
        }


# ---------------------------------------------------------------------------
# Frame I/O (pluggable, test-friendly)
# ---------------------------------------------------------------------------
class FrameSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> "np.ndarray | None": ...
    def is_open(self) -> bool: ...
    def close(self) -> None: ...


class FrameSink(Protocol):
    def open(self) -> None: ...
    def write(self, frame: "np.ndarray") -> None: ...
    def is_open(self) -> bool: ...
    def close(self) -> None: ...


class ListFrameSource:
    """Yields frames from an in-memory list. For tests and offline demos."""

    def __init__(self, frames: list[Any], loop: bool = False) -> None:
        self._frames = list(frames)
        self._idx = 0
        self._loop = loop
        self._open = True

    def open(self) -> None:
        self._open = True
        self._idx = 0

    def read(self) -> "np.ndarray | None":
        if not self._open or not self._frames:
            return None
        if self._idx >= len(self._frames):
            if self._loop:
                self._idx = 0
            else:
                return None
        frame = self._frames[self._idx]
        self._idx += 1
        return frame

    def is_open(self) -> bool:
        if not self._open:
            return False
        return self._loop or self._idx < len(self._frames)

    def close(self) -> None:
        self._open = False


class ListFrameSink:
    """Collects frames in memory. For tests and inspection."""

    def __init__(self) -> None:
        self.frames: list[Any] = []
        self._open = True

    def open(self) -> None:
        self._open = True

    def write(self, frame: "np.ndarray") -> None:
        if self._open:
            self.frames.append(frame)

    def is_open(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False


class OpenCVFrameSource:
    """Captures frames from a URL, device, or file via OpenCV.

    OpenCV is imported lazily so the adapter still imports in environments
    without ``cv2`` (e.g. CI). Constructing this source when cv2 is missing
    raises at :meth:`open`, which callers must guard.
    """

    def __init__(self, source: str, frame_rate: int = 25) -> None:
        self.source = source
        self.frame_rate = frame_rate
        self._cap = None
        self._cv2 = None

    def open(self) -> None:
        import cv2  # lazy

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open frame source: {self.source}")

    def read(self) -> "np.ndarray | None":
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VirtualCamSink:
    """Writes frames to a v4l2loopback virtual camera (or stream) via FFmpeg.

    Only used on a live, online backend. OpenCV is imported lazily.
    """

    def __init__(self, output_target: str, width: int = 1280, height: int = 720, fps: int = 25) -> None:
        self.output_target = output_target
        self.width = width
        self.height = height
        self.fps = fps
        self._writer = None

    def open(self) -> None:
        import cv2  # lazy

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(self.output_target, fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open virtual camera: {self.output_target}")

    def write(self, frame: "np.ndarray") -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def is_open(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class FaceSwapPipeline:
    """Feeds source frames through the backend manager into a sink.

    Supports both a one-shot :meth:`process_frame` (used by unit tests and
    per-frame callers) and a streaming :meth:`run` loop (used in production).
    """

    def __init__(
        self,
        source: FrameSource,
        sink: FrameSink,
        manager: BackendManager,
        backend: str = "facefusion",
    ) -> None:
        self.source = source
        self.sink = sink
        self.manager = manager
        self.backend = backend
        self.frames_processed = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_ts = time.time()

    def process_frame(self, frame: Any) -> Any:
        out = self.manager.swap(frame)
        record_swap(self.backend, mode="swap" if not self.manager.passthrough else "passthrough")
        self.frames_processed += 1
        return out

    def step(self) -> bool:
        frame = self.source.read()
        if frame is None:
            return False
        self.sink.write(self.process_frame(frame))
        return True

    def run(self, max_frames: int | None = None) -> int:
        count = 0
        self._last_ts = time.time()
        while self._running:
            if not self.step():
                break
            count += 1
            if max_frames and count >= max_frames:
                break
        if count > 1 and self._last_ts:
            set_swap_fps(count / max(time.time() - self._last_ts, 1e-6))
        return count

    def start_loop(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class FaceSwapAdapter(Renderer):
    """Renderer that performs face swapping on the avatar video stream.

    Backend-agnostic (FaceFusion or Deep-Live-Cam). On activation it detects
    backend / model / GPU availability and either spawns the backend or degrades
    to a transparent passthrough. Implements the :class:`Renderer` interface so
    it can be selected as a drop-in renderer in the orchestrator.

    Lifecycle / DI: ``backend_manager``, ``frame_source``, ``frame_sink`` and
    ``swap_callable`` are injectable so the pipeline can be unit-tested without
    spawning a real backend or requiring cv2 / numpy video stacks.
    """

    def __init__(
        self,
        config: FaceSwapConfig | None = None,
        *,
        backend: str | None = None,
        enabled: bool | None = None,
        vendor_dir: str | None = None,
        source_face_path: str | None = None,
        device: str | None = None,
        input_source: str | None = None,
        output_virtual_cam: str | None = None,
        output_stream_url: str | None = None,
        frame_rate: int | None = None,
        require_gpu: bool | None = None,
        backend_manager: BackendManager | None = None,
        frame_source: FrameSource | None = None,
        frame_sink: FrameSink | None = None,
        swap_callable: Callable[[Any], Any] | None = None,
    ) -> None:
        cfg = config or FaceSwapConfig()
        overrides = {
            k: v
            for k, v in dict(
                backend=backend,
                enabled=enabled,
                vendor_dir=vendor_dir,
                source_face_path=source_face_path,
                device=device,
                input_source=input_source,
                output_virtual_cam=output_virtual_cam,
                output_stream_url=output_stream_url,
                frame_rate=frame_rate,
                require_gpu=require_gpu,
            ).items()
            if v is not None
        }
        if overrides:
            cfg = cfg.model_copy(update=overrides)
        self.config = cfg

        self.character_index: CharacterIndex | None = None
        self.source_reference: TrainingReference | None = None
        self.source_image_path: str | None = self.config.source_face_path
        self.behavior: AvatarBehaviorState | None = None
        self.active_style: VisualStyle | None = None
        self.active_background: BackgroundSpec | None = None
        self.watermark = "Synthetic avatar output - consent required for real identities"

        self.manager = backend_manager or BackendManager(self.config, swap_callable=swap_callable)
        self.frame_source = frame_source
        self.frame_sink = frame_sink
        self.pipeline: FaceSwapPipeline | None = None

        self.replacement_active = False
        self.last_error: str | None = None
        self._activate()

    # ---- source face selection ------------------------------------------
    def _select_source_face(self, character_index: CharacterIndex) -> TrainingReference | None:
        identity_anchor = next(
            (
                ref
                for ref in character_index.training_references
                if ref.role == "identity_anchor" and Path(ref.path).exists()
            ),
            None,
        )
        if identity_anchor is not None:
            return identity_anchor
        canonical = Path(character_index.canonical_image)
        if canonical.exists():
            return TrainingReference(
                id="canonical_identity_anchor",
                path=str(canonical),
                role="identity_anchor",
                state="neutral",
                weight=1.0,
                tags=["canonical", "identity", "neutral"],
            )
        return None

    # ---- activation ------------------------------------------------------
    def _activate(self) -> None:
        self.last_error = None
        set_degraded(self.config.backend, True)

        if not self.config.enabled:
            self.replacement_active = False
            self.last_error = "Face-swap renderer selected but disabled."
            logger.info(
                "faceswap disabled",
                extra={"audit": {"event": "faceswap.disabled", "backend": self.config.backend}},
            )
            return

        if self.character_index is None and self.source_image_path is None:
            self.replacement_active = False
            self.last_error = "No character / source face loaded."
            return

        if self.source_image_path is None and self.character_index is not None:
            ref = self._select_source_face(self.character_index)
            if ref is not None:
                self.source_reference = ref
                self.source_image_path = ref.path

        self.manager.config = self.config
        self.manager.start(self.source_image_path)

        self.replacement_active = self.manager.available and not self.manager.degraded
        set_degraded(self.config.backend, self.manager.degraded)

        if self.replacement_active:
            self._build_pipeline()
            logger.info(
                "faceswap active",
                extra={"audit": {"event": "faceswap.active", "backend": self.config.backend}},
            )
        else:
            self.last_error = self.manager.error or "Face-swap backend unavailable; passthrough active."
            logger.warning(
                "faceswap degraded/passthrough",
                extra={
                    "audit": {
                        "event": "faceswap.passthrough",
                        "backend": self.config.backend,
                        "reason": self.last_error,
                    }
                },
            )

    def _build_pipeline(self) -> None:
        if self.frame_source is None:
            try:
                self.frame_source = OpenCVFrameSource(self.config.input_source, self.config.frame_rate)
            except Exception as exc:  # pragma: no cover - needs cv2 + a live source
                logger.warning("faceswap could not build frame source", extra={"audit": {"event": "faceswap.no_source", "error": str(exc)}})
                return
        if self.frame_sink is None:
            target = self.config.output_virtual_cam or self.config.output_stream_url
            if target:
                try:
                    self.frame_sink = VirtualCamSink(target, fps=self.config.frame_rate)
                except Exception as exc:  # pragma: no cover
                    logger.warning("faceswap could not build frame sink", extra={"audit": {"event": "faceswap.no_sink", "error": str(exc)}})
                    return
        if self.frame_source is not None and self.frame_sink is not None:
            self.pipeline = FaceSwapPipeline(
                self.frame_source, self.frame_sink, self.manager, self.config.backend
            )
            try:
                self.frame_source.open()
                self.frame_sink.open()
                self.pipeline.start_loop()
            except Exception as exc:  # pragma: no cover
                logger.warning("faceswap pipeline could not start", extra={"audit": {"event": "faceswap.pipeline_failed", "error": str(exc)}})
                self.pipeline = None

    # ---- Renderer interface ---------------------------------------------
    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index
        self.source_reference = self._select_source_face(character_index)
        self.source_image_path = self.source_reference.path if self.source_reference else None
        self._activate()

    def set_theme(
        self,
        character_index: CharacterIndex,
        style: VisualStyle | None,
        background: BackgroundSpec | None,
    ) -> None:
        self.character_index = character_index
        self.active_style = style
        self.active_background = background
        self.source_reference = self._select_source_face(character_index)
        self.source_image_path = self.source_reference.path if self.source_reference else None
        self._activate()

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        if behavior.lip_sync_enabled:
            return
        self.behavior = behavior

    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None:
        self.behavior = behavior
        if self.pipeline is None and self.config.enabled:
            self._activate()

    def interrupt(self) -> None:
        self.behavior = AvatarBehaviorState(mode="recovering", affect="reset", gaze_target="soft_forward")
        if self.pipeline is not None:
            self.pipeline.stop()
        self.replacement_active = False

    # ---- observability ---------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        m = self.manager
        return {
            "backend": self.config.backend,
            "enabled": self.config.enabled,
            "online": m.available and not m.degraded,
            "model_present": m.model_present,
            "degraded": m.degraded,
            "passthrough": m.passthrough,
            "replacement_active": self.replacement_active,
            "source_image_present": bool(self.source_image_path and Path(self.source_image_path).exists()),
            "source_image_path": self.source_image_path,
            "source_reference_id": self.source_reference.id if self.source_reference else None,
            "source_reference_role": self.source_reference.role if self.source_reference else None,
            "device": self.config.device,
            "require_gpu": self.config.require_gpu,
            "gpu_present": m.gpu_present,
            "input_source": self.config.input_source,
            "output_target": self.config.output_virtual_cam or self.config.output_stream_url,
            "output_virtual_cam": self.config.output_virtual_cam,
            "output_stream_url": self.config.output_stream_url,
            "frame_rate": self.config.frame_rate,
            "swap_threshold": self.config.swap_threshold,
            "vendor_dir": str(self.config.vendor_dir),
            "vendor_dir_exists": Path(self.config.vendor_dir).exists(),
            "backend_binary": " ".join(m.binary) if m.binary else None,
            "process_running": m.process is not None and m.process.poll() is None,
            "watermark": self.watermark,
            "error": self.last_error or m.error,
            # backward-compatible aliases used by older tests/UI
            "replacement_active_legacy": self.replacement_active,
        }

    def health(self) -> dict[str, Any]:
        caps = self.capabilities()
        status = "ok" if caps["online"] else ("degraded" if self.config.enabled else "ok")
        return {"status": status, "backend": self.config.backend, "detail": caps}

    def shutdown(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.manager.stop()
        if self.frame_source is not None:
            try:
                self.frame_source.close()
            except Exception:
                pass
        if self.frame_sink is not None:
            try:
                self.frame_sink.close()
            except Exception:
                pass
