from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import (
    BackgroundSpec,
    CharacterIndex,
    TrainingReference,
    VisualStyle,
)
from .base import Renderer

logger = logging.getLogger(__name__)

# Signature-level defaults. These are referenced both by FaceSwapAdapter.__init__
# and by DeepLiveCamAdapter (a subclass) so that explicit overrides win over
# config-provided values while unset arguments fall back to config then to these.
_DEFAULT_VENDOR_DIR = "vendor/Deep-Live-Cam"
_DEFAULT_FACEFUSION_DIR = "vendor/facefusion"
_DEFAULT_BACKEND = "auto"  # auto | facefusion | deeplivecam
_DEFAULT_DEVICE = "cpu"  # cpu | cuda
_DEFAULT_WATERMARK = "Synthetic avatar output - consent required for real identities"


def _device_to_provider(device: str) -> str:
    """Map a logical device ("cpu"/"cuda") to an execution-provider token.

    Both FaceFusion and Deep-Live-Cam accept ``cpu`` / ``cuda`` execution
    providers; this keeps the mapping in one place so callers pass a logical
    device rather than backend-specific strings.
    """
    return "cuda" if str(device).lower().startswith("cuda") else "cpu"


class BackendManager:
    """Locates and (optionally) supervises a local face-swap backend.

    Two backends are supported:

    * **facefusion** -- OpenRAIL-AS headless CLI with a job-based architecture.
    * **deeplivecam** -- the vendored ``Deep-Live-Cam`` (AGPL-3.0) ``run.py``.

    Discovery is purely filesystem-based: we look for a known CLI entrypoint
    under the configured vendor directories. Nothing is imported from the
    vendored code at module load time (its heavy deps -- insightface, onnx,
    cv2 -- are absent in many environments), so importing this module is always
    safe.

    The manager operates in one of two modes:

    * **on-demand** (default): no persistent process. Each frame/image swap is
      performed by spawning a short-lived subprocess. ``start()`` simply marks
      the manager ready; ``online`` tracks backend availability.
    * **supervised**: if ``launch_command`` is provided, a long-lived backend
      process is spawned and tracked (heartbeat + exit detection + shutdown).
      This lets an operator run a sidecar server (e.g. facefusion's job API)
      on a GPU host while this adapter supervises it.
    """

    def __init__(
        self,
        vendor_dir: str = _DEFAULT_VENDOR_DIR,
        facefusion_dir: str = _DEFAULT_FACEFUSION_DIR,
        backend: str = _DEFAULT_BACKEND,
        device: str = _DEFAULT_DEVICE,
        models_dir: str | None = None,
        process_timeout: float = 30.0,
        heartbeat_interval: float = 5.0,
        extra_args: list[str] | None = None,
        launch_command: list[str] | None = None,
    ) -> None:
        self.vendor_dir = Path(vendor_dir)
        self.facefusion_dir = Path(facefusion_dir)
        self.backend_choice = backend
        self.device = device
        self.models_dir = Path(models_dir) if models_dir else None
        self.process_timeout = process_timeout
        self.heartbeat_interval = heartbeat_interval
        self.extra_args = list(extra_args or [])
        self.launch_command = list(launch_command) if launch_command else None

        self._discovered: dict[str, Any] | None = self._discover()
        self._process: subprocess.Popen | None = None
        self._online: bool = False
        self._last_health: float | None = None
        self._last_error: str | None = None

    # ------------------------------------------------------------------ discovery
    def _discover(self) -> dict[str, Any] | None:
        """Return the first usable backend entrypoint, or ``None`` if none found.

        When ``backend_choice`` is ``auto`` we prefer facefusion (its headless
        job architecture is the most server-friendly) and fall back to
        Deep-Live-Cam. Explicit choices only consider that backend.
        """
        candidates: list[tuple[str, Path]] = []
        if self.backend_choice in ("auto", "facefusion"):
            candidates.append(("facefusion", self.facefusion_dir))
        if self.backend_choice in ("auto", "deeplivecam"):
            candidates.append(("deeplivecam", self.vendor_dir))

        for name, base in candidates:
            entrypoint = self._find_entrypoint(name, base)
            if entrypoint is not None:
                return {"name": name, "entrypoint": entrypoint, "dir": base}
        return None

    @staticmethod
    def _find_entrypoint(name: str, base: Path) -> Path | None:
        """Locate a CLI entrypoint for the named backend under ``base``."""
        if name == "facefusion":
            for cand in (base / "facefusion.py", base / "run.py", base / "__main__.py"):
                if cand.is_file():
                    return cand
            # facefusion installed as a package module
            if (base / "facefusion" / "__main__.py").is_file():
                return base / "facefusion" / "__main__.py"
        else:  # deeplivecam
            for cand in (base / "run.py", base / "DeepLiveCam.py", base / "main.py"):
                if cand.is_file():
                    return cand
            # Vendored checkout with a modules/ package but no top-level run.py
            # (common). The canonical entrypoint is modules/run or run.py.
            if (base / "modules").is_dir():
                alt = base / "run.py"
                if alt.is_file():
                    return alt
        return None

    def is_available(self) -> bool:
        """True if a CLI entrypoint was discovered (regardless of model presence)."""
        return self._discovered is not None

    def availability_detail(self) -> dict[str, Any] | None:
        """Honest description of what was discovered (or why nothing was)."""
        if self._discovered is None:
            return {
                "name": None,
                "entrypoint": None,
                "dir": None,
                "searched": [str(self.facefusion_dir), str(self.vendor_dir)],
            }
        detail = dict(self._discovered)
        detail["entrypoint"] = str(detail["entrypoint"])
        detail["dir"] = str(detail["dir"])
        detail["models_available"] = self.models_available()
        return detail

    def models_available(self) -> bool:
        """Best-effort check for required ONNX models.

        If an explicit ``models_dir`` is configured we require it to contain at
        least one ``.onnx`` file. For Deep-Live-Cam we look for the inswapper
        model under ``<dir>/models``. For facefusion the model cache layout is
        backend-managed and not reliably detectable, so we optimistically
        return ``True`` and let the subprocess surface a real error if a model
        is genuinely missing (fail-honest at execution time, not at probe time).
        """
        if self.models_dir is not None:
            models = Path(self.models_dir)
            if not models.exists():
                return False
            return any(models.glob("*.onnx")) or any(models.glob("*.onnx.*"))
        if self._discovered is None:
            return False
        if self._discovered["name"] == "deeplivecam":
            inswapper = self._discovered["dir"] / "models" / "inswapper_128.onnx"
            return inswapper.exists()
        return True

    # ------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Bring the backend online.

        In supervised mode this spawns ``launch_command`` and tracks it. In
        on-demand mode there is no persistent process, so this simply records
        readiness. Returns ``True`` if the backend is usable afterwards.
        """
        if not self.is_available():
            self._last_error = "Cannot start: face-swap backend not available."
            self._online = False
            logger.warning(
                "faceswap backend start skipped",
                extra={"audit": {"event": "faceswap.backend_unavailable", "error": self._last_error}},
            )
            return False

        if self._process is not None:
            return True

        if self.launch_command:
            try:
                self._process = subprocess.Popen(
                    self.launch_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._online = True
                self._last_health = time.time()
                logger.info(
                    "faceswap backend process started",
                    extra={"audit": {"event": "faceswap.backend_started", "cmd": self.launch_command}},
                )
            except Exception as exc:  # pragma: no cover - depends on host
                self._online = False
                self._last_error = f"Failed to launch backend: {exc}"
                logger.error(self._last_error)
                return False
        else:
            # On-demand mode: ready to swap per-frame via subprocess.
            self._online = True
            self._last_health = time.time()
        return True

    def healthcheck(self) -> bool:
        """Update and return ``online`` status (supervised mode only really needs this)."""
        if self._process is not None:
            if self._process.poll() is not None:
                self._online = False
                self._last_error = f"Backend process exited (code {self._process.returncode})."
                logger.warning(self._last_error)
            else:
                self._online = True
                self._last_health = time.time()
        return self._online

    def stop(self) -> None:
        """Terminate a supervised backend process if one is running."""
        if self._process is not None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except Exception:
                    self._process.kill()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("error stopping backend: %s", exc)
            finally:
                self._process = None
        self._online = False

    @property
    def online(self) -> bool:
        return self._online

    # --------------------------------------------------------- command building
    def _build_swap_command(self, source: str, target: str, output: str) -> list[str]:
        """Build the CLI command for a one-shot image/video swap.

        Raises ``RuntimeError`` if no backend is available so callers fail
        honestly instead of silently producing no-op output.
        """
        if not self.is_available():
            raise RuntimeError("No face-swap backend available; cannot build swap command.")
        provider = _device_to_provider(self.device)
        name = self._discovered["name"]
        python = sys.executable or "python"
        entry = str(self._discovered["entrypoint"])

        if name == "facefusion":
            cmd = [
                python, entry, "headless-run",
                "--source", source,
                "--target", target,
                "--output", output,
                "--execution-providers", provider,
            ]
        else:  # deeplivecam
            cmd = [
                python, entry,
                "--source", source,
                "--target", target,
                "--output", output,
                "--frame-processor", "inswapper_128",
                "--execution-provider", provider,
                "--many-faces",
                "--skip-download",
            ]
        cmd.extend(self.extra_args)
        return cmd

    def swap_image(self, source: str, target: str, output: str) -> dict[str, Any]:
        """Run a one-shot swap of ``target`` using ``source`` face -> ``output``.

        Returns a structured result dict. Raises ``RuntimeError`` only when the
        backend is entirely unavailable; transient subprocess failures are
        returned as ``{"ok": False, ...}`` so callers can degrade gracefully.
        """
        cmd = self._build_swap_command(source, target, output)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.process_timeout,
                check=False,
            )
            ok = proc.returncode == 0 and Path(output).exists()
            result = {
                "ok": ok,
                "returncode": proc.returncode,
                "output": output,
                "cmd": cmd,
                "stdout": (proc.stdout or "")[-2000:],
                "stderr": (proc.stderr or "")[-2000:],
            }
            if not ok:
                logger.warning(
                    "faceswap swap_image failed",
                    extra={"audit": {"event": "faceswap.swap_failed", "stderr": result["stderr"]}},
                )
            return result
        except subprocess.TimeoutExpired:
            logger.warning("faceswap swap_image timed out after %ss", self.process_timeout)
            return {"ok": False, "error": "timeout", "cmd": cmd}
        except Exception as exc:
            logger.warning("faceswap swap_image error: %s", exc)
            return {"ok": False, "error": str(exc), "cmd": cmd}

    def swap_frame(self, frame: Any, source: str) -> Any:
        """Swap a single in-memory frame (numpy array) using ``source``.

        ``cv2`` is imported lazily inside this method so the module imports
        cleanly even when OpenCV is absent. The frame is written to a temp file,
        swapped through the CLI, and read back -- correct (if not the fastest)
        for both supported backends. If OpenCV is missing we raise so the caller
        can fall back to pass-through.
        """
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - env-dependent
            raise RuntimeError("cv2/numpy unavailable; cannot process frame in-memory.") from exc

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tgt = Path(td) / "frame.png"
            out = Path(td) / "swap.png"
            ok_encode = cv2.imwrite(str(tgt), frame)
            if not ok_encode:
                raise RuntimeError("cv2.imwrite failed to serialize frame.")
            res = self.swap_image(str(tgt), str(tgt), str(out))
            if not res.get("ok"):
                raise RuntimeError(f"backend swap failed: {res.get('stderr') or res.get('error')}")
            swapped = cv2.imread(str(out))
            if swapped is None:
                raise RuntimeError("cv2.imread failed to load swapped output.")
            return swapped


class FaceSwapAdapter(Renderer):
    """Local-subprocess face-swap renderer backed by FaceFusion or Deep-Live-Cam.

    Unlike the prior ``DeepLiveCamAdapter`` stub, this adapter drives a real
    backend: it discovers a CLI entrypoint, supervises (or on-demand invokes) the
    backend, and performs actual frame/image swaps. When no backend, models, or
    GPU is present it degrades **gracefully and honestly** -- reporting
    ``replacement_active=False`` with a truthful ``last_error`` and passing frames
    through unchanged rather than faking success.

    The full :class:`Renderer` interface is implemented; ``load_character`` /
    ``set_theme`` / ``set_behavior`` / ``speak`` refresh the swap session state,
    and :meth:`process_frame` / :meth:`swap_image` are the actual processing paths
    a compositor would call.
    """

    backend_label = "faceswap"

    def __init__(
        self,
        enabled: bool = False,
        vendor_dir: str = _DEFAULT_VENDOR_DIR,
        config: Any = None,
        backend: str = _DEFAULT_BACKEND,
        device: str = _DEFAULT_DEVICE,
        source_image: str | None = None,
        models_dir: str | None = None,
        facefusion_dir: str = _DEFAULT_FACEFUSION_DIR,
        process_timeout: float = 30.0,
        heartbeat_interval: float = 5.0,
        watermark: str = _DEFAULT_WATERMARK,
        extra_args: list[str] | None = None,
        launch_command: list[str] | None = None,
    ) -> None:
        cfg = config
        # enabled: explicit True wins; otherwise fall back to config.
        self.enabled = enabled or (bool(getattr(cfg, "enabled", False)) if cfg is not None else False)

        # Resolve each setting: explicit kwarg wins over config over signature default.
        vd = vendor_dir if vendor_dir != _DEFAULT_VENDOR_DIR else (getattr(cfg, "vendor_dir", _DEFAULT_VENDOR_DIR) if cfg else _DEFAULT_VENDOR_DIR)
        ffd = facefusion_dir if facefusion_dir != _DEFAULT_FACEFUSION_DIR else (getattr(cfg, "facefusion_dir", _DEFAULT_FACEFUSION_DIR) if cfg else _DEFAULT_FACEFUSION_DIR)
        bk = backend if backend != _DEFAULT_BACKEND else (getattr(cfg, "backend", _DEFAULT_BACKEND) if cfg else _DEFAULT_BACKEND)
        dv = device if device != _DEFAULT_DEVICE else (getattr(cfg, "device", _DEFAULT_DEVICE) if cfg else _DEFAULT_DEVICE)
        md = models_dir if models_dir is not None else (getattr(cfg, "models_dir", None) if cfg else None)
        si = source_image if source_image is not None else (getattr(cfg, "source_image", None) if cfg else None)
        pto = process_timeout if process_timeout != 30.0 else (getattr(cfg, "process_timeout", 30.0) if cfg else 30.0)
        hb = heartbeat_interval if heartbeat_interval != 5.0 else (getattr(cfg, "heartbeat_interval", 5.0) if cfg else 5.0)
        wm = watermark if watermark != _DEFAULT_WATERMARK else (getattr(cfg, "watermark", _DEFAULT_WATERMARK) if cfg else _DEFAULT_WATERMARK)
        ea = extra_args if extra_args is not None else (list(getattr(cfg, "extra_args", []) or []) if cfg else None)
        lc = launch_command if launch_command is not None else (list(getattr(cfg, "launch_command", []) or []) if cfg else None)

        self.vendor_dir = Path(vd)
        self.facefusion_dir = Path(ffd)
        self.backend_choice = bk
        self.device = dv
        self.models_dir = Path(md) if md else None
        self.source_image_override = si
        self.process_timeout = float(pto)
        self.heartbeat_interval = float(hb)
        self.watermark = wm
        self.extra_args = list(ea or [])
        self.launch_command = list(lc) if lc else None

        self.backend = BackendManager(
            vendor_dir=str(self.vendor_dir),
            facefusion_dir=str(self.facefusion_dir),
            backend=self.backend_choice,
            device=self.device,
            models_dir=str(self.models_dir) if self.models_dir else None,
            process_timeout=self.process_timeout,
            heartbeat_interval=self.heartbeat_interval,
            extra_args=self.extra_args,
            launch_command=self.launch_command,
        )

        self.character_index: CharacterIndex | None = None
        self.active_style: VisualStyle | None = None
        self.active_background: BackgroundSpec | None = None
        self.behavior: AvatarBehaviorState | None = None
        self.source_reference: TrainingReference | None = None
        self.source_image_path: str | None = None
        self.replacement_active = False
        self._interrupted = False
        self.last_error: str | None = None

        # Bring the backend online if it is available; otherwise stay honest.
        if self.enabled and self.backend.is_available():
            self.backend.start()

    # ----------------------------------------------------------- capabilities
    def capabilities(self) -> dict[str, Any]:
        detail = self.backend.availability_detail()
        if detail is not None:
            backend_name = detail.get("name")
            backend_discovered = detail.get("entrypoint")
        else:
            backend_name = None
            backend_discovered = None
        return {
            "backend": self.backend_label,
            "enabled": self.enabled,
            "backend_name": backend_name,
            "backend_available": self.backend.is_available(),
            "backend_discovered": backend_discovered,
            "models_available": self.backend.models_available(),
            "online": self.backend.online,
            "replacement_active": self.replacement_active,
            "interrupted": self._interrupted,
            "source_image_path": self.source_image_path,
            "source_reference_id": self.source_reference.id if self.source_reference else None,
            "source_reference_role": self.source_reference.role if self.source_reference else None,
            "canonical_image": self.character_index.canonical_image if self.character_index else None,
            "vendor_dir_exists": self.vendor_dir.exists(),
            "device": self.device,
            "watermark": self.watermark,
            "error": self.last_error,
        }

    # -------------------------------------------------------------- Renderer API
    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index
        self._interrupted = False
        self.source_reference = self._select_source_face(character_index)
        self.source_image_path = self.source_reference.path if self.source_reference else None
        self._recompute()

    def set_theme(
        self,
        character_index: CharacterIndex,
        style: VisualStyle | None,
        background: BackgroundSpec | None,
    ) -> None:
        self.character_index = character_index
        self.active_style = style
        self.active_background = background
        self._interrupted = False
        self.source_reference = self._select_source_face(character_index)
        self.source_image_path = self.source_reference.path if self.source_reference else None
        self._recompute()

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        # Lip-sync renderers provide their own face; do not swap over them.
        if behavior.lip_sync_enabled:
            return
        self.behavior = behavior
        self._interrupted = False
        self._recompute()

    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None:
        self.behavior = behavior
        self._interrupted = False
        self._recompute()
        if self.replacement_active:
            logger.info(
                "faceswap engaged for utterance",
                extra={"audit": {"event": "faceswap.speak", "audio_path": audio_path, "text_len": len(text)}},
            )

    def interrupt(self) -> None:
        self.behavior = AvatarBehaviorState(mode="recovering", affect="reset", gaze_target="soft_forward")
        self._interrupted = True
        self._recompute()

    # --------------------------------------------------------- processing paths
    def process_frame(self, frame: Any) -> Any:
        """Swap a single live frame. Passes through unchanged when inactive.

        Honest degradation: if the swap session is not active (backend missing,
        models absent, not enabled, or interrupted) the original frame is
        returned untouched -- we never claim a swap occurred.
        """
        if not self.replacement_active or self.source_image_path is None:
            return frame
        try:
            return self.backend.swap_frame(frame, self.source_image_path)
        except Exception as exc:
            logger.warning(
                "faceswap process_frame failed; passing frame through",
                extra={"audit": {"event": "faceswap.frame_error", "error": str(exc)}},
            )
            self.last_error = f"swap_frame failed: {exc}"
            return frame

    def swap_image(self, target: str, output: str, source: str | None = None) -> dict[str, Any]:
        """One-shot image/video swap. Returns a structured result dict."""
        src = source or self.source_image_path
        if not src:
            return {"ok": False, "error": "No source face image configured."}
        return self.backend.swap_image(src, target, output)

    # --------------------------------------------------------------- internals
    def _recompute(self) -> None:
        """Recompute ``replacement_active`` honestly from current state.

        ``True`` only when: enabled, a character+source is loaded, not
        interrupted, AND a real backend (with models) is available. Anything
        else yields ``False`` with a truthful ``last_error``.
        """
        self.last_error = None
        if not self.enabled:
            self.replacement_active = False
            self.last_error = "Face-swap renderer selected but not enabled."
            return
        if self.character_index is None:
            self.replacement_active = False
            self.last_error = "No character loaded."
            return
        if self.source_reference is None:
            self.source_reference = self._select_source_face(self.character_index)
            self.source_image_path = self.source_reference.path if self.source_reference else None
        if not self.source_image_path or not Path(self.source_image_path).exists():
            self.replacement_active = False
            self.last_error = f"Source face image not available: {self.source_image_path}"
            return
        if self._interrupted:
            self.replacement_active = False
            self.last_error = "Face replacement interrupted (session paused)."
            return
        if not self.backend.is_available():
            self.replacement_active = False
            self.last_error = (
                "Face-swap backend not available. No FaceFusion or Deep-Live-Cam CLI "
                f"found under {self.vendor_dir} or {self.facefusion_dir}."
            )
            return
        if not self.backend.models_available():
            self.replacement_active = False
            self.last_error = (
                "Face-swap backend present but required models (e.g. inswapper_128.onnx) "
                "are missing. Download models before enabling replacement."
            )
            return
        # All checks passed: a real swap session can be driven.
        self.replacement_active = True

    def _select_source_face(self, character_index: CharacterIndex) -> TrainingReference | None:
        """Pick the best identity-anchor face for swapping.

        Prefers an explicit ``identity_anchor`` training reference that exists on
        disk; falls back to the canonical character image (a single source face
        is sufficient for inswapper-style one-shot swapping).
        """
        if self.source_image_override and Path(self.source_image_override).exists():
            return TrainingReference(
                id="override_identity_anchor",
                path=self.source_image_override,
                role="identity_anchor",
                state="neutral",
                weight=1.0,
                tags=["override", "identity", "neutral"],
            )
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

    def __del__(self) -> None:
        try:
            self.backend.stop()
        except Exception:
            pass


class DeepLiveCamAdapter(FaceSwapAdapter):
    """Backward-compatible alias for the Deep-Live-Cam / FaceSwap backend.

    Preserves the historical ``DeepLiveCamAdapter(enabled=..., vendor_dir=...)``
    constructor signature and default behavior (``enabled=True`` when selected
    via the demo CLI) while inheriting the real backend-driven implementation.
    """

    backend_label = "deeplivecam"

    def __init__(
        self,
        enabled: bool = False,
        vendor_dir: str = _DEFAULT_VENDOR_DIR,
    ) -> None:
        super().__init__(enabled=enabled, vendor_dir=vendor_dir)
