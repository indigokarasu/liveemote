"""``DeepLiveCamVendorDaemon`` — wraps the vendored Deep-Live-Cam pipeline.

The vendored repo (``vendor/Deep-Live-Cam``) provides the production pipeline:

* ``modules.face_analyser`` — RetinaFace detection + ArcFace recognition
* ``modules.processors.frame.face_swapper`` — Inswapper ONNX + Poisson blending
* ``modules.processors.frame.face_enhancer`` — GFPGAN enhancement (optional)

This daemon lazy-imports those modules on first :meth:`swap` call, extracts
the source/target face embeddings, and applies the swap to the per-frame
``frame`` numpy array. On any failure (vendor missing, vendor dir absent,
``, no ``cv2`` / ``onnxruntime`` / ``insightface`` importable, no face detected
in source image) the daemon falls back to passthrough and reports the reason
through :meth:`health`.

The exact face-swap primitive invoked is:

    modules.processors.frame.face_swapper.get_face_swapper().get(
        target_frame=<numpy bgr frame>,
        source_face=<face analyser result from source image>,
    )

which mirrors Deep-Live-Cam's documented single-frame helper. That helper is
not always exposed as a public function in every Deep-Live-Cam release — we
keep the call site behind :meth:`_apply_swap` for clean test overrides and
release-drift tolerance.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base_daemon import _VendorDaemonBase
from .facefusion_adapter import VendorDaemon  # re-export for type clarity

logger = logging.getLogger(__name__)


# Root of the vendored Deep-Live-Cam checkout, relative to the repo root.
DEFAULT_VENDOR_DIR = Path("vendor/Deep-Live-Cam")


class DeepLiveCamVendorDaemon(_VendorDaemonBase):
    """Face-swap backend daemon wrapping the vendored Deep-Live-Cam code."""

    backend_label = "deeplivecam"

    def __init__(
        self,
        vendor_dir: Path | str | None = None,
        enable_enhancer: bool = False,
    ) -> None:
        super().__init__(backend_name="deeplivecam")
        self._configured_vendor_dir = Path(vendor_dir) if vendor_dir else DEFAULT_VENDOR_DIR
        self.enable_enhancer = enable_enhancer

    # ----------------------------------------------------------- vendored load --
    def _load_vendor(self) -> Path:
        """Lightweight probe: confirm ``vendor/Deep-Live-Cam`` is on disk +
        its top-level ``modules`` package imports cleanly.

        We deliberately do NOT import ``modules.face_analyser`` (heavy
        sklearn + insightface + buffalo_l model pack) or
        ``modules.processors.frame.face_swapper`` (inswapper_128.onnx on
        GPU provider chains) here. Both are lazily resolved in
        :meth:`_extract_face` and :meth:`_apply_swap`. That keeps the
        ``health()["loaded"]`` contract meaningful on CPU-only / no-model
        CI boxes while still letting the same daemon do real inference
        once a machine has ``make setup`` + the model pack on disk.
        """
        vendor_path = self._configured_vendor_dir.resolve()
        if not vendor_path.exists():
            raise FileNotFoundError(
                f"vendor directory not present: {vendor_path}. "
                f"Run `make setup` (pulls hacksider/Deep-Live-Cam into {vendor_path})."
            )

        self._append_vendor_to_syspath(vendor_path)

        try:
            modules_pkg = self._safe_import("modules")
        except (ImportError, ModuleNotFoundError, Exception) as exc:
            raise ImportError(
                f"Deep-Live-Cam 'modules' not importable: {exc}"
            ) from exc

        self.modules["modules_pkg"] = modules_pkg
        return vendor_path

    # ----------------------------------------------------------------- helpers --
    def _extract_face(self, image: Any, *, role: str) -> Any:
        """Lazy-load ``modules.face_analyser`` and call
        :func:`get_one_face` on the image.

        On any failure (analyser not importable on this Python build,
        sklearn missing, buffalo_l model pack absent, no face detected)
        we return ``None`` so the base class falls back to passthrough.
        """
        if image is None:
            return None
        face_analyser = self.modules.get("face_analyser")
        if face_analyser is None:
            try:
                face_analyser = self._safe_import("modules.face_analyser")
                self.modules["face_analyser"] = face_analyser
            except (ImportError, ModuleNotFoundError, Exception) as exc:
                logger.warning(
                    "Deep-Live-Cam 'modules.face_analyser' unavailable; "
                    "face extraction passthrough: %s",
                    exc,
                )
                return None
        try:
            return face_analyser.get_one_face(image)
        except Exception:
            return None

    def _apply_swap(self, frame: Any, source_face: Any, target_face: Any) -> Any:
        """Lazy-load ``modules.processors.frame.face_swapper`` and call its
        real public :func:`swap_face(source_face, target_face, temp_frame)`
        primitive.

        On any failure (the single-file module not importable on this
        build, ONNX model missing on disk, GPU provider chain invalid,
        ``swap_face`` raises during inference) the frame is returned
        unchanged so passthrough stays consistent across every failure.
        """
        if source_face is None or target_face is None or frame is None:
            return frame
        face_swapper_mod = self.modules.get("face_swapper")
        if face_swapper_mod is None:
            try:
                face_swapper_mod = self._safe_import(
                    "modules.processors.frame.face_swapper"
                )
                self.modules["face_swapper"] = face_swapper_mod
            except (ImportError, ModuleNotFoundError, Exception) as exc:
                logger.warning(
                    "Deep-Live-Cam 'modules.processors.frame.face_swapper' "
                    "unavailable; swap passthrough: %s",
                    exc,
                )
                return frame
        try:
            return face_swapper_mod.swap_face(source_face, target_face, frame)
        except Exception:
            return frame
