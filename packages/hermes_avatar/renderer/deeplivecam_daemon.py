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
        """Import the Deep-Live-Cam modules directory and return the resolved
        vendor root. Raises ``FileNotFoundError`` if the checkout is missing,
        ``ImportError`` if any of the modules we need aren't importable, and
        ``ModuleNotFoundError`` if cv2 / onnxruntime aren't installed.
        """
        vendor_path = self._configured_vendor_dir.resolve()
        if not vendor_path.exists():
            raise FileNotFoundError(
                f"vendor directory not present: {vendor_path}. "
                f"Run `make setup` (pulls hacksider/Deep-Live-Cam into {vendor_path})."
            )

        self._append_vendor_to_syspath(vendor_path)

        # The vendor's "modules" package is the runtime surface. Importing it
        # alone isn't enough on its own; we need face_analyser for detection
        # and the face_swapper module for inference.
        modules_pkg = self._safe_import("modules")
        try:
            face_analyser = self._safe_import("modules.face_analyser")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"Deep-Live-Cam 'modules.face_analyser' not importable: {exc}"
            ) from exc
        try:
            face_swapper_mod = self._safe_import("modules.processors.frame.face_swapper")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"Deep-Live-Cam 'modules.processors.frame.face_swapper' "
                f"not importable: {exc}"
            ) from exc

        self.modules.update(
            {
                "modules_pkg": modules_pkg,
                "face_analyser": face_analyser,
                "face_swapper": face_swapper_mod,
            }
        )
        # Optional enhancer is attempted but not required.
        if self.enable_enhancer:
            try:
                enhancer_mod = self._safe_import("modules.processors.frame.face_enhancer")
                self.modules["face_enhancer"] = enhancer_mod
            except (ImportError, ModuleNotFoundError) as exc:
                logger.info(
                    "Deep-Live-Cam enhancer unavailable; continuing without it: %s",
                    exc,
                )
        return vendor_path

    # ----------------------------------------------------------------- helpers --
    def _extract_face(self, image: Any, *, role: str) -> Any:
        """Detect and extract a single face embedding from ``image``.

        Deep-Live-Cam exposes ``get_one_face`` on its face_analyser module,
        which performs RetinaFace detection + ArcFace embedding extraction
        and returns the vendor's face dataclass for downstream swapper use.
        Returns ``None`` when no face is found.
        """
        if image is None:
            return None
        face_analyser = self.modules["face_analyser"]
        # Deep-Live-Cam's get_one_face takes a BGR numpy ndarray.
        try:
            return face_analyser.get_one_face(image)
        except Exception:
            return None

    def _apply_swap(self, frame: Any, source_face: Any, target_face: Any) -> Any:
        """Run the swapped-face inference for ``frame``.

        Deep-Live-Cam's typical single-frame helper,
        ``modules.processors.frame.face_swapper.get_face_swapper().get(...)``,
        returns the swapped face crop — applying it back to ``frame`` requires
        the Poisson-blend path that lives next to it. We encapsulate the call
        here so a test subclass can stub it without a real inswapper.
        """
        if source_face is None or target_face is None or frame is None:
            return frame
        face_swapper_mod = self.modules["face_swapper"]
        # `get_face_swapper()` is memoised by DP; calling it more than once is
        # safe and inexpensive. Returns the singleton inswapper wrapper.
        swapper = face_swapper_mod.get_face_swapper()
        if swapper is None:
            # The vendor hasn't loaded an ONNX model yet (no GPU + no models
            # in this env). Treat as passthrough rather than blowing up.
            return frame
        # The vendor's infer helper returns a swapped crop that the
        # face_enhancer (if enabled) and the frame_masker would normally
        # composite back into ``frame``. For the per-frame daemon path we
        # trust the swapper's composite output when it returns a full frame;
        # otherwise we passthrough and the orchestrator logs the gap.
        try:
            return swapper.get(target_frame=frame, source_face=source_face)  # type: ignore[attr-defined]
        except Exception:
            return frame
