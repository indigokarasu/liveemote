"""``FaceFusionVendorDaemon`` — wraps the vendored FaceFusion pipeline.

When ``vendor/FaceFusion`` is checked in (i.e. after a `make setup` of the
upstream facefusion/facefusion repo), this daemon lazy-imports the canonical
``facefusion`` import root and uses:

* ``facefusion.processors.frame.modules.face_swapper.process_frame`` — the
  per-frame swap primitive
* ``facefusion.modules.face_analyser.get_one_face`` — RetinaFace detection
  + ArcFace embedding extraction

Both calls are wrapped behind :meth:`_extract_face` and :meth:`_apply_swap`
respectively so a test subclass can stub them without an actual ONNX stack.

In this environment ``vendor/FaceFusion`` is not present, so construction
itself succeeds but :meth:`health` reports ``degraded`` and every
:meth:`swap` call returns ``req.frame`` unchanged. The first ``swap`` call
also triggers :meth:`_ensure_loaded`, which logs a clear warning about the
missing vendor and stays in passthrough mode forever after.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base_daemon import _VendorDaemonBase
from .facefusion_adapter import VendorDaemon  # re-export for type clarity

logger = logging.getLogger(__name__)


# Root of the vendored FaceFusion checkout, relative to the repo root.
DEFAULT_VENDOR_DIR = Path("vendor/FaceFusion")


class FaceFusionVendorDaemon(_VendorDaemonBase):
    """Face-swap backend daemon wrapping the vendored FaceFusion code."""

    backend_label = "facefusion"

    def __init__(
        self,
        vendor_dir: Path | str | None = None,
        enable_face_enhancer: bool = False,
    ) -> None:
        super().__init__(backend_name="facefusion")
        self._configured_vendor_dir = Path(vendor_dir) if vendor_dir else DEFAULT_VENDOR_DIR
        self.enable_face_enhancer = enable_face_enhancer

    # ----------------------------------------------------------- vendored load --
    def _load_vendor(self) -> Path:
        """Import the FaceFusion ``facefusion`` package root. Raises
        ``FileNotFoundError`` if ``vendor/FaceFusion`` is missing,
        ``ImportError`` if the facefusion package isn't on sys.path,
        ``ModuleNotFoundError`` if cv2/onnxruntime/insightface aren't
        installed.
        """
        vendor_path = self._configured_vendor_dir.resolve()
        if not vendor_path.exists():
            raise FileNotFoundError(
                f"vendor directory not present: {vendor_path}. "
                f"Run `make setup` (add a facefusion/facefusion clone at {vendor_path})."
            )

        self._append_vendor_to_syspath(vendor_path)

        try:
            facefusion_root = self._safe_import("facefusion")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"facefusion package not importable: {exc}"
            ) from exc

        # FaceFusion's per-frame swapper lives at the canonical path. We
        # don't eagerly resolve the entire package — only the modules we'll
        # actually call.
        try:
            face_analyser = self._safe_import("facefusion.modules.face_analyser")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"facefusion.modules.face_analyser not importable: {exc}"
            ) from exc
        try:
            face_swapper_mod = self._safe_import(
                "facefusion.processors.frame.modules.face_swapper"
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"facefusion.processors.frame.modules.face_swapper "
                f"not importable: {exc}"
            ) from exc

        self.modules.update(
            {
                "facefusion_root": facefusion_root,
                "face_analyser": face_analyser,
                "face_swapper": face_swapper_mod,
            }
        )

        if self.enable_face_enhancer:
            try:
                enhancer_mod = self._safe_import(
                    "facefusion.processors.frame.modules.face_enhancer"
                )
                self.modules["face_enhancer"] = enhancer_mod
            except (ImportError, ModuleNotFoundError) as exc:
                logger.info(
                    "FaceFusion enhancer unavailable; continuing without it: %s",
                    exc,
                )
        return vendor_path

    # ----------------------------------------------------------------- helpers --
    def _extract_face(self, image: Any, *, role: str) -> Any:
        """Detect + extract a single face embedding via FaceFusion's
        ``face_analyser.get_one_face`` (RetinaFace + ArcFace). Returns the
        vendor's face-vision-frame dict that ``process_frame`` consumes.
        """
        if image is None:
            return None
        face_analyser = self.modules["face_analyser"]
        try:
            # FaceFusion accepts a numpy BGR ndarray; ``position`` selects the
            # face by index when multiple are present. Most avatars have a
            # single dominant face; if not, callers can pre-filter upstream.
            return face_analyser.get_one_face(image, position=0)
        except Exception:
            return None

    def _apply_swap(self, frame: Any, source_face: Any, target_face: Any) -> Any:
        """Run FaceFusion's per-frame swap primitive.

        The documented signature is::

            process_frame(
                source_face: VisionFrame,
                reference_face: VisionFrame | None,
                target_vision_frame: VisionFrame,
            ) -> VisionFrame

        In this codebase ``source_face`` is the character's identity anchor
        (the canonical / identity_anchor PNG) and ``target_face`` is the
        active emote PNG selected from ``CharacterIndex.emotes``. We pass it
        through as the reference face so the vendored pipeline gets an actual
        face-similarity signal per frame.

        Returns the frame unchanged when the call or its prerequisites are
        unavailable (e.g. vendored face_swapper functions aren't loaded onto
        a registered processor chain).
        """
        if source_face is None or frame is None:
            return frame
        face_swapper_mod = self.modules["face_swapper"]

        # FaceFusion's process_frame is only wired up after its multi-step
        # job manager registers ``face_swapper`` as an active processor. We
        # call the underlying module function which exists even before
        # registration, so a test subclass can stub it freely.
        process_fn = getattr(face_swapper_mod, "process_frame", None)
        if process_fn is None:
            return frame

        try:
            return process_fn(source_face, target_face, frame)
        except Exception:
            return frame
