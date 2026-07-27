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
import typing
from pathlib import Path
from typing import Any

from .base_daemon import _VendorDaemonBase
from .facefusion_adapter import VendorDaemon  # re-export for type clarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- typing shim --
# FaceFusion 3.x's vendored ``types.py`` uses ``typing.NotRequired`` which
# only landed in Python 3.11. On 3.10/3.9 the import crashes the moment any
# FF module is touched (``'type' object is not subscriptable`` because a
# plain ``type()`` proxy doesn't support ``[List[Face]]`` subscripting like
# a real ``_SpecialForm`` does). ``typing_extensions`` ships a proper
# ``NotRequired`` SpecialForm for older Python versions; we forward-install
# it under the ``typing`` namespace before any FF import succeeds. If
# ``typing_extensions`` itself is missing the vendored FF won't import on
# 3.10 — the daemon degrades to passthrough via the existing fallback in
# :meth:`_load_vendor`.
try:
    from typing_extensions import NotRequired as _TERNotRequired
    if not hasattr(typing, "NotRequired"):
        typing.NotRequired = _TERNotRequired  # type: ignore[attr-defined]
except ImportError:
    pass  # typing_extensions missing — vendored FF won't import on 3.10, daemon degrades passthrough


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
        """Lightweight probe: verify the FaceFusion package root can be
        imported under our sys.path.

        We deliberately do NOT import ``facefusion.face_analyser`` or
        ``facefusion.processors.modules.face_swapper.core`` here. The core
        module uses ``typing.NotRequired`` (Python 3.11+) and the analyser
        pulls in heavy ONNX/insightface deps. Both are imported lazily in
        :meth:`_extract_face` and :meth:`_apply_swap` instead. That keeps
        ``health()["loaded"]`` honest on CPU-only boxes where the heavy
        modules aren't usable yet, and lets the same daemon serve on
        ``make setup`` machines without a 280MB buffalo_l model pack.
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
        except (ImportError, ModuleNotFoundError, Exception) as exc:
            # ``Exception`` covers the Python 3.10 typing.NotRequired case
            # before our module-level shim is fully effective in race-prone
            # imports; degrade with a clear reason instead of crashing.
            raise ImportError(
                f"facefusion package not importable: {exc}"
            ) from exc

        self.modules["facefusion_root"] = facefusion_root
        return vendor_path

    # ----------------------------------------------------------------- helpers --
    def _extract_face(self, image: Any, *, role: str) -> Any:
        """Lazy-load ``facefusion.face_analyser`` on first call and run
        :func:`get_one_face` (or a guarded fallback) on the image.

        On any failure (vendored analyser not importable on this Python
        build, missing buffalo_l model pack, no face detected) we return
        ``None`` so the base class's caller falls through to passthrough.
        """
        if image is None:
            return None
        face_analyser = self.modules.get("face_analyser")
        if face_analyser is None:
            try:
                face_analyser = self._safe_import("facefusion.face_analyser")
                self.modules["face_analyser"] = face_analyser
            except (ImportError, ModuleNotFoundError, Exception) as exc:
                logger.warning(
                    "facefusion.face_analyser unavailable; face extraction passthrough: %s",
                    exc,
                )
                return None
        try:
            # FaceFusion accepts a numpy BGR ndarray; ``position`` selects the
            # face by index when multiple are present.
            return face_analyser.get_one_face(image, position=0)
        except Exception:
            return None

    def _apply_swap(self, frame: Any, source_face: Any, target_face: Any) -> Any:
        """Lazy-load ``facefusion.processors.modules.face_swapper.core`` on
        first call and invoke its ``process_frame`` per the documented
        ``inputs``-dict contract.

        On any failure (vendor's ``core`` module can't import on this
        build — even with our ``NotRequired`` shim, downstream
        ``state_manager.get_item(...)`` will KeyError if no CLI has run —
        no face detected, ONNX model unavailable, GPU absent) the frame is
        returned unchanged so the upstream render path never blocks.
        """
        if source_face is None or target_face is None or frame is None:
            return frame
        face_swapper_core = self.modules.get("face_swapper_core")
        if face_swapper_core is None:
            try:
                face_swapper_core = self._safe_import(
                    "facefusion.processors.modules.face_swapper.core"
                )
                self.modules["face_swapper_core"] = face_swapper_core
            except (ImportError, ModuleNotFoundError, Exception) as exc:
                logger.warning(
                    "facefusion.processors.modules.face_swapper.core unavailable; "
                    "swap passthrough: %s",
                    exc,
                )
                return frame
        process_fn = getattr(face_swapper_core, "process_frame", None)
        if process_fn is None:
            return frame
        try:
            out_frame, _out_mask = process_fn(
                {
                    "reference_vision_frame": target_face,
                    "source_vision_frames": [source_face],
                    "target_vision_frames": [target_face],
                    "temp_vision_frame": frame,
                    "temp_vision_mask": None,
                }
            )
            return out_frame
        except Exception:
            return frame
