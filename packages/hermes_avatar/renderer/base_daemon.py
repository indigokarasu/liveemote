"""Shared base for vendored face-swap ``VendorDaemon`` implementations.

The two concrete daemons live in :mod:`packages.hermes_avatar.renderer.facefusion_daemon`
and :mod:`packages.hermes_avatar.renderer.deeplivecam_daemon`. They share enough
behaviour (lazy import, image caching, passthrough-on-failure, health reporting)
that a small base class is the right factoring.

Design contract (matches ``VendorDaemon`` ABC in :mod:`facefusion_adapter`):

* Constructor does **no** vendor-side work, no GPU checks, no cv2 / onnxruntime
  imports. Modules are only loaded the first time :meth:`swap` is called.
* Vendor dir absent → ``_is_degraded = True`` with the reason exposed via
  :meth:`health`. Frames are returned unchanged.
* cv2 / onnxruntime / insightface absent → same passthrough behaviour. The base
  never imports any of them at module load time.
* The first ``swap(req)`` call against a fresh daemon triggers
  :meth:`_load_vendor`. Subsequent calls reuse the cached extracted face for
  matching ``req.source_face`` / ``req.target_face`` paths. Path change =
  invalidation.
* Every state-changing step is independently overridable so tests can stub
  inference without standing up the heavy ONNX stack:

    - :meth:`_load_vendor`
    - :meth:`_load_image`
    - :meth:`_extract_face`
    - :meth:`_apply_swap`
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

from .facefusion_adapter import SwapRequest, VendorDaemon

logger = logging.getLogger(__name__)


class _VendorDaemonBase(VendorDaemon):
    """Shared lazy-load + image-cache + passthrough behaviour.

    Subclasses MUST override :meth:`_load_vendor` to import their backend's
    Python modules and return a truthy value on success. All vendor-specific
    inference happens inside :meth:`_apply_swap`, which receives already-loaded
    numpy frames (or any data type the subclass returns from
    :meth:`_extract_face`).
    """

    #: Set by :meth:`_load_vendor` with the absolute vendor root once imports
    #: succeed. ``None`` means we haven't tried loading yet.
    vendor_dir: Path | None = None

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        # Lazy: not loaded until first swap().
        self._is_loaded = False
        self._is_degraded = True
        self._degraded_reason = "vendor not yet probed"
        # Simple path -> face cache so repeated swap() calls with the same
        # source/target paths don't re-extract faces.
        self._last_source_path: str | None = None
        self._cached_source_face: Any = None
        self._last_target_path: str | None = None
        self._cached_target_face: Any = None
        # Vendor module surface populated by subclasses (e.g.
        # ``self.modules["face_analyser"] = fa``). Used by tests to inspect
        # that the right submodule actually got imported.
        self.modules: dict[str, Any] = {}

    # ------------------------------------------------------------------ ABC --
    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "loaded": self._is_loaded,
            "degraded": self._is_degraded,
            "reason": self._degraded_reason,
            "vendor_dir": str(self.vendor_dir) if self.vendor_dir else None,
        }

    def swap(self, req: SwapRequest) -> Any:
        """Run inference on ``req.frame`` using ``req.source_face`` (identity)
        and ``req.target_face`` (active emote) when both are present and on disk.

        Every failure mode short-circuits to passthrough: missing vendor dir,
        failed import, missing cv2 / onnxruntime, missing face in image, ONNX
        inference error. The frame is returned byte-identical to ``req.frame``
        in those cases so the upstream render path is never blocked.
        """
        # First-time vendor probe. Idempotent after success.
        if not self._is_loaded and not self._ensure_loaded():
            return req.frame

        try:
            source_img = self._load_image(req.source_face) if req.source_face else None
            target_img = (
                self._load_image(req.target_face) if req.target_face else None
            )
        except Exception as exc:  # pragma: no cover - exercised with cv2
            logger.warning(
                "vendor %s image load failed; passthrough",
                self.backend_name,
                extra={"audit": {"event": "faceswap.image_load_failed", "error": str(exc)}},
            )
            return req.frame

        if source_img is None or target_img is None:
            # Source is required (it's the identity anchor). Target being None
            # is fine — it falls back to "swap against the source identity
            # only" which vendors handle differently; we treat it as not enough
            # signal and pass through.
            return req.frame

        try:
            source_face = self._get_cached_face(
                path=req.source_face,
                image=source_img,
                cache_attr="_cached_source_face",
                last_attr="_last_source_path",
            )
            target_face = self._get_cached_face(
                path=req.target_face,
                image=target_img,
                cache_attr="_cached_target_face",
                last_attr="_last_target_path",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "vendor %s face extraction failed; passthrough",
                self.backend_name,
                extra={"audit": {"event": "faceswap.extract_failed", "error": str(exc)}},
            )
            return req.frame

        try:
            return self._apply_swap(req.frame, source_face, target_face)
        except Exception as exc:
            logger.warning(
                "vendor %s swap failed; passthrough",
                self.backend_name,
                extra={"audit": {"event": "faceswap.swap_failed", "error": str(exc)}},
            )
            return req.frame

    # ------------------------------------------------------------ overridable --
    def _ensure_loaded(self) -> bool:
        """Idempotent vendor import probe. Returns True if inference is
        available, False if we should passthrough forever.
        """
        if self._is_loaded:
            return True
        try:
            vendor_dir = self._load_vendor()
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
            self._is_degraded = True
            self._degraded_reason = f"vendor import failed: {exc}"
            logger.info(
                "vendor %s not available: %s",
                self.backend_name,
                exc,
                extra={"audit": {"event": "faceswap.vendor_unavailable", "backend": self.backend_name, "error": str(exc)}},
            )
            return False
        except Exception as exc:  # pragma: no cover - defensive
            self._is_degraded = True
            self._degraded_reason = f"vendor init error: {exc}"
            return False

        self.vendor_dir = vendor_dir
        self._is_loaded = True
        self._is_degraded = False
        self._degraded_reason = ""
        logger.info(
            "vendor %s loaded",
            self.backend_name,
            extra={"audit": {"event": "faceswap.vendor_loaded", "backend": self.backend_name}},
        )
        return True

    def _load_vendor(self) -> Path:
        """Subclasses implement: import the vendored Python modules, populate
        ``self.modules`` with handles to each imported submodule, return the
        resolved vendor root path. Raise ``(ImportError|ModuleNotFoundError|
        FileNotFoundError)`` on any failure.
        """
        raise NotImplementedError

    def _load_image(self, path: str | None) -> Any:
        """Read image at ``path`` to a numpy BGR ndarray. Returns ``None`` for
        a missing file or unsupported extension.
        """
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            import cv2  # type: ignore # lazy
        except Exception:
            return None
        return cv2.imread(str(p))

    def _extract_face(self, image: Any, *, role: str) -> Any:
        """Subclasses implement: detect + extract a face embedding from
        ``image``. Return any object the underlying vendor's ``_apply_swap``
        understands. Return ``None`` when no face is detected.
        """
        raise NotImplementedError

    def _apply_swap(self, frame: Any, source_face: Any, target_face: Any) -> Any:
        """Subclasses implement: run the actual ONNX swap on ``frame`` using
        the cached ``source_face`` and ``target_face`` embeddings. Must return
        a numpy ndarray of the same shape as ``frame`` (or the frame
        unchanged if the vendor can't operate).
        """
        raise NotImplementedError

    # -------------------------------------------------------------- internals --
    def _get_cached_face(self, *, path: str | None, image: Any, cache_attr: str, last_attr: str):
        """Extract a face embedding for ``image``, but reuse the cached result
        when ``path`` hasn't changed. Invalidates the cache automatically when
        the path shifts.
        """
        cached_path = getattr(self, last_attr)
        if path == cached_path and getattr(self, cache_attr) is not None:
            return getattr(self, cache_attr)
        role = "source" if cache_attr == "_cached_source_face" else "target"
        face = self._extract_face(image, role=role)
        setattr(self, cache_attr, face)
        setattr(self, last_attr, path)
        return face

    @staticmethod
    def _append_vendor_to_syspath(vendor_dir: Path) -> None:
        """Prepend the resolved vendor directory to ``sys.path`` so absolute
        imports like ``modules.face_analyser`` work. Uses a sentinel comment
        marker so downstream code can detect we did this.
        """
        s = str(vendor_dir.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)

    @staticmethod
    def _safe_import(name: str) -> Any:
        """Wrapped importlib call so tests can monkey-patch a single entry
        point.
        """
        return importlib.import_module(name)
