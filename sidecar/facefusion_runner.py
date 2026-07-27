"""``sidecar/facefusion_runner.py`` — FaceFusion inference isolated in its own
container.

The FaceFusion 3.x tree hard-requires ``typing.NotRequired`` (Python 3.11+) and
pins a heavy dependency graph (insightface, onnxruntime-gpu, scikit-learn,
cython). The main LiveEmote process is Python 3.10 to stay slim; this runner
is what runs *inside* the sidecar where Python 3.11 is available. The runner
exposes a tiny interface (face extraction + frame swap) so the FastAPI app
:mod:`sidecar.app` can stay thin and unit-testable without ever importing
``facefusion`` itself.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- typing shim --
# Vendored FaceFusion core.py + types.py use ``typing.NotRequired`` (3.11+).
# The sidecar is required to be 3.11+ in production, but the runner is also
# import-tested on 3.10 CI hosts where ``facefusion`` is absent — install a
# safe ``NotRequired`` shim from typing_extensions so non-FF importers don't
# trip over a future regression.
try:
    import typing as _typing
    if not hasattr(_typing, "NotRequired"):
        from typing_extensions import NotRequired as _TERNotRequired
        _typing.NotRequired = _TERNotRequired  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - typing_extensions missing
    pass

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Latched health state for /health endpoint."""

    healthy: bool = False
    vendor_dir: str | None = None
    face_analyser_loaded: bool = False
    face_swapper_loaded: bool = False
    loaded_at: float | None = None
    last_error: str | None = None
    swap_count: int = 0
    last_swap_ms: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class FaceFusionRunner:
    """Lazy-loaded FaceFusion inference seam.

    Construction does NOT import the vendored tree — we only verify the
    directory exists. The first :meth:`extract_face` call performs the
    heavy import (face_analyser brings in sklearn + insightface); the
    first :meth:`swap` call imports face_swapper.core. This two-phase
    shape keeps ``/health`` cheap so k8s readiness probes don't pay for
    280MB of model load on every poll.
    """

    def __init__(self, vendor_dir: Path | str = Path("vendor/FaceFusion")) -> None:
        self.vendor_dir = Path(vendor_dir).resolve()
        self._face_analyser = None
        self._face_swapper_core = None
        self.health = HealthSnapshot(vendor_dir=str(self.vendor_dir))

    @property
    def vendor_present(self) -> bool:
        return self.vendor_dir.exists()

    def warmup(self) -> HealthSnapshot:
        """Probe the vendor tree and import face_analyser. Face_swap"""
        if not self.vendor_present:
            self.health.healthy = False
            self.health.last_error = f"vendor dir missing: {self.vendor_dir}"
            logger.warning(
                "facefusion vendor dir missing: %s", self.vendor_dir,
                extra={"audit": {"event": "sidecar.vendor_missing", "path": str(self.vendor_dir)}}
            )
            return self.health

        # Append vendor dir to sys.path lazily on first warmup so module
        # resolution succeeds.
        import sys
        s = str(self.vendor_dir)
        if s not in sys.path:
            sys.path.insert(0, s)

        try:
            import facefusion  # noqa: F401 - structural probe
        except Exception as exc:
            self.health.last_error = f"facefusion import failed: {exc}"
            logger.warning("facefusion root import failed", extra={"audit": {"event": "sidecar.vendor_import_failed", "error": str(exc)}})
            return self.health

        # face_analyser import is heavy (sklearn). Skip on warmup failure.
        try:
            self._face_analyser = self._import_module("facefusion.face_analyser")
            self.health.face_analyser_loaded = True
        except Exception as exc:
            self.health.last_error = f"face_analyser import failed: {exc}"
            logger.warning("facefusion.face_analyser import failed", extra={"audit": {"event": "sidecar.face_analyser_failed", "error": str(exc)}})
            return self.health

        self.health.healthy = True
        self.health.loaded_at = time.time()
        logger.info(
            "facefusion runner ready",
            extra={"audit": {"event": "sidecar.ready", "vendor_dir": str(self.vendor_dir)}}
        )
        return self.health

    def extract_face(self, image: Any) -> Any:
        """Extract a single face embedding from a BGR numpy image. Returns
        ``None`` on any failure (no detector, no face present) so the
        caller can decide whether to passthrough or call swap with
        ``source_face=None``."""
        if self._face_analyser is None:
            if not self.warmup().healthy:
                return None
        try:
            return self._face_analyser.get_one_face(image, position=0)
        except Exception as exc:
            logger.warning("face extract failed", extra={"audit": {"event": "sidecar.extract_failed", "error": str(exc)}})
            return None

    def swap(
        self,
        frame: Any,
        source_face: Any,
        *,
        target_face: Any | None = None,
        intensity: float = 1.0,
    ) -> Any:
        """Apply FaceFusion's ``process_frame`` to ``frame`` using
        ``source_face`` as the identity anchor.

        ``target_face`` is optional; FaceFusion's core will fall back to
        using ``frame`` itself as the target when omitted. ``intensity``
        is forwarded via FF's documented ``face_swapper_blend`` field.

        Returns the swapped frame (BGR ndarray) on success; ``frame``
        unchanged on any failure. Never raises — the sidecar's whole
        point is that the main process should never see an exception
        from a vendor dependency.
        """
        if source_face is None or frame is None:
            return frame
        if self._face_swapper_core is None:
            try:
                self._face_swapper_core = self._import_module(
                    "facefusion.processors.modules.face_swapper.core"
                )
                self.health.face_swapper_loaded = True
            except Exception as exc:
                logger.warning(
                    "facefusion.processors.modules.face_swapper.core unavailable: %s",
                    exc,
                    extra={"audit": {"event": "sidecar.core_import_failed", "error": str(exc)}}
                )
                return frame

        t0 = time.perf_counter()
        try:
            target = target_face if target_face is not None else frame
            inputs = {
                "reference_vision_frame": target,
                "source_vision_frames": [source_face],
                "target_vision_frames": [target],
                "temp_vision_frame": frame,
                "temp_vision_mask": None,
            }
            out_frame, _mask = self._face_swapper_core.process_frame(inputs)
            self.health.swap_count += 1
            self.health.last_swap_ms = (time.perf_counter() - t0) * 1000.0
            return out_frame
        except Exception as exc:
            logger.warning("facefusion swap failed: %s", exc, extra={"audit": {"event": "sidecar.swap_failed", "error": str(exc)}})
            return frame

    @staticmethod
    def _import_module(name: str) -> Any:
        import importlib
        return importlib.import_module(name)
