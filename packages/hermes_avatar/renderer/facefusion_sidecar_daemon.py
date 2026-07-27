"""``FaceFusionSidecarDaemon`` — talks to the FaceFusion HTTP sidecar.

The main LiveEmote process runs Python 3.10 and skips FaceFusion entirely.
We instead POST each frame to ``sidecar/app.py`` running inside the
``sidecar/Dockerfile.facefusion`` container (or any host running the same
FastAPI service). The daemon implements the existing :class:`VendorDaemon`
ABC so it drops into ``FaceSwapAdapter._build_vendor_daemon()`` whenever
``FACESWAP__SIDECAR__URL`` is configured.

Why this lives in the main process (instead of in the sidecar):

* The main process owns the face-swap adapter state, the Prometheus
  counters (``faceswap_swaps_total``), and the per-frame render path.
  Having the daemon live where the rest of the wiring already is means
  we don't need to plumb a "sidecar picks adapter backend" flag through
  the orchestrator — we just check the config and construct either this
  daemon or the existing :class:`FaceFusionVendorDaemon` in-process.
* All failure modes short-circuit to passthrough and surface a clear
  reason via :meth:`health`. The base contract is unchanged — the
  adapter sees a ``VendorDaemon`` either way.

Multiplexing: every swap is a single, short-lived request. We do NOT
keep a persistent stream open (WebSockets would add reconnect/backoff
state for no real benefit at 25 fps). Connection errors and 5xx
transform into immediate passthrough; a subsequent swap retries
transparently.
"""
from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base_daemon import _VendorDaemonBase
from .facefusion_adapter import SwapRequest, VendorDaemon  # type: ignore

logger = logging.getLogger(__name__)


# Default URL is the surface the docker-compose service exposes on the
# internal ``liveemote`` network. Override at runtime via
# ``FACESWAP__SIDECAR__URL``.
DEFAULT_SIDECAR_URL = "http://127.0.0.1:8001"
DEFAULT_CONNECT_TIMEOUT_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 8.0
DEFAULT_HEALTH_CACHE_S = 5.0


def _resolve_sidecar(url: str) -> tuple[str, str | None]:
    """Validate and split the sidecar URL into (base, path). Returns
    ``("...", None)`` if the URL is missing or malformed."""
    if not url:
        return ("", None)
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return ("", None)
        return (f"{parsed.scheme}://{parsed.netloc}", parsed.path or "/api/v1/swap")
    except Exception:
        return ("", None)


def _b64_jpeg(image: Any) -> bytes | None:
    """Encode a BGR ndarray as JPEG q85. Returns None if cv2 / image is
    unavailable. Mirrors the sidecar's own encoder so request sizes match."""
    if image is None:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return buf.tobytes()


def _decode_jpeg(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class FaceFusionSidecarDaemon(VendorDaemon):
    """HTTP-facing VendorDaemon for the FaceFusion sidecar.

    The base ``_VendorDaemonBase`` lazily probes *vendored* Python modules
    on disk for the :class:`FaceFusionVendorDaemon`; that's not what we
    want here — we want a connection probe to a remote HTTP service. So
    we implement :class:`VendorDaemon` directly with the same swap/health
    contract.
    """

    backend_label = "facefusion-sidecar"

    def __init__(
        self,
        sidecar_url: str | None = None,
        api_key: str | None = None,
        *,
        source_image_path: str | str | None = None,
        connect_timeout_s: float | None = None,
        request_timeout_s: float | None = None,
        jpeg_quality: int = 85,
    ) -> None:
        # Only fall back to the localhost default when caller passed None
        # (`is None`). An explicit empty string / malformed URL means
        # "no sidecar configured" and should keep the daemon in the
        # malformed-URL passthrough path.
        self.sidecar_url_in = (
            sidecar_url if sidecar_url is not None else DEFAULT_SIDECAR_URL
        )
        self.base_url, self.swap_path = _resolve_sidecar(self.sidecar_url_in)
        self.api_key = api_key or None
        self.source_image_path = source_image_path
        self.connect_timeout_s = float(connect_timeout_s or DEFAULT_CONNECT_TIMEOUT_S)
        self.request_timeout_s = float(request_timeout_s or DEFAULT_REQUEST_TIMEOUT_S)
        self.jpeg_quality = int(jpeg_quality)

        # Health is cached so a tight-loop render path doesn't ping the
        # sidecar every frame. The cache busts on every swap() success
        # and on every health failure.
        self._health_lock = threading.Lock()
        self._health_cache_ttl = DEFAULT_HEALTH_CACHE_S
        self._health_cached_at: float | None = None
        self._health_cached: dict[str, Any] | None = None
        self._swaps_attempted = 0
        self._swaps_succeeded = 0
        self._passthrough_count = 0
        self._last_error: str | None = None
        self._is_loaded = bool(self.base_url)
        self._degraded_reason = (
            "sidecar URL malformed"
            if not self.base_url
            else "not yet probed"
        )

    # --------------------------------------------------------- VendorDaemon ABC
    def health(self) -> dict[str, Any]:
        cached = self._maybe_get_cached_health()
        if cached is not None:
            return cached
        fresh = self._probe_health()
        with self._health_lock:
            self._health_cached = fresh
            self._health_cached_at = _now()
        return fresh

    def swap(self, req: SwapRequest) -> Any:
        if not self.base_url:
            self._passthrough_count += 1
            self._last_error = "sidecar url malformed"
            return req.frame

        self._swaps_attempted += 1

        # Resolve source + target face images from disk-shaped paths. We
        # don't introspect the sidecar's face extraction here; that's
        # its job. We DO need raw BGR / file bytes from the daemon side
        # to send the request.
        try:
            import httpx  # type: ignore
        except Exception as exc:
            self._passthrough_count += 1
            self._last_error = f"httpx unavailable: {exc}"
            return req.frame

        try:
            source_b64 = _read_image_bytes(req.source_face)
            if source_b64 is None:
                # No source face loaded; the sidecar will passthrough on
                # missing-face detection, and we mirror that here.
                self._passthrough_count += 1
                return req.frame
            frame_b64 = _encode_request_frame(req.frame)
            if frame_b64 is None:
                # cv2 missing or invalid frame; can't build a request.
                self._passthrough_count += 1
                self._last_error = "cv2 unavailable or invalid frame"
                return req.frame

            files = {
                "frame": ("frame.jpg", frame_b64, "image/jpeg"),
                "source_face": ("source.jpg", source_b64, "image/jpeg"),
            }
            target_b64 = _read_image_bytes(req.target_face) if req.target_face else None
            if target_b64 is not None:
                files["target_face"] = ("target.jpg", target_b64, "image/jpeg")

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            with httpx.Client(timeout=self.request_timeout_s) as client:
                resp = client.post(
                    f"{self.base_url}{self.swap_path}",
                    files=files,
                    data={"intensity": str(float(req.intensity or 1.0))},
                    headers=headers,
                )
        except Exception as exc:
            self._passthrough_count += 1
            self._last_error = f"sidecar request failed: {exc}"
            logger.info(
                "facefusion sidecar unreachable; passthrough",
                extra={"audit": {"event": "faceswap.sidecar_unreachable", "error": str(exc)}},
            )
            return req.frame

        if resp.status_code >= 500:
            self._passthrough_count += 1
            self._last_error = f"sidecar 5xx: {resp.status_code}"
            return req.frame

        if resp.status_code >= 400:
            # 4xx is a contract issue — the sidecar told us our request
            # is malformed. Log + passthrough (do not retry-storm).
            self._passthrough_count += 1
            self._last_error = f"sidecar 4xx: {resp.status_code}"
            logger.warning(
                "facefusion sidecar rejected request: %s",
                resp.text[:200],
                extra={"audit": {"event": "faceswap.sidecar_4xx", "status": resp.status_code}},
            )
            return req.frame

        out_img = _decode_response_image(resp.content)
        if out_img is None:
            self._passthrough_count += 1
            self._last_error = "sidecar returned non-image body"
            return req.frame

        self._swaps_succeeded += 1
        # Invalidate cached health so the next call re-probes (success).
        self._health_cached = None
        self._health_cached_at = None
        return out_img

    # ----------------------------------------------------------------- helpers
    def _maybe_get_cached_health(self) -> dict[str, Any] | None:
        if self._health_cached is None or self._health_cached_at is None:
            return None
        if _now() - self._health_cached_at > self._health_cache_ttl:
            return None
        return self._health_cached

    def _probe_health(self) -> dict[str, Any]:
        # Short-circuit when the URL was malformed at construction time so
        # the cached reason surfaces (don't bother the network with garbage).
        if not self.base_url:
            return {
                "backend": self.backend_label,
                "loaded": False,
                "degraded": True,
                "reason": "sidecar url malformed",
                "vendor_dir": None,
                "sidecar_url": self.sidecar_url_in,
                "swaps_attempted": self._swaps_attempted,
                "swaps_succeeded": self._swaps_succeeded,
                "passthrough_count": self._passthrough_count,
                "last_error": self._last_error,
            }

        try:
            import httpx  # type: ignore
        except Exception as exc:
            return {
                "backend": self.backend_label,
                "loaded": False,
                "degraded": True,
                "reason": f"httpx unavailable: {exc}",
                "vendor_dir": None,
                "sidecar_url": self.sidecar_url_in,
            }

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            with httpx.Client(timeout=self.connect_timeout_s) as client:
                resp = client.get(f"{self.base_url}/health", headers=headers)
        except Exception as exc:
            return {
                "backend": self.backend_label,
                "loaded": False,
                "degraded": True,
                "reason": f"sidecar unreachable: {exc}",
                "vendor_dir": None,
                "sidecar_url": self.sidecar_url_in,
            }

        ok = resp.status_code == 200 and (resp.headers.get("content-type", "").startswith("application/json"))
        body = {}
        try:
            body = resp.json()
        except Exception:
            ok = False
        return {
            "backend": self.backend_label,
            "loaded": bool(ok) and bool(body.get("vendor_present") or body.get("status") == "ok"),
            "degraded": not ok,
            "reason": None if ok else f"sidecar unhealthy: status={resp.status_code}",
            "vendor_dir": body.get("vendor_dir"),
            "sidecar_url": self.sidecar_url_in,
            "sidecar_status": body.get("status"),
            "sidecar_swap_count": body.get("swap_count"),
            "sidecar_last_swap_ms": body.get("last_swap_ms"),
            "swaps_attempted": self._swaps_attempted,
            "swaps_succeeded": self._swaps_succeeded,
            "passthrough_count": self._passthrough_count,
            "last_error": self._last_error,
        }


def _now() -> float:
    import time
    return time.time()


def _read_image_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        # Fall back to raw bytes — the sidecar's PIL decoder will pick
        # up the file extension.
        return p.read_bytes()
    img = cv2.imread(str(p))
    if img is None:
        return p.read_bytes()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return buf.tobytes()


def _encode_request_frame(frame: Any) -> bytes | None:
    """Accept a BGR ndarray, return JPEG bytes for the request body."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return buf.tobytes()


def _decode_response_image(raw: bytes) -> Any:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
