"""``sidecar/app.py`` — FastAPI service exposing FaceFusion over HTTP.

Why HTTP (not gRPC, not WebSockets, not subprocess stdout):

* Multipart JPEG (q85, ~80KB per 640x480 frame) travels an internal Docker
  network in 1–5 ms one-way. gRPC would add protobuf codegen + a new
  dependency for ~no measurable win on a 25 fps pipeline.
* HTTP request/response semantics make the daemon's reconnect logic
  trivial: a 5xx or ConnectError falls through to passthrough straight
  away. No state machine to manage.
* ``Bearer`` shared-secret in ``Authorization`` keeps the door closed if
  someone accidentally publishes the Docker network behind a public
  load balancer. Token is read from ``FACESWAP__SIDECAR__API_KEY`` so it
  flows through the same env-var overlay as the main process.

Run inside the sidecar: ``uvicorn sidecar.app:app --host 0.0.0.0 --port 8001``.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, Response

from .facefusion_runner import FaceFusionRunner

logger = logging.getLogger("sidecar.facefusion")
logging.basicConfig(level=os.getenv("SIDECAR_LOG_LEVEL", "INFO"))


# Lazy-encoded JPEG quality. Same q85 we use elsewhere in LiveEmote so a
# request encoding cost is amortised; tweak via env without redeploys.
JPEG_QUALITY = int(os.getenv("FACESWAP__SIDECAR__JPEG_QUALITY", "85"))
VENDOR_DIR = Path(os.getenv("FACESWAP__SIDECAR__VENDOR_DIR", "/app/vendor/FaceFusion"))
API_KEY = os.getenv("FACESWAP__SIDECAR__API_KEY", "")
AUTH_REQUIRED = os.getenv("FACESWAP__SIDECAR__AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")

app = FastAPI(title="LiveEmote FaceFusion Sidecar", version="0.1.0")
_runner = FaceFusionRunner(vendor_dir=VENDOR_DIR)


# -----------------------------------------------------------------------------
# Auth — Bearer shared-secret when FACESWAP__SIDECAR__AUTH_REQUIRED=true. We
# keep auth OPT-IN so a localhost-only docker-compose deployment doesn't need
# any secret wiring to develop against. In production set both env vars.
# -----------------------------------------------------------------------------
def _require_bearer(authorization: str | None = None) -> None:
    if not AUTH_REQUIRED:
        return
    # FastAPI request injection below — see Depends.
    pass


async def _bearer_dep(authorization: str | None = None) -> None:
    if not AUTH_REQUIRED:
        return
    if not API_KEY:
        # Misconfiguration: auth required but no key set. Lock everything out.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="sidecar auth not configured (FACESWAP__SIDECAR__API_KEY missing)",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid bearer token",
        )


# -----------------------------------------------------------------------------
# Health + readiness — used by main process startup probe and k8s readiness.
# -----------------------------------------------------------------------------
@app.get("/health")
async def health(dep: None = Depends(_bearer_dep)) -> JSONResponse:
    snap = _runner.health
    body = {
        "status": "ok" if snap.healthy else "degraded",
        "vendor_present": _runner.vendor_present,
        "vendor_dir": snap.vendor_dir,
        "face_analyser_loaded": snap.face_analyser_loaded,
        "face_swapper_loaded": snap.face_swapper_loaded,
        "loaded_at": snap.loaded_at,
        "swap_count": snap.swap_count,
        "last_swap_ms": snap.last_swap_ms,
        "last_error": snap.last_error,
    }
    return JSONResponse(body, status_code=200 if snap.healthy else 503)


@app.post("/warmup")
async def warmup(dep: None = Depends(_bearer_dep)) -> JSONResponse:
    """Force the runner to load face_analyser + face_swapper ahead of the
    first swap. Useful in production to absorb the 1.5–2.5 s ONNX warmup
    into container startup instead of the first user frame."""
    snap = _runner.warmup()
    return JSONResponse(
        {"healthy": snap.healthy, "last_error": snap.last_error},
        status_code=200 if snap.healthy else 503,
    )


# -----------------------------------------------------------------------------
# Inference — POST multipart: frame + source_face (+ optional target_face
# + intensity form field). Returns image/jpeg bytes.
# -----------------------------------------------------------------------------
def _decode_image(raw: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG bytes into a BGR ndarray (OpenCV convention used by
    FaceFusion). Returns ``None`` if cv2 is missing or bytes are invalid —
    the caller should 4xx in that case."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpeg(image: np.ndarray) -> bytes:
    import cv2  # type: ignore
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode returned False")
    return buf.tobytes()


@app.post("/api/v1/swap")
async def swap(
    frame: UploadFile,
    source_face: UploadFile,
    target_face: UploadFile | None = None,
    intensity: float = Form(1.0),
    dep: None = Depends(_bearer_dep),
):
    """Accept three image uploads (frame / source_face / optional
    target_face) and an ``intensity`` form field, return JPEG bytes of
    the swapped frame. Always returns 200 + image bytes even when the
    swap was a no-op passthrough so the main process can naively
    write whatever we send back; we ALSO set ``X-Swap-Mode: passthrough``
    on the response when we couldn't run inference so observability
    surfaces the silent no-op."""
    try:
        frame_img = _decode_image(await frame.read())
        source_img = _decode_image(await source_face.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode uploaded images: {exc}")

    if frame_img is None or source_img is None:
        raise HTTPException(status_code=400, detail="cv2 unavailable or invalid image format")

    target_img: np.ndarray | None = None
    if target_face is not None:
        try:
            target_img = _decode_image(await target_face.read())
        except Exception:
            target_img = None  # optional — swap uses source_face as the target on miss

    # ---------------- inference ----------------
    source_face_obj = _runner.extract_face(source_img)
    target_face_obj = _runner.extract_face(target_img) if target_img is not None else None

    if source_face_obj is None:
        # No identity anchor available; return the frame JPEG passthrough.
        body = _encode_jpeg(frame_img)
        return Response(
            content=body,
            media_type="image/jpeg",
            headers={"X-Swap-Mode": "passthrough", "X-Swap-Reason": "no_source_face"},
        )

    out = _runner.swap(
        frame=frame_img,
        source_face=source_face_obj,
        target_face=target_face_obj,
        intensity=float(intensity),
    )
    if out is frame_img:
        return Response(
            content=_encode_jpeg(out),
            media_type="image/jpeg",
            headers={"X-Swap-Mode": "passthrough"},
        )

    try:
        body = _encode_jpeg(out)
    except Exception as exc:
        # Encoding fallback so the main pipeline never blocks.
        logger.warning("could not encode swap result: %s", exc)
        body = _encode_jpeg(frame_img)
        return Response(
            content=body,
            media_type="image/jpeg",
            headers={"X-Swap-Mode": "passthrough"},
        )
    return Response(
        content=body,
        media_type="image/jpeg",
        headers={"X-Swap-Mode": "swap", "X-Latency-Ms": str(int(_runner.health.last_swap_ms or 0))},
    )


# -----------------------------------------------------------------------------
# Run via ``uvicorn sidecar.app:app --host 0.0.0.0 --port 8001``.
# Eager-warm on startup so the readiness probe reaps the boot cost once.
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def _on_startup() -> None:  # pragma: no cover - exercised on container start
    try:
        _runner.warmup()
    except Exception as exc:  # pragma: no cover - never bring the container down
        logger.warning("eager warmup at startup failed: %s", exc)
