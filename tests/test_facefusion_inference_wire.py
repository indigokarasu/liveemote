"""``tests/test_facefusion_inference_wire.py`` — proves the inference-side
wire contract inflates ``X-Swap-Mode: swap`` + ``X-Latency-Ms: NNN``
correctly when the ONNX step returns a real frame.

Substitutes for the GPU-required end-to-end: on this box we cannot
actually run ``facefusion.process_frame`` with real ONNX (Python 3.10 + no
docker + no GPU). The most informative test substitutes deterministic
stubs for the lazy-imported ``face_analyser`` + ``face_swapper.core``
modules, then exercises both the runner directly AND the FastAPI
``/api/v1/swap`` route end-to-end over :class:`httpx.ASGITransport`. When
the response carries ``X-Swap-Mode: swap`` + a real ``X-Latency-Ms``
header AND the response bytes differ from input, the inference wire
contract is proven — the GPU host only adds the actual ONNX step that
this test substitutes for.

Confidence basis (verified by re-reading the vendored source just now):
the runner's ``inputs`` dict keys + return tuple shape match exactly the
real signature at ``vendor/FaceFusion/facefusion/processors/modules/
face_swapper/core.py:process_frame``. Key names ``reference_vision_frame``,
``source_vision_frames``, ``target_vision_frames``, ``temp_vision_frame``,
``temp_vision_mask`` are all consumed by line with the same names inside
the vendored ONNX call. Return shape is ``(out_frame, mask)`` — two-tuple.
This stub matches what ONNX would have consumed; on a GPU host the only
thing that changes is that ``process_frame`` runs ONNX graph execution
instead of returning ``np.full_like(frame, 42)``.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

# ``sidecar/app.py`` reads these at import time, BEFORE any test runs.
# AUTH_REQUIRED=false so we don't need to add ``Authorization: Bearer`` to
# every request. The vendor dir env override path doesn't matter for this
# test — the lazy import is bypassed via monkeypatch.
os.environ.setdefault("FACESWAP__SIDECAR__AUTH_REQUIRED", "false")
os.environ.setdefault("FACESWAP__SIDECAR__API_KEY", "")
os.environ.setdefault("FACESWAP__SIDECAR__VENDOR_DIR", "/home/daytona/codebase/vendor/FaceFusion")

import numpy as np
import pytest
from PIL import Image
from httpx import ASGITransport, AsyncClient

import sidecar.app as sidecar_app  # noqa: E402


# ---------------------------------------------------------------------------
# Runner stubs — bypass the vendored FaceFusion import path entirely.
# ---------------------------------------------------------------------------
class _StubFaceAnalyser:
    """Always returns a sentinel truthy object — never inspects the image."""

    def get_one_face(self, image, position):  # noqa: ARG002 — positional matches FF
        return object()


class _StubFaceSwapperCore:
    """Always returns a solid frame distinguishable from any real input.
    Records invocations so the test asserts the runner fed it the
    documented FaceFusion ``inputs`` dict."""

    def __init__(self, fill: int = 42) -> None:
        self.fill = fill
        self.calls: list[dict] = []

    def process_frame(self, inputs: dict):
        self.calls.append(dict(inputs))
        frame = inputs["reference_vision_frame"]
        return np.full_like(frame, self.fill), None


@pytest.fixture
def stub_runner():
    """Replace the lazy-loaded heavy attributes on ``sidecar_app._runner``
    for the duration of one test. Yields the ``_StubFaceSwapperCore``
    recorder so the test asserts on what the runner fed the swapper."""
    pre_analyser = sidecar_app._runner._face_analyser
    pre_swapper = sidecar_app._runner._face_swapper_core
    core = _StubFaceSwapperCore(fill=42)
    sidecar_app._runner._face_analyser = _StubFaceAnalyser()
    sidecar_app._runner._face_swapper_core = core
    try:
        yield core
    finally:
        sidecar_app._runner._face_analyser = pre_analyser
        sidecar_app._runner._face_swapper_core = pre_swapper


# ---------------------------------------------------------------------------
# Byte fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def frame_bytes() -> bytes:
    """Encode a 480x640 BGR ndarray as JPEG q85, matching the sidecar's
    own ``JPEG_QUALITY=85`` so byte-level comparisons are not affected
    by a quality-setting mismatch on either side."""
    img = Image.fromarray(np.full((480, 640, 3), 100, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def source_face_bytes() -> bytes:
    """Encode a 240x320 PNG that the sidecar will accept and decode.
    We don't care whether the stub ``get_one_face`` actually finds a
    real face — it returns an ``object()`` regardless."""
    img = Image.fromarray(np.full((240, 320, 3), 200, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1 — Runner-layer: stubbed core mutates frame, health latches.
# ---------------------------------------------------------------------------
async def test_runner_swap_emits_inference_side_health(stub_runner):
    """With a stubbed ``_face_analyser`` + ``_face_swapper_core``,
    ``FaceFusionRunner.swap()`` invokes the stubbed core, MUTATES the
    frame, increments ``health.swap_count``, and latches
    ``health.last_swap_ms``.

    This isolates the seam between the runner and the vendored FF —
    no Flask / FastAPI / httpx involvement, so any regression in the
    runner-or-vendor contract surfaces here first.
    """
    import cv2  # noqa: F401 — used inline below

    bgr = np.full((480, 640, 3), 100, dtype=np.uint8)
    out = sidecar_app._runner.swap(
        frame=bgr, source_face=object(), intensity=1.0,
    )

    # Stubbed process_frame returned np.full_like(frame, 42) — a *new*
    # ndarray, not the input by object identity. The mutation must
    # propagate through the runner unchanged.
    assert out is not bgr
    assert out.shape == (480, 640, 3)
    assert int(out.sum()) == 42 * 480 * 640 * 3, "stubbed swapper output bytes did not flow through"

    # Health snapshot caught the swap + the timer:
    assert sidecar_app._runner.health.swap_count == 1
    last_ms = sidecar_app._runner.health.last_swap_ms
    assert last_ms is not None and last_ms >= 0.0
    # Note: ``face_swapper_loaded`` is a *lazy-import-tracking* flag set ONLY
    # inside ``runner.swap()``'s ``if self._face_swapper_core is None`` branch
    # — i.e. when the runner imports the real ``facefusion....core`` itself.
    # When monkeypatch sets ``_face_swapper_core`` directly that branch is
    # bypassed and the flag stays ``False`` even though inference is
    # operational. We deliberately do NOT assert it here.

    # The runner fed FaceFusion's documented inputs dict to the swapper.
    # Key set is the exact contract from
    # vendor/FaceFusion/facefusion/processors/modules/face_swapper/core.py:process_frame.
    assert len(stub_runner.calls) == 1
    inputs = stub_runner.calls[0]
    assert set(inputs.keys()) == {
        "reference_vision_frame",
        "source_vision_frames",
        "target_vision_frames",
        "temp_vision_frame",
        "temp_vision_mask",
    }, f"unexpected inputs shape: {sorted(inputs.keys())}"
    assert isinstance(inputs["source_vision_frames"], list) and inputs["source_vision_frames"], (
        "source_vision_frames must be a non-empty list-of-Face per FF contract"
    )


# ---------------------------------------------------------------------------
# Test 2 — HTTP-layer: inference path emits the right response headers.
# ---------------------------------------------------------------------------
async def test_http_swap_emits_swap_headers_when_inference_differs_input(
    frame_bytes, source_face_bytes, stub_runner,
):
    """With stubbed runner, POST ``/api/v1/swap`` with a real multipart
    payload returns 200 + image/jpeg + ``X-Swap-Mode: swap`` +
    ``X-Latency-Ms: NNN``, AND the response bytes differ from input.

    Together with :func:`test_runner_swap_emits_inference_side_health` and
    the bearer/auth matrix in ``tests/test_sidecar_app.py``, this is the
    full chain: HTTP request → multipart parsed → extract_face stubbed
    to return a face → swap() invoked → stubbed process_frame mutated
    the frame → encode → response with documented headers.
    """
    transport = ASGITransport(app=sidecar_app.app)
    async with AsyncClient(transport=transport, base_url="http://sidecar.local") as client:
        resp = await client.post(
            "/api/v1/swap",
            files={
                "frame": ("frame.jpg", frame_bytes, "image/jpeg"),
                "source_face": ("source.png", source_face_bytes, "image/png"),
            },
            data={"intensity": "1.0"},
        )

    assert resp.status_code == 200, (resp.status_code, resp.text)
    assert resp.headers["content-type"].startswith("image/jpeg"), dict(resp.headers)
    assert resp.headers.get("x-swap-mode") == "swap", (
        f"X-Swap-Mode must reflect the inference-path emission: {dict(resp.headers)}"
    )
    assert "x-latency-ms" in resp.headers, (
        f"X-Latency-Ms missing under inference path: {dict(resp.headers)}"
    )
    assert int(resp.headers["x-latency-ms"]) >= 0

    # The response bytes MUST differ from the input. If they match, the
    # passthrough branch leaked and the X-Swap-Mode=swap header is a lie.
    assert resp.content != frame_bytes, (
        "inference bit flipped output should differ from input frame bytes"
    )
    # JPEG SOI marker = start of a valid JPEG stream.
    assert resp.content[:3] == b"\xff\xd8\xff", "response is not a valid JPEG"

    # Stubbed process_frame was actually invoked.
    assert len(stub_runner.calls) == 1, "stubbed process_frame was not invoked"


# ---------------------------------------------------------------------------
# Test 3 — Regression: passthrough branch emits different headers, no latency.
# ---------------------------------------------------------------------------
async def test_http_swap_passthrough_when_extract_face_returns_none(
    frame_bytes, source_face_bytes, monkeypatch,
):
    """Regression guard: when ``extract_face`` returns None (e.g. no real
    face in the source image at 3.10's vendored FF load failure), the
    route MUST emit ``X-Swap-Mode: passthrough`` + ``X-Swap-Reason:
    no_source_face`` and MUST NOT carry ``X-Latency-Ms`` (we didn't
    measure latency because we didn't run inference)."""
    class _BadAnalyser:
        def get_one_face(self, image, position):  # noqa: ARG002
            return None

    monkeypatch.setattr(sidecar_app._runner, "_face_analyser", _BadAnalyser())

    transport = ASGITransport(app=sidecar_app.app)
    async with AsyncClient(transport=transport, base_url="http://sidecar.local") as client:
        resp = await client.post(
            "/api/v1/swap",
            files={
                "frame": ("frame.jpg", frame_bytes, "image/jpeg"),
                "source_face": ("source.png", source_face_bytes, "image/png"),
            },
            data={"intensity": "1.0"},
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert resp.headers.get("x-swap-mode") == "passthrough", dict(resp.headers)
    assert resp.headers.get("x-swap-reason") == "no_source_face", dict(resp.headers)
    assert "x-latency-ms" not in resp.headers, (
        "passthrough MUST NOT carry X-Latency-Ms (no inference ran)"
    )
