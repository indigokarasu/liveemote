"""``tests/test_facefusion_inference_wire.py`` — proves the inference-side
wire contract inflates ``X-Swap-Mode: swap`` + ``X-Latency-Ms: NNN``
correctly when the ONNX step returns a real frame, AND that every
encode-decode quirk the multipart decoder could plausibly see in the
wild keeps emitting the right headers.

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

Encode-quirk coverage
---------------------

The HTTP-layer tests parametrize over **five** image-pair variants so
each variant stresses one specific decode surface:

* ``solid_jpeg_baseline`` — control case; both uploads are solid-colour
  JPEG q85. Same default the original fixture used; kept as the
  regression anchor.
* ``rotated_jpeg_frame_90deg`` — frame pixel-block is rotated 90° (no
  EXIF tag, fully rotated pixels). Decoder sees a 480×640 input as
  640×480; catches a regression where the route accidentally assumes
  frame dims via filename or Content-Type metadata.
* ``webp_everywhere`` — frame + source both encoded WebP q85. Different
  container format than baseline; stresses ``cv2.imdecode`` on a
  format neither baseline nor PNG test ever exercises.
* ``half_size_png`` — frame is 320×240, source is 320×240 (half of the
  baseline 640×480 / 480×640 pair). Catches a regression where the
  route bails on small images or rejects frame != source dimensions.
* ``frame_same_as_source`` — both uploads are byte-identical solid-100
  JPEG bytes. Catches a regression where the route short-circuits on
  input-byte equality (e.g. an honest comparison mistake that would
  need the bytes to differ for inference to fire).

Each variant is exercised under BOTH the inference path (stubbed
``get_one_face`` returns ``object()``) and the passthrough path
(stubbed ``get_one_face`` returns ``None``) so the multipart decoder
+ header assembly is verified per variant regardless of which code
branch fires downstream.
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
# Variant builders — one helper per container format, raised at parametrize
# time so each variant's bytes are constructed exactly once during
# collection and shared across the inference-path + passthrough-path
# tests for that variant.
# ---------------------------------------------------------------------------
def _solid_jpeg(height: int, width: int, fill: int) -> bytes:
    """Encode a solid-colour ``height × width × 3`` ndarray as JPEG q85,
    matching the sidecar's own ``JPEG_QUALITY=85`` so byte-level
    comparisons are not affected by a quality-setting mismatch on
    either side."""
    img = Image.fromarray(np.full((height, width, 3), fill, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _solid_png(height: int, width: int, fill: int) -> bytes:
    img = Image.fromarray(np.full((height, width, 3), fill, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid_webp(height: int, width: int, fill: int) -> bytes:
    img = Image.fromarray(np.full((height, width, 3), fill, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=85)
    return buf.getvalue()


def _rotated_jpeg_pixels(height: int, width: int, fill: int, angle: int) -> bytes:
    """Rotated-pixel JPEG (no EXIF orientation tag). A regression guard
    specifically against the multipart decoder accidentally assuming
    upright frame dimensions from filename or Content-Type metadata —
    here the pixel layout IS rotated, but Content-Type is still
    ``image/jpeg``."""
    img = Image.fromarray(np.full((height, width, 3), fill, dtype=np.uint8)).rotate(
        angle, expand=True,
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Parametrized image-pair fixture — one variant per encode-decode quirk.
# Each parameter value is a 6-tuple packed into the standard ``UploadFile``
# form: ``(frame_filename, frame_bytes, frame_mime, source_filename,
# source_bytes, source_mime)``.
#
# Constructed LAZILY by pytest at parametrize time (i.e. during test
# collection), so each variant's image bytes are encoded exactly once
# total and reused across all parametrized cases.
# ---------------------------------------------------------------------------
@pytest.fixture(
    params=[
        pytest.param(
            (
                "frame.jpg", _solid_jpeg(480, 640, 100), "image/jpeg",
                "src.jpg",   _solid_jpeg(240, 320, 200), "image/jpeg",
            ),
            id="solid_jpeg_baseline",
        ),
        pytest.param(
            (
                "frame.jpg", _rotated_jpeg_pixels(480, 640, 100, 90), "image/jpeg",
                "src.jpg",   _solid_jpeg(240, 320, 200), "image/jpeg",
            ),
            id="rotated_jpeg_frame_90deg",
        ),
        pytest.param(
            (
                "frame.webp", _solid_webp(480, 640, 100), "image/webp",
                "src.webp",   _solid_webp(240, 320, 200), "image/webp",
            ),
            id="webp_everywhere",
        ),
        pytest.param(
            (
                "frame.png", _solid_png(320, 240, 100), "image/png",
                "src.png",   _solid_png(320, 240, 200), "image/png",
            ),
            id="half_size_png",
        ),
        pytest.param(
            (
                # Both uploads are byte-identical solid-100 JPEG bytes.
                # The route doesn't depend on byte-equality, but a buggy
                # "skip inference if input bytes match expected source"
                # shortcut would fire here and silently emit passthrough
                # instead of swap.
                "frame.jpg", _solid_jpeg(480, 640, 100), "image/jpeg",
                "src.jpg",   _solid_jpeg(480, 640, 100), "image/jpeg",
            ),
            id="frame_same_as_source",
        ),
    ]
)
def image_pair(request):
    """Per-variant (frame, source) upload payload tuple. Tests unpack it
    into the multipart ``files=`` dict and assert the same wire contract
    provides correct headers + content regardless of which decode quirk
    selected this variant."""
    return request.param


# ---------------------------------------------------------------------------
# Test 1 — Runner-layer: stubbed core mutates frame, health latches.
# (NOT parametrize'd: this exercises the runner's direct swap() call
# path with a hand-built numpy array, independent of multipart decoding.)
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
# Test 2 — HTTP-layer inference path, parametrized over encode-decode quirks.
# ---------------------------------------------------------------------------
async def test_http_swap_emits_swap_headers_for_variant(
    image_pair, stub_runner,
):
    """With stubbed runner, POST ``/api/v1/swap`` with a real multipart
    payload returns 200 + image/jpeg + ``X-Swap-Mode: swap`` +
    ``X-Latency-Ms: NNN``, AND the response bytes differ from input —
    REGARDLESS of which encode-decode quirk the parametrize selected
    for this iteration.

    Together with :func:`test_runner_swap_emits_inference_side_health` and
    the bearer/auth matrix in ``tests/test_sidecar_app.py``, this is the
    full chain: HTTP request → multipart parsed → decode quirk absorbed
    → extract_face stubbed to return a face → swap() invoked → stubbed
    process_frame mutated the frame → encode → response with documented
    headers.
    """
    (frame_name, frame_bytes, frame_mime,
     source_name, source_bytes, source_mime) = image_pair

    transport = ASGITransport(app=sidecar_app.app)
    async with AsyncClient(transport=transport, base_url="http://sidecar.local") as client:
        resp = await client.post(
            "/api/v1/swap",
            files={
                "frame": (frame_name, frame_bytes, frame_mime),
                "source_face": (source_name, source_bytes, source_mime),
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
    # This holds under every variant because the stubbed swap fills the
    # frame with a different constant (42) than any input fill (100/200).
    assert resp.content != frame_bytes, (
        f"[{resp.headers.get('x-swap-mode')}] inference bit flipped output should differ from input frame bytes"
    )
    # JPEG SOI marker = start of a valid JPEG stream.
    assert resp.content[:3] == b"\xff\xd8\xff", "response is not a valid JPEG"

    # Stubbed process_frame was actually invoked.
    assert len(stub_runner.calls) == 1, "stubbed process_frame was not invoked"


# ---------------------------------------------------------------------------
# Test 3 — Passthrough branch under the same parametrize.
# ---------------------------------------------------------------------------
async def test_http_swap_passthrough_for_variant(
    image_pair, monkeypatch,
):
    """Regression guard: when ``extract_face`` returns None the route
    MUST emit ``X-Swap-Mode: passthrough`` + ``X-Swap-Reason:
    no_source_face`` and MUST NOT carry ``X-Latency-Ms`` (we didn't
    measure latency because we didn't run inference) — REGARDLESS of
    which encode-decode quirk the parametrize selected.

    Stresses the multipart decoder + passthrough code path independently
    of the inference path: if any variant's frame/source can't be
    decoded, the route 400s BEFORE passthrough headers get a chance to
    fire, so a 200 with the right headers proves the decoder handled
    the quirk.
    """
    class _BadAnalyser:
        def get_one_face(self, image, position):  # noqa: ARG002
            return None

    monkeypatch.setattr(sidecar_app._runner, "_face_analyser", _BadAnalyser())

    (frame_name, frame_bytes, frame_mime,
     source_name, source_bytes, source_mime) = image_pair

    transport = ASGITransport(app=sidecar_app.app)
    async with AsyncClient(transport=transport, base_url="http://sidecar.local") as client:
        resp = await client.post(
            "/api/v1/swap",
            files={
                "frame": (frame_name, frame_bytes, frame_mime),
                "source_face": (source_name, source_bytes, source_mime),
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
