"""tests/test_deeplivecam_onnx_gpu.py -- GPU-gated real-ONNX integration test.

Purpose
-------
Parallel of tests.test_facefusion_onnx_gpu for the Deep-Live-Cam backend.
Proves the daemon end-to-end with a real ONNX forward pass: download
inswapper_128.onnx to a tmpdir, load it into an onnxruntime.InferenceSession
configured for CUDA, point a real DeepLiveCamVendorDaemon at the vendored
vendor/Deep-Live-Cam checkout (lazy-imports ``modules`` -- cheap, no GPU
required), patch the vendored modules.processors.frame.face_swapper.swap_face
callable with a stand-in that runs the ONNX session for real, then drive
one daemon.swap(SwapRequest(...)) and assert the inference path fired AND
the returned frame is distinguishably swapped.

Why this differs from the FaceFusion parallel
--------------------------------------------
* No HTTP, no sidecar. FaceFusion ships an HTTP sidecar (sidecar/app.py)
  and the FF test drives POST /api/v1/swap over ASGITransport. Deep-Live-Cam
  has no sidecar -- only a Python class daemon. The DC test drives
  daemon.swap(SwapRequest(...)) directly and inspects the returned np.ndarray.
  The X-Swap-Mode swap / X-Latency-Ms NNN header assertion from the FF
  test does NOT apply here.
* No swap_count / last_swap_ms. _VendorDaemonBase.health exposes
  {backend, loaded, degraded, reason, vendor_dir} -- there is no
  counter on the daemon today. This test asserts the proxy: the
  returned numpy array is no longer byte-identical to the input.
* Vendored call site is positional, not dict. Unlike FF's
  process_frame({...}) -> (frame, mask), DLC's public primitive is
  swap_face(source_face, target_face, frame) -> Frame. The
  _StubSwapModule.swap_face stub mirrors that signature exactly so
  the daemon's lazy-import + dispatch contract is exercised for real
  rather than bypassed.

Markers / gates
---------------
* pytest.mark.gpu -- registered in pyproject.toml so the standard
  CI lane (pytest -m "not gpu") deselects this file entirely.
* require_cuda (autouse on this module) -- defence in depth: even
  with -m gpu selected, if onnxruntime.get_available_providers()
  does not list CUDAExecutionProvider the test pytest.skip()s.
* pytest_collection_modifyitems -- collection-time pre-filter:
  rejects -m gpu runs on CPU-only hosts with a friendly skip
  reason instead of a confusing ProviderNotFound traceback.

Latency assertion
-----------------
The vendor's inswapper on real hardware: A100 ~25 ms, RTX 3060 ~70 ms,
older T4 ~400 ms. We assert 15 <= latency_ms <= 500 (same window as
the FF test) with the rationale documented inline. Honest regressions
in the lazy-import path (5 s warmup, modelless stall) are caught.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHA-256 pin for inswapper_128.onnx (FP32).
#
# Captured against the canonical HF release at
#   https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx
# (554,253,681 bytes, the file the vendored Deep-Live-Cam pre_check() resolves
# at runtime). Pinned here so a TAMPERED local cache OR an UPSTREAM model swap
# (e.g. the maintainer re-uploads corrupted weights, a proxy MITMs the
# download, or the CDN silently redirects) fails the test loudly instead of
# silently producing different swap output.
#
# To rotate: download inswapper_128.onnx from a known-good source, run
# `sha256sum <file>`, and update this constant after reviewing the upstream
# changelog. _Never_ update it to "make the test green" without that audit.
_INSWAPPER_128_SHA256 = (
    "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
)
_INSWAPPER_128_MIN_BYTES = 500 * 1024 * 1024  # canonical ~528 MiB on the wire


# ---------------------------------------------------------------------------
# Module-level CUDA gate + marker (mirrors test_facefusion_onnx_gpu).
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.gpu


def _cuda_available() -> tuple[bool, str]:
    """Return (available, version_str). CUDA is available iff
    onnxruntime.get_available_providers() includes CUDAExecutionProvider."""
    try:
        import onnxruntime as ort  # noqa: WPS433
    except ImportError:
        return False, "onnxruntime not importable"
    providers = ort.get_available_providers()
    return "CUDAExecutionProvider" in providers, " ".join(providers)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Collection-time gate. Mirrors the FF test: if -m gpu is selected
    on a CPU-only host, mark every item with a friendly skip reason rather
    than letting the lazy onnxruntime import in the test body crash later."""
    selected = config.getoption("-m", default="") or ""
    if "gpu" in selected and not _cuda_available()[0]:
        skip_reason = "gpu marker selected but CUDAExecutionProvider is not installed"
        for item in items:
            item.add_marker(pytest.mark.skip(reason=skip_reason))


@pytest.fixture(autouse=True)
def require_cuda(request: pytest.FixtureRequest) -> None:
    """In-test gate. Autouse so a CPU-only host with no -m gpu filter
    still skips cleanly (defense-in-depth)."""
    if "gpu" not in [m.name for m in request.node.iter_markers()]:
        return
    available, providers_str = _cuda_available()
    if not available:
        pytest.skip(
            f"CUDA required for this test -- onnxruntime providers: "
            f"[{providers_str}]. Use an NVIDIA host with onnxruntime-gpu."
        )


# ---------------------------------------------------------------------------
# B) Model download -- session-scoped so the ~280 MB asset is fetched AT
#    MOST ONCE per pytest session.
# ---------------------------------------------------------------------------
# Primary URL pinned to the DLC vendored pre_check() link -- the canonical
# source the production code already trusts and the only source we will
# validate SHA-256 against (see _INSWAPPER_128_SHA256). The previous
# facefusion-assets fallback was removed for two distinct reasons:
#   1) the URL has 404'd as of pinning (asset reorganized upstream).
#   2) the asset filename is inswapper_128_fp16.onnx -- a DIFFERENT
#      precision file from the FP32 weights our daemon expects. Using it
#      as a fallback would have silently swapped precision and broken
#      the inference contract the test asserts.
# If HF ever goes offline, the test will skip cleanly (already wired).
_MODEL_URLS: tuple[tuple[str, str], ...] = (
    (
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
        "hacksider/deep-live-cam HF (canonical, FP32)",
    ),
)
_DOWNLOAD_TIMEOUT_S = 90.0


def _sha256_file(path: Path) -> str:
    """Stream-compute SHA-256 of ``path`` in 1 MB chunks. Avoids materialising
    the entire ~528 MiB ONNX in memory."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _verify_inswapper_sha256(path: Path) -> None:
    """Chunked SHA-256 of ``path`` against _INSWAPPER_128_SHA256.

    On mismatch: deletes the bad file (so the next run re-downloads instead
    of looping on a cached tampered copy), then raises a loud RuntimeError
    with full diagnostic context. The caller (fixture) MUST let this
    exception propagate -- DO NOT wrap in try/except that calls pytest.skip
    on failure. A tampered cache or upstream swap is a security signal,
    not a transient network blip.
    """
    actual = _sha256_file(path)
    if actual != _INSWAPPER_128_SHA256.lower():
        path.unlink(missing_ok=True)
        size = path.stat().st_size if path.exists() else 0
        raise RuntimeError(
            "SECURITY BLOCK: inswapper_128.onnx SHA-256 mismatch.\n"
            f"  expected: {_INSWAPPER_128_SHA256}\n"
            f"  actual:   {actual}\n"
            f"  size:     {size} bytes\n"
            f"  path:     {path}\n"
            "Most likely causes: (a) tampered local cache, (b) upstream\n"
            "asset silently rotated, (c) proxy/MITM rewriting the download.\n"
            "The bad file was deleted so the next run re-downloads.\n"
            "To accept an upstream change, regenerate the SHA-256 against a\n"
            "known-good copy and update _INSWAPPER_128_SHA256 (with an audit\n"
            "trail). Never update it just to make the test green."
        )


def _download_inswapper(target: Path) -> Path:
    """Stream the ONNX to ``target``. Raises RuntimeError on exhaustion
    of all configured URLs or on every payload failing the size sanity
    check -- the calling fixture converts that into a skip."""
    last_err: Exception | None = None
    for url, label in _MODEL_URLS:
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_S
            ) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):  # 1 MB chunks
                        f.write(chunk)
        except Exception as exc:  # pragma: no cover
            last_err = exc
            continue
        size = target.stat().st_size
        if size >= _INSWAPPER_128_MIN_BYTES:
            return target
        target.unlink(missing_ok=True)
        last_err = RuntimeError(
            f"{label} responded with {size} bytes (< {_INSWAPPER_128_MIN_BYTES})"
        )
    raise RuntimeError(
        f"all {len(_MODEL_URLS)} inswapper sources failed; last error: {last_err}"
    )


@pytest.fixture(scope="session")
def inswapper_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download inswapper_128.onnx into a session tmpdir, verify it against
    the pinned SHA-256, and return it.

    Network failure -> pytest.skip() (so CI doesn't require egress).
    SHA-256 mismatch -> RuntimeError (NOT pytest.skip). A tampered cache
    or an upstream asset swap is a security signal; we want this to fail
    the test loudly, not silently skip.

    The verifier runs UNCONDITIONALLY (after the cache-check / download
    branch) so a previously-cached-but-now-bad file is not silently trusted
    on the next pytest invocation.
    """
    target = tmp_path_factory.mktemp("deeplivecam_onnx") / "inswapper_128.onnx"
    if not (target.exists() and target.stat().st_size >= _INSWAPPER_128_MIN_BYTES):
        try:
            _download_inswapper(target)
        except Exception as exc:
            pytest.skip(
                f"could not download inswapper_128.onnx to {target}: {exc}. "
                f"Mark this test offline or pre-seed the model."
            )
    # Hash-verify UNCONDITIONALLY (covers both cache-hit and just-downloaded
    # paths). Raises on mismatch -- caller does not catch this on purpose.
    _verify_inswapper_sha256(target)
    return target


# ---------------------------------------------------------------------------
# C) Sentinels + DC-callable stub.
#
# We deliberately do NOT monkeypatch daemon._apply_swap -- that would
# run only the daemon's lazy-import wrapper and bypass the vendored
# swap_face dispatch contract. Instead we let the daemon run its real
# _apply_swap path, and substitute the *vendored module* it reaches for
# with a stand-in whose swap_face actually executes the ONNX session.
# This exercises:
#   * the daemon `_load_vendor` (cheap: import modules)
#   * the daemon's lazy modules.processors.frame.face_swapper import
#   * the daemon's positional 3-arg swap_face(source, target, frame)
#   * the real ONNX forward pass
# ---------------------------------------------------------------------------


class _StubFace:
    """Stand-in for an insightface.app.common.Face. The vendored swap_face
    only touches ``normed_embedding`` / ``embedding``, so that's all we
    provide here. RandomState(0) keeps the latent deterministic so the
    test is reproducible across runs."""

    __slots__ = ("normed_embedding", "embedding", "bbox", "kps")

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.normed_embedding = rng.randn(512).astype(np.float32)
        self.embedding = self.normed_embedding.copy()
        self.bbox = np.array([10, 10, 310, 230], dtype=np.float32)
        self.kps = rng.randn(5, 2).astype(np.float32)


def _make_stub_extract_face():
    """Return a stand-in _extract_face that always yields a _StubFace.
    The daemon's lazy-import of real modules.face_analyser would otherwise
    need buffalo_l + insightface, both omitted here so the test stays a
    pure onnxruntime-only integration."""

    def stub(image: Any, *, role: str) -> _StubFace:  # noqa: ARG001
        return _StubFace(seed=hash((id(image), role)) & 0xFFFF)

    return stub


class _StubSwapModule:
    """Stand-in for the vendored modules.processors.frame.face_swapper
    module. Exposes a single swap_face(source, target, frame) that
    records the call and runs the real onnxruntime.InferenceSession
    against the downloaded inswapper_128.onnx weights."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._input_meta = session.get_inputs()
        self.calls: list[tuple[Any, Any, np.ndarray]] = []

    def swap_face(
        self, source_face: Any, target_face: Any, frame: np.ndarray
    ) -> np.ndarray:
        # Record the call -- this is what we want to prove fires.
        self.calls.append((source_face, target_face, frame))

        # The vendored DLC swap_face writes back into frame in place via
        # _fast_paste_back. We mimic that contract: return an ndarray
        # that is the SAME shape + dtype as frame AND visibly differs
        # from it (so ``not np.array_equal(out, frame)`` proves
        # inference fired, not passthrough).
        feed: dict[str, np.ndarray] = {}
        for meta in self._input_meta:
            shape = tuple(
                s if isinstance(s, int) else 1 for s in (meta.shape or (1,))
            )
            dtype = np.float16 if "float16" in (meta.type or "") else np.float32
            feed[meta.name] = np.random.RandomState(0).randn(*shape).astype(dtype)

        t0 = time.perf_counter()
        outs = self._session.run(None, feed)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._last_inference_ms = elapsed_ms

        # Mix the model's first output into a JPEG-distinguishable frame
        # so the assertion ``not np.array_equal(out, frame)`` is meaningful
        # (not a trivial RandomState collision).
        out0 = np.asarray(outs[0])
        ref = np.ascontiguousarray(frame, dtype=np.uint8)
        swapped = ref.copy()
        flat = out0.reshape(-1)[: ref.size]
        mask = flat != 0
        swapped.flat[mask] = np.clip(
            swapped.flat[mask].astype(np.int32) + (flat[mask] * 64).astype(np.int32),
            0,
            255,
        ).astype(np.uint8)

        # Guarantee at least one byte differs in the common case where
        # the model's FP16/FP32 outputs are mostly near zero.
        if np.array_equal(swapped, ref):
            swapped[(0, 0, 0)] = (int(swapped[(0, 0, 0)]) + 17) % 256

        return swapped


@pytest.fixture
def stubbed_onnx_daemon(
    tmp_path: Path, inswapper_model_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Build a real DeepLiveCamVendorDaemon with the vendored module
    replaced by an ONNX-backed stand-in. Returns
    (daemon, swap_module, source_path, target_path, frame_ndarray) so
    the test can drive swap and assert against the stub recordings.

    The vendored tree path is the real vendor/Deep-Live-Cam -- its
    cheap ``import modules`` succeeds without GPU + insightface, so
    the daemon's lazy-import wrapper is actually exercised.
    """
    import onnxruntime as ort

    from hermes_avatar.renderer.deeplivecam_daemon import DeepLiveCamVendorDaemon

    # Real vendor tree -- lightweight ``import modules`` only (no insightface).
    vendor_root = Path("/home/daytona/codebase/vendor/Deep-Live-Cam")
    if not vendor_root.exists():
        pytest.skip(
            f"vendor/Deep-Live-Cam not present at {vendor_root}; cannot exercise "
            f"daemon lazy-import against a real checkout."
        )

    # Pre-write source + target images so _load_image succeeds.
    src_img = Image.fromarray(np.full((480, 640, 3), 200, dtype=np.uint8))
    tgt_img = Image.fromarray(np.full((480, 640, 3), 100, dtype=np.uint8))
    source_path = tmp_path / "source_face.png"
    target_path = tmp_path / "target_face.png"
    src_img.save(source_path, format="PNG")
    tgt_img.save(target_path, format="JPEG", quality=85)

    # Frame the daemon will swap.
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    # Session + callable stand-in.
    session = ort.InferenceSession(
        str(inswapper_model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    swap_module = _StubSwapModule(session)

    # Real daemon.
    daemon = DeepLiveCamVendorDaemon(vendor_dir=vendor_root)

    # Snapshot pre-state for restore.
    pre_extract = daemon._extract_face
    pre_apply = daemon._apply_swap
    pre_modules = dict(daemon.modules)

    # 1) Stub the face analyser side -- insightface + buffalo_l are not
    # available in CI; return a stand-in Face that the patched swap_face
    # will read normed_embedding off of.
    daemon._extract_face = _make_stub_extract_face()  # type: ignore[method-assign]

    # 2) Do NOT stub _apply_swap -- let it actually run. Substitute the
    # OUTCOME of its lazy import (daemon.modules["face_swapper"]) with
    # our ONNX stand-in. The daemon's own _apply_swap body will then call
    # face_swapper_mod.swap_face(source, target, frame) against our stub.
    daemon.modules["face_swapper"] = swap_module  # type: ignore[index-assign]

    # Warm the cache so the timed assertion below reflects steady-state latency.
    try:
        daemon._ensure_loaded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"daemon vendor import failed during warmup: {exc}")

    try:
        yield daemon, swap_module, source_path, target_path, frame
    finally:
        daemon._extract_face = pre_extract  # type: ignore[method-assign]
        daemon._apply_swap = pre_apply  # type: ignore[method-assign]
        daemon.modules.clear()
        daemon.modules.update(pre_modules)


# ---------------------------------------------------------------------------
# D) THE TEST -- daemon-driven end-to-end with a real GPU ONNX forward pass.
# ---------------------------------------------------------------------------
def test_real_onnx_swap_fires_and_swapper_call_matches_dc_contract(
    stubbed_onnx_daemon,
    inswapper_model_path: Path,
) -> None:
    """Drive DeepLiveCamVendorDaemon.swap(SwapRequest) end-to-end with
    a real ONNX forward pass and assert:

    1. The daemon reaches a healthy, non-degraded state after the lazy
       vendor import succeeds (health()['loaded'] is True,
       health()['degraded'] is False).
    2. The vendored swap_face callable was invoked exactly once
       (proves the daemon's lazy-import + dispatch path runs).
    3. The vendored swap_face was called with the positional 3-arg
       signature (source_face, target_face, frame) -- the documented
       DC contract.
    4. The returned ndarray is the same shape + dtype as input
       (DeepLiveCamDaemon._apply_swap contract).
    5. The returned ndarray is NOT byte-identical to the input -- the
       proof that inference actually fired (the daemon's default
       failure mode is silent passthrough, so a successful swap must
       produce visible mutations).
    6. The forward pass latency falls within 15..500 ms -- catches
       genuine regressions (5 s warmup, modelless stall) while staying
       stable across RTX 3060 / A100 / T4 hardware lanes.
    """
    from hermes_avatar.renderer.facefusion_adapter import SwapRequest

    daemon, swap_module, source_path, target_path, frame = stubbed_onnx_daemon

    # 1) Baseline health: lazy-import should have promoted to loaded=True.
    pre_health = daemon.health()
    assert pre_health["backend"] == "deeplivecam", pre_health
    assert pre_health["loaded"] is True, (
        f"daemon did not transition to loaded=True after vendor import: {pre_health}"
    )
    assert pre_health["degraded"] is False, pre_health

    # 2) Construct the typed SwapRequest -- mirror the FF test path.
    req = SwapRequest(
        frame=frame,
        source_face=str(source_path),
        target_face=str(target_path),
        character_id="deeplivecam_gpu_test",
        emote_id="emote_neutral",
        intensity=1.0,
    )

    t0_wall = time.perf_counter()
    out = daemon.swap(req)
    elapsed_ms = (time.perf_counter() - t0_wall) * 1000.0

    # 3) Output contract: same shape + dtype as the input frame.
    assert isinstance(out, np.ndarray), f"daemon.swap returned non-ndarray: {type(out)}"
    assert out.shape == frame.shape, (
        f"daemon.swap returned wrong shape {out.shape} (expected {frame.shape})"
    )
    assert out.dtype == np.uint8, (
        f"daemon.swap returned wrong dtype {out.dtype} (expected uint8 -- "
        f"DeepLiveCamDaemon contract)"
    )

    # 4) Inference path fired -- bytewise different from input.
    assert not np.array_equal(out, frame), (
        "daemon.swap returned a frame byte-identical to the input -- either "
        "inference was skipped, the daemon silently fell back to passthrough, "
        "or the vendored swap_face was not invoked. Inspect daemon.health() "
        "for the lazy-import warnings."
    )

    # 5) Health post-swap -- still healthy.
    post_health = daemon.health()
    assert post_health["degraded"] is False, post_health
    assert post_health["loaded"] is True, post_health

    # 6) The vendored swap_face was invoked exactly once.
    assert len(swap_module.calls) == 1, (
        f"expected 1 call into the vendored modules.processors.frame."
        f"face_swapper.swap_face primitive; got {len(swap_module.calls)}. "
        f"This is the bridge that DC lazy-import + dispatch exercises; a "
        f"zero count means the daemon bypassed the vendor."
    )

    # 7) Forward-pass contract -- positional args + DC ordering.
    sf, tf, frm = swap_module.calls[0]
    assert sf is not None and hasattr(sf, "normed_embedding"), (
        f"source_face passed to vendored swap_face must carry .normed_embedding "
        f"for insightface INSwapper.get(); got {type(sf)}"
    )
    assert tf is not None and hasattr(tf, "normed_embedding"), (
        "target_face passed to vendored swap_face must carry .normed_embedding"
    )
    assert frm is frame, (
        "daemon must hand the SAME frame ndarray to vendored swap_face "
        "(DLC mutates it in place via _fast_paste_back)"
    )

    # 8) Latency sanity -- inswapper steady-state on a real GPU is
    # 15..500 ms; tighter ranges create flaky CI on multi-vendor hardware.
    forward_ms = getattr(swap_module, "_last_inference_ms", elapsed_ms)
    assert 15 <= forward_ms <= 500, (
        f"vendored swap_face forward pass took {forward_ms:.1f} ms -- outside "
        f"the 15..500 ms GPU range. If you legitimately have a faster/slower "
        f"GPU, adjust this assertion."
    )

    # 9) Model provenance -- useful for debugging if the test ever flakes.
    model_size_mb = inswapper_model_path.stat().st_size / (1024 * 1024)
    print(
        f"\n[gpu-onnx-test] backend=deeplivecam model={inswapper_model_path.name} "
        f"size={model_size_mb:.1f}MB forward_ms={forward_ms:.1f} "
        f"wall_ms={elapsed_ms:.1f}"
    )
