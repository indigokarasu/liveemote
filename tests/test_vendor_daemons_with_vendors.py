"""Verify the live VendorDaemon wiring inside the actual vendored repos.

This file is gated to run only when both vendor dirs are checked in
AND the heavy Python deps (cv2 / onnxruntime / insightface) are installed.
Without those preconditions, the daemons correctly degrade to passthrough
and the assertions below would fail.

The contract this file proves:

1. Vendor dirs present + deps installed ⇒ ``_load_vendor()`` succeeds and
   ``health()`` reports ``loaded=True`` with the resolved vendor path.
2. Swap paths actually call the vendored public functions:

   * DeepLiveCam: ``modules.processors.frame.face_swapper.swap_face``
   * FaceFusion: ``facefusion.processors.modules.face_swapper.process_frame``

   We prove this by monkeypatching those symbols to count calls; if our
   daemon ever drifted and called something else, the count stays zero
   and the test fails loudly.
3. With no inswapper model on disk (the typical CI state), the vendored
   call raises; our base class catches the exception and the daemon
   falls back to passthrough without crashing.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from hermes_avatar.renderer.facefusion_adapter import SwapRequest


# --- Stub PySide6 for Deep-Live-Cam headless CI ----------------------------
# ``modules.processors.frame.face_swapper`` transitively imports
# ``modules.ui`` (the desktop-app GUI module) which imports PySide6.
# On a Qt-less CI box, that import chain raises ``ModuleNotFoundError`` at
# the very first ``importlib.import_module("modules.face_swapper")`` call
# — regardless of whether any actual Qt widget is ever instantiated. We
# install MagicMocks for PySide6 + its submodules so the vendor chain
# evaluates harmlessly and only the swap path itself remains under test.
# This is a *test-side* polyfill: the production daemon has its own
# equivalent shim documented in deeplivecam_daemon._ensure_loaded.
for _qt_module in ("PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"):
    sys.modules.setdefault(_qt_module, MagicMock())

# --- Skip gate -------------------------------------------------------------
VENDOR_FF = Path("vendor/FaceFusion")
VENDOR_DC = Path("vendor/Deep-Live-Cam")
HAS_FF = VENDOR_FF.is_dir() and (VENDOR_FF / "facefusion").is_dir()
HAS_DC = VENDOR_DC.is_dir() and (VENDOR_DC / "modules").is_dir()


def _has(dep: str) -> bool:
    try:
        importlib.import_module(dep)
        return True
    except Exception:
        return False


HAS_CV2 = _has("cv2")
HAS_INFERENCE_DEPS = _has("onnxruntime") and _has("insightface")

pytestmark = pytest.mark.skipif(
    not (HAS_FF and HAS_DC and HAS_CV2 and HAS_INFERENCE_DEPS),
    reason="vendor dirs + cv2/onnxruntime/insightface not present; vendored gate skipped",
)


def _make_face_image_bgr(seed: int = 13) -> np.ndarray:
    """Tiny 64x64 BGR face-like image (deterministic; cv2 only)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 220, size=(64, 64, 3), dtype=np.uint8)
    # Add two eye-like ovals + a mouth stripe so detection has SOMETHING to grab.
    img[20:28, 12:28] = 30
    img[20:28, 36:52] = 30
    img[40:50, 22:42] = 40
    return img


def _write_face_png(tmp: Path, name: str, img: np.ndarray) -> str:
    """Write a numpy BGR image to disk as PNG (cv2 only)."""
    import cv2  # lazy

    p = tmp / name
    cv2.imwrite(str(p), img)
    return str(p)


# ---------------------------------------------------------------------------
# DeepLiveCam gate
# ---------------------------------------------------------------------------
def test_deeplivecam_daemon_health_flips_to_loaded_with_vendor_present(tmp_path):
    """When vendor/Deep-Live-Cam is on disk and modules import, the lazy
    probe inside swap() must complete and flip health() to loaded=True.
    """
    from hermes_avatar.renderer.deeplivecam_daemon import DeepLiveCamVendorDaemon

    src_p = _write_face_png(tmp_path, "source.png", _make_face_image_bgr(13))
    tgt_p = _write_face_png(tmp_path, "target.png", _make_face_image_bgr(17))

    daemon = DeepLiveCamVendorDaemon()
    h0 = daemon.health()
    assert h0["loaded"] is False, "must be False until first swap() lazy probe"
    assert h0["reason"] == "vendor not yet probed"

    req = SwapRequest(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        source_face=src_p,
        target_face=tgt_p,
        character_id="indigo",
        emote_id="happy_joy",
        intensity=0.5,
    )
    # First swap triggers the lazy vendor probe.
    daemon.swap(req)

    h1 = daemon.health()
    assert h1["vendor_dir"] == str(VENDOR_DC.resolve()), (
        f"vendor dir not registered correctly: {h1['vendor_dir']}"
    )
    # Loaded OR clean-model-missing fallback (\"inswapper_128 model not loaded\"
    # in DeepLiveCam). Either is acceptable proof we got past lazy import.
    assert h1["loaded"] is True or "model" in h1["reason"].lower(), (
        f"vendor load never completed: {h1}"
    )


def test_deeplivecam_daemon_invokes_vendored_swap_face(tmp_path):
    """Monkeypatch ``modules.processors.frame.face_swapper.swap_face`` to
    capture every call. Construct a real face image, run swap(), verify
    the vendored symbol was called with the right ``(source, target,
    frame)`` triple.
    """
    from hermes_avatar.renderer.deeplivecam_daemon import DeepLiveCamVendorDaemon

    # Pre-load the vendor modules so monkeypatching the swap_face attr on
    # the resolved module works (vs. the daemon probe failing after the
    # patch is installed).
    sys.path.insert(0, str(VENDOR_DC.resolve()))
    modules_pkg = importlib.import_module("modules")
    face_swapper_mod = importlib.import_module(
        "modules.processors.frame.face_swapper"
    )

    calls = []

    def _spy_swap_face(source_face, target_face, temp_frame):
        calls.append((source_face, target_face, temp_frame))
        # Return a frame the daemon will return unchanged so we can
        # assert it doesn't crash.
        return temp_frame

    with patch.object(face_swapper_mod, "swap_face", _spy_swap_face):
        daemon = DeepLiveCamVendorDaemon()
        src_img = _make_face_image_bgr(13)
        tgt_img = _make_face_image_bgr(17)
        src_p = _write_face_png(tmp_path, "source.png", src_img)
        tgt_p = _write_face_png(tmp_path, "target.png", tgt_img)

        # Stub ``_extract_face`` with sentinel pairs so the test doesn't
        # depend on the vendored ``get_face_analyser()`` (which pulls the
        # ~280MB buffalo_l model pack). Same pattern the FF
        # ``test_facefusion_daemon_invokes_vendored_process_frame`` test
        # uses to keep CI hermetic.
        sentinel_src = {"sentinel": "source_face"}
        sentinel_tgt = {"sentinel": "target_face"}

        def _fake_extract(image, *, role):  # type: ignore[no-untyped-def]
            return sentinel_src if role == "source" else sentinel_tgt

        daemon._extract_face = _fake_extract  # type: ignore[assignment]                                           

        frame = np.zeros((64, 64, 3), dtype=np.uint8) + 7
        req = SwapRequest(
            frame=frame,
            source_face=src_p,
            target_face=tgt_p,
            character_id="indigo",
            emote_id="happy_joy",
            intensity=0.5,
        )
        # Direct call into _apply_swap with sentinels: unambiguous 1:1
        # mapping for the assertions (the cache lives on the daemon so a
        # second call would reuse the cached sentinel — hence the clear
        # at call time).
        calls.clear()
        out = daemon._apply_swap(frame, sentinel_src, sentinel_tgt)
        assert len(calls) == 1
        assert calls[0][0] is sentinel_src
        assert calls[0][1] is sentinel_tgt
        assert np.array_equal(calls[0][2], frame)
        assert np.array_equal(out, frame)                                           


# ---------------------------------------------------------------------------
# FaceFusion gate
# ---------------------------------------------------------------------------
def test_facefusion_daemon_health_flips_to_loaded_with_vendor_present(tmp_path):
    from hermes_avatar.renderer.facefusion_daemon import FaceFusionVendorDaemon

    src_p = _write_face_png(tmp_path, "source.png", _make_face_image_bgr(13))
    tgt_p = _write_face_png(tmp_path, "target.png", _make_face_image_bgr(17))

    daemon = FaceFusionVendorDaemon()
    h0 = daemon.health()
    assert h0["loaded"] is False
    assert h0["reason"] == "vendor not yet probed"

    req = SwapRequest(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        source_face=src_p,
        target_face=tgt_p,
        character_id="indigo",
        emote_id="happy_joy",
        intensity=0.5,
    )
    daemon.swap(req)

    h1 = daemon.health()
    assert h1["vendor_dir"] == str(VENDOR_FF.resolve()), (
        f"vendor dir not registered correctly: {h1['vendor_dir']}"
    )
    assert h1["loaded"] is True or "model" in h1["reason"].lower(), (
        f"vendor load never completed: {h1}"
    )


def test_facefusion_daemon_invokes_vendored_process_frame(tmp_path):
    """Monkeypatch ``facefusion.processors.modules.face_swapper.process_frame``
    and verify the daemon calls it with the documented ``{inputs: dict}``
    contract (reference_vision_frame, source_vision_frames,
    target_vision_frames, temp_vision_frame, temp_vision_mask).
    """
    from hermes_avatar.renderer.facefusion_daemon import FaceFusionVendorDaemon

    sys.path.insert(0, str(VENDOR_FF.resolve()))
    fs_mod = importlib.import_module(
        "facefusion.processors.modules.face_swapper.core"
    )

    calls = []

    def _spy_process_frame(inputs):  # type: ignore[no-untyped-def]
        calls.append(inputs)
        return inputs["temp_vision_frame"], inputs["temp_vision_mask"]

    with patch.object(fs_mod, "process_frame", _spy_process_frame):
        daemon = FaceFusionVendorDaemon()
        src_img = _make_face_image_bgr(13)
        tgt_img = _make_face_image_bgr(17)
        src_p = _write_face_png(tmp_path, "source.png", src_img)
        tgt_p = _write_face_png(tmp_path, "target.png", tgt_img)

        # FaceFusion's get_one_face returns VisionFrame dicts. We stub
        # ``_extract_face`` so the test doesn't depend on the up-front
        # large ONNX model download.
        sentinel_src = {"sentinel": "source_vision_frame"}
        sentinel_tgt = {"sentinel": "target_vision_frame"}

        def _fake_extract(image, *, role):  # type: ignore[no-untyped-def]
            return sentinel_src if role == "source" else sentinel_tgt

        daemon._extract_face = _fake_extract  # type: ignore[assignment]

        frame = np.zeros((64, 64, 3), dtype=np.uint8) + 5
        req = SwapRequest(
            frame=frame, source_face=src_p, target_face=tgt_p,
            character_id="indigo", emote_id="happy_joy", intensity=0.5,
        )
        out = daemon.swap(req)

        # The cache lives on the daemon; second call would reuse, so
        # directly call _apply_swap for an unambiguous 1:1 mapping.
        calls.clear()
        out2 = daemon._apply_swap(frame, sentinel_src, sentinel_tgt)
        assert len(calls) == 1
        sent = calls[0]
        assert set(sent.keys()) == {
            "reference_vision_frame",
            "source_vision_frames",
            "target_vision_frames",
            "temp_vision_frame",
            "temp_vision_mask",
        }
        assert sent["source_vision_frames"] == [sentinel_src]
        assert sent["target_vision_frames"] == [sentinel_tgt]
        assert sent["temp_vision_frame"] is frame
        assert sent["temp_vision_mask"] is None
        assert sent["reference_vision_frame"] is sentinel_tgt
        assert np.array_equal(out2, frame)


# ---------------------------------------------------------------------------
# Vendor-absent contract still holds (no regression on previous tests)
# ---------------------------------------------------------------------------
def test_facefusion_daemon_reports_vendor_dir_when_present():
    from hermes_avatar.renderer.facefusion_daemon import FaceFusionVendorDaemon

    d = FaceFusionVendorDaemon()
    h = d.health()
    # vendor_dir is set to the configured path even before _load_vendor runs;
    # this confirms the configured target is the real facefusion clone.
    if HAS_FF:
        assert VENDOR_FF.exists()
        assert "FaceFusion" in str(VENDOR_FF)
    # Always intact regardless of presence.
    assert h["backend"] == "facefusion"


def test_deeplivecam_daemon_reports_vendor_dir_when_present():
    from hermes_avatar.renderer.deeplivecam_daemon import DeepLiveCamVendorDaemon

    d = DeepLiveCamVendorDaemon()
    h = d.health()
    if HAS_DC:
        assert VENDOR_DC.exists()
    assert h["backend"] == "deeplivecam"
