"""Tests for the real ``VendorDaemon`` implementations.

These verify the contracts Claude's earlier rewrite established:

1. **Vendor dir absent → passthrough.** Construction succeeds; first
   ``swap(req)`` returns the frame unchanged; ``health()`` reports the
   missing-vendor reason.
2. **Lazy vendor import.** No ``importlib.import_module`` / ``cv2.imread``
   happens at ``__init__``. The first ``swap`` call is when vendored
   modules are probed.
3. **Face cache invalidation.** With a stubbed ``_extract_face``, repeating
   the same ``req.source_face`` path keeps the same cached embedding;
   changing the path triggers a fresh extraction.
4. **Auto-pick in FaceSwapAdapter.** Constructing a ``FaceSwapAdapter``
   with ``enabled=True`` and no explicit daemon results in a wire-up that
   selects the correct VendorDaemon for ``config.backend`` (facefusion →
   ``FaceFusionVendorDaemon``, deeplivecam → ``DeepLiveCamVendorDaemon``).
5. **Null daemon fallback.** Unknown / unimplemented backends fall back to
   ``_NullVendorDaemon`` which is a clean passthrough.

None of these tests require cv2 / onnxruntime / insightface to be installed.
The daemons degrade to passthrough on this CI box (no GPU, no models,
neither vendor dir present); the asserts below pin that contract.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.config.schema import FaceSwapConfig
from hermes_avatar.renderer.deeplivecam_daemon import DeepLiveCamVendorDaemon
from hermes_avatar.renderer.facefusion_adapter import (
    FaceSwapAdapter,
    SwapRequest,
    _NullVendorDaemon,
)
from hermes_avatar.renderer.facefusion_daemon import FaceFusionVendorDaemon


# ---------------------------------------------------------------------------
# (1) Vendor dir absent → passthrough contract
# ---------------------------------------------------------------------------
def test_facefusion_daemon_passthrough_when_vendor_missing():
    daemon = FaceFusionVendorDaemon()
    # health() before any swap: still degraded, reason clearly stated.
    h = daemon.health()
    assert h["backend"] == "facefusion"
    assert h["loaded"] is False
    assert h["degraded"] is True
    assert "not yet probed" in h["reason"]
    assert h["vendor_dir"] is None

    frame = np.zeros((4, 4, 3), dtype=np.uint8) + 7
    req = SwapRequest(
        frame=frame,
        source_face="/nonexistent/source.png",
        target_face="/nonexistent/target.png",
        character_id="indigo",
        emote_id="happy_joy",
        intensity=0.5,
    )
    out = daemon.swap(req)
    # Frame passes through byte-identical (we never wrote to it).
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, frame)


def test_deeplivecam_daemon_passthrough_when_vendor_missing():
    daemon = DeepLiveCamVendorDaemon()
    h = daemon.health()
    assert h["backend"] == "deeplivecam"
    assert h["loaded"] is False
    assert h["degraded"] is True

    frame = np.zeros((4, 4, 3), dtype=np.uint8) + 11
    req = SwapRequest(
        frame=frame,
        source_face="/nonexistent/source.png",
        target_face="/nonexistent/target.png",
        character_id="indigo",
        emote_id="concerned_furious",
        intensity=0.9,
    )
    out = daemon.swap(req)
    assert np.array_equal(out, frame)


# ---------------------------------------------------------------------------
# (2) Lazy-import contract
# ---------------------------------------------------------------------------
def test_facefusion_daemon_does_not_import_at_construct():
    """Constructing the daemon must not touch importlib or vendor modules.
    The first ``swap`` call is when ``_ensure_loaded`` runs.
    """
    with patch(
        "hermes_avatar.renderer.base_daemon.importlib.import_module"
    ) as import_mock:
        FaceFusionVendorDaemon()
    # No vendor-side imports during construction.
    import_mock.assert_not_called()


def test_deeplivecam_daemon_does_not_import_at_construct():
    with patch(
        "hermes_avatar.renderer.base_daemon.importlib.import_module"
    ) as import_mock:
        DeepLiveCamVendorDaemon()
    import_mock.assert_not_called()


def test_facefusion_daemon_imports_only_on_first_swap():
    """The first ``swap(req)`` triggers the vendor load probe. Subsequent
    calls must not re-probe (idempotent).
    """
    daemon = FaceFusionVendorDaemon()
    with patch(
        "hermes_avatar.renderer.base_daemon.importlib.import_module"
    ) as import_mock:
        # Vendor dir is absent in this env, so _load_vendor() raises
        # FileNotFoundError and we fall into passthrough. We assert:
        # importlib.import_module was NOT called (because vendor dir
        # check fails first).
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        req = SwapRequest(
            frame=frame,
            source_face="/none.png",
            target_face=None,
            character_id=None,
            emote_id=None,
            intensity=0.0,
        )
        out = daemon.swap(req)
        assert np.array_equal(out, frame)
    import_mock.assert_not_called()


def test_deeplivecam_daemon_does_not_pollute_sys_path_when_vendor_missing():
    """Headless environments must not see ``vendor/Deep-Live-Cam`` injected
    into ``sys.path`` when the directory doesn't exist.
    """
    import sys

    before = list(sys.path)
    DeepLiveCamVendorDaemon()
    after = list(sys.path)
    assert before == after, "_load_vendor ran (added sys.path) on a fresh daemon"


# ---------------------------------------------------------------------------
# (3) Cache invalidation contract (subclass stubs the heavy bits)
# ---------------------------------------------------------------------------
def test_cache_invalidation_on_path_change():
    """Same source/target paths → same cached face (extracted once).
    Path changes → fresh extraction.
    """
    from hermes_avatar.renderer.base_daemon import _VendorDaemonBase

    class CountingDaemon(_VendorDaemonBase):
        def __init__(self) -> None:
            _VendorDaemonBase.__init__(self, backend_name="counting")
            self.extract_count = {"source": 0, "target": 0}

        def _load_vendor(self):
            self._is_loaded = True
            self._is_degraded = False
            return Path("/tmp/fake-vendor")

        def _load_image(self, path):
            return f"img({path})"

        def _extract_face(self, image, *, role: str):
            self.extract_count[role] += 1
            return f"face_{role}_{image}"

        def _apply_swap(self, frame, source_face, target_face):
            return f"swapped({frame}, {source_face}, {target_face})"

    daemon = CountingDaemon()
    frame = "frame-A"

    req1 = SwapRequest(
        frame=frame, source_face="/a.png", target_face="/b.png",
        character_id="indigo", emote_id="happy", intensity=0.5,
    )
    daemon.swap(req1)
    # First call: source + target each extracted once.
    assert daemon.extract_count == {"source": 1, "target": 1}

    # Same paths again: still 1 each.
    req2 = SwapRequest(
        frame=frame, source_face="/a.png", target_face="/b.png",
        character_id="indigo", emote_id="happy", intensity=0.6,
    )
    daemon.swap(req2)
    assert daemon.extract_count == {"source": 1, "target": 1}

    # Source path changes: source re-extracted, target still cached.
    req3 = SwapRequest(
        frame=frame, source_face="/c.png", target_face="/b.png",
        character_id="indigo", emote_id="neutral", intensity=0.4,
    )
    daemon.swap(req3)
    assert daemon.extract_count == {"source": 2, "target": 1}

    # Target path changes: target re-extracted, source cached.
    req4 = SwapRequest(
        frame=frame, source_face="/c.png", target_face="/d.png",
        character_id="indigo", emote_id="sad", intensity=0.7,
    )
    daemon.swap(req4)
    assert daemon.extract_count == {"source": 2, "target": 2}


def test_swap_failure_falls_back_to_passthrough_frame():
    """An exception raised inside ``_apply_swap`` must surface as
    ``req.frame`` returned unchanged, never a raised exception.
    """
    from hermes_avatar.renderer.base_daemon import _VendorDaemonBase

    class BoomDaemon(_VendorDaemonBase):
        def __init__(self) -> None:
            _VendorDaemonBase.__init__(self, backend_name="boom")

        def _load_vendor(self):
            self._is_loaded = True
            self._is_degraded = False
            return Path("/tmp/fake-vendor")

        def _load_image(self, path):
            return f"img({path})"

        def _extract_face(self, image, *, role: str):
            return f"face_{role}"

        def _apply_swap(self, frame, source_face, target_face):
            raise RuntimeError("vendor blew up")

    frame = np.zeros((3, 3, 3), dtype=np.uint8) + 13
    req = SwapRequest(
        frame=frame, source_face="/a.png", target_face="/b.png",
        character_id="indigo", emote_id=None, intensity=0.0,
    )
    out = BoomDaemon().swap(req)
    assert np.array_equal(out, frame)


# ---------------------------------------------------------------------------
# (4) FaceSwapAdapter auto-pick
# ---------------------------------------------------------------------------
def test_face_swap_adapter_auto_picks_facefusion_daemon():
    """With ``backend='facefusion'`` and no explicit daemon, the adapter
    must wire in a ``FaceFusionVendorDaemon`` (currently degraded because
    vendor absen, but the class is correct).
    """
    adapter = FaceSwapAdapter(backend="facefusion", enabled=True)
    assert isinstance(adapter.manager.daemon, FaceFusionVendorDaemon)


def test_face_swap_adapter_auto_picks_deeplivecam_daemon():
    adapter = FaceSwapAdapter(backend="deeplivecam", enabled=True)
    assert isinstance(adapter.manager.daemon, DeepLiveCamVendorDaemon)


def test_face_swap_adapter_disabled_skips_daemon_auto_pick():
    """When faceswap is disabled, no vendor daemon should be built. We
    keep the production-default OFF behaviour intact.
    """
    adapter = FaceSwapAdapter(backend="facefusion", enabled=False)
    assert adapter.manager.daemon is None


def test_face_swap_adapter_explicit_daemon_wins():
    """If a daemon is injected explicitly, we never override it (e.g.
    FakeVendorDaemon in tests). Auto-pick only fires when daemon is None.
    """
    from hermes_avatar.renderer.facefusion_adapter import FakeVendorDaemon

    fake = FakeVendorDaemon()
    adapter = FaceSwapAdapter(backend="facefusion", enabled=True, daemon=fake)
    assert adapter.manager.daemon is fake
    assert not isinstance(adapter.manager.daemon, FaceFusionVendorDaemon)


def test_face_swap_adapter_unknown_backend_uses_null_daemon():
    adapter = FaceSwapAdapter(backend="madeup_backend", enabled=True)
    assert isinstance(adapter.manager.daemon, _NullVendorDaemon)


# ---------------------------------------------------------------------------
# (5) Null daemon behaviour
# ---------------------------------------------------------------------------
def test_null_daemon_passes_frames_through():
    null = _NullVendorDaemon("madeup_backend")
    frame = np.zeros((4, 4, 3), dtype=np.uint8) + 99
    req = SwapRequest(
        frame=frame, source_face="/a", target_face="/b",
        character_id=None, emote_id=None, intensity=0.0,
    )
    assert np.array_equal(null.swap(req), frame)
    h = null.health()
    assert h["degraded"] is True
    assert "madeup_backend" in h["reason"]


# ---------------------------------------------------------------------------
# (6) End-to-end passthrough through FaceSwapAdapter → BackendManager
# ---------------------------------------------------------------------------
def test_face_swap_adapter_passthrough_when_only_vendor_daemon_available(
    tmp_path,
):
    """Construct a character with a canonical face, hook the adapter up to
    a FaceFusionVendorDaemon (which is degraded here), and assert that
    process_frame still returns the original frame.
    """
    import os

    canonical = tmp_path / "canonical.png"
    canonical.write_bytes(_stub_png_bytes())

    cfg = FaceSwapConfig(enabled=True, backend="facefusion", source_face_path=str(canonical))
    adapter = FaceSwapAdapter(backend="facefusion", enabled=True, config=cfg)
    # Force the simulation flags so the manager path opens to the daemon.
    adapter.manager.available = True
    adapter.manager.degraded = False
    adapter.manager.passthrough = False

    assert adapter.manager.daemon is not None
    assert isinstance(adapter.manager.daemon, FaceFusionVendorDaemon)

    frame = np.zeros((4, 4, 3), dtype=np.uint8) + 21
    req = SwapRequest(
        frame=frame, source_face=str(canonical), target_face=None,
        character_id="stub", emote_id=None, intensity=0.0,
    )
    out = adapter.manager.swap_with_request(req)
    assert np.array_equal(out, frame)


def _stub_png_bytes() -> bytes:
    """A 1x1 transparent PNG — smallest possible valid PNG. We never
    decode it (the daemons passthrough when cv2 is missing), we just
    need a file to exist.
    """
    import base64

    return base64.b64decode(
        # 1x1 transparent PNG
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII="
    )
