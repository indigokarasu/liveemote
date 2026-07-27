"""Tests for ``FaceFusionSidecarDaemon``.

These tests do NOT require a real FaceFusion sidecar container — they
exercise the daemon's httpx surface via :class:`unittest.mock.MagicMock` so
the test suite stays self-contained. Real face-swap correctness against a
running ONNX model pack is covered by the vendored integration tests.

What we cover:

* URL parsing — malformed / empty URLs degrade to passthrough, not crash.
* Swap passthrough on missing source face, 4xx, 5xx, and connection error.
* Successful swap round-trip — 200 + JPEG bytes come back as a BGR ndarray,
  Authorization bearer header delivered, intensity form field set.
* Health probe success + connection error + cache TTL expiry.
* Auto-pick by ``FaceSwapAdapter`` when ``sidecar_url`` is configured; falls
  back to the in-process ``FaceFusionVendorDaemon`` when ``sidecar_url`` is
  empty.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---- ensure cv2 stub doesn't break the daemon if it's missing on this box
cv2_stub = MagicMock()
cv2_stub.imencode = MagicMock(return_value=(True, np.zeros((1, 1), dtype=np.uint8)))
cv2_stub.imdecode = MagicMock(side_effect=lambda arr, _flag: np.zeros((8, 8, 3), dtype=np.uint8))
cv2_stub.IMREAD_COLOR = 1
cv2_stub.IMWRITE_JPEG_QUALITY = 95
sys.modules.setdefault("cv2", cv2_stub)


class _Resp:
    """Stand-in for an httpx.Response with the few attributes the daemon
    reads (status_code, content, headers, json(), text)."""

    def __init__(self, status_code=200, content=b"", headers=None, json_body=None, text=""):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self._json = json_body
        self.text = text

    def json(self):
        return self._json if self._json is not None else {}


@pytest.fixture
def fake_httpx(monkeypatch):
    """Patch ``httpx.Client`` with a MagicMock factory. Tests configure
    ``httpx.Client.return_value.post.return_value`` / ``.get.return_value``
    directly to script the sidecar's responses. The factory also behaves as a
    context manager so ``with httpx.Client(...) as client:`` calls pass."""
    import httpx

    mock = MagicMock()
    # Make the context-manager protocol transparent.
    mock.return_value.__enter__.return_value = mock.return_value
    mock.return_value.__exit__.return_value = False
    monkeypatch.setattr(httpx, "Client", mock)
    return mock


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------
def test_resolve_sidecar_accepts_full_url():
    from hermes_avatar.renderer.facefusion_sidecar_daemon import _resolve_sidecar

    base, path = _resolve_sidecar("http://liveemote-facefusion:8001")
    assert base == "http://liveemote-facefusion:8001"
    assert path == "/api/v1/swap"


def test_resolve_sidecar_rejects_empty():
    from hermes_avatar.renderer.facefusion_sidecar_daemon import _resolve_sidecar

    base, path = _resolve_sidecar("")
    assert (base, path) == ("", None)


def test_resolve_sidecar_rejects_garbage():
    from hermes_avatar.renderer.facefusion_sidecar_daemon import _resolve_sidecar

    base, path = _resolve_sidecar("not-a-url")
    assert (base, path) == ("", None)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_construction_with_no_url_keeps_degraded():
    from hermes_avatar.renderer.facefusion_sidecar_daemon import FaceFusionSidecarDaemon

    d = FaceFusionSidecarDaemon(sidecar_url="")
    h = d.health()
    assert h["degraded"] is True
    assert "malformed" in (h.get("reason") or "")


def test_construction_sets_url_and_metadata():
    from hermes_avatar.renderer.facefusion_sidecar_daemon import FaceFusionSidecarDaemon

    d = FaceFusionSidecarDaemon(
        sidecar_url="http://localhost:8001",
        api_key="abc",
        connect_timeout_s=1.5,
        request_timeout_s=4.0,
    )
    assert d.base_url == "http://localhost:8001"
    assert d.api_key == "abc"
    assert d.connect_timeout_s == 1.5
    assert d.request_timeout_s == 4.0


def test_swap_with_malformed_url_passthrough():
    from hermes_avatar.renderer.facefusion_sidecar_daemon import FaceFusionSidecarDaemon, SwapRequest

    d = FaceFusionSidecarDaemon(sidecar_url="garbage")
    req = SwapRequest(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        source_face="/tmp/does-not-exist.png",
        target_face=None,
        character_id=None,
        emote_id=None,
        intensity=1.0,
    )
    out = d.swap(req)
    assert out is req.frame
    assert d._passthrough_count == 1


def test_swap_with_missing_source_face_passthrough(fake_httpx):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import FaceFusionSidecarDaemon, SwapRequest

    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001")
    req = SwapRequest(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        source_face="/tmp/definitely-not-on-disk.png",
        target_face=None,
        character_id=None,
        emote_id=None,
        intensity=1.0,
    )
    out = d.swap(req)
    assert out is req.frame
    assert d._passthrough_count == 1
    fake_httpx.return_value.post.assert_not_called()


# ---------------------------------------------------------------------------
# Health probe behavior
# ---------------------------------------------------------------------------
def test_health_probe_success_caches_response(fake_httpx):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
    )

    fake_httpx.return_value.get.return_value = _Resp(
        200,
        json_body={"status": "ok", "vendor_present": True, "vendor_dir": "/x"},
        headers={"content-type": "application/json"},
    )

    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001")
    h = d.health()
    assert h["loaded"] is True
    assert h["degraded"] is False
    assert h["sidecar_status"] == "ok"
    # The second call returns the cached snapshot without a second HTTP poll.
    fake_httpx.return_value.get.reset_mock()
    h2 = d.health()
    assert h2 == h
    fake_httpx.return_value.get.assert_not_called()


def test_health_probe_connection_error_degrades(fake_httpx):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
    )

    fake_httpx.return_value.get.side_effect = ConnectionError("nope")
    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001")
    h = d.health()
    assert h["degraded"] is True
    assert "unreachable" in (h.get("reason") or "")


def test_health_cache_ttl_expires(fake_httpx):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
    )

    fake_httpx.return_value.get.return_value = _Resp(
        200,
        json_body={"status": "ok", "vendor_present": True, "vendor_dir": "/x"},
        headers={"content-type": "application/json"},
    )
    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001")
    d._health_cache_ttl = 0.01
    d.health()
    time.sleep(0.05)
    assert d._maybe_get_cached_health() is None


# ---------------------------------------------------------------------------
# Swap round-trip — uses real on-disk placeholder files so _read_image_bytes
# succeeds when cv2 is missing in CI.
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_source_face(tmp_path: Path) -> str:
    p = tmp_path / "source.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(p)


def test_swap_2xx_returns_decoded_image(fake_httpx, fake_source_face):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
        SwapRequest,
    )

    fake_httpx.return_value.post.return_value = _Resp(
        200,
        content=b"\xff\xd8\xff\xe0\x00\x10JFIF",  # minimal magic bytes; cv2 stub decodes anyway
        headers={"content-type": "image/jpeg", "X-Swap-Mode": "swap"},
    )

    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001", api_key="tok")
    req = SwapRequest(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        source_face=fake_source_face,
        target_face=None,
        character_id=None,
        emote_id=None,
        intensity=0.7,
    )
    out = d.swap(req)
    # cv2 stub decodes to np.zeros((8,8,3))
    assert out is not None
    assert d._swaps_succeeded == 1
    kwargs = fake_httpx.return_value.post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer tok"
    assert kwargs["data"]["intensity"] == "0.7"


def test_swap_4xx_passthrough(fake_httpx, fake_source_face):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
        SwapRequest,
    )

    fake_httpx.return_value.post.return_value = _Resp(400, text="bad face")
    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001")
    req = SwapRequest(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        source_face=fake_source_face,
        target_face=None,
        character_id=None,
        emote_id=None,
        intensity=1.0,
    )
    out = d.swap(req)
    assert out is req.frame
    assert d._passthrough_count == 1
    assert d._last_error and "4xx" in d._last_error


def test_swap_5xx_passthrough(fake_httpx, fake_source_face):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
        SwapRequest,
    )

    fake_httpx.return_value.post.return_value = _Resp(500, text="boom")
    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001")
    req = SwapRequest(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        source_face=fake_source_face,
        target_face=None,
        character_id=None,
        emote_id=None,
        intensity=1.0,
    )
    out = d.swap(req)
    assert out is req.frame
    assert d._passthrough_count == 1
    assert d._last_error and "5xx" in d._last_error


def test_swap_connect_error_passthrough(fake_httpx, fake_source_face):
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
        SwapRequest,
    )

    fake_httpx.return_value.post.side_effect = ConnectionError("container down")
    d = FaceFusionSidecarDaemon(sidecar_url="http://localhost:8001")
    req = SwapRequest(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        source_face=fake_source_face,
        target_face=None,
        character_id=None,
        emote_id=None,
        intensity=1.0,
    )
    out = d.swap(req)
    assert out is req.frame
    assert d._passthrough_count == 1
    assert d._last_error and "failed" in d._last_error


# ---------------------------------------------------------------------------
# Adapter auto-pick wiring
# ---------------------------------------------------------------------------
def test_adapter_autopicks_sidecar_when_sidecar_url_set():
    from hermes_avatar.config.schema import FaceSwapConfig
    from hermes_avatar.renderer.facefusion_adapter import FaceSwapAdapter
    from hermes_avatar.renderer.facefusion_sidecar_daemon import (
        FaceFusionSidecarDaemon,
    )

    cfg = FaceSwapConfig(
        enabled=True,
        vendor_dir="vendor/does-not-exist",
        source_face_path=None,
        sidecar_url="http://localhost:8001",
    )
    adapter = FaceSwapAdapter(config=cfg, source_face_path=None, daemon=None)
    assert isinstance(adapter.manager.daemon, FaceFusionSidecarDaemon)


def test_adapter_falls_back_to_in_process_when_no_sidecar_url():
    from hermes_avatar.config.schema import FaceSwapConfig
    from hermes_avatar.renderer.facefusion_adapter import FaceSwapAdapter
    from hermes_avatar.renderer.facefusion_daemon import FaceFusionVendorDaemon

    cfg = FaceSwapConfig(
        enabled=True,
        vendor_dir="vendor/does-not-exist",
        source_face_path=None,
        sidecar_url=None,
    )
    adapter = FaceSwapAdapter(config=cfg, source_face_path=None, daemon=None)
    assert isinstance(adapter.manager.daemon, FaceFusionVendorDaemon)
