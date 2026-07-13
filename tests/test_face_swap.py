from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.config.schema import AppConfig, FaceSwapConfig, load_config
from hermes_avatar.renderer.deeplivecam_adapter import DeepLiveCamAdapter
from hermes_avatar.renderer.facefusion_adapter import (
    BackendManager,
    FaceSwapAdapter,
    FaceSwapPipeline,
    ListFrameSink,
    ListFrameSource,
)
from scripts.create_sample_character import PNG_1X1_RGBA


# ---------------------------------------------------------------------------
# (a) Adapter construction
# ---------------------------------------------------------------------------
def test_faceswap_adapter_constructs_with_defaults():
    adapter = FaceSwapAdapter()
    assert adapter.config.backend == "facefusion"
    # Disabled by default -> not active, no crash.
    caps = adapter.capabilities()
    assert caps["enabled"] is False
    assert caps["replacement_active"] is False
    # Required capability keys are always present and accurate.
    for key in ("backend", "online", "model_present", "degraded", "source_image_present", "output_target"):
        assert key in caps


def test_deeplivecam_adapter_is_facefusion_adapter_subclass():
    adapter = DeepLiveCamAdapter(enabled=True)
    assert isinstance(adapter, FaceSwapAdapter)
    assert adapter.config.backend == "deeplivecam"
    assert adapter.config.vendor_dir == "vendor/Deep-Live-Cam"


def test_faceswap_adapter_applies_overrides():
    adapter = FaceSwapAdapter(
        backend="deeplivecam",
        enabled=True,
        device="cuda",
        input_source="rtsp://127.0.0.1:8554/x",
        output_virtual_cam="/dev/video10",
    )
    assert adapter.config.backend == "deeplivecam"
    assert adapter.config.enabled is True
    assert adapter.config.device == "cuda"
    assert adapter.config.input_source == "rtsp://127.0.0.1:8554/x"
    assert adapter.config.output_virtual_cam == "/dev/video10"


# ---------------------------------------------------------------------------
# (b) Capability reporting when backend / model is ABSENT (expected CI state)
# ---------------------------------------------------------------------------
def test_faceswap_degraded_without_source_face(tmp_path):
    char = tmp_path / "empty"
    char.mkdir()
    index = build_asset_index(char)

    adapter = FaceSwapAdapter(backend="facefusion", enabled=True)
    adapter.load_character(index)
    caps = adapter.capabilities()

    assert caps["source_image_present"] is False
    assert caps["degraded"] is True
    assert caps["passthrough"] is True
    assert caps["online"] is False
    assert caps["replacement_active"] is False


def test_faceswap_degraded_when_models_absent(tmp_path):
    """Source face exists on disk but no backend binary / models -> the adapter
    detects this at activation, reports it accurately, and degrades to
    passthrough without spawning anything or raising."""
    char = tmp_path / "canonical_only"
    (char / "canonical").mkdir(parents=True)
    (char / "canonical" / "canonical.png").write_bytes(PNG_1X1_RGBA)
    index = build_asset_index(char)

    adapter = FaceSwapAdapter(backend="facefusion", enabled=True)
    # No subprocess should be spawned in the degraded path.
    assert adapter.manager.process is None
    adapter.load_character(index)
    caps = adapter.capabilities()

    assert caps["source_image_present"] is True
    assert caps["model_present"] is False
    assert caps["degraded"] is True
    assert caps["passthrough"] is True
    assert caps["online"] is False
    assert caps["replacement_active"] is False
    assert caps["process_running"] is False
    assert isinstance(caps["error"], str) and caps["error"]


def test_backend_manager_detect_reports_absent_backend():
    cfg = FaceSwapConfig(backend="facefusion", vendor_dir="vendor/DoesNotExist")
    mgr = BackendManager(cfg)
    mgr.detect()
    assert mgr.available is False
    assert mgr.degraded is True
    assert mgr.passthrough is True
    assert mgr.error is not None


# ---------------------------------------------------------------------------
# (c) Config parsing + FACESWAP__ env var overrides
# ---------------------------------------------------------------------------
def test_faceswap_env_overrides(monkeypatch):
    monkeypatch.setenv("FACESWAP__BACKEND", "deeplivecam")
    monkeypatch.setenv("FACESWAP__ENABLED", "true")
    monkeypatch.setenv("FACESWAP__DEVICE", "cuda")
    monkeypatch.setenv("FACESWAP__INPUT_SOURCE", "rtsp://127.0.0.1:8554/avatar")
    monkeypatch.setenv("FACESWAP__OUTPUT_VIRTUAL_CAM", "/dev/video10")
    monkeypatch.setenv("FACESWAP__FRAME_RATE", "30")

    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.faceswap.backend == "deeplivecam"
    assert cfg.faceswap.enabled is True
    assert cfg.faceswap.device == "cuda"
    assert cfg.faceswap.input_source == "rtsp://127.0.0.1:8554/avatar"
    assert cfg.faceswap.output_virtual_cam == "/dev/video10"
    assert cfg.faceswap.frame_rate == 30


def test_faceswap_config_defaults():
    cfg = FaceSwapConfig()
    assert cfg.backend == "facefusion"
    assert cfg.enabled is False
    assert cfg.device == "cpu"
    assert cfg.input_source == "http://127.0.0.1:8010"
    assert cfg.frame_rate == 25


# ---------------------------------------------------------------------------
# (d) Frame-pipeline wiring with a MOCKED backend subprocess
# ---------------------------------------------------------------------------
def _mock_manager(cfg: FaceSwapConfig) -> BackendManager:
    """Build a manager that pretends the backend is live, bypassing real
    detection/startup so the pipeline can be exercised offline."""
    mgr = BackendManager(cfg)
    mgr.available = True
    mgr.degraded = False
    mgr.passthrough = False
    mgr.model_present = True
    mgr.process = MagicMock(poll=MagicMock(return_value=None))
    mgr.detect = lambda: None  # type: ignore[assignment]
    mgr.start = lambda source_face=None: None  # type: ignore[assignment]
    mgr.swap_callable = lambda frame: frame + 1
    return mgr


def test_frame_pipeline_process_frame_applies_swap():
    cfg = FaceSwapConfig(backend="facefusion", enabled=True, source_face_path="fake.png")
    mgr = _mock_manager(cfg)
    adapter = FaceSwapAdapter(
        config=cfg,
        backend_manager=mgr,
        frame_source=ListFrameSource([]),
        frame_sink=ListFrameSink(),
    )
    assert adapter.replacement_active is True
    assert adapter.pipeline is not None

    frame = np.zeros((4, 4, 3), dtype=np.uint8) + 5
    out = adapter.pipeline.process_frame(frame)
    assert np.array_equal(out, frame + 1)


def test_frame_pipeline_full_run_with_mocked_backend():
    cfg = FaceSwapConfig(backend="facefusion", enabled=True, source_face_path="fake.png")
    mgr = _mock_manager(cfg)
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) + i for i in range(3)]
    src = ListFrameSource(frames)
    sink = ListFrameSink()
    adapter = FaceSwapAdapter(
        config=cfg,
        backend_manager=mgr,
        frame_source=src,
        frame_sink=sink,
    )
    adapter.pipeline.run(max_frames=3)
    assert len(sink.frames) == 3
    # Each frame was swapped (fake.png stub adds 1) on the way through.
    assert np.array_equal(sink.frames[0], frames[0] + 1)
    assert np.array_equal(sink.frames[2], frames[2] + 1)
    adapter.shutdown()


def test_frame_pipeline_passthrough_when_backend_absent():
    """With no backend, process_frame returns the frame unchanged (passthrough)."""
    cfg = FaceSwapConfig(backend="facefusion", vendor_dir="vendor/DoesNotExist")
    mgr = BackendManager(cfg)
    mgr.detect()  # sets degraded / passthrough
    frame = np.zeros((2, 2, 3), dtype=np.uint8) + 7
    out = mgr.swap(frame)
    assert np.array_equal(out, frame)


def test_interrupt_stops_pipeline_without_error(tmp_path):
    char = tmp_path / "canonical_only"
    (char / "canonical").mkdir(parents=True)
    (char / "canonical" / "canonical.png").write_bytes(PNG_1X1_RGBA)
    index = build_asset_index(char)

    adapter = FaceSwapAdapter(backend="facefusion", enabled=True)
    adapter.load_character(index)
    # interrupt must be safe even in the degraded/passthrough state.
    adapter.interrupt()
    assert adapter.replacement_active is False
    adapter.shutdown()


def test_health_surface_present():
    adapter = FaceSwapAdapter(backend="deeplivecam", enabled=True)
    health = adapter.health()
    assert "status" in health
    assert health["backend"] == "deeplivecam"
    assert "detail" in health
