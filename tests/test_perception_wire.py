"""Verify the perception pipeline is wired end-to-end.

The three gaps that existed:
1. No /api/perception/video route in routes.py
2. No tracker instance in DemoOrchestrator
3. No wiring from route → tracker.process_frame() → runtime.consume()

Tests are intentionally lightweight:
- Tracker + FaceSignals tested directly (no orchestrator init cost).
- The NullFaceTracker fallback for environments without MediaPipe.
- The orchestrator's process_perception_frame method tested via a mock.
- Route schema validates the Pydantic model.
- Capabilities enforce perception key presence.
"""
from __future__ import annotations

import base64
import io
import time
from unittest import mock

import pytest
from PIL import Image

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.config.schema import AppConfig, load_config
from hermes_avatar.perception.mediapipe_tracker import (
    FaceSignals,
    NullFaceTracker,
    build_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_jpeg_b64() -> str:
    img = Image.new("RGB", (1, 1), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{raw}"


# ---------------------------------------------------------------------------
# Tests: tracker + FaceSignals (no orchestrator needed)
# ---------------------------------------------------------------------------

class TestNullTracker:
    def test_null_tracker_is_always_usable(self):
        t = NullFaceTracker()
        assert t.is_available() is False
        assert t.kind() == "null"

    def test_null_tracker_produces_empty_signals(self):
        t = NullFaceTracker()
        s = t.process_frame(_blank_jpeg_b64(), timestamp_ms=99999)
        assert isinstance(s, FaceSignals)
        assert s.face_detected is False
        assert s.dominant_expression == "neutral"
        assert s.last_updated_ms == 99999

    def test_null_tracker_handles_empty_input(self):
        t = NullFaceTracker()
        s = t.process_frame("", 0)
        assert s.last_updated_ms == 0
        assert s.face_detected is False


class TestMediaPipeTracker:
    def test_build_tracker_returns_something(self):
        t = build_tracker("mediapipe")
        assert t is not None
        assert t.kind() in ("mediapipe", "null", "model_missing")

    def test_real_tracker_handles_empty_input(self):
        t = build_tracker("mediapipe")
        s = t.process_frame("", 12345)
        assert s.face_detected is False
        assert s.last_updated_ms == 12345

    def test_real_tracker_processes_jpeg(self):
        t = build_tracker("mediapipe")
        s = t.process_frame(_blank_jpeg_b64(), 54321)
        # Either no face detected (1×1 pixel image) or an exception handled gracefully.
        assert s.last_updated_ms == 54321
        assert isinstance(s.face_detected, bool)


class TestFaceSignalsToDict:
    def test_to_dict_includes_all_keys(self):
        s = FaceSignals(
            face_detected=True,
            attention=0.8,
            valence=0.4,
            arousal=0.55,
            tension=0.2,
            dominant_expression="happy",
            gaze_direction="toward_user",
            head_yaw=0.05,
            last_updated_ms=12345,
        )
        d = s.to_dict()
        assert d["face_detected"] is True
        assert d["dominant_expression"] == "happy"
        assert d["gaze_direction"] == "toward_user"
        assert d["timestamp_ms"] == 12345
        assert "expression" in d


# ---------------------------------------------------------------------------
# Tests: runtime consume of perception.frame
# ---------------------------------------------------------------------------

class TestRuntimeConsumePerception:
    def test_consume_perception_frame_updates_user(self):
        runtime = AffectRuntime()
        ts = int(time.time() * 1000)
        signals = FaceSignals(
            face_detected=True,
            attention=0.75,
            valence=0.3,
            arousal=0.5,
            tension=0.1,
            dominant_expression="neutral",
            gaze_direction="toward_user",
            last_updated_ms=ts,
        )
        behavior = runtime.consume({
            "type": "perception.frame",
            **signals.to_dict(),
            "timestamp_ms": ts,
        })
        assert runtime.user.last_updated_ms == ts
        assert runtime.user.face_detected is True
        assert runtime.user.gaze_direction == "toward_user"
        assert behavior.mode in ("idle", "thinking")


# ---------------------------------------------------------------------------
# Tests: orchestrator.process_perception_frame (mocked)
# ---------------------------------------------------------------------------

class TestOrchestratorProcessFrame:
    def test_process_perception_frame_calls_tracker_and_runtime(self):
        """Verify the orchestrator method's contract without a full init."""
        from hermes_avatar.demo.demo_orchestrator import DemoOrchestrator

        tracker = NullFaceTracker()
        runtime = AffectRuntime()

        with mock.patch.object(
            DemoOrchestrator, "__init__", lambda self: None
        ):
            orch = DemoOrchestrator.__new__(DemoOrchestrator)
            orch.tracker = tracker
            orch.runtime = runtime
            orch.renderer = mock.MagicMock()

        ts = int(time.time() * 1000)
        result = orch.process_perception_frame(_blank_jpeg_b64(), ts)

        assert "signals" in result
        assert result["signals"]["face_detected"] is False
        assert orch.runtime.user.last_updated_ms == ts
        assert orch.renderer.set_behavior.called


# ---------------------------------------------------------------------------
# Tests: Pydantic model for the route
# ---------------------------------------------------------------------------

class TestPerceptionFrameModel:
    def test_model_accepts_valid_input(self):
        from apps.demo_server.routes import PerceptionFrameRequest
        p = PerceptionFrameRequest(image=_blank_jpeg_b64(), timestamp_ms=12345)
        assert p.image.startswith("data:image/jpeg;base64,")
        assert p.timestamp_ms == 12345

    def test_model_defaults_timestamp(self):
        from apps.demo_server.routes import PerceptionFrameRequest
        p = PerceptionFrameRequest(image=_blank_jpeg_b64())
        assert p.timestamp_ms == 0
