"""Tests for the perception trackers.

These tests verify the *graceful degradation* contract the redesign depends
on: if mediapipe / opencv aren't installed, the package still imports and
the runtime still runs because the tracker degrades to ``NullFaceTracker``.
We do NOT require mediapipe to be installed in CI.
"""
from __future__ import annotations

from hermes_avatar.perception.mediapipe_tracker import (
    FaceSignals,
    NullFaceTracker,
    build_tracker,
)


def test_null_tracker_is_safe_no_op():
    tracker = NullFaceTracker()
    assert tracker.is_available() is False
    assert tracker.kind() == "null"
    signals = tracker.process_frame("aGVsbG8=", timestamp_ms=123)
    assert isinstance(signals, FaceSignals)
    assert signals.face_detected is False
    assert signals.last_updated_ms == 123


def test_build_tracker_returns_null_or_mediapipe(monkeypatch):
    """build_tracker never raises and always returns an initialised tracker."""
    tracker = build_tracker(prefer="mediapipe")
    assert tracker is not None
    assert tracker.kind() in {"mediapipe", "null", "unavailable", "init_failed"}
    tracker = build_tracker(prefer="null")
    assert isinstance(tracker, NullFaceTracker)


def test_signals_to_dict_matches_runtime_event_contract():
    """AffectRuntime consumes ``perception.frame`` events with these exact keys."""
    signals = FaceSignals(
        face_detected=True,
        attention=0.82,
        valence=0.4,
        arousal=0.55,
        tension=0.18,
        dominant_expression="happy",
        gaze_direction="toward_user",
        head_yaw=0.05,
        head_pitch=-0.02,
        eye_aspect_ratio=0.30,
        smile_ratio=0.7,
        blink_rate=0.1,
        mouth_open=0.2,
        emotion_confidence=0.7,
        gaze_confidence=0.85,
        last_updated_ms=999,
    )
    payload = signals.to_dict()
    # Required keys the runtime reads.
    for key in (
        "face_detected",
        "head_yaw",
        "head_pitch",
        "gaze_confidence",
        "emotion_confidence",
        "expression",
        "timestamp_ms",
        "attention",
        "valence",
        "arousal",
        "tension",
        "dominant_expression",
        "gaze_direction",
    ):
        assert key in payload, key
    # Expression keys policy functions read.
    for key in ("smile", "frown", "brow_raise", "eye_open"):
        assert key in payload["expression"], key
    assert payload["expression"]["smile"] == 0.7
    assert payload["timestamp_ms"] == 999
