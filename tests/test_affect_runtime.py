from hermes_avatar.affect.policy import AffectRuntime


def test_user_speaking_switches_to_listening():
    runtime = AffectRuntime(emote_lookup=lambda state: f"{state}_001")
    behavior = runtime.consume({"type": "audio.vad", "timestamp_ms": 1000, "speaking": True, "energy": 0.7, "speech_rate": 0.5})
    assert behavior.mode == "listening"
    assert behavior.lip_sync_enabled is False
    assert behavior.gaze_target == "toward_user"


def test_reflect_frustration_moderates_anger():
    runtime = AffectRuntime(emote_lookup=lambda state: f"{state}_001")
    runtime.expression_latch.dwell_ms = 0
    runtime.consume({"type": "perception.frame", "timestamp_ms": 1000, "face_detected": True, "face_center": [0.5, 0.5], "head_yaw": 0, "head_pitch": 0, "expression": {"frown": 0.8, "brow_raise": 0.5, "eye_open": 0.7}})
    behavior = runtime.tick(1300)
    assert behavior.affect in {"apologetic_grounded", "attentive_soft"}
    assert "angry" not in behavior.affect


def test_assistant_speaking_enables_lipsync():
    runtime = AffectRuntime(emote_lookup=lambda state: f"{state}_001")
    runtime.conversation.turn_state = "assistant_speaking"
    runtime.hermes_tags = {"affect": "focused", "voice": {"intensity": 0.35}}
    behavior = runtime.tick(2000)
    assert behavior.mode == "speaking"
    assert behavior.lip_sync_enabled is True
    assert behavior.mirror_strength == 0.0
