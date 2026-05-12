from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.mirror_policy import mirrored_affect
from hermes_avatar.affect.reflect_policy import reflected_affect
from hermes_avatar.affect.state import UserAffectState


def test_user_speaking_switches_to_listening():
    runtime = AffectRuntime(emote_lookup=lambda state: f"{state}_001")
    behavior = runtime.consume({"type": "audio.vad", "timestamp_ms": 1000, "speaking": True, "energy": 0.7, "speech_rate": 0.5})
    assert behavior.mode == "listening"
    assert behavior.lip_sync_enabled is False
    assert behavior.gaze_target == "toward_user"


def test_reflect_frustration_validates_without_copying_anger():
    runtime = AffectRuntime(emote_lookup=lambda state: f"{state}_001")
    runtime.expression_latch.dwell_ms = 0
    runtime.consume({"type": "perception.frame", "timestamp_ms": 1000, "face_detected": True, "face_center": [0.5, 0.5], "head_yaw": 0, "head_pitch": 0, "expression": {"frown": 0.8, "brow_raise": 0.5, "eye_open": 0.7}})
    behavior = runtime.tick(1300)
    assert behavior.affect in {"validating_grounded", "attentive_soft"}
    assert "angry" not in behavior.affect
    assert behavior.mirror_strength == 0.0


def test_mirroring_prefers_positive_affiliation_over_negative_contagion():
    happy_affect, happy_intensity = mirrored_affect(UserAffectState(face_detected=True, attention=0.9, valence=0.5, arousal=0.4, dominant_expression="happy"))
    frustrated_affect, frustrated_intensity = mirrored_affect(UserAffectState(face_detected=True, attention=0.9, valence=-0.5, arousal=0.7, tension=0.8, dominant_expression="frustrated"))
    assert happy_affect == "small_delayed_smile"
    assert frustrated_affect == "grounded_concern_soft_brow"
    assert happy_intensity > frustrated_intensity
    assert frustrated_intensity <= 0.16


def test_reflection_gives_space_when_attention_is_low():
    affect, intensity = reflected_affect(UserAffectState(face_detected=True, attention=0.1, gaze_direction="away", dominant_expression="neutral", arousal=0.4))
    assert affect == "spacious_attentive"
    assert intensity == 0.22


def test_audio_updates_arousal_for_policy_context():
    runtime = AffectRuntime(emote_lookup=lambda state: f"{state}_001")
    runtime.consume({"type": "audio.vad", "timestamp_ms": 1000, "speaking": True, "energy": 0.9, "speech_rate": 0.8})
    assert runtime.user.arousal > 0.0


def test_assistant_speaking_enables_lipsync():
    runtime = AffectRuntime(emote_lookup=lambda state: f"{state}_001")
    runtime.conversation.turn_state = "assistant_speaking"
    runtime.hermes_tags = {"affect": "focused", "voice": {"intensity": 0.35}}
    behavior = runtime.tick(2000)
    assert behavior.mode == "speaking"
    assert behavior.lip_sync_enabled is True
    assert behavior.mirror_strength == 0.0
