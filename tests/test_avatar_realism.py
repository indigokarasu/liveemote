"""
test_avatar_realism.py — full coverage of all 6 avatar-realism subsystems.

Tests are organised by improvement number:

  1. Emotional inertia — AffectBlender crossfade timing, progress, intensity blend
  2. Idle micro-movements — CSS-only; validated via AvatarBehaviorState defaults
  3. Breathing entrainment — _breath_rate_from_arousal() mapping
  4. Head tilt mirroring — _ema() smoothing + UserAffectState EMA persistence
  5. Gaze triangle — GazePolicy cycle distribution, dwell ranges, mode switching
  6. Anticipatory micro-expressions — pre_speech_affect + is_speaking from hermes_tags
  I. Integration — Full AffectRuntime.tick() pipeline end-to-end
"""

import math
import random
import pytest  # type: ignore

# Ensure the packages tree is on sys.path before importing hermes_avatar.
# (Run with  PYTHONPATH=packages  from the repo root.)
from hermes_avatar.affect.state import (
    AvatarBehaviorState,
    FaceSignals,
    UserAffectState,
)
from hermes_avatar.affect.affect_blender import AffectBlender, BlendResult
from hermes_avatar.affect.gaze_policy import (
    GazePolicy,
    GazeResult,
    _GAZE_DWELLS,
    _GAZE_TRANSITIONS,
)
from hermes_avatar.affect.policy import (
    AffectRuntime,
    ExpressionLatch,
    _avatar_affect_for,
    _breath_rate_from_arousal,
    _classify_affect,
    _ema,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _new_runtime(mode: str = "reflect") -> AffectRuntime:
    """Return a fresh AffectRuntime with a pre-seeded random to make
    gaze distributions deterministic in tests."""
    random.seed(42)
    return AffectRuntime(mode=mode)


# ═══════════════════════════════════════════════════════════════════════
# 1 — Emotional inertia: AffectBlender
# ═══════════════════════════════════════════════════════════════════════

class TestAffectBlender:

    # ── construction ─────────────────────────────────────────────────────

    def test_default_state_is_neutral(self):
        b = AffectBlender()
        assert b.current_affect == "neutral"
        assert not b.is_transitioning

    def test_custom_default_fade_ms(self):
        b = AffectBlender(default_fade_ms=400)
        r = b.tick("happy", 0.7, 10)
        assert r.affect_fade_ms == 400

    # ── transition trigger ────────────────────────────────────────────────

    def test_target_change_triggers_transition(self):
        b = AffectBlender()
        r = b.tick("happy", 0.7, 10)
        assert b.is_transitioning
        assert r.prev_affect == "neutral"
        assert r.target_affect == "happy"
        assert r.transition_progress < 1.0

    def test_transition_completes(self):
        b = AffectBlender()
        b.tick("happy", 0.7, 10)       # trigger: 600 ms fade
        r = b.tick("happy", 0.7, 600)
        assert not b.is_transitioning
        assert r.transition_progress == 1.0
        assert r.prev_affect == "happy"
        assert r.target_affect == "happy"

    def test_intensity_blend_linear(self):
        b = AffectBlender(default_fade_ms=600)
        # neutral→happy maps to 400 ms; use a transition with default 600 ms instead
        # neutral→sad: "sad" is not in the custom fade map, stays at default 600 ms
        b.tick("sad", 0.7, 10)          # trigger; prev_intensity = 0.5 (neutral)
        r_half = b.tick("sad", 0.7, 300)   # ~50 % through the 600 ms fade
        # blended intensity ≈ 0.5 + (0.7-0.5)*0.5 = 0.6
        assert 0.58 <= r_half.intensity <= 0.62

    def test_negative_to_positive_longer_fade(self):
        b = AffectBlender()
        r = b.tick("warm_steady_consoling", 0.6, 10)
        assert r.affect_fade_ms == 600  # no specific mapping: default

    def test_excited_to_neutral_fade_duration(self):
        b = AffectBlender()
        b.tick("neutral", 0.5, 10)     # start
        b.tick("excited", 0.8, 10)     # neutral→excited
        r = b.tick("neutral", 0.5, 10) # excited→neutral — should be 650 ms
        assert r.affect_fade_ms == 650

    # ── reset ─────────────────────────────────────────────────────────────

    def test_reset_clears_transition(self):
        b = AffectBlender()
        b.tick("happy", 0.7, 10)
        b.reset()
        assert b.current_affect == "neutral"
        assert not b.is_transitioning

    # ── same affect, no transition ────────────────────────────────────────

    def test_no_transition_when_affect_unchanged(self):
        b = AffectBlender()
        b.tick("neutral", 0.5, 500)   # settle
        r = b.tick("neutral", 0.5, 50)
        assert not b.is_transitioning
        assert r.transition_progress == 1.0

    def test_intensity_only_tweak_short_crossfade(self):
        b = AffectBlender()
        b.tick("neutral", 0.5, 600)    # fully settled
        r = b.tick("neutral", 0.7, 10) # intensity-only change
        assert r.affect_fade_ms == 200


# ═══════════════════════════════════════════════════════════════════════
# 2 — Idle micro-movements (CSS-only; validated via state defaults)
# ═══════════════════════════════════════════════════════════════════════

class TestIdleMicro:
    """CSS animations (saccades, staggered blinks, idle-sway) need no
    server-side logic.  These tests verify the state fields that the
    browser uses to modulate those animations (intensity, breath rate)."""

    def test_default_avatar_state_has_neutral_values(self):
        s = AvatarBehaviorState()
        assert s.intensity == 0.5          # baseline for micro-movement scaling
        assert s.breath_rate_hz == 0.25    # calm baseline for sway speed
        assert s.target_affect == "neutral"

    def test_intensity_carried_through_tick(self):
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "happy",
                     "valence": 0.7, "arousal": 0.4, "last_updated_ms": 1000})
        s = rt.tick(50)
        assert s.intensity >= 0.5       # happy is in mirror set; mapped intensity > neutral

    def test_breath_rate_default_is_calm(self):
        hz = _breath_rate_from_arousal(0.0)
        assert hz == 0.25               # neutral → 15 breaths/min


# ═══════════════════════════════════════════════════════════════════════
# 3 — Breathing entrainment: arousal → breath rate
# ═══════════════════════════════════════════════════════════════════════

class TestBreathEntrainment:

    def test_neutral_arousal(self):
        assert _breath_rate_from_arousal(0.0) == 0.25

    def test_high_arousal_fast_breath(self):
        hz = _breath_rate_from_arousal(1.0)
        assert hz == 0.37               # ~22.2 breaths/min

    def test_low_arousal_slow_breath(self):
        hz = _breath_rate_from_arousal(-1.0)
        assert hz == 0.13               # ~7.8 breaths/min — very slow, soothing

    def test_clamps_out_of_range_high(self):
        hz = _breath_rate_from_arousal(3.0)
        assert hz == 0.37               # clamped to 1.0

    def test_clamps_out_of_range_low(self):
        hz = _breath_rate_from_arousal(-3.0)
        assert hz == 0.13               # clamped to -1.0

    def test_breath_rate_monotonic_with_arousal(self):
        """Higher arousal must always produce a faster (or equal) breath rate."""
        rates = [_breath_rate_from_arousal(a) for a in [-0.8, -0.3, 0.0, 0.3, 0.8]]
        assert rates == sorted(rates)

    def test_breath_rate_on_tick_output(self):
        rt = _new_runtime()
        # Feed multiple high-arousal frames so EMA converges
        for _ in range(20):
            rt.consume({"type": "perception.frame", "dominant_expression": "excited",
                         "valence": 0.8, "arousal": 0.9, "last_updated_ms": 1000})
        rt.tick(50)
        s = rt.tick(50)
        # After 20 EMA steps: _ema_arousal ≈ 0.9*0.15 each + compounding ≈ 0.86
        # breath = 0.25 + 0.86*0.12 ≈ 0.353
        assert s.breath_rate_hz > 0.28  # excited → faster breath


# ═══════════════════════════════════════════════════════════════════════
# 4 — Head tilt mirroring: EMA smoothing
# ═══════════════════════════════════════════════════════════════════════

class TestHeadTiltMirroring:

    def test_ema_converges_to_steady_state(self):
        val = 0.0
        for _ in range(40):
            val = _ema(val, 10.0)
        assert 9.5 < val < 10.5

    def test_ema_alpha_high_fast_response(self):
        val = _ema(0.0, 5.0, alpha=0.8)
        assert val == 4.0               # 0 + 0.8*(5-0)

    def test_face_signals_populate_raw_yaw(self):
        u = UserAffectState()
        # Simulate what _update_face does:
        u.head_yaw = 3.5
        u._ema_head_yaw = _ema(u._ema_head_yaw, 3.5)
        assert u.head_yaw == 3.5
        assert 0.4 < u._ema_head_yaw < 0.6  # 0.15 * 3.5 ≈ 0.525

    def test_user_affect_state_ema_fields_exist(self):
        u = UserAffectState()
        assert hasattr(u, "_ema_head_yaw")
        assert hasattr(u, "_ema_head_pitch")
        assert hasattr(u, "_ema_valence")
        assert hasattr(u, "_ema_arousal")

    def test_multiple_frames_smooth_toward_truth(self):
        u = UserAffectState()
        # Feed 10 frames with yaw = 5.0; EMA should approach 5.0
        for _ in range(10):
            u._ema_head_yaw = _ema(u._ema_head_yaw, 5.0)
        assert 3.5 < u._ema_head_yaw <= 5.0  # after 10 frames, well converged

    def test_tick_outputs_smoothed_head_pose(self):
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "neutral",
                     "head_yaw": 4.2, "head_pitch": -1.8, "last_updated_ms": 1000})
        s = rt.tick(50)
        assert s.head_yaw == pytest.approx(0.63, abs=0.01)   # 4.2 * 0.15
        assert s.head_pitch == pytest.approx(-0.27, abs=0.01) # -1.8 * 0.15


# ═══════════════════════════════════════════════════════════════════════
# 5 — Gaze triangle: GazePolicy
# ═══════════════════════════════════════════════════════════════════════

class TestGazePolicyUnit:

    # ── construction ─────────────────────────────────────────────────────

    def test_default_point_is_eyes(self):
        g = GazePolicy()
        assert g.current_point == "eyes"

    def test_default_mode_is_listening(self):
        g = GazePolicy()
        assert g.cognitive_mode == "listening"

    # ── transition graph integrity ────────────────────────────────────────

    def test_all_named_points_have_transitions(self):
        for point in ("eyes", "mouth", "away", "soft_forward"):
            assert point in _GAZE_TRANSITIONS

    def test_all_transitions_sum_to_1(self):
        for point, weights in _GAZE_TRANSITIONS.items():
            total = sum(weights.values())
            assert math.isclose(total, 1.0, rel_tol=0.01), \
                f"{point} transitions sum to {total}, not 1.0"

    def test_eyes_to_mouth_possible(self):
        assert "mouth" in _GAZE_TRANSITIONS["eyes"]

    # ── dwell ranges ──────────────────────────────────────────────────────

    def test_all_modes_define_dwells_for_known_points(self):
        for mode in ("listening", "speaking", "thinking"):
            dwells = _GAZE_DWELLS[mode]
            for point in ("eyes", "mouth", "away"):
                assert point in dwells
                assert dwells[point].min_ms <= dwells[point].max_ms

    def test_listening_eyes_dwell_longest(self):
        listening_eyes = _GAZE_DWELLS["listening"]["eyes"]
        speaking_eyes  = _GAZE_DWELLS["speaking"]["eyes"]
        thinking_eyes  = _GAZE_DWELLS["thinking"]["eyes"]
        assert listening_eyes.min_ms > thinking_eyes.min_ms

    def test_thinking_away_dwell_longest(self):
        thinking_away = _GAZE_DWELLS["thinking"]["away"]
        listening_away = _GAZE_DWELLS["listening"]["away"]
        assert thinking_away.max_ms >= listening_away.max_ms

    # ── tick cycle ────────────────────────────────────────────────────────

    def test_tick_returns_current_point_before_expiry(self):
        random.seed(123)
        g = GazePolicy()
        r1 = g.tick("listening", 50)
        r2 = g.tick("listening", 50)
        # Should stay on the same point until dwell expires
        assert r1.gaze_point == r2.gaze_point

    def test_runs_through_multiple_points_over_time(self):
        random.seed(99)
        g = GazePolicy()
        g.reset(point="eyes")
        points_seen = set()
        # Run for 60 seconds — should visit all 3 main points
        for _ in range(1200):           # 1200 * 50 ms = 60 s
            r = g.tick("listening", 50)
            points_seen.add(r.gaze_point)
        assert {"eyes", "mouth", "away"}.issubset(points_seen)

    def test_gaze_point_classification_can_reach_all(self):
        """Run multiple random seeds to ensure the gaze policy visits all points."""
        all_seen = set()
        for seed in range(20):
            random.seed(seed)
            g = GazePolicy()
            for _ in range(200):
                r = g.tick("listening", 50)
                all_seen.add(r.gaze_point)
        assert {"eyes", "mouth", "away"}.issubset(all_seen)

    # ── mode switching ────────────────────────────────────────────────────

    def test_mode_change_preserves_point(self):
        random.seed(7)
        g = GazePolicy()
        g.reset(point="eyes", mode="listening")
        r = g.tick("speaking", 10)      # switch mode
        assert r.gaze_point == "eyes"   # point not lost on mode switch

    def test_speaking_mode_reported_in_result(self):
        g = GazePolicy()
        r = g.tick("speaking", 50)
        assert r.cognitive_mode == "speaking"


# ═══════════════════════════════════════════════════════════════════════
# 5b — Gaze integration through AffectRuntime.tick()
# ═══════════════════════════════════════════════════════════════════════

class TestGazeIntegration:

    def test_tick_output_includes_gaze_point(self):
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "calm",
                     "attention": 0.8, "last_updated_ms": 1000})
        s = rt.tick(50)
        assert s.gaze_point in ("eyes", "mouth", "away", "soft_forward")

    def test_thinking_mode_when_attention_low(self):
        rt = _new_runtime()
        # Feed many low-attention frames so EMA decays below 0.3
        for _ in range(30):
            rt.consume({"type": "perception.frame", "dominant_expression": "tired",
                         "attention": 0.15, "gaze_direction": "away", "last_updated_ms": 1000})
        s = rt.tick(50)
        assert s.cognitive_mode == "thinking"

    def test_listening_mode_when_attention_high(self):
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "interested",
                     "attention": 0.9, "last_updated_ms": 1000})
        s = rt.tick(50)
        assert s.cognitive_mode == "listening"


# ═══════════════════════════════════════════════════════════════════════
# 6 — Anticipatory micro-expressions
# ═══════════════════════════════════════════════════════════════════════

class TestPreSpeech:

    def test_is_speaking_true_when_hermes_tags_have_affect(self):
        rt = _new_runtime()
        rt.consume({"type": "llm.tags", "tags": {"affect": "excited"}})
        s = rt.tick(50)
        assert s.is_speaking
        assert s.pre_speech_affect == "excited"

    def test_is_speaking_false_when_no_tags(self):
        rt = _new_runtime()
        s = rt.tick(50)
        assert not s.is_speaking
        assert s.pre_speech_affect == ""

    def test_is_speaking_false_in_recovering_mode(self):
        """Pre-speech should suppress during ambient recovery (no source signal)."""
        rt = _new_runtime(mode="recovering")
        rt.consume({"type": "llm.tags", "tags": {"affect": "happy"}})
        s = rt.tick(50)
        assert not s.is_speaking

    def test_hermes_tags_persist_across_ticks(self):
        rt = _new_runtime()
        rt.consume({"type": "llm.tags", "tags": {"affect": "warm"}})
        rt.tick(50)
        s = rt.tick(50)
        assert s.pre_speech_affect == "warm"


# ═══════════════════════════════════════════════════════════════════════
# 7 — Affect classification + mapping
# ═══════════════════════════════════════════════════════════════════════

class TestAffectClassification:

    def test_counter_emotions(self):
        for expr in ("angry", "frustrated", "annoyed", "irritated", "enraged"):
            assert _classify_affect(expr) == "counter"

    def test_validate_emotions(self):
        for expr in ("sad", "disappointed", "grief", "hurt", "lonely"):
            assert _classify_affect(expr) == "validate"

    def test_mirror_emotions(self):
        for expr in ("happy", "joy", "excited", "calm", "content",
                     "interested", "curious", "amused", "surprised_positive"):
            assert _classify_affect(expr) == "mirror"

    def test_unknown_emotion_defaults_to_reflect(self):
        assert _classify_affect("puzzled") == "reflect"
        assert _classify_affect("") == "reflect"

    def test_avatar_affect_angry_is_counter_not_mirror(self):
        """Psychology: mirroring anger is harmful. The avatar must counter, not mirror."""
        affect, intensity = _avatar_affect_for("angry", mode="mirror")
        # Even in mirror mode, angry gets a counter response
        assert affect in ("grounded_concern_soft_brow", "validating_grounded")

    def test_avatar_affect_happy_is_mirrored(self):
        affect, intensity = _avatar_affect_for("happy", mode="mirror")
        assert intensity > 0.7            # mirror amplifies warmth
        assert "smile" in affect.lower()

    def test_avatar_affect_sad_is_validated(self):
        """Sadness must be validated (hold space), never countered or mirrored."""
        affect, intensity = _avatar_affect_for("sad", mode="reflect")
        assert "consoling" in affect or "concern" in affect


# ═══════════════════════════════════════════════════════════════════════
# 8 — ExpressionLatch
# ═══════════════════════════════════════════════════════════════════════

class TestExpressionLatch:

    def test_first_call_returns_desired(self):
        latch = ExpressionLatch(min_dwell_ms=250)
        assert latch.try_switch("happy", 0) == "happy"

    def test_latch_blocks_within_dwell(self):
        latch = ExpressionLatch(min_dwell_ms=250)
        latch.try_switch("happy", 0)
        assert latch.try_switch("sad", 100) == "happy"  # still locked

    def test_latch_allows_after_dwell(self):
        latch = ExpressionLatch(min_dwell_ms=250)
        latch.try_switch("happy", 0)
        assert latch.try_switch("sad", 300) == "sad"

    def test_latch_allows_same_emote(self):
        latch = ExpressionLatch(min_dwell_ms=250)
        latch.try_switch("happy", 0)
        assert latch.try_switch("happy", 100) == "happy"


# ═══════════════════════════════════════════════════════════════════════
# I — Integration: Full AffectRuntime pipeline
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_pipeline_no_events(self):
        """Tick without any prior consume should produce a valid default state."""
        rt = _new_runtime()
        s = rt.tick(50)
        assert isinstance(s, AvatarBehaviorState)
        # "neutral" user expression maps to patient_low_energy (reflect category)
        assert s.target_affect in ("neutral", "patient_low_energy", "calm_attentive")
        assert 0.0 <= s.intensity <= 1.0
        assert s.breath_rate_hz >= 0.1
        assert s.gaze_point in ("eyes", "mouth", "away", "soft_forward")

    def test_pipeline_perception_then_tick(self):
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "happy",
                     "valence": 0.7, "arousal": 0.3, "attention": 0.85,
                     "last_updated_ms": 1000})
        s = rt.tick(50)
        assert s.target_affect != "neutral"
        assert s.intensity > 0.4

    def test_pipeline_llm_then_tick(self):
        rt = _new_runtime()
        rt.consume({"type": "llm.tags", "tags": {"affect": "excited", "voice": {"intensity": 0.8}}})
        s = rt.tick(50)
        assert s.is_speaking
        assert s.pre_speech_affect == "excited"

    def test_pipeline_perception_then_llm_then_tick(self):
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "interested",
                     "valence": 0.5, "arousal": 0.2, "attention": 0.9,
                     "last_updated_ms": 500})
        rt.consume({"type": "llm.tags", "tags": {"affect": "calm"}})
        s = rt.tick(50)
        # Both perception and LLM data should coexist in the output
        assert s.is_speaking
        assert s.cognitive_mode == "thinking"  # LLM tags → thinking mode

    def test_accumulate_dt_flag_gates_timer(self):
        """When accumulate_dt=False, internal timers should not advance."""
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "happy",
                     "valence": 0.7, "arousal": 0.3, "last_updated_ms": 1000})

        # Capture blend state after first tick with accumulation
        s1 = rt.tick(50, accumulate_dt=True)

        # Tick again without accumulation — blend should not advance
        s2 = rt.tick(500, accumulate_dt=False)
        assert s2.transition_progress == pytest.approx(s1.transition_progress, abs=0.05)

    def test_stale_perception_triggers_recovering_mode(self):
        rt = _new_runtime()
        rt.user.stale_after_ms = 500    # short staleness window for test
        rt.consume({"type": "perception.frame", "dominant_expression": "happy",
                     "valence": 0.7, "arousal": 0.3, "last_updated_ms": 100})
        rt.tick(16)                     # first tick: now_ms ≈ 16
        rt.tick(16)                     # now_ms ≈ 32
        s = rt.tick(1000)               # now_ms ≈ 1048; 1048-100=948ms > 500ms stale window
        assert s.mode == "recovering"
        assert s.intensity < 0.3

    def test_consumed_emotion_itergrates_with_blender(self):
        """Verify that rapid consume→tick cycles produce stable blended output."""
        rt = _new_runtime()
        # Feed multiple emotions rapidly
        emotions = ["happy", "calm", "interested", "calm"]
        for expr in emotions:
            rt.consume({"type": "perception.frame", "dominant_expression": expr,
                         "valence": 0.5, "arousal": 0.3, "last_updated_ms": 0})
        s = rt.tick(100)
        # After rapid-fire consumes, the state should still be valid
        assert s.target_affect != ""
        assert 0.0 <= s.intensity <= 1.0

    def test_face_signals_empty_constructor(self):
        fs = FaceSignals.empty()
        assert not fs.face_detected
        assert fs.dominant_expression == "neutral"

    def test_user_affect_state_stale_detection(self):
        u = UserAffectState(last_updated_ms=500, stale_after_ms=2000)
        assert (3000 - u.last_updated_ms) > u.stale_after_ms

    def test_runtime_mode_changes_reflected_in_tick(self):
        rt = AffectRuntime(mode="mirror")
        rt.consume({"type": "perception.frame", "dominant_expression": "happy",
                     "valence": 0.8, "arousal": 0.5, "last_updated_ms": 1000})
        s = rt.tick(50)
        assert s.mode == "mirror"
        assert s.mirror_strength == 1.0

        rt2 = AffectRuntime(mode="reflect")
        rt2.consume({"type": "perception.frame", "dominant_expression": "happy",
                      "valence": 0.8, "arousal": 0.5, "last_updated_ms": 1000})
        s2 = rt2.tick(50)
        assert s2.mode == "reflect"
        assert s2.mirror_strength == 0.0

    def test_all_avatars_behavior_state_fields_are_present(self):
        """Regression: every tick output must have all AvatarBehaviorState fields filled."""
        rt = _new_runtime()
        rt.consume({"type": "perception.frame", "dominant_expression": "happy",
                     "valence": 0.6, "arousal": 0.4, "attention": 0.75,
                     "head_yaw": 2.0, "head_pitch": -1.0, "last_updated_ms": 1000})
        rt.consume({"type": "llm.tags", "tags": {"affect": "warm"}})
        s = rt.tick(50)

        required_fields = [
            "target_affect", "prev_affect", "intensity", "prev_intensity",
            "transition_progress", "affect_fade_ms", "emote_id",
            "gaze_point", "cognitive_mode", "gaze_target",
            "head_yaw", "head_pitch", "breath_rate_hz",
            "is_speaking", "pre_speech_affect", "mode", "mirror_strength",
        ]
        for f in required_fields:
            assert hasattr(s, f), f"AvatarBehaviorState missing field: {f}"
            assert getattr(s, f) is not None, f"Field {f} is None"
