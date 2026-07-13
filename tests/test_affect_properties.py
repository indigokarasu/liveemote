from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from hermes_avatar.affect.smoothing import clamp, ema
from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.state import AvatarBehaviorState

# Bounded strategies keep the suite fast.
FLOATS_01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
                        allow_infinity=False)
ALPHAS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
                      allow_infinity=False, exclude_min=False)


@given(FLOATS_01, FLOATS_01, ALPHAS)
@settings(max_examples=200, deadline=None)
def test_ema_stays_within_unit_interval(prev, cur, alpha):
    out = ema(prev, cur, alpha)
    assert 0.0 <= out <= 1.0


@given(FLOATS_01, FLOATS_01, ALPHAS)
@settings(max_examples=200, deadline=None)
def test_ema_none_previous_returns_current(cur, alpha, _ignored):
    # When no previous value exists, ema returns the current input unchanged.
    assert ema(None, cur, alpha) == cur


@given(FLOATS_01, FLOATS_01, ALPHAS, ALPHAS)
@settings(max_examples=200, deadline=None)
def test_ema_alpha_monotonic_smoothing_strength(prev, cur, alpha_low, alpha_high):
    # Larger alpha => output closer to current (less smoothing).
    # When current >= previous this is monotonically increasing in alpha.
    if not (prev <= cur):
        prev, cur = cur, prev  # make current the larger of the two
    a_lo = min(alpha_low, alpha_high)
    a_hi = max(alpha_low, alpha_high)
    if a_lo == a_hi:
        return
    lo = ema(prev, cur, a_lo)
    hi = ema(prev, cur, a_hi)
    assert lo <= hi <= cur + 1e-9


@given(FLOATS_01, FLOATS_01)
@settings(max_examples=200, deadline=None)
def test_clamp_bounds(prev, cur):
    assert clamp(cur, 0.0, 1.0) == max(0.0, min(1.0, cur))


@given(FLOATS_01, FLOATS_01, ALPHAS)
@settings(max_examples=200, deadline=None)
def test_ema_converges_toward_current_with_full_alpha(prev, cur, alpha):
    # At alpha == 1.0, ema snaps exactly to the current value.
    out = ema(prev, cur, 1.0)
    assert abs(out - cur) < 1e-9


@given(
    st.lists(
        st.one_of(
            st.fixed_dictionaries(
                {"type": st.just("perception.frame"),
                 "face_detected": st.booleans(),
                 "head_yaw": st.floats(-15.0, 15.0, allow_nan=False),
                 "head_pitch": st.floats(-10.0, 10.0, allow_nan=False),
                 "gaze_confidence": FLOATS_01,
                 "expression": st.fixed_dictionaries({
                     "smile": FLOATS_01, "frown": FLOATS_01,
                     "brow_raise": FLOATS_01, "eye_open": FLOATS_01}),
                 "timestamp_ms": st.integers(0, 10_000_000)}),
            st.fixed_dictionaries({
                "type": st.just("audio.vad"),
                "speaking": st.booleans(),
                "energy": FLOATS_01, "speech_rate": FLOATS_01,
                "timestamp_ms": st.integers(0, 10_000_000)}),
        ),
        max_size=40,
    )
)
@settings(max_examples=40, deadline=None)
def test_runtime_ticks_stay_within_declared_bounds(events):
    rt = AffectRuntime()
    for ev in events:
        rt.consume(ev)
        avatar = rt.tick()
        # intensity & mirror_strength must remain within sane [0,1] range
        assert 0.0 <= avatar.intensity <= 1.0
        assert 0.0 <= avatar.mirror_strength <= 1.0
        assert avatar.mode in {"idle", "listening", "thinking",
                              "speaking", "recovering"}
        assert isinstance(avatar, AvatarBehaviorState)
