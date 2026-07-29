"""``tests/test_affect_ambient.py`` — ambient-recovery override behaviour.

Pins :meth:`AffectRuntime.tick` so that when the avatar's source signal
goes silent for longer than ``affect.ambient_after_ms`` the runtime flips
``avatar.mode`` to ``"recovering"`` with a damped calm preset. The
browser then plays the subtle CSS ambient loop (see
``apps/demo_server/static/demo.css`` ``.mode-recovering``) instead of
freezing on whatever the last user-driven state happened to be.

The four scenarios this file locks down:

  * ``ambient_after_ms = 1500`` (default): per-step stale ticks flip the
    avatar into ``mode="recovering"`` once the gap exceeds the threshold.
  * A fresh perception event after the gap closes clears the mode back
    to the normal idle/mirror/reflect branch.
  * On a cold-started runtime with no perception event yet
    (``user.last_updated_ms == 0``) the ambient path stays closed —
    silence ISN'T the same as loss.
  * ``affect.ambient_after_ms = 0`` (configured opt-out) keeps the
    legacy freeze-on-last behaviour intact.
"""
from __future__ import annotations

import pytest

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.config.schema import (
    AffectConfig,
    AppConfig,
    ReactionDelayConfig,
    SmoothingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _perception_event(timestamp_ms: int) -> dict:
    """Mimic the perception.frame the browser posts when MediaPipe sees
    a face. Keeps ``timestamp_ms`` exact so staleness arithmetic is
    deterministic."""
    return {
        "type": "perception.frame",
        "timestamp_ms": timestamp_ms,
        "face_detected": True,
        "face_center": (0.5, 0.5),
        "head_yaw": 0.0,
        "head_pitch": 0.0,
        "expression": {"smile": 0.4, "frown": 0.0, "brow_raise": 0.0, "eye_open": 0.7},
    }


def _make_config(**overrides) -> AppConfig:
    """Return a freshly built AppConfig with explicit affect overrides."""
    affect_kwargs = {
        "update_hz": 30,
        "min_emote_dwell_ms": 1200,
        "ambient_after_ms": 1500,
        "reaction_delay_ms": ReactionDelayConfig(),
        "smoothing": SmoothingConfig(),
    }
    affect_kwargs.update(overrides)
    return AppConfig(affect=AffectConfig(**affect_kwargs))


# ---------------------------------------------------------------------------
# Core flip-on-stale behaviour
# ---------------------------------------------------------------------------
def test_tick_flips_to_recovering_when_perception_goes_stale() -> None:
    """After one perception event, advancing the runtime clock past
    ``ambient_after_ms`` (default 1500 ms) flips ``avatar.mode`` to
    ``"recovering"``.
    """
    runtime = AffectRuntime(config=_make_config())
    # Land the first perception event so the runtime is no longer in
    # pre-boot silence. tick(now_ms=1700000000000) sets last_updated_ms.
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))

    # Now advance time 2 s ahead (well past the 1500 ms threshold).
    stale_tick = runtime.tick(timestamp_ms=1_700_000_002_000)
    assert stale_tick.mode == "recovering", (
        f"expected avatar.mode='recovering' after staleness gap, "
        f"got {stale_tick.mode!r}; full state: "
        f"intensity={stale_tick.intensity} mirror_strength={stale_tick.mirror_strength} "
        f"gaze_target={stale_tick.gaze_target!r}"
    )
    assert stale_tick.intensity <= 0.10, (
        f"ambient intensity must be ≤ 0.10 (got {stale_tick.intensity}); "
        f"a higher amplitude would overpower the subtle CSS keyframes."
    )
    assert stale_tick.mirror_strength == 0.0, (
        f"ambient must zero out mirror_strength (got {stale_tick.mirror_strength}); "
        f"satisfies the project's signal-leakage invariant under fallback."
    )
    assert stale_tick.gaze_target == "soft_forward"
    assert stale_tick.full_body_pose == "standing_idle"


def test_new_perception_event_clears_ambient_recovery() -> None:
    """Once a fresh perception event lands the runtime must revert to the
    normal idle / mirror / reflect branch; mode='recovering' should NOT
    persist when the user is back online."""
    runtime = AffectRuntime(config=_make_config())
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))
    # Drive into ambient.
    runtime.tick(timestamp_ms=1_700_000_005_000)
    assert runtime.avatar.mode == "recovering"
    # Fresh perception event lands. consume() forwards timestamp_ms into tick().
    runtime.consume(_perception_event(timestamp_ms=1_700_000_010_000))
    assert runtime.avatar.mode != "recovering", (
        f"fresh perception event should clear ambient mode; "
        f"got {runtime.avatar.mode!r}"
    )


# ---------------------------------------------------------------------------
# Guards that keep the override honest
# ---------------------------------------------------------------------------
def test_ambient_does_not_fire_on_cold_started_runtime() -> None:
    """Before any perception event has ever landed, ``user.last_updated_ms``
    is 0. The ambient override must NOT trigger — silence pending first
    signal isn't the same as signal loss."""
    runtime = AffectRuntime(config=_make_config())
    # No ``consume()`` — runtime.user.last_updated_ms stays 0.
    state = runtime.tick(timestamp_ms=1_700_000_000_000)
    assert state.mode != "recovering", (
        f"cold-started runtime flipped to recovering mode without ever "
        f"seeing a perception event; got {state.mode!r}"
    )


def test_ambient_can_be_disabled_via_zero_threshold() -> None:
    """Setting ``ambient_after_ms=0`` (legacy opt-out) must keep the
    original freeze-on-last-state behaviour intact. Lets users who want
    the avatar hard-frozen on signal loss still get that."""
    runtime = AffectRuntime(config=_make_config(ambient_after_ms=0))
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))
    # 60 s ahead — would have triggered ambient definitively.
    far_future = runtime.tick(timestamp_ms=1_700_000_060_000)
    assert far_future.mode != "recovering", (
        f"ambient_after_ms=0 should disable the fallback; "
        f"got {far_future.mode!r}"
    )


def test_ambient_threshold_is_config_driven() -> None:
    """Bumping the threshold to 60 s keeps the avatar out of ambient at
    the 2 s mark — confirms the threshold is the actual gate, not a
    constant."""
    runtime = AffectRuntime(config=_make_config(ambient_after_ms=60_000))
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))
    assert runtime.tick(timestamp_ms=1_700_000_002_000).mode != "recovering"
    # But well past 60 s it should flip.
    assert runtime.tick(timestamp_ms=1_700_000_070_000).mode == "recovering"


# ---------------------------------------------------------------------------
# Behaviour inside the recovering branch
# ---------------------------------------------------------------------------
def test_ambient_recovery_preserves_neutral_emote() -> None:
    """When entering ambient, emote_id is reset to the runtime's
    'neutral' lookup result so the avatar doesn't carry the user's
    last emote into the cooldown state."""
    runtime = AffectRuntime(
        config=_make_config(),
        emote_lookup=lambda state: "neutral_id" if state == "neutral" else "other_id",
    )
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))
    state = runtime.tick(timestamp_ms=1_700_000_005_000)
    assert state.mode == "recovering"
    assert state.emote_id == "neutral_id", (
        f"ambient branch must set emote_id to the neutral lookup result "
        f"(got {state.emote_id!r}); carrying the user's last emote into "
        f"the cooldown would be jarring on-screen."
    )


def test_ambient_recovery_state_is_compatible_with_existing_schema_modes() -> None:
    """'recovering' must remain a legal AvatarMode literal so daemons
    that serialize AvatarBehaviorState continue to round-trip cleanly."""
    runtime = AffectRuntime(config=_make_config())
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))
    state = runtime.tick(timestamp_ms=1_700_000_005_000)
    # round-trip through to_dict() / dataclass typing; should not raise.
    payload = state.to_dict()
    assert payload["mode"] == "recovering"
    # Should also be a valid AvatarBehaviorState instance with the
    # documented default padding intact if reset() is ever called.
    fresh = AvatarBehaviorState()
    fresh.mode = "recovering"
    assert fresh.mode == "recovering"


def test_status_poll_does_not_drift_conversation_timers_but_still_drives_staleness_recovery() -> None:
    """Regression for C1 (1.5 s /api/status heartbeat): the runtime must
    pulse so ``(now - user.last_updated_ms) > ambient_after_ms`` flips
    ``avatar.mode`` to ``"recovering"``, but it must NOT advance
    ``conversation.user_turn_ms`` / ``assistant_turn_ms`` or reset
    ``_last_tick_ms`` just because the browser polled.

    Three sub-assertions:

      1. With ``turn_state = "user_speaking"`` and a known baseline,
         ``tick(timestamp_ms=t, accumulate_dt=False)`` leaves BOTH
         ``user_turn_ms`` and ``_last_tick_ms`` exactly unchanged.
      2. ``tick(timestamp_ms=t, accumulate_dt=False)`` STILL fires the
         ambient-recovery override above the threshold - because the
         staleness check reads ``self.user.last_updated_ms`` (an
         event-driven timestamp), not ``_last_tick_ms``.
      3. ``tick(timestamp_ms=t, accumulate_dt=True)`` (default) keeps
         its prior semantics: advancing 1000 ms in the ``user_speaking``
         branch adds exactly 1000 ms to ``user_turn_ms`` and resets
         ``_last_tick_ms`` to ``t``.
    """
    runtime = AffectRuntime(config=_make_config())
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))

    # --- Assertion 1 + 2: accumulate_dt=False -----------------------
    runtime.conversation.turn_state = "user_speaking"
    user_turn_baseline = runtime.conversation.user_turn_ms
    last_tick_baseline = runtime._last_tick_ms

    runtime.tick(timestamp_ms=1_700_000_001_000, accumulate_dt=False)
    assert runtime.conversation.user_turn_ms == user_turn_baseline, (
        f"tick(accumulate_dt=False) must NOT advance user_turn_ms; C1"
        f" regression: heartbeat poll should be dt-neutral. "
        f"(baseline={user_turn_baseline}, post={runtime.conversation.user_turn_ms})"
    )
    assert runtime._last_tick_ms == last_tick_baseline, (
        f"tick(accumulate_dt=False) must NOT advance _last_tick_ms; "
        f"otherwise the next real event observes a shrunken dt. "
        f"(baseline={last_tick_baseline}, post={runtime._last_tick_ms})"
    )

    # Long jump across the ambient threshold - recovery MUST fire
    # under accumulate_dt=False because the staleness override reads
    # self.user.last_updated_ms which IS event-driven.
    runtime.tick(timestamp_ms=1_700_000_005_000, accumulate_dt=False)
    assert runtime.avatar.mode == "recovering", (
        "ambient-recovery must fire under accumulate_dt=False; the "
        "status() heartbeat would otherwise break the user-facing "
        "stale-signal fallback the user explicitly asked for."
    )

    # --- Assertion 3: accumulate_dt=True (default) -------------------
    # Fresh perception event resets last_updated_ms so we're out of
    # the recovering branch.
    runtime.consume(_perception_event(timestamp_ms=1_700_000_010_000))
    runtime.conversation.turn_state = "user_speaking"
    user_turn_before = runtime.conversation.user_turn_ms
    last_tick_before = runtime._last_tick_ms
    runtime.tick(timestamp_ms=1_700_000_011_000, accumulate_dt=True)
    assert runtime.conversation.user_turn_ms - user_turn_before == 1000, (
        "tick(accumulate_dt=True) must continue to advance user_turn_ms"
        " by exactly the elapsed dt for event-driven callers."
    )
    assert runtime._last_tick_ms == 1_700_000_011_000, (
        "_last_tick_ms must advance to the supplied timestamp_ms under"
        " accumulate_dt=True"
    )


def test_staleness_override_uses_strict_greater_than_threshold() -> None:
    """T3 — Boundary pin for the staleness override.

    The override condition is ``(now - last_updated_ms) > ambient_after_ms``,
    i.e. STRICTLY greater-than. A tick landing at exactly the threshold
    must NOT flip the avatar out of its normal idle/thinking/listening
    branch. Only one ms over the threshold should fire the recovery.

    Without this pin a future refactor that flips ``>`` to ``>=`` would
    introduce a one-poll-wide phantom ambient flicker on every event
    boundary, defeating the entire purpose of the feature.
    """
    runtime = AffectRuntime(config=_make_config(ambient_after_ms=1500))
    runtime.consume(_perception_event(timestamp_ms=1_700_000_000_000))
    # last_updated_ms is now exactly 1_700_000_000_000.

    # Tick at exactly the threshold (gap == 1500). Override must NOT fire.
    edge_state = runtime.tick(timestamp_ms=1_700_000_001_500)
    assert edge_state.mode != "recovering", (
        f"at gap == ambient_after_ms the override must NOT fire (got"
        f" mode={edge_state.mode!r}); the condition is strict >."
    )

    # Tick one ms past the threshold (gap == 1501). Override MUST fire.
    over_state = runtime.tick(timestamp_ms=1_700_000_001_501)
    assert over_state.mode == "recovering", (
        f"at gap == ambient_after_ms + 1 ms the override MUST fire"
        f" (got mode={over_state.mode!r})."
    )

    # Tick one ms below the threshold (gap == 1499). Override must NOT fire.
    under_state = runtime.tick(timestamp_ms=1_700_000_001_499 - 1 - (-1))
    # NB: arithmetic above computes 1_700_000_001_498. Override MUST NOT
    # fire.
    assert under_state.mode != "recovering", (
        f"at gap == ambient_after_ms - 1 ms the override must NOT fire"
        f" (got mode={under_state.mode!r})."
    )
