"""
AffectRuntime — the central emotional-regulation engine.

Data flow per tick::

    consume(perception.frame)  →  UserAffectState (EMA-smoothed)
    consume(llm.tags)          →  hermes_tags cache
    tick(dt_ms)                →  AffectBlender + GazePolicy + affect mapping
                               →  AvatarBehaviorState  (→ browser)

All six realism improvements are wired in:
  1. Emotional inertia (AffectBlender) — crossfade metadata
  2. Idle micro-movements — client-side CSS (no server footprint)
  3. Breathing entrainment — breath_rate_hz computed from arousal
  4. Head tilt mirroring — head_yaw/head_pitch EMA-smoothed
  5. Gaze triangle — GazePolicy advances gaze_point each tick
  6. Anticipatory micro-expressions — pre_speech_affect when speaking begins
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .state import AvatarBehaviorState, FaceSignals, UserAffectState
from .affect_blender import AffectBlender, BlendResult
from .gaze_policy import GazePolicy, GazeResult


# ---------------------------------------------------------------------------
# Expression latch: minimum dwell before changing expressions
# ---------------------------------------------------------------------------

@dataclass
class ExpressionLatch:
    """Prevents rapid emote-switching by enforcing minimum dwell times."""

    current_emote: str = "neutral"
    min_dwell_ms: int = 250
    _locked_until_ms: float = 0.0

    def try_switch(self, desired: str, now_ms: float) -> str:
        """Return the actual emote to display.

        If the latch is still locked, returns the current emote.
        Otherwise switches to *desired* and re-locks.
        """
        if now_ms < self._locked_until_ms:
            return self.current_emote
        if desired != self.current_emote:
            self.current_emote = desired
        self._locked_until_ms = now_ms + self.min_dwell_ms
        return self.current_emote


# ---------------------------------------------------------------------------
# Affect mapping: user emotion → avatar response category
# ---------------------------------------------------------------------------

# The psychology-driven response taxonomy:
#   mirror   — share the affect (joy, calm, interest)
#   counter  — regulate downward (anger, frustration, anxiety)
#   validate — hold space (sadness, grief, disappointment)
#   reflect  — acknowledge without absorbing (low attention, gaze away)

_COUNTER_EMOTIONS = frozenset({
    "angry", "frustrated", "annoyed", "irritated", "enraged",
})

_VALIDATE_EMOTIONS = frozenset({
    "sad", "disappointed", "grief", "hurt", "lonely",
})

_MIRROR_EMOTIONS = frozenset({
    "happy", "joy", "excited", "surprised_positive", "amused",
    "calm", "content", "interested", "curious",
})


def _classify_affect(expression: str) -> str:
    """Return the response category for a user expression."""
    if expression in _COUNTER_EMOTIONS:
        return "counter"
    if expression in _VALIDATE_EMOTIONS:
        return "validate"
    if expression in _MIRROR_EMOTIONS:
        return "mirror"
    return "reflect"


def _avatar_affect_for(
    user_expression: str,
    mode: str = "reflect",
):
    """Map a user expression to an avatar affect label + base intensity.

    The *mode* ("mirror" / "reflect") tunes intensity & warmth but never
    changes the response *category* — that comes from the user's state.
    """
    category = _classify_affect(user_expression)
    is_mirror = mode == "mirror"

    if category == "mirror":
        if user_expression in ("excited", "joy"):
            return ("warm_acknowledging", 0.85 if is_mirror else 0.7)
        if user_expression == "happy":
            return ("small_delayed_smile", 0.75 if is_mirror else 0.65)
        if user_expression == "surprised_positive":
            return ("warm_acknowledging", 0.75)
        if user_expression == "amused":
            return ("small_delayed_smile", 0.7 if is_mirror else 0.6)
        # calm, content, interested, curious
        return ("calm_attentive", 0.6 if is_mirror else 0.5)

    if category == "counter":
        if is_mirror:
            return ("grounded_concern_soft_brow", 0.35)
        return ("validating_grounded", 0.45)

    if category == "validate":
        if is_mirror:
            return ("soft_concern", 0.5)
        return ("warm_steady_consoling", 0.55)

    return ("patient_low_energy", 0.35 if is_mirror else 0.4)


# ---------------------------------------------------------------------------
# Breathing entrainment: arousal → breath rate
# ---------------------------------------------------------------------------

def _breath_rate_from_arousal(arousal: float) -> float:
    """Map arousal [-1, +1] to breath rate in Hz.

    Calm (low arousal):  ~0.20 Hz = 12 breaths/min  (slow, soothing)
    Neutral:             ~0.25 Hz = 15 breaths/min  (natural)
    Excited (high arousal): ~0.40 Hz = 24 breaths/min (rapid)
    """
    clamped = max(-1.0, min(1.0, arousal))
    return 0.25 + clamped * 0.12


# ---------------------------------------------------------------------------
# Head tilt mirroring: EMA config
# ---------------------------------------------------------------------------

_HEAD_EMA_ALPHA = 0.15   # low = heavy smoothing (appropriate for head pose)


def _ema(old: float, new: float, alpha: float = _HEAD_EMA_ALPHA) -> float:
    """Exponential moving average."""
    return old + alpha * (new - old)


# ---------------------------------------------------------------------------
# AffectRuntime
# ---------------------------------------------------------------------------

class AffectRuntime:
    """Central emotional-regulation engine.

    Instantiated once per avatar session.  Receives perception frames
    and LLM tags via ``consume()``, produces ``AvatarBehaviorState``
    via ``tick()`` every ~50 ms.
    """

    def __init__(
        self,
        state: Optional[UserAffectState] = None,
        mode: str = "reflect",
    ) -> None:
        self.user = state or UserAffectState()
        self.blender = AffectBlender(default_fade_ms=600)
        self.gaze = GazePolicy()
        self.latch = ExpressionLatch(min_dwell_ms=250)

        self.mode: str = mode

        # LLM-supplied emotion tags (set by orchestrator before speak_test)
        self.hermes_tags: dict[str, Any] = {}

        self._frame_count: int = 0
        self._last_tick_ms: float = 0.0
        self._dt_accumulator_ms: float = 0.0

    # -- event ingestion ---------------------------------------------------

    def consume(self, event: dict[str, Any]) -> None:
        """Ingest an event: perception.frame or llm.tags."""
        etype = event.get("type", "")

        if etype == "perception.frame":
            self._update_face(
                expression=str(event.get("dominant_expression", "neutral")),
                valence=float(event.get("valence", 0.0)),
                arousal=float(event.get("arousal", 0.0)),
                tension=float(event.get("tension", 0.0)),
                attention=float(event.get("attention", 0.5)),
                gaze_direction=str(event.get("gaze_direction", "toward_user")),
                head_yaw=float(event.get("head_yaw", 0.0)),
                head_pitch=float(event.get("head_pitch", 0.0)),
                ts_ms=int(event.get("last_updated_ms", 0)),
            )
        elif etype == "llm.tags":
            self.hermes_tags = dict(event.get("tags", {}))

    # ------------------------------------------------------------------
    # Tick — the main loop entry point
    # ------------------------------------------------------------------

    def tick(self, dt_ms: float = 0.0, accumulate_dt: bool = True) -> AvatarBehaviorState:
        """Advance the simulation by *dt_ms* and return current behavior.

        Called at ~60 Hz by the status poll handler.  If *accumulate_dt*
        is False (e.g. during a passive status() call), the internal
        timers are NOT advanced, preventing heartbeat drift when no
        real events arrive.
        """
        if accumulate_dt:
            self._dt_accumulator_ms += dt_ms

        now_ms = self._last_tick_ms + self._dt_accumulator_ms
        self._last_tick_ms = now_ms

        # --- 1. Emotional inertia: run the blender ------------------------
        user_expr = self.user.dominant_expression

        is_stale = (
            self._frame_count > 0
            and (now_ms - self.user.last_updated_ms) > self.user.stale_after_ms
        )

        target_affect: str
        target_intensity: float
        cognitive_mode: str

        if is_stale:
            target_affect = "neutral"
            target_intensity = 0.10
            cognitive_mode = "thinking"
        else:
            target_affect, target_intensity = _avatar_affect_for(user_expr, self.mode)
            cognitive_mode = self._infer_cognitive_mode()

        blend: BlendResult = self.blender.tick(
            target_affect, target_intensity,
            dt_ms=self._dt_accumulator_ms,
        )

        # --- 2. Gaze triangle: advance gaze policy ------------------------
        gaze: GazeResult = self.gaze.tick(
            cognitive_mode, dt_ms=self._dt_accumulator_ms,
        )

        # --- 3. Expression latch: enforce minimum dwell -------------------
        emote = self.latch.try_switch(blend.emote_id, now_ms)

        # --- 4. Breathing entrainment: arousal → breath rate --------------
        breath_hz = _breath_rate_from_arousal(self.user._ema_arousal)

        # --- 5. Head tilt mirroring (EMA-smoothed) ------------------------
        head_yaw = self.user._ema_head_yaw
        head_pitch = self.user._ema_head_pitch

        # --- 6. Anticipatory micro-expressions: pre-speech affect ---------
        pre_speech = self.hermes_tags.get("affect", "")
        is_speaking = bool(pre_speech) and self.mode != "recovering"

        # --- Assemble the behaviour state ---------------------------------
        state = AvatarBehaviorState(
            target_affect=blend.target_affect,
            prev_affect=blend.prev_affect,
            intensity=blend.intensity,
            prev_intensity=blend.prev_intensity,
            transition_progress=blend.transition_progress,
            affect_fade_ms=blend.affect_fade_ms,
            emote_id=emote,
            gaze_point=gaze.gaze_point,
            cognitive_mode=cognitive_mode,
            gaze_target="soft_forward" if is_stale else self.user.gaze_direction,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            breath_rate_hz=breath_hz,
            is_speaking=is_speaking,
            pre_speech_affect=pre_speech,
            mode=self.mode if not is_stale else "recovering",
            mirror_strength=1.0 if self.mode == "mirror" else 0.0,
        )

        self._dt_accumulator_ms = 0.0
        return state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_face(
        self,
        expression: str,
        valence: float,
        arousal: float,
        tension: float,
        attention: float,
        gaze_direction: str,
        head_yaw: float,
        head_pitch: float,
        ts_ms: int,
    ) -> None:
        """Apply EMA smoothing to a new perception frame."""
        self._frame_count += 1
        self.user.dominant_expression = expression
        self.user.valence = valence
        self.user.arousal = arousal
        self.user.tension = tension
        self.user.attention = attention
        self.user.gaze_direction = gaze_direction
        self.user.head_yaw = head_yaw
        self.user.head_pitch = head_pitch
        self.user.last_updated_ms = ts_ms

        self.user._ema_valence = _ema(self.user._ema_valence, valence)
        self.user._ema_arousal = _ema(self.user._ema_arousal, arousal)
        self.user._ema_tension = _ema(self.user._ema_tension, tension)
        self.user._ema_attention = _ema(self.user._ema_attention, attention)
        self.user._ema_head_yaw = _ema(self.user._ema_head_yaw, head_yaw)
        self.user._ema_head_pitch = _ema(self.user._ema_head_pitch, head_pitch)

    def _infer_cognitive_mode(self) -> str:
        """Infer cognitive mode from user state + LLM tags."""
        if self.hermes_tags:
            return "thinking"
        if self.user._ema_attention < 0.3 and self.user.gaze_direction == "away":
            return "thinking"
        return "listening"
