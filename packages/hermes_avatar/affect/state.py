"""
Core state dataclasses for the Hermes affect runtime.

UserAffectState tracks the emotional state detected from the user's webcam.
AvatarBehaviorState carries the computed avatar behavior — consumed by the
browser rendering layer to drive CSS animations and emote selection.

All six realism dimensions are represented:
  1. Emotional inertia — prev_affect + transition fields for client-side crossfade
  2. Idle micro-movements — driven entirely client-side (CSS, no state needed)
  3. Breathing entrainment — breath_rate_hz modulated by user arousal
  4. Head tilt mirroring — smoothed head_yaw/head_pitch from MediaPipe
  5. Gaze triangle — gaze_point + cognitive_mode for client-side saccade loop
  6. Anticipatory micro-expressions — pre_speech_affect + is_speaking
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# User state (detected from the webcam / MediaPipe)
# ---------------------------------------------------------------------------

@dataclass
class UserAffectState:
    """Continuously updated emotional state of the human user.

    Populated by AffectRuntime._update_face() from FaceSignals
    produced by MediaPipeFaceTracker.  All values are EMA-smoothed
    so single-frame noise does not cause jitter.
    """

    # -- Raw (latest frame) ------------------------------------------------
    dominant_expression: str = "neutral"
    valence: float = 0.0          # pleasure ⇔ displeasure    [-1.0, +1.0]
    arousal: float = 0.0          # activation ⇔ deactivation [-1.0, +1.0]
    tension: float = 0.0          # cognitive load / stress   [ 0.0,  1.0]
    attention: float = 0.5        # engagement level          [ 0.0,  1.0]
    gaze_direction: str = "toward_user"

    # -- Head pose (raw from tracker) --------------------------------------
    head_yaw: float = 0.0         # left⇔right degrees
    head_pitch: float = 0.0       # up⇔down degrees

    # -- Smoothing state (maintained by EMA in _update_face) ----------------
    _ema_valence: float = field(default=0.0, repr=False)
    _ema_arousal: float = field(default=0.0, repr=False)
    _ema_tension: float = field(default=0.0, repr=False)
    _ema_attention: float = field(default=0.5, repr=False)
    _ema_head_yaw: float = field(default=0.0, repr=False)
    _ema_head_pitch: float = field(default=0.0, repr=False)

    last_updated_ms: int = 0
    stale_after_ms: int = 2000    # after this many ms, treat perception as stale

    def to_dict(self) -> dict:
        return {
            "dominant_expression": self.dominant_expression,
            "valence": self.valence,
            "arousal": self.arousal,
            "tension": self.tension,
            "attention": self.attention,
            "gaze_direction": self.gaze_direction,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "last_updated_ms": self.last_updated_ms,
        }


# ---------------------------------------------------------------------------
# Avatar behavior state (sent to the browser every status poll)
# ---------------------------------------------------------------------------

@dataclass
class AvatarBehaviorState:
    """Computed avatar behaviour, emitted by AffectRuntime.tick().

    The browser reads these fields every ~50 ms status poll and maps
    them to CSS classes, CSS custom properties, and expression assets.

    Separation of concerns:
      * Fields here describe WHAT the avatar should express.
      * The browser (demo.js + demo.css) decides HOW to render it.
    """

    # ── Emotional state ──────────────────────────────────────────────────
    target_affect: str = "neutral"       # primary emotion label
    prev_affect: str = "neutral"         # previous label (for crossfade)
    intensity: float = 0.5               # 0.0 → 1.0  expression strength
    prev_intensity: float = 0.5          # previous intensity (for crossfade)

    # -- Transition control (1: emotional inertia) -------------------------
    transition_progress: float = 1.0     # 0.0 = old, 1.0 = fully settled
    affect_fade_ms: int = 600            # client should crossfade over this many ms
    emote_id: str = "neutral"            # resolved emote asset identifier

    # ── Gaze & attention (5: gaze triangle) ──────────────────────────────
    gaze_point: str = "soft_forward"     # "eyes" | "mouth" | "away" | "soft_forward"
    cognitive_mode: str = "listening"    # "listening" | "speaking" | "thinking"
    gaze_target: str = "soft_forward"    # high-level target (legacy compat)

    # ── Head mirroring (4: head tilt) ────────────────────────────────────
    head_yaw: float = 0.0               # server-EMA-smoothed, client CSS-mapped
    head_pitch: float = 0.0

    # ── Breathing (3: entrainment) ───────────────────────────────────────
    breath_rate_hz: float = 0.25         # ~15 breaths/min = calm baseline
    #                                    # mapped client-side to CSS animation-duration

    # ── Speaking state (6: anticipatory micro-expressions) ───────────────
    is_speaking: bool = False
    pre_speech_affect: str = ""          # non-empty ⇒ avatar shows anticipatory micro-expression

    # ── Mode / mirror control ────────────────────────────────────────────
    mode: str = "reflect"                # "reflect" | "mirror" | "recovering"
    mirror_strength: float = 0.0         # 0.0 = regulated, 1.0 = full mirror

    def to_dict(self) -> dict:
        return {
            "target_affect": self.target_affect,
            "prev_affect": self.prev_affect,
            "intensity": self.intensity,
            "prev_intensity": self.prev_intensity,
            "transition_progress": self.transition_progress,
            "affect_fade_ms": self.affect_fade_ms,
            "emote_id": self.emote_id,
            "gaze_point": self.gaze_point,
            "cognitive_mode": self.cognitive_mode,
            "gaze_target": self.gaze_target,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "breath_rate_hz": self.breath_rate_hz,
            "is_speaking": self.is_speaking,
            "pre_speech_affect": self.pre_speech_affect,
            "mode": self.mode,
            "mirror_strength": self.mirror_strength,
        }


# ---------------------------------------------------------------------------
# Perception event (feeds into runtime.consume)
# ---------------------------------------------------------------------------

@dataclass
class FaceSignals:
    """Raw or EMA-smoothed signals from the face tracker (MediaPipe).

    This is the contract between the perception layer and the affect runtime.
    """

    face_detected: bool = False
    dominant_expression: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    attention: float = 0.5
    gaze_direction: str = "toward_user"
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    last_updated_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "face_detected": self.face_detected,
            "dominant_expression": self.dominant_expression,
            "valence": self.valence,
            "arousal": self.arousal,
            "tension": self.tension,
            "attention": self.attention,
            "gaze_direction": self.gaze_direction,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "last_updated_ms": self.last_updated_ms,
        }

    @classmethod
    def empty(cls) -> "FaceSignals":
        """Return a neutral / no-face result (NullFaceTracker fallback)."""
        return cls()


# ---------------------------------------------------------------------------
# Lightweight object pool for AvatarBehaviorState (perf: avoids alloc churn)
# ---------------------------------------------------------------------------

_POOL: list[AvatarBehaviorState] = []
_MAX_POOL_SIZE: int = 8


def acquire_behavior_state() -> AvatarBehaviorState:
    """Return a recycled or new AvatarBehaviorState instance.

    Callers MUST call :func:`release_behavior_state` when done so the
    instance can be reused by the next tick.
    """
    if _POOL:
        return _POOL.pop()
    return AvatarBehaviorState()


def release_behavior_state(state: AvatarBehaviorState) -> None:
    """Return *state* to the pool for reuse."""
    if len(_POOL) < _MAX_POOL_SIZE:
        _POOL.append(state)


def reset_behavior_state_pool() -> None:
    """Clear the pool (useful between test runs)."""
    _POOL.clear()
