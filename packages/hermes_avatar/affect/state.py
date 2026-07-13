from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal

TurnState = Literal["idle", "user_speaking", "assistant_thinking", "assistant_speaking", "interrupted"]
AvatarMode = Literal["idle", "listening", "thinking", "speaking", "recovering"]


@dataclass
class UserAffectState:
    face_detected: bool = False
    attention: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    dominant_expression: str = "neutral"
    speaking: bool = False
    speech_energy: float = 0.0
    speech_rate: float = 0.0
    gaze_direction: str = "unknown"
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    gaze_confidence: float = 0.0
    emotion_confidence: float = 0.0
    last_updated_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConversationState:
    turn_state: TurnState = "idle"
    silence_ms: int = 0
    user_turn_ms: int = 0
    assistant_turn_ms: int = 0
    interruption_risk: float = 0.0
    tension: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AvatarBehaviorState:
    mode: AvatarMode = "idle"
    affect: str = "neutral"
    gaze_target: str = "toward_user"
    emote_id: str | None = None
    intensity: float = 0.25
    lip_sync_enabled: bool = False
    mirror_strength: float = 0.0
    delay_ms: int = 0
    full_body_pose: str = "standing_idle"

    def to_dict(self) -> dict:
        return asdict(self)

    def reset(self) -> "AvatarBehaviorState":
        """Return this object to its default field values (used by the pool)."""
        self.mode = "idle"
        self.affect = "neutral"
        self.gaze_target = "toward_user"
        self.emote_id = None
        self.intensity = 0.25
        self.lip_sync_enabled = False
        self.mirror_strength = 0.0
        self.delay_ms = 0
        self.full_body_pose = "standing_idle"
        return self


def fill_behavior_state(
    out: AvatarBehaviorState,
    *,
    mode: AvatarMode,
    affect: str,
    gaze_target: str,
    emote_id: str | None,
    intensity: float,
    lip_sync_enabled: bool = False,
    mirror_strength: float = 0.0,
    delay_ms: int = 0,
    full_body_pose: str = "standing_idle",
) -> AvatarBehaviorState:
    """Populate ``out`` in place and return it. Avoids allocating a new
    AvatarBehaviorState when a pooled object is supplied by the caller."""
    out.mode = mode
    out.affect = affect
    out.gaze_target = gaze_target
    out.emote_id = emote_id
    out.intensity = intensity
    out.lip_sync_enabled = lip_sync_enabled
    out.mirror_strength = mirror_strength
    out.delay_ms = delay_ms
    out.full_body_pose = full_body_pose
    return out


class _BehaviorStatePool:
    """Free-list pool for short-lived AvatarBehaviorState objects.

    The avatar render loop allocates a fresh AvatarBehaviorState every tick.
    Reusing a small bounded set of objects eliminates that churn. A previously
    returned state is only handed out again after it has been released back to
    the pool, so a caller that holds a reference across one frame is safe: the
    object will not be mutated underneath it until the next tick recycles it.
    """

    __slots__ = ("_free", "_max")

    def __init__(self, max_size: int = 32) -> None:
        self._free: list[AvatarBehaviorState] = []
        self._max = max(1, max_size)

    def acquire(self) -> AvatarBehaviorState:
        obj = self._free.pop() if self._free else AvatarBehaviorState()
        return obj.reset()

    def release(self, obj: AvatarBehaviorState | None) -> None:
        if obj is not None and len(self._free) < self._max:
            self._free.append(obj)


_default_pool = _BehaviorStatePool()


def acquire_behavior_state() -> AvatarBehaviorState:
    return _default_pool.acquire()


def release_behavior_state(obj: AvatarBehaviorState | None) -> None:
    _default_pool.release(obj)


def reset_behavior_state_pool() -> None:
    """Test/diagnostic helper: drop all pooled instances."""
    _default_pool._free.clear()
