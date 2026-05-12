from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal

TurnState = Literal["idle", "user_speaking", "assistant_thinking", "assistant_speaking", "interrupted"]
AvatarMode = Literal["idle", "listening", "thinking", "speaking", "recovering"]

@dataclass
class UserAffectState:
    face_detected: bool = False
    attention: float = 0.0
    engagement: float = 0.0
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
    topic_weight: float = 0.0
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

    def to_dict(self) -> dict:
        return asdict(self)
