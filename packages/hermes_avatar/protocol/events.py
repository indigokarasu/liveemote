from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class BaseEvent(BaseModel):
    type: str
    timestamp_ms: int

class PerceptionFrameEvent(BaseEvent):
    type: Literal["perception.frame"] = "perception.frame"
    face_detected: bool = False
    face_center: tuple[float, float] | None = None
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    expression: dict[str, float] = Field(default_factory=dict)
    emotion_confidence: float = 0.0
    gaze_confidence: float = 0.0

class AudioVADEvent(BaseEvent):
    type: Literal["audio.vad"] = "audio.vad"
    speaking: bool
    energy: float = 0.0
    speech_rate: float = 0.0

class HermesResponseEvent(BaseEvent):
    type: Literal["hermes.response"] = "hermes.response"
    text: str
    tags: dict[str, Any] = Field(default_factory=dict)

class AvatarBehaviorEvent(BaseEvent):
    type: Literal["avatar.behavior"] = "avatar.behavior"
    mode: str
    affect: str
    gaze_target: str
    emote_id: str | None = None
    intensity: float = 0.0
    lip_sync_enabled: bool = False
    full_body_pose: str = "standing_idle"

class AvatarSpeakEvent(BaseEvent):
    type: Literal["avatar.speak"] = "avatar.speak"
    text: str
    audio_path: str
    affect: str
    voice_style: dict[str, Any] = Field(default_factory=dict)

def parse_event(payload: dict[str, Any]) -> BaseEvent:
    mapping = {
        "perception.frame": PerceptionFrameEvent,
        "audio.vad": AudioVADEvent,
        "hermes.response": HermesResponseEvent,
        "avatar.behavior": AvatarBehaviorEvent,
        "avatar.speak": AvatarSpeakEvent,
    }
    model = mapping.get(payload.get("type"), BaseEvent)
    return model.model_validate(payload)
