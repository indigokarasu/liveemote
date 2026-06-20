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

class AgentResponseEvent(BaseEvent):
    type: Literal["agent.response"] = "agent.response"
    text: str
    tags: dict[str, Any] = Field(default_factory=dict)





def parse_event(payload: dict[str, Any]) -> BaseEvent:
    mapping = {
        "perception.frame": PerceptionFrameEvent,
        "audio.vad": AudioVADEvent,
        "hermes.response": AgentResponseEvent,
        "agent.response": AgentResponseEvent,
    }
    model = mapping.get(payload.get("type"), BaseEvent)
    return model.model_validate(payload)
