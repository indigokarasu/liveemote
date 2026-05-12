from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field

class SmoothingConfig(BaseModel):
    face_alpha: float = 0.35
    audio_alpha: float = 0.45
    affect_alpha: float = 0.25

class ReactionDelayConfig(BaseModel):
    mirror_min: int = 250
    mirror_max: int = 900
    reflect_min: int = 600
    reflect_max: int = 1600

class AffectConfig(BaseModel):
    update_hz: int = 30
    min_emote_dwell_ms: int = 1200
    reaction_delay_ms: ReactionDelayConfig = Field(default_factory=ReactionDelayConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)

class GazeConfig(BaseModel):
    enabled: bool = True
    eye_lead_ms: int = 120
    head_follow_ms: int = 450
    max_yaw_deg: float = 12
    max_pitch_deg: float = 8
    micro_saccades: bool = True

class BehaviorConfig(BaseModel):
    default_mode: str = "reflect"
    mirroring_strength: float = 0.22
    expressiveness: float = 0.42
    avoid_constant_eye_contact: bool = True

class HermesConfig(BaseModel):
    mode: str = "fake"
    url: str = "ws://127.0.0.1:18789/avatar"
    send_events: list[str] = Field(default_factory=lambda: ["user.transcript", "affect.summary", "interruption"])
    receive_events: list[str] = Field(default_factory=lambda: ["hermes.response", "hermes.behavior_hint"])

class RendererConfig(BaseModel):
    livetalking_url: str = "http://127.0.0.1:8010"

class VoiceConfig(BaseModel):
    backend: str = "luxtts"
    device: str = "cpu"
    cache_dir: str = "cache/voice"

class AppConfig(BaseModel):
    affect: AffectConfig = Field(default_factory=AffectConfig)
    gaze: GazeConfig = Field(default_factory=GazeConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    hermes: HermesConfig = Field(default_factory=HermesConfig)
    renderer: RendererConfig = Field(default_factory=RendererConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(path: str | Path | None = None) -> AppConfig:
    defaults = Path(__file__).with_name("defaults.yaml")
    data = yaml.safe_load(defaults.read_text()) or {}
    if path:
        data = deep_merge(data, yaml.safe_load(Path(path).read_text()) or {})
    return AppConfig.model_validate(data)
