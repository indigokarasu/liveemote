from __future__ import annotations
import os
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
    max_yaw_deg: float = 12
    max_pitch_deg: float = 8

class BehaviorConfig(BaseModel):
    default_mode: str = "reflect"
    mirroring_strength: float = 0.22

class AgentConfig(BaseModel):
    mode: str = "fake"
    harness: str = "generic"
    url: str = "ws://127.0.0.1:18789/avatar"


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
    agent: AgentConfig = Field(default_factory=AgentConfig)
    renderer: RendererConfig = Field(default_factory=RendererConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)


def _parse_env_value(value: str) -> Any:
    """Parse an env var string into a typed Python value.

    Supports: bool (true/false/1/0/yes/no), int, float, and string fallback.
    """
    lowered = value.lower().strip()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    # int
    try:
        return int(value)
    except ValueError:
        pass
    # float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _load_env_config() -> dict[str, Any]:
    """Read env vars with ``__`` as nested key separator.

    Convention: uppercase env var names use ``__`` to separate nesting levels.
    Each part is lowercased when building the nested dict.

    Examples:
        ``AFFECT__UPDATE_HZ=60``     -> ``{"affect": {"update_hz": 60}}``
        ``GAZE__ENABLED=false``      -> ``{"gaze": {"enabled": False}}``
        ``VOICE__DEVICE=cuda``       -> ``{"voice": {"device": "cuda"}}``
    """
    result: dict[str, Any] = {}
    sep = "__"  # double underscore = nesting separator
    for key, raw in os.environ.items():
        if sep not in key:
            continue
        parts = key.split(sep)
        parsed = _parse_env_value(raw)
        target = result
        for i, part in enumerate(parts):
            key_lower = part.lower()
            if i == len(parts) - 1:
                target[key_lower] = parsed
            else:
                if key_lower not in target:
                    target[key_lower] = {}
                target = target[key_lower]
    return result


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
    # Environment variables with __ separator override YAML values
    env_data = _load_env_config()
    data = deep_merge(data, env_data)
    return AppConfig.model_validate(data)
