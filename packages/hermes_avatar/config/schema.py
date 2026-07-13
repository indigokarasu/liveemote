from __future__ import annotations
from pathlib import Path
from typing import Any
import os
import yaml
from pydantic import BaseModel, Field, field_validator

class SmoothingConfig(BaseModel):
    face_alpha: float = 0.35
    audio_alpha: float = 0.45
    affect_alpha: float = 0.25

    @field_validator('face_alpha', 'audio_alpha', 'affect_alpha')
    @classmethod
    def check_alpha_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('alpha must be between 0.0 and 1.0')
        return v

class ReactionDelayConfig(BaseModel):
    mirror_min: int = 250
    mirror_max: int = 900
    reflect_min: int = 600
    reflect_max: int = 1600

    @field_validator('mirror_max')
    @classmethod
    def check_mirror_max(cls, v, info):
        if v <= info.data.get('mirror_min'):
            raise ValueError('mirror_max must be greater than mirror_min')
        return v

    @field_validator('reflect_max')
    @classmethod
    def check_reflect_max(cls, v, info):
        if v <= info.data.get('reflect_min'):
            raise ValueError('reflect_max must be greater than reflect_min')
        return v

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
    request_timeout: float = 1.5
    connect_timeout: float = 1.0

    @field_validator("request_timeout", "connect_timeout")
    @classmethod
    def check_timeout_positive(cls, v):
        if v <= 0:
            raise ValueError("timeout must be greater than 0")
        return v

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
    hardware_profile: dict[str, Any] | None = None

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def _leaf_paths(d: dict[str, Any], prefix: str = "") -> set[str]:
    """Return the set of dotted leaf-paths present in a nested dict (e.g. 'affect.update_hz')."""
    paths: set[str] = set()
    for key, value in d.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            paths |= _leaf_paths(value, full)
        else:
            paths.add(full)
    return paths


def detect_hardware_profile() -> dict[str, Any]:
    """Safely detect basic hardware capabilities. Detection is non-fatal: any failure
    leaves the relevant field as None / False rather than raising."""
    profile: dict[str, Any] = {"cpu_count": None, "memory_gb": None, "gpu_available": False}
    try:
        profile["cpu_count"] = os.cpu_count()
    except Exception:
        profile["cpu_count"] = None

    mem_gb: float | None = None
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            with meminfo.open() as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                mem_gb = int(parts[1]) / (1024 * 1024)
                            except ValueError:
                                mem_gb = None
                        break
    except Exception:
        mem_gb = None
    if mem_gb is None:
        try:
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            mem_gb = None
    profile["memory_gb"] = mem_gb

    gpu = False
    try:
        import torch
        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False
    profile["gpu_available"] = gpu
    return profile


def load_config(path: str | Path | None = None) -> AppConfig:
    defaults = Path(__file__).with_name("defaults.yaml")
    data = yaml.safe_load(defaults.read_text()) or {}
    overridden_keys: set[str] = set()
    if path:
        file_data = yaml.safe_load(Path(path).read_text()) or {}
        data = deep_merge(data, file_data)
        overridden_keys |= _leaf_paths(file_data)
    # Overlay environment variables with nested prefix parsing (e.g., AFFECT__UPDATE_HZ=60)
    for key, value in os.environ.items():
        if "__" not in key:
            continue
        parts = key.split("__")
        # Convert to lowercase to match field names
        parts = [p.lower() for p in parts]
        # Navigate/create nested dict
        d = data
        for part in parts[:-1]:
            if part not in d or not isinstance(d.get(part), dict):
                d[part] = {}
            d = d[part]
        last_part = parts[-1]
        # Attempt to convert value to appropriate simple types for better compatibility
        try:
            # Try int
            converted = int(value)
        except ValueError:
            try:
                # Try float
                converted = float(value)
            except ValueError:
                # Try bool (case-insensitive)
                if value.lower() in ("true", "false"):
                    converted = value.lower() == "true"
                else:
                    converted = value
        d[last_part] = converted
        overridden_keys.add(".".join(parts))

    # Hardware-aware tuning (non-fatal). ON by default; opt out with
    # HERMES_DISABLE_HW_AWARE=true. Only adjusts keys NOT explicitly overridden by
    # env/file, so user intent always wins.
    hw_aware = os.environ.get("HERMES_DISABLE_HW_AWARE", "").lower() not in ("1", "true", "yes", "on")
    profile = detect_hardware_profile()
    if hw_aware:
        low_spec = (
            (profile["cpu_count"] is not None and profile["cpu_count"] <= 2)
            or (profile["memory_gb"] is not None and profile["memory_gb"] <= 4)
        )
        if low_spec:
            if "affect.update_hz" not in overridden_keys:
                data.setdefault("affect", {})["update_hz"] = min(
                    int(data.get("affect", {}).get("update_hz", 30)), 15
                )
            if "affect.smoothing.face_alpha" not in overridden_keys:
                data.setdefault("affect", {}).setdefault("smoothing", {})["face_alpha"] = 0.5
            if "affect.smoothing.audio_alpha" not in overridden_keys:
                data.setdefault("affect", {}).setdefault("smoothing", {})["audio_alpha"] = 0.6
            if "affect.smoothing.affect_alpha" not in overridden_keys:
                data.setdefault("affect", {}).setdefault("smoothing", {})["affect_alpha"] = 0.4
    # Always expose the detected profile for observability.
    data["hardware_profile"] = profile

    return AppConfig.model_validate(data)

def reload_config() -> AppConfig:
    """Reload configuration from defaults.yaml and environment variables.

    The returned AppConfig carries the freshly detected ``hardware_profile`` for
    observability.
    """
    return load_config()