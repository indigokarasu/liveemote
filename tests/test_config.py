from __future__ import annotations

import pytest
from pydantic import ValidationError

from hermes_avatar.config.schema import (
    AppConfig,
    detect_hardware_profile,
    load_config,
    reload_config,
)


def test_load_config_defaults(monkeypatch):
    # Disable hardware-aware tuning so defaults are not mutated on low-spec boxes.
    monkeypatch.setenv("HERMES_DISABLE_HW_AWARE", "true")
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.affect.update_hz == 30
    assert cfg.affect.smoothing.face_alpha == 0.35
    assert cfg.affect.smoothing.audio_alpha == 0.45
    assert cfg.affect.smoothing.affect_alpha == 0.25
    assert cfg.renderer.request_timeout == 1.5
    assert cfg.renderer.connect_timeout == 1.0
    assert cfg.gaze.enabled is True


def test_env_overlay_update_hz_int_coercion(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_HW_AWARE", "true")
    monkeypatch.setenv("AFFECT__UPDATE_HZ", "60")
    cfg = load_config()
    assert cfg.affect.update_hz == 60
    assert isinstance(cfg.affect.update_hz, int)


def test_env_overlay_nested_smoothing_face_alpha(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_HW_AWARE", "true")
    monkeypatch.setenv("AFFECT__SMOOTHING__FACE_ALPHA", "0.5")
    cfg = load_config()
    assert cfg.affect.smoothing.face_alpha == 0.5


def test_env_bool_coercion_true_false(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_HW_AWARE", "true")
    monkeypatch.setenv("GAZE__ENABLED", "TRUE")
    assert load_config().gaze.enabled is True
    monkeypatch.setenv("GAZE__ENABLED", "false")
    assert load_config().gaze.enabled is False


def test_env_invalid_alpha_raises(monkeypatch):
    monkeypatch.setenv("AFFECT__SMOOTHING__FACE_ALPHA", "1.5")
    with pytest.raises(ValidationError):
        load_config()


def test_env_invalid_cross_field_mirror_max_le_mirror_min(monkeypatch):
    # mirror_max default is 900; force it below mirror_min (250).
    monkeypatch.setenv("AFFECT__REACTION_DELAY_MS__MIRROR_MAX", "100")
    with pytest.raises(ValidationError):
        load_config()


def test_renderer_negative_request_timeout_raises(monkeypatch):
    monkeypatch.setenv("RENDERER__REQUEST_TIMEOUT", "-1.0")
    with pytest.raises(ValidationError):
        load_config()


def test_detect_hardware_profile_returns_dict_and_never_raises():
    # Should always return a dict, regardless of platform specifics.
    profile = detect_hardware_profile()
    assert isinstance(profile, dict)
    for key in ("cpu_count", "memory_gb", "gpu_available"):
        assert key in profile


def test_reload_config_returns_appconfig_with_hardware_profile():
    cfg = reload_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.hardware_profile, dict)
    # update_hz must be a sane positive integer.
    assert isinstance(cfg.affect.update_hz, int)
    assert cfg.affect.update_hz > 0
