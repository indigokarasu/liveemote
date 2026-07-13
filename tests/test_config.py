"""Tests for configuration loading with environment variable support.

Tests cover:
- _parse_env_value: type coercion from env var strings
- _load_env_config: double-underscore nesting convention
- load_config: env vars override YAML defaults
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from hermes_avatar.config.schema import (
    _parse_env_value,
    _load_env_config,
    load_config,
    AppConfig,
)


# ── _parse_env_value ───────────────────────────────────────────────────


class TestParseEnvValue:
    def test_true_values(self):
        assert _parse_env_value("true") is True
        assert _parse_env_value("TRUE") is True
        assert _parse_env_value("True") is True
        assert _parse_env_value("1") is True
        assert _parse_env_value("yes") is True

    def test_false_values(self):
        assert _parse_env_value("false") is False
        assert _parse_env_value("FALSE") is False
        assert _parse_env_value("False") is False
        assert _parse_env_value("0") is False
        assert _parse_env_value("no") is False

    def test_integer_values(self):
        assert _parse_env_value("42") == 42
        assert _parse_env_value("0") is False  # bool wins for "0"
        assert _parse_env_value("-7") == -7
        assert _parse_env_value("1000") == 1000

    def test_float_values(self):
        assert _parse_env_value("3.14") == 3.14
        assert _parse_env_value("0.5") == 0.5
        assert _parse_env_value("-2.7") == -2.7
        assert _parse_env_value("1e3") == 1000.0

    def test_string_fallback(self):
        assert _parse_env_value("hello") == "hello"
        assert _parse_env_value("") == ""
        assert _parse_env_value("ws://127.0.0.1:8010") == "ws://127.0.0.1:8010"
        assert _parse_env_value("cuda") == "cuda"


# ── _load_env_config ───────────────────────────────────────────────────


class TestLoadEnvConfig:
    def test_single_level(self):
        """AFFECT__UPDATE_HZ=60 -> affect.update_hz: 60"""
        with mock.patch.dict(os.environ, {"AFFECT__UPDATE_HZ": "60"}, clear=True):
            result = _load_env_config()
            assert result == {"affect": {"update_hz": 60}}

    def test_two_level_nesting(self):
        """AFFECT__REACTION_DELAY_MS__MIRROR_MIN=500 -> affect.reaction_delay_ms.mirror_min: 500"""
        with mock.patch.dict(
            os.environ, {"AFFECT__REACTION_DELAY_MS__MIRROR_MIN": "500"}, clear=True
        ):
            result = _load_env_config()
            assert result == {"affect": {"reaction_delay_ms": {"mirror_min": 500}}}

    def test_multiple_env_vars(self):
        """Multiple env vars create merged nested dict"""
        with mock.patch.dict(
            os.environ,
            {
                "AFFECT__UPDATE_HZ": "60",
                "GAZE__ENABLED": "false",
                "VOICE__DEVICE": "cuda",
            },
            clear=True,
        ):
            result = _load_env_config()
            assert result["affect"]["update_hz"] == 60
            assert result["gaze"]["enabled"] is False
            assert result["voice"]["device"] == "cuda"

    def test_env_vars_without_double_underscore_are_ignored(self):
        """Env vars without __ should not appear in config"""
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "HOME": "/root",
                "SHELL": "/bin/bash",
            },
            clear=True,
        ):
            result = _load_env_config()
            assert result == {}

    def test_mixed_double_and_single_underscore(self):
        """Keys with both single and double underscores parse correctly"""
        with mock.patch.dict(
            os.environ,
            {
                "AFFECT__MIN_EMOTE_DWELL_MS": "2000",
                "BEHAVIOR__MIRRORING_STRENGTH": "0.5",
            },
            clear=True,
        ):
            result = _load_env_config()
            assert result["affect"]["min_emote_dwell_ms"] == 2000
            assert result["behavior"]["mirroring_strength"] == 0.5


# ── load_config ────────────────────────────────────────────────────────


class TestLoadConfigIntegration:
    def test_defaults_without_env_vars(self):
        """load_config() returns defaults when no env vars override"""
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.affect.update_hz == 30
        assert config.gaze.enabled is True
        assert config.behavior.default_mode == "reflect"

    def test_env_var_overrides_default(self):
        """AFFECT__UPDATE_HZ=60 overrides the YAML default of 30"""
        with mock.patch.dict(os.environ, {"AFFECT__UPDATE_HZ": "60"}, clear=False):
            config = load_config()
            assert config.affect.update_hz == 60
            assert config.affect.min_emote_dwell_ms == 1200  # unchanged

    def test_env_var_overrides_bool_default(self):
        """GAZE__ENABLED=false overrides the YAML default"""
        with mock.patch.dict(os.environ, {"GAZE__ENABLED": "false"}, clear=False):
            config = load_config()
            assert config.gaze.enabled is False
            assert config.gaze.max_yaw_deg == 12  # unchanged

    def test_env_var_overrides_float_default(self):
        """BEHAVIOR__MIRRORING_STRENGTH=0.5 overrides the default 0.22"""
        with mock.patch.dict(
            os.environ, {"BEHAVIOR__MIRRORING_STRENGTH": "0.5"}, clear=False
        ):
            config = load_config()
            assert config.behavior.mirroring_strength == 0.5

    def test_env_var_overrides_yaml_file(self):
        """Env vars take precedence over user-provided YAML file"""
        config_path = Path(__file__).parent / "fixtures" / "override.yaml"
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text("affect:\n  update_hz: 15\n")
        with mock.patch.dict(os.environ, {"AFFECT__UPDATE_HZ": "45"}, clear=False):
            config = load_config(config_path)
            assert config.affect.update_hz == 45
        config_path.unlink()
        config_path.parent.rmdir()

    def test_env_var_deeply_nested(self):
        """Three-level nesting via __"""
        with mock.patch.dict(
            os.environ,
            {"AFFECT__REACTION_DELAY_MS__REFLECT_MIN": "800"},
            clear=False,
        ):
            config = load_config()
            assert config.affect.reaction_delay_ms.reflect_min == 800
            assert config.affect.reaction_delay_ms.mirror_min == 250  # unchanged

    def test_multiple_env_vars_combined(self):
        """Multiple env vars work together"""
        with mock.patch.dict(
            os.environ,
            {
                "AFFECT__UPDATE_HZ": "60",
                "GAZE__ENABLED": "false",
                "VOICE__DEVICE": "cuda",
                "BEHAVIOR__MIRRORING_STRENGTH": "0.5",
            },
            clear=False,
        ):
            config = load_config()
            assert config.affect.update_hz == 60
            assert config.gaze.enabled is False
            assert config.voice.device == "cuda"
            assert config.behavior.mirroring_strength == 0.5
            assert config.renderer.livetalking_url == "http://127.0.0.1:8010"


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_env(self):
        """No env vars -> empty config merge"""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _load_env_config()
            assert result == {}

    def test_unrelated_double_underscore_env_var(self):
        """An env var with __ but non-config key is still parsed, it just won't affect validation"""
        with mock.patch.dict(os.environ, {"WACKY__KEY": "20"}, clear=True):
            result = _load_env_config()
            assert result == {"wacky": {"key": 20}}

    def test_string_value_string_config(self):
        """String env var value is preserved"""
        with mock.patch.dict(os.environ, {"AGENT__URL": "ws://custom:8080/path"}, clear=False):
            config = load_config()
            assert config.agent.url == "ws://custom:8080/path"