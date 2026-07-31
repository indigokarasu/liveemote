"""Tests for EmotionTagPreprocessor — parametrized across the affect→prosody
mapping, SSML wrapping per backend, and integration with VoiceStyle."""

from __future__ import annotations

import pytest

from hermes_avatar.voice.base import VoiceStyle
from hermes_avatar.voice.emotion_tag_processor import (
    EmotionTagProcessor,
    EmotionResult,
    _ssml_wrap,
    _AFFECT_MAP,
)


@pytest.fixture
def processor():
    return EmotionTagProcessor()


class TestAffectMap:
    def test_all_known_affects_have_complete_mapping(self):
        """Every affect label must define pace, warmth, intensity, pitch, rate, volume."""
        required = {"pace", "warmth", "intensity", "pitch", "rate", "volume"}
        for label, mapping in _AFFECT_MAP.items():
            missing = required - set(mapping.keys())
            assert not missing, f"affect={label!r} missing keys: {missing}"

    def test_neutral_is_identity(self):
        m = _AFFECT_MAP["neutral"]
        assert m["pace"] == 1.0
        assert m["warmth"] == 1.0
        assert m["intensity"] == 0.6

    def test_excited_increases_pace_and_pitch(self):
        m = _AFFECT_MAP["excited"]
        assert m["pace"] > 1.0
        assert "+" in str(m["pitch"])

    def test_sad_decreases_pace_and_pitch(self):
        m = _AFFECT_MAP["sad"]
        assert m["pace"] < 1.0
        assert "-" in str(m["pitch"])


class TestSSMLWrapping:
    def test_default_prosody_produces_plain_text(self):
        result = _ssml_wrap("Hello", pitch="default", rate="medium", volume="medium")
        assert result == "Hello"

    def test_custom_prosody_wraps_in_speak(self):
        result = _ssml_wrap("Hello!", pitch="+10%", rate="fast", volume="loud")
        assert result.startswith("<speak><prosody ")
        assert "pitch=\"+10%\"" in result
        assert "rate=\"fast\"" in result
        assert "volume=\"loud\"" in result
        assert "Hello!" in result
        assert result.endswith("</prosody></speak>")

    def test_partial_prosody_only_includes_non_default(self):
        result = _ssml_wrap("Hi", pitch="+5%", rate="medium", volume="medium")
        assert "pitch=\"+5%\"" in result
        assert "rate=" not in result
        assert "volume=" not in result


class TestProcessPlainText:
    def test_neutral_affect_returns_unchanged_text_for_luxtts(self, processor):
        tags = {"affect": "neutral", "voice": {"pace": 0.44, "warmth": 0.62, "intensity": 0.35}}
        result = processor.process(tags, "Hello world", backend="luxtts")
        assert result.text == "Hello world"
        assert not result.text.startswith("<speak>")
        assert result.affect == "neutral"

    def test_excited_affect_returns_ssml_for_elevenlabs(self, processor):
        tags = {"affect": "excited", "voice": {"pace": 0.5, "warmth": 0.7, "intensity": 0.8}}
        result = processor.process(tags, "Amazing!", backend="elevenlabs")
        assert result.text.startswith("<speak>")
        assert "Amazing!" in result.text
        assert result.affect == "excited"

    def test_ssml_is_always_populated_regardless_of_backend(self, processor):
        tags = {"affect": "sad", "voice": {}}
        result = processor.process(tags, "Oh no...", backend="luxtts")
        assert result.ssml.startswith("<speak>")
        assert result.text == "Oh no..."  # luxtts gets plain


class TestVoiceStyleEnrichment:
    def test_excited_multiplies_intensity_up(self, processor):
        tags = {"affect": "excited", "voice": {"intensity": 0.5}}
        result = processor.process(tags, "Wow!", backend="luxtts")
        assert result.voice_style.intensity > 0.5

    def test_sad_multiplies_intensity_down(self, processor):
        tags = {"affect": "sad", "voice": {"intensity": 0.5}}
        result = processor.process(tags, "Oh...", backend="luxtts")
        assert result.voice_style.intensity < 0.5

    def test_voice_style_preserves_provided_values_scaled_by_affect(self, processor):
        tags = {"affect": "warm", "voice": {"pace": 0.5, "warmth": 0.6, "intensity": 0.4}}
        mapping = _AFFECT_MAP["warm"]
        result = processor.process(tags, "Hello", backend="luxtts")
        vs = result.voice_style
        # pace is scaled by mapping multiplier, clamped to [0,1]
        expected_pace = min(1.0, max(0.0, 0.5 * mapping["pace"]))
        assert vs.pace == expected_pace
        expected_warmth = min(1.0, max(0.0, 0.6 * mapping["warmth"]))
        assert vs.warmth == expected_warmth

    def test_extra_dict_carries_prosody_hints(self, processor):
        tags = {"affect": "reassuring", "voice": {}}
        result = processor.process(tags, "It's okay.", backend="luxtts")
        extra = result.voice_style.extra
        assert extra["pitch"] == "-2%"
        assert extra["rate"] == "slow"
        assert extra["volume"] == "soft"
        assert extra["affect"] == "reassuring"

    def test_base_style_is_layered(self, processor):
        base = VoiceStyle(pace=0.3, warmth=0.8, intensity=0.2, extra={"character": "indigo"})
        tags = {"affect": "happy", "voice": {}}
        result = processor.process(tags, "Yay!", backend="luxtts", base_style=base)
        assert result.voice_style.extra.get("character") == "indigo"


class TestUnknownAffectFallback:
    def test_unknown_affect_falls_back_to_neutral(self, processor):
        tags = {"affect": "nonexistent_emotion_xyz", "voice": {}}
        result = processor.process(tags, "text", backend="luxtts")
        assert result.affect == "nonexistent_emotion_xyz"  # we keep what was given
        # but the mapping falls back to neutral multipliers
        assert result.voice_style.pace == VoiceStyle().pace  # 1.0 * 0.44

    def test_missing_affect_defaults_to_neutral(self, processor):
        tags: dict = {"voice": {}}
        result = processor.process(tags, "text", backend="luxtts")
        assert result.affect == "neutral"


class TestBackendSelection:
    @pytest.mark.parametrize("backend,expects_ssml", [
        ("elevenlabs", True),
        ("fishaudio", False),
        ("luxtts", False),
        ("none", False),
    ])
    def test_ssml_only_applied_to_named_backends(self, processor, backend, expects_ssml):
        tags = {"affect": "excited", "voice": {}}
        result = processor.process(tags, "Test", backend=backend)
        if expects_ssml:
            assert result.text.startswith("<speak>")
        else:
            assert not result.text.startswith("<speak>")
            assert result.text == "Test"


class TestEmotionResultDataclass:
    def test_emotion_result_is_importable_and_fields_exist(self):
        r = EmotionResult(text="t", voice_style=VoiceStyle(), ssml="<speak>t</speak>", affect="neutral")
        assert r.text == "t"
        assert r.affect == "neutral"


class TestLegacyVoiceTagsPassthrough:
    """The LLM's existing ``voice.{pace,warmth,intensity}`` tags must still work
    — the preprocessor layers on top of them, never replaces them."""

    def test_llm_voice_tags_are_honored(self, processor):
        tags = {"affect": "neutral", "voice": {"pace": 0.9, "warmth": 0.3, "intensity": 0.1}}
        result = processor.process(tags, "Hmm", backend="luxtts")
        # Neutral multipliers: pace=1.0, warmth=1.0, intensity=0.6.
        # The LLM's values are scaled by these multipliers, not replaced.
        assert result.voice_style.pace == 0.9  # 0.9 * 1.0
        assert result.voice_style.warmth == 0.3  # 0.3 * 1.0
        assert result.voice_style.intensity == 0.06  # 0.1 * 0.6

    def test_llm_voice_tags_are_scaled_by_affect(self, processor):
        # excited pace multiplier = 1.1
        tags = {"affect": "excited", "voice": {"pace": 0.5}}
        result = processor.process(tags, "Fast!", backend="luxtts")
        assert result.voice_style.pace == pytest.approx(0.55)  # 0.5 * 1.1
