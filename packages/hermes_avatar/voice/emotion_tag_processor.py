"""Emotion tag → TTS prosody preprocessor.

Bridges the gap between the LLM's structured emotion tags and the TTS
backends.  The LLM returns a JSON contract with ``{text, tags: {affect,
voice: {pace, warmth, intensity}}}``, but only the ``voice`` sub-dict
(3 generic floats) reaches the TTS.  The ``affect`` label — which carries
richer semantic information like "excited", "validating_grounded", or
"spacious" — never influences how the voice actually *sounds*.

This module fixes that by:

1. Mapping each ``affect`` label to a set of prosody adjustments (pitch,
   rate, volume) and an SSML template.
2. Producing an enriched ``VoiceStyle`` (the existing 3-float contract
   is preserved; the preprocessor *adds* backend-specific hints).
3. Optionally wrapping the text in SSML for backends that support it
   (ElevenLabs, Amazon Polly) and leaving plain text for backends that
   don't (Fish Audio, LuxTTS).

Usage (in orchestrator)::

    from hermes_avatar.voice.emotion_tag_processor import EmotionTagProcessor

    processor = EmotionTagProcessor()
    result = processor.process(
        response.tags, response.text,
        backend=orchestrator.voice_backend_name,
    )
    # result.text          → plain or SSML-wrapped text
    # result.voice_style   → enriched VoiceStyle
    # result.ssml          → SSML fragment (always populated, even if
    #                        the backend doesn't consume it)

    speech = self.voice.synthesize(result.text, result.voice_style, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import VoiceStyle


# ---------------------------------------------------------------------------
# Emotion → prosody mapping
# ---------------------------------------------------------------------------
# Each entry defines:
#   pace      → multiplier applied to VoiceStyle.pace  (clamped [0.0, 1.0])
#   warmth    → multiplier applied to VoiceStyle.warmth
#   intensity → multiplier applied to VoiceStyle.intensity
#   pitch     → SSML prosody pitch  (e.g. "+10%", "-5%", "default")
#   rate      → SSML prosody rate   (e.g. "fast", "slow", "medium")
#   volume    → SSML prosody volume (e.g. "loud", "soft", "medium")
_AFFECT_MAP: dict[str, dict[str, Any]] = {
    # -- warm / positive ------------------------------------------------
    "warm": {
        "pace": 0.85, "warmth": 1.2, "intensity": 0.9,
        "pitch": "+5%", "rate": "medium", "volume": "medium",
    },
    "amused": {
        "pace": 0.9, "warmth": 1.15, "intensity": 1.0,
        "pitch": "+3%", "rate": "medium", "volume": "medium",
    },
    "reassuring": {
        "pace": 0.8, "warmth": 1.3, "intensity": 0.7,
        "pitch": "-2%", "rate": "slow", "volume": "soft",
    },
    "focused": {
        "pace": 1.0, "warmth": 1.0, "intensity": 0.8,
        "pitch": "default", "rate": "medium", "volume": "medium",
    },
    "curious": {
        "pace": 0.9, "warmth": 1.1, "intensity": 0.75,
        "pitch": "+2%", "rate": "medium", "volume": "medium",
    },
    "happy": {
        "pace": 0.95, "warmth": 1.25, "intensity": 0.95,
        "pitch": "+8%", "rate": "medium", "volume": "medium",
    },
    "excited": {
        "pace": 1.1, "warmth": 1.2, "intensity": 1.1,
        "pitch": "+12%", "rate": "fast", "volume": "loud",
    },
    # -- grounding / validating -----------------------------------------
    "grounded": {
        "pace": 0.85, "warmth": 1.1, "intensity": 0.65,
        "pitch": "default", "rate": "medium", "volume": "medium",
    },
    "grounded_steady": {
        "pace": 0.8, "warmth": 1.05, "intensity": 0.6,
        "pitch": "-2%", "rate": "slow", "volume": "medium",
    },
    "validating_grounded": {
        "pace": 0.82, "warmth": 1.25, "intensity": 0.55,
        "pitch": "-3%", "rate": "slow", "volume": "soft",
    },
    "grounded_concern_soft_brow": {
        "pace": 0.78, "warmth": 1.3, "intensity": 0.5,
        "pitch": "-5%", "rate": "slow", "volume": "soft",
    },
    # -- spacious / calm ------------------------------------------------
    "spacious": {
        "pace": 0.7, "warmth": 1.1, "intensity": 0.4,
        "pitch": "-2%", "rate": "slow", "volume": "soft",
    },
    "spacious_attentive": {
        "pace": 0.72, "warmth": 1.15, "intensity": 0.45,
        "pitch": "default", "rate": "slow", "volume": "soft",
    },
    "attentive_soft": {
        "pace": 0.85, "warmth": 1.1, "intensity": 0.5,
        "pitch": "default", "rate": "medium", "volume": "soft",
    },
    # -- neutral / idle -------------------------------------------------
    "neutral": {
        "pace": 1.0, "warmth": 1.0, "intensity": 0.6,
        "pitch": "default", "rate": "medium", "volume": "medium",
    },
    "idle": {
        "pace": 0.95, "warmth": 1.0, "intensity": 0.5,
        "pitch": "default", "rate": "medium", "volume": "medium",
    },
    # -- concern / empathy ----------------------------------------------
    "concerned": {
        "pace": 0.85, "warmth": 1.2, "intensity": 0.6,
        "pitch": "-3%", "rate": "slow", "volume": "soft",
    },
    "sad": {
        "pace": 0.75, "warmth": 1.1, "intensity": 0.4,
        "pitch": "-8%", "rate": "slow", "volume": "soft",
    },
    "empathetic": {
        "pace": 0.82, "warmth": 1.3, "intensity": 0.55,
        "pitch": "-4%", "rate": "slow", "volume": "soft",
    },
}


# Backends that natively understand SSML.
_SSML_BACKENDS = frozenset({"elevenlabs"})


def _ssml_wrap(text: str, pitch: str, rate: str, volume: str) -> str:
    """Wrap plain text in an SSML `<speak>` block with prosody hints."""
    prosody: list[str] = []
    if pitch and pitch != "default":
        prosody.append(f'pitch="{pitch}"')
    if rate and rate != "medium":
        prosody.append(f'rate="{rate}"')
    if volume and volume != "medium":
        prosody.append(f'volume="{volume}"')

    if prosody:
        return f'<speak><prosody {" ".join(prosody)}>{text}</prosody></speak>'
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class EmotionResult:
    """Output of :meth:`EmotionTagProcessor.process`.

    Attributes
    ----------
    text:
        The text the TTS backend should speak.  For SSML-capable backends
        this is the SSML-wrapped version; otherwise it's the original plain
        text.
    voice_style:
        Enriched :class:`VoiceStyle` with the affect-driven adjustments
        merged into the existing style.  The ``extra`` dict carries the
        raw prosody hints so downstream code can inspect them.
    ssml:
        The SSML fragment (always populated for observability, even when
        the backend doesn't consume it).
    affect:
        The resolved affect label (fallback: ``"neutral"``).
    """

    text: str
    voice_style: VoiceStyle
    ssml: str = ""
    affect: str = "neutral"


class EmotionTagProcessor:
    """Stateless emotion-tag → TTS prosody converter.

    Parameters
    ----------
    ssml_backends:
        Set of backend names that receive SSML-wrapped text instead of
        plain text.  Default: ``{"elevenlabs"}``.
    affect_map:
        Override the built-in emotion→prosody mapping.  Pass a dict of
        ``{affect_label: {pace, warmth, intensity, pitch, rate, volume}}``
        to customize or extend the defaults.
    """

    def __init__(
        self,
        ssml_backends: set[str] | None = None,
        affect_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._ssml_backends = ssml_backends or set(_SSML_BACKENDS)
        self._affect_map = affect_map or _AFFECT_MAP

    # -- main entry point --------------------------------------------------

    def process(
        self,
        tags: dict[str, Any],
        text: str,
        backend: str = "luxtts",
        base_style: VoiceStyle | None = None,
    ) -> EmotionResult:
        """Convert LLM emotion tags into enriched text + VoiceStyle.

        Parameters
        ----------
        tags:
            The LLM's ``response.tags`` dict.  Expected keys: ``affect``
            (string label), ``voice`` (dict with ``pace``, ``warmth``,
            ``intensity``).
        text:
            The plain-text utterance.
        backend:
            TTS backend name (``"elevenlabs"``, ``"fishaudio"``,
            ``"luxtts"``, ...).  Controls whether SSML is applied.
        base_style:
            Optional pre-existing VoiceStyle (from character config) to
            layer the emotion adjustments on top of.  When ``None``,
            starts from the VoiceStyle defaults.

        Returns
        -------
        EmotionResult
            ``.text`` is SSML-wrapped for SSML backends, plain otherwise.
            ``.voice_style`` is the enriched VoiceStyle.
        """
        affect_label = (tags.get("affect") or tags.get("emotion") or "neutral")
        affect_label = affect_label.lower().replace(" ", "_").replace("-", "_")

        mapping = self._affect_map.get(
            affect_label,
            self._affect_map.get("neutral", {}),
        )

        # Extract LLM-provided voice overrides.
        voice_tags = tags.get("voice", {}) if isinstance(tags.get("voice"), dict) else {}

        # Start from the base_style (character defaults) or VoiceStyle defaults.
        base = base_style or VoiceStyle()
        base_pace = voice_tags.get("pace", base.pace)
        base_warmth = voice_tags.get("warmth", base.warmth)
        base_intensity = voice_tags.get("intensity", base.intensity)

        # Apply affect multipliers, clamping to valid ranges.
        enriched = VoiceStyle(
            pace=max(0.0, min(1.0, base_pace * mapping.get("pace", 1.0))),
            warmth=max(0.0, min(1.0, base_warmth * mapping.get("warmth", 1.0))),
            intensity=max(0.0, min(1.0, base_intensity * mapping.get("intensity", 1.0))),
            extra={
                **getattr(base, "extra", {}),
                "pitch": mapping.get("pitch", "default"),
                "rate": mapping.get("rate", "medium"),
                "volume": mapping.get("volume", "medium"),
                "affect": affect_label,
            },
        )

        pitch = mapping.get("pitch", "default")
        rate = mapping.get("rate", "medium")
        volume = mapping.get("volume", "medium")
        ssml_text = _ssml_wrap(text, str(pitch), str(rate), str(volume))

        use_ssml = backend.lower() in self._ssml_backends

        return EmotionResult(
            text=ssml_text if use_ssml else text,
            voice_style=enriched,
            ssml=ssml_text,
            affect=affect_label,
        )
