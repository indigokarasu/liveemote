from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import (
    BackgroundSpec,
    CharacterIndex,
    EmoteAsset,
    TrainingReference,
    VisualStyle,
)
from hermes_avatar.renderer.base import Renderer


# ---------------------------------------------------------------------------
# Env cleanup: ensure no stray AFFECT__/GAZE__/RENDERER__/HERMES__ vars leak
# between tests, since load_config() reads os.environ at call time.
# ---------------------------------------------------------------------------
_RELEVANT_PREFIXES = ("AFFECT__", "GAZE__", "RENDERER__", "BEHAVIOR__",
                       "AGENT__", "VOICE__", "HERMES_")


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ[k] for k in list(os.environ)
              if any(k.startswith(p) for p in _RELEVANT_PREFIXES)}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k in [k2 for k2 in os.environ
               if any(k2.startswith(p) for p in _RELEVANT_PREFIXES)]:
        os.environ.pop(k, None)
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# Sample in-memory CharacterIndex
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_character_index() -> CharacterIndex:
    return CharacterIndex(
        character_id="test_char",
        display_name="Test Character",
        canonical_image="/tmp/nonexistent_canonical.png",
        styles=[
            VisualStyle(id="neutral", name="Neutral",
                       default_background_id="studio"),
            VisualStyle(id="cyberpunk", name="Cyberpunk",
                       default_background_id="cyberpunk-city"),
        ],
        backgrounds=[
            BackgroundSpec(id="studio", name="Studio"),
            BackgroundSpec(id="cyberpunk-city", name="Cyberpunk City"),
        ],
        emotes=[
            EmoteAsset(id="neutral_001", path="/tmp/neutral.png",
                       state="neutral", loopable=True),
            EmoteAsset(id="thinking_001", path="/tmp/thinking.png",
                       state="thinking", loopable=True),
            EmoteAsset(id="listening_001", path="/tmp/listening.png",
                       state="listening", loopable=True),
        ],
        training_references=[
            TrainingReference(
                id="identity_anchor_001",
                path="/tmp/canonical.png",
                role="identity_anchor",
                state="neutral",
                weight=1.0,
                tags=["canonical", "identity", "neutral"],
            ),
        ],
        default_style_id="neutral",
        default_background_id="studio",
    )


# ---------------------------------------------------------------------------
# Fake renderer implementing the Renderer ABC, recording calls.
# ---------------------------------------------------------------------------
class FakeRenderer(Renderer):
    def __init__(self):
        self.load_character_calls = []
        self.set_behavior_calls = []
        self.speak_calls = []
        self.interrupt_calls = 0
        self.set_theme_calls = []

    def load_character(self, character_index):
        self.load_character_calls.append(character_index)
        return None

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        self.set_behavior_calls.append(behavior)

    def speak(self, audio_path: str, text: str,
              behavior: AvatarBehaviorState) -> None:
        self.speak_calls.append((audio_path, text, behavior))

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def set_theme(self, character_index, style, background) -> None:
        self.set_theme_calls.append((character_index, style, background))

    def capabilities(self) -> dict:
        return {
            "backend": "fake",
            "online": True,
            "circuit_breaker": {"state": "closed", "failure_count": 0,
                                "last_failure_time": None},
        }


@pytest.fixture
def fake_renderer() -> FakeRenderer:
    return FakeRenderer()


# ---------------------------------------------------------------------------
# Temp on-disk character directory for building a real DemoOrchestrator.
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_character_dir(tmp_path: Path) -> Path:
    root = tmp_path / "character"
    (root / "canonical").mkdir(parents=True)
    (root / "canonical" / "canonical.png").write_text("placeholder")
    return root
