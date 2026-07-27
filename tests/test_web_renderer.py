"""Tests for the self-driven WebRenderer.

The renderer is the default backend for the LiveEmote redesign: the avatar
has its own face and is animated autonomously in the browser. These tests
verify it stays self-consistent (always-on, no face reenactment / face
swap), correctly maps ``AvatarBehaviorState`` into an avatar visual hint,
and never opens a face-reenactment connection to an external server.
"""
from __future__ import annotations

from pathlib import Path

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import (
    BackgroundSpec,
    CharacterIndex,
    EmoteAsset,
    VisualStyle,
    VoiceStyleSpec,
)
from hermes_avatar.renderer.web_renderer import WebRenderer
from scripts.create_sample_character import PNG_1X1_RGBA


def _build_index(tmp_path: Path) -> CharacterIndex:
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir(parents=True)
    canonical_png = canonical_dir / "canonical.png"
    # Placeholder tiny PNG — renderer should NOT treat this as visually meaningful.
    canonical_png.write_bytes(PNG_1X1_RGBA)
    emotes_root = tmp_path / "emotes"
    for state in ("neutral", "listening", "thinking", "happy"):
        (emotes_root / state).mkdir(parents=True, exist_ok=True)
        (emotes_root / state / f"{state}_001.png").write_bytes(PNG_1X1_RGBA)
    return CharacterIndex(
        character_id="test_char",
        canonical_image=str(canonical_png),
        emotes=[
            EmoteAsset(id=f"{state}_001", path=str(emotes_root / state / f"{state}_001.png"), state=state)
            for state in ("neutral", "listening", "thinking", "happy")
        ],
        styles=[VisualStyle(id="neutral", name="Neutral", voice=VoiceStyleSpec(), default_background_id="studio")],
        backgrounds=[BackgroundSpec(id="studio", name="Soft studio")],
        default_style_id="neutral",
        default_background_id="studio",
    )


def test_web_renderer_capabilities_never_pretend_to_be_face_reenactment(tmp_path):
    index = _build_index(tmp_path)
    renderer = WebRenderer()
    renderer.load_character(index)
    caps = renderer.capabilities()
    assert caps["backend"] == "web"
    assert caps["online"] is True
    assert caps["no_face_reenactment"] is True
    assert caps["no_face_swap"] is True
    # Default state is no avatar yet.
    assert caps["avatar_visual"]["portrait_kind"] == "svg_fallback"
    # Canonical image is the 1x1 placeholder — should NOT be exposed as canonical_url.
    assert caps["avatar_visual"]["canonical_url"] is None


def test_web_renderer_set_behavior_picks_appropriate_emote(tmp_path):
    index = _build_index(tmp_path)
    renderer = WebRenderer()
    renderer.load_character(index)

    listening_behavior = AvatarBehaviorState(mode="listening", affect="attentive_soft", emote_id=None, gaze_target="toward_user", intensity=0.3)
    renderer.set_behavior(listening_behavior)
    visual = renderer.avatar_visual()
    # No emote_id was supplied — fallback to listening state.
    assert visual["active_emote"] is not None
    assert visual["active_emote"]["state"] == "listening"

    thinking_behavior = AvatarBehaviorState(mode="thinking", affect="reflective", emote_id=None, gaze_target="soft_forward", intensity=0.2)
    renderer.set_behavior(thinking_behavior)
    visual = renderer.avatar_visual()
    assert visual["active_emote"]["state"] == "thinking"


def test_web_renderer_records_audio_for_audio_route(tmp_path):
    index = _build_index(tmp_path)
    renderer = WebRenderer()
    renderer.load_character(index)
    behavior = AvatarBehaviorState(mode="speaking", affect="warm", intensity=0.4)
    renderer.speak("/tmp/not-cache.wav", "Hello world", behavior)
    caps = renderer.capabilities()
    assert caps["last_audio_path"] == "/tmp/not-cache.wav"
    assert caps["last_behavior"]["mode"] == "speaking"


def test_web_renderer_interrupt_flips_to_recovering(tmp_path):
    index = _build_index(tmp_path)
    renderer = WebRenderer()
    renderer.load_character(index)
    renderer.set_behavior(AvatarBehaviorState(mode="speaking", affect="warm", intensity=0.4))
    renderer.interrupt()
    caps = renderer.capabilities()
    assert caps["last_behavior"]["mode"] == "recovering"
    assert caps["last_behavior"]["intensity"] == 0.0
