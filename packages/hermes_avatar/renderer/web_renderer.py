"""Browser-driven renderer for the LiveEmote avatar.

The WebRenderer is the default backend for the redesign requested by the
project owner: the avatar has its own identity (the character's canonical stills
and emotes) and its own autonomous motion driven by the affect runtime, NOT a
live face-reenactment or face-swap of the user on webcam. Webcam input is used
only as a perceptual signal (focus, energy) that shapes the avatar's behavior;
it is not copied frame-by-frame.

Concretely the renderer:

* stores the latest AvatarBehaviorState so the demo server's GET /api/status
  endpoint can publish it to the browser, where the CSS/SVG avatar animates
  autonomously (breathing, gaze shifts, emote overlays, intensity-modulated
  pose transitions);
* records the most recent synth audio_path so /api/audio can stream it back;
* reports an always-online posture against a local in-process renderer (no
  external daemon required) and exposes a structured capability surface for
  feature-detection on the client.

The LiveTalking / Deep-Live-Cam adapters remain available as opt-in choices via
``--renderer livetalking`` / ``--renderer deeplivecam``. They are face
re-enactment / swap tools and are not the user's intent for the default demo.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .base import Renderer
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import (
    BackgroundSpec,
    CharacterIndex,
    EmoteAsset,
    VisualStyle,
)


class WebRenderer(Renderer):
    """Self-driven renderer; the avatar animates autonomously in the browser."""

    backend_name: str = "web"

    def __init__(self, static_root: str | Path = "cache/renderer") -> None:
        self.static_root = Path(static_root)
        self.static_root.mkdir(parents=True, exist_ok=True)
        self.character_index: CharacterIndex | None = None
        self.active_style: VisualStyle | None = None
        self.active_background: BackgroundSpec | None = None
        self.last_behavior: AvatarBehaviorState | None = None
        self.last_audio_path: str | None = None
        self.last_audio_text: str | None = None
        self.last_error: str | None = None

    # --- Renderer ABC ----------------------------------------------------

    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        self.last_behavior = behavior

    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None:
        self.last_behavior = behavior
        self.last_audio_path = audio_path or None
        self.last_audio_text = text
        # If a real audio file was produced by the voice backend, the demo
        # server can stream it back via /api/audio. WebRenderer does not
        # require the file to exist locally — it accepts the path as-is.

    def interrupt(self) -> None:
        if not self.last_behavior:
            return
        from hermes_avatar.affect.state import AvatarBehaviorState

        self.last_behavior = AvatarBehaviorState(
            mode="recovering",
            affect="reset",
            gaze_target="soft_forward",
            emote_id=self.last_behavior.emote_id,
            intensity=0.0,
            mirror_strength=0.0,
        )

    # --- Theme + extras --------------------------------------------------

    def set_theme(
        self,
        character_index: CharacterIndex,
        style: VisualStyle | None,
        background: BackgroundSpec | None,
    ) -> None:
        self.character_index = character_index
        self.active_style = style
        self.active_background = background

    # --- Capability surface ---------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "online": True,
            "no_face_reenactment": True,
            "no_face_swap": True,
            "driven_by": "AvatarBehaviorState",
            "avatar_visual": self._avatar_visual_payload(),
            "watermark": "Synthetic avatar output - identity taken from the active character profile.",
            "active_style": asdict(self.active_style) if self.active_style else None,
            "active_background": asdict(self.active_background) if self.active_background else None,
            "last_audio_path": self.last_audio_path,
            "last_behavior": self.last_behavior.to_dict() if self.last_behavior else None,
            "error": self.last_error,
        }

    # --- Avatar visual payload ------------------------------------------

    def avatar_visual(self) -> dict[str, Any]:
        """Public version of the visual hint sent to the client."""
        return self._avatar_visual_payload()

    def _avatar_visual_payload(self) -> dict[str, Any]:
        if not self.character_index:
            return {
                "canonical": None,
                "active_emote": None,
                "emote_options": [],
                "portrait_kind": "svg_fallback",
            }
        canonical = self.character_index.canonical_image
        canonical_exists = bool(canonical) and Path(canonical).exists()
        canonical_size_ok = canonical_exists and self._is_meaningful_image(Path(canonical))
        emote = self._select_emote_for(self.last_behavior, self.character_index)
        emote_options = [emote.id for emote in self.character_index.emotes[:24]]
        return {
            "canonical": canonical if canonical_size_ok else None,
            "canonical_url": self._asset_url(canonical) if canonical_size_ok else None,
            "active_emote": asdict(emote) if emote else None,
            "active_emote_url": self._asset_url(emote.path) if emote else None,
            "emote_options": emote_options,
            "portrait_kind": "canonical" if canonical_size_ok else "svg_fallback",
        }

    @staticmethod
    def _select_emote_for(
        behavior: AvatarBehaviorState | None, index: CharacterIndex
    ) -> EmoteAsset | None:
        if behavior and behavior.emote_id:
            emote = next((e for e in index.emotes if e.id == behavior.emote_id), None)
            if emote:
                return emote
        if behavior is None:
            return index.find_emote("neutral")
        state_lookup = {
            "listening": "listening",
            "thinking": "thinking",
            "speaking": "happy",
            "idle": "neutral",
            "recovering": "error_recovery",
        }
        state = state_lookup.get(behavior.mode, "neutral")
        return index.find_emote(state) or index.find_emote("neutral")

    def _asset_url(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        # Routes are mounted at /api/character/asset — see apps/demo_server/routes.py.
        encoded = raw_path.replace("\\", "/")
        return f"/api/character/asset?path={encoded}"

    @staticmethod
    def _is_meaningful_image(path: Path) -> bool:
        """Treat tiny generated placeholders as not visually meaningful."""
        try:
            if path.stat().st_size < 96:
                return False
            # We avoid pulling PIL into the renderer; rely on a minimum size.
            return True
        except OSError:
            return False
