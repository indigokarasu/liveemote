from __future__ import annotations
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import BackgroundSpec, CharacterIndex, VisualStyle
from .base import Renderer

class DeepLiveCamAdapter(Renderer):
    """Optional puppeting experiment; off by default and never overrides TTS lip-sync."""
    def __init__(self, enabled: bool = False, allow_identity_output: bool = False) -> None:
        self.enabled = enabled
        self.allow_identity_output = allow_identity_output
        self.watermark = "Synthetic avatar output - consent required for real identities"

    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index

    def set_theme(self, character_index: CharacterIndex, style: VisualStyle | None, background: BackgroundSpec | None) -> None:
        self.character_index = character_index
        self.active_style = style
        self.active_background = background

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        if behavior.lip_sync_enabled:
            return
        self.behavior = behavior

    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None:
        self.behavior = behavior

    def interrupt(self) -> None:
        self.behavior = AvatarBehaviorState(mode="recovering", affect="reset", gaze_target="soft_forward")
