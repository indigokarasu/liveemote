from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import (
    BackgroundSpec,
    CharacterIndex,
    TrainingReference,
    VisualStyle,
)
from .base import Renderer


class DeepLiveCamAdapter(Renderer):
    """Optional face-replacement adapter for Deep-Live-Cam-style runtimes.

    Deep-Live-Cam only needs one source face image to start a replacement
    session. The canonical character image is therefore a valid source even when
    a character has no emote/expression references. Additional emote stills can
    improve downstream conditioning, but they must not be required for basic
    face replacement.
    """

    def __init__(
        self,
        enabled: bool = False,
        vendor_dir: str = "vendor/Deep-Live-Cam",
    ) -> None:
        self.enabled = enabled
        self.vendor_dir = Path(vendor_dir)
        self.watermark = "Synthetic avatar output - consent required for real identities"
        self.character_index: CharacterIndex | None = None
        self.active_style: VisualStyle | None = None
        self.active_background: BackgroundSpec | None = None
        self.behavior: AvatarBehaviorState | None = None
        self.source_reference: TrainingReference | None = None
        self.source_image_path: str | None = None
        self.replacement_active = False
        self.last_error: str | None = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "deeplivecam",
            "enabled": self.enabled,
            "online": self.replacement_active,
            "replacement_active": self.replacement_active,
            "source_image_path": self.source_image_path,
            "source_reference_id": self.source_reference.id if self.source_reference else None,
            "source_reference_role": self.source_reference.role if self.source_reference else None,
            "canonical_image": self.character_index.canonical_image if self.character_index else None,
            "vendor_dir_exists": self.vendor_dir.exists(),
            "watermark": self.watermark,
            "error": self.last_error,
        }

    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index
        self.source_reference = self._select_source_face(character_index)
        self.source_image_path = self.source_reference.path if self.source_reference else None
        self._activate_face_replacement()

    def set_theme(
        self,
        character_index: CharacterIndex,
        style: VisualStyle | None,
        background: BackgroundSpec | None,
    ) -> None:
        self.character_index = character_index
        self.active_style = style
        self.active_background = background
        self.source_reference = self._select_source_face(character_index)
        self.source_image_path = self.source_reference.path if self.source_reference else None
        self._activate_face_replacement()

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        if behavior.lip_sync_enabled:
            return
        self.behavior = behavior
        self._activate_face_replacement()

    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None:
        self.behavior = behavior
        self._activate_face_replacement()

    def interrupt(self) -> None:
        self.behavior = AvatarBehaviorState(mode="recovering", affect="reset", gaze_target="soft_forward")
        self.replacement_active = False

    def _activate_face_replacement(self) -> None:
        self.last_error = None
        if not self.enabled:
            self.replacement_active = False
            self.last_error = "Deep-Live-Cam renderer is selected but not enabled."
            return
        if self.character_index is None:
            self.replacement_active = False
            self.last_error = "No character loaded."
            return
        if self.source_reference is None:
            self.source_reference = self._select_source_face(self.character_index)
            self.source_image_path = self.source_reference.path if self.source_reference else None
        if not self.source_image_path:
            self.replacement_active = False
            self.last_error = "No source face image found. Expected canonical/canonical.png or an identity_anchor reference."
            return
        if not Path(self.source_image_path).exists():
            self.replacement_active = False
            self.last_error = f"Source face image does not exist: {self.source_image_path}"
            return
        self.replacement_active = True

    def _select_source_face(self, character_index: CharacterIndex) -> TrainingReference | None:
        identity_anchor = next(
            (
                ref
                for ref in character_index.training_references
                if ref.role == "identity_anchor" and Path(ref.path).exists()
            ),
            None,
        )
        if identity_anchor is not None:
            return identity_anchor
        canonical = Path(character_index.canonical_image)
        if canonical.exists():
            return TrainingReference(
                id="canonical_identity_anchor",
                path=str(canonical),
                role="identity_anchor",
                state="neutral",
                weight=1.0,
                tags=["canonical", "identity", "neutral"],
            )
        return None

