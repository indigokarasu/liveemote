from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

SUPPORTED_EMOTE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm"}
SUPPORTED_TRAINING_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class VoiceStyleSpec:
    pace: float = 0.44
    warmth: float = 0.62
    intensity: float = 0.35
    backend: str | None = None
    reference_audio: str | None = None
    elevenlabs_voice_id: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class BackgroundSpec:
    id: str
    name: str
    kind: str = "gradient"
    value: str = "radial-gradient(circle,#374151,#030712)"
    synced_style_id: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class VisualStyle:
    id: str
    name: str
    description: str = ""
    voice: VoiceStyleSpec = field(default_factory=VoiceStyleSpec)
    default_background_id: str | None = None
    workflow_tags: list[str] = field(default_factory=list)
    renderer_prompt: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class WorkflowStyleRule:
    workflow: str
    style_id: str
    background_id: str | None = None
    description: str = ""


@dataclass
class EmoteAsset:
    id: str
    path: str
    state: str
    intensity: float = 0.35
    loopable: bool = True
    duration_ms: int | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TrainingReference:
    id: str
    path: str
    role: str
    state: str | None = None
    weight: float = 1.0
    tags: list[str] = field(default_factory=list)


@dataclass
class CharacterIndex:
    character_id: str
    canonical_image: str
    display_name: str | None = None
    voice_reference: str | None = None
    elevenlabs_voice_id: str | None = None
    emotes: list[EmoteAsset] = field(default_factory=list)
    training_references: list[TrainingReference] = field(default_factory=list)
    styles: list[VisualStyle] = field(default_factory=list)
    backgrounds: list[BackgroundSpec] = field(default_factory=list)
    workflow_style_rules: list[WorkflowStyleRule] = field(default_factory=list)
    default_style_id: str = "neutral"
    default_background_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def find_emote(self, state: str) -> EmoteAsset | None:
        return next((e for e in self.emotes if e.state == state), None)

    def find_style(self, style_id: str | None) -> VisualStyle | None:
        if style_id is None:
            return None
        return next((style for style in self.styles if style.id == style_id), None)

    def find_background(self, background_id: str | None) -> BackgroundSpec | None:
        if background_id is None:
            return None
        return next((bg for bg in self.backgrounds if bg.id == background_id), None)

    def expression_references(
        self, state: str | None = None
    ) -> list[TrainingReference]:
        return [
            ref
            for ref in self.training_references
            if ref.role == "expression_reference"
            and (state is None or ref.state == state)
        ]
