from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

SUPPORTED_EMOTE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm"}
SUPPORTED_TRAINING_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


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
    voice_reference: str | None = None
    elevenlabs_voice_id: str | None = None
    emotes: list[EmoteAsset] = field(default_factory=list)
    training_references: list[TrainingReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["emotes"] = [asdict(e) for e in self.emotes]
        data["training_references"] = [asdict(ref) for ref in self.training_references]
        return data

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def find_emote(self, state: str) -> EmoteAsset | None:
        return next((e for e in self.emotes if e.state == state), None)

    def expression_references(self, state: str | None = None) -> list[TrainingReference]:
        return [
            ref
            for ref in self.training_references
            if ref.role == "expression_reference" and (state is None or ref.state == state)
        ]
