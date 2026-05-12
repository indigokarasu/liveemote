from __future__ import annotations
from pathlib import Path
import yaml
from .asset_index import (
    CharacterIndex,
    EmoteAsset,
    SUPPORTED_EMOTE_EXTS,
    SUPPORTED_TRAINING_IMAGE_EXTS,
    TrainingReference,
)
from .emote_tagger import tags_for
from .validate import validate_character_folder

DEFAULT_STATES = ["neutral", "listening", "thinking", "happy", "concerned", "apologetic", "amused", "sad", "error_recovery"]


def _reference_weight(state: str) -> float:
    if state == "neutral":
        return 0.9
    if state in DEFAULT_STATES:
        return 0.75
    return 0.65


def build_asset_index(root: str | Path, character_id: str | None = None) -> CharacterIndex:
    root = Path(root)
    validate_character_folder(root)
    profile_path = root / "canonical" / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text()) if profile_path.exists() else {}
    cid = character_id or profile.get("character_id") or profile.get("name") or root.name
    voice = root / "canonical" / "voice_reference.wav"
    eleven_file = root / "canonical" / "elevenlabs_voice_id.txt"
    eleven = eleven_file.read_text().strip() if eleven_file.exists() else None
    canonical_image = root / "canonical" / "canonical.png"
    index = CharacterIndex(
        character_id=str(cid),
        canonical_image=str(canonical_image),
        voice_reference=str(voice) if voice.exists() else None,
        elevenlabs_voice_id=eleven,
        training_references=[
            TrainingReference(
                id="identity_anchor_001",
                path=str(canonical_image),
                role="identity_anchor",
                state="neutral",
                weight=1.0,
                tags=["canonical", "identity", "neutral"],
            )
        ],
    )
    emote_root = root / "emotes"
    if emote_root.exists():
        for state_dir in sorted(p for p in emote_root.iterdir() if p.is_dir()):
            state = state_dir.name
            expression_ref_idx = 1
            for idx, asset in enumerate(sorted(state_dir.iterdir()), 1):
                if asset.is_file() and asset.suffix.lower() in SUPPORTED_EMOTE_EXTS:
                    tags = tags_for(asset, state)
                    index.emotes.append(EmoteAsset(
                        id=f"{state}_{idx:03d}", path=str(asset), state=state,
                        loopable=asset.suffix.lower() in {".mp4", ".mov", ".webm"},
                        duration_ms=2800 if asset.suffix.lower() in {".mp4", ".mov", ".webm"} else None,
                        tags=tags,
                    ))
                    if asset.suffix.lower() in SUPPORTED_TRAINING_IMAGE_EXTS:
                        index.training_references.append(TrainingReference(
                            id=f"{state}_expression_{expression_ref_idx:03d}",
                            path=str(asset),
                            role="expression_reference",
                            state=state,
                            weight=_reference_weight(state),
                            tags=tags + ["expression", state],
                        ))
                        expression_ref_idx += 1
    return index
