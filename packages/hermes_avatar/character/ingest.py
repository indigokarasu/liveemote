from __future__ import annotations
from pathlib import Path
import yaml
from .asset_index import CharacterIndex, EmoteAsset, SUPPORTED_EMOTE_EXTS
from .emote_tagger import tags_for
from .validate import validate_character_folder

DEFAULT_STATES = ["neutral", "listening", "thinking", "happy", "concerned", "apologetic", "amused", "sad", "error_recovery"]

def build_asset_index(root: str | Path, character_id: str | None = None) -> CharacterIndex:
    root = Path(root)
    validate_character_folder(root)
    profile_path = root / "canonical" / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text()) if profile_path.exists() else {}
    cid = character_id or profile.get("character_id") or profile.get("name") or root.name
    voice = root / "canonical" / "voice_reference.wav"
    eleven_file = root / "canonical" / "elevenlabs_voice_id.txt"
    eleven = eleven_file.read_text().strip() if eleven_file.exists() else None
    index = CharacterIndex(
        character_id=str(cid),
        canonical_image=str(root / "canonical" / "canonical.png"),
        voice_reference=str(voice) if voice.exists() else None,
        elevenlabs_voice_id=eleven,
    )
    emote_root = root / "emotes"
    if emote_root.exists():
        for state_dir in sorted(p for p in emote_root.iterdir() if p.is_dir()):
            state = state_dir.name
            for idx, asset in enumerate(sorted(state_dir.iterdir()), 1):
                if asset.is_file() and asset.suffix.lower() in SUPPORTED_EMOTE_EXTS:
                    index.emotes.append(EmoteAsset(
                        id=f"{state}_{idx:03d}", path=str(asset), state=state,
                        loopable=asset.suffix.lower() in {".mp4", ".mov", ".webm"},
                        duration_ms=2800 if asset.suffix.lower() in {".mp4", ".mov", ".webm"} else None,
                        tags=tags_for(asset, state),
                    ))
    return index
