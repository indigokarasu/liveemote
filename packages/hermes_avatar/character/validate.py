from __future__ import annotations
from pathlib import Path
from .asset_index import SUPPORTED_EMOTE_EXTS

class CharacterValidationError(ValueError):
    pass

def validate_character_folder(root: str | Path, require_voice: bool = False) -> list[str]:
    root = Path(root)
    warnings: list[str] = []
    canonical = root / "canonical" / "canonical.png"
    if not canonical.exists():
        raise CharacterValidationError(f"Missing required canonical image: {canonical}")
    voice = root / "canonical" / "voice_reference.wav"
    eleven = root / "canonical" / "elevenlabs_voice_id.txt"
    if require_voice and not voice.exists() and not eleven.exists():
        raise CharacterValidationError("Local voice clone requires canonical/voice_reference.wav or ElevenLabs voice id")
    emotes = root / "emotes"
    if not emotes.exists():
        warnings.append("No emotes directory found; runtime will use canonical still image.")
        return warnings
    for item in emotes.rglob("*"):
        if item.is_file() and item.suffix.lower() not in SUPPORTED_EMOTE_EXTS and item.suffix.lower() != ".wav":
            warnings.append(f"Ignoring unsupported emote file: {item}")
    return warnings
