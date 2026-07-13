from __future__ import annotations
from pathlib import Path
from .asset_index import SUPPORTED_EMOTE_EXTS

class CharacterValidationError(ValueError):
    pass

def validate_character_folder(root: str | Path) -> list[str]:
    root = Path(root)
    warnings: list[str] = []
    canonical = root / "canonical" / "canonical.png"
    if not canonical.exists():
        warnings.append(f"Missing canonical image: {canonical}")
        return warnings
    emotes = root / "emotes"
    if not emotes.exists():
        warnings.append("No emotes directory found; runtime will use canonical still image.")
        return warnings
    for item in emotes.rglob("*"):
        if item.is_file() and item.suffix.lower() not in SUPPORTED_EMOTE_EXTS and item.suffix.lower() != ".wav":
            warnings.append(f"Ignoring unsupported emote file: {item}")
    return warnings
