from __future__ import annotations
from pathlib import Path
import hashlib

class VoiceCache:
    def __init__(self, root: str | Path = "cache/voice") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, text: str, backend: str, suffix: str = ".wav") -> Path:
        digest = hashlib.sha1(f"{backend}:{text}".encode()).hexdigest()[:16]
        return self.root / f"{backend}_{digest}{suffix}"
