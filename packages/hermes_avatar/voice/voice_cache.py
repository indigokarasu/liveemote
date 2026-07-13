from __future__ import annotations
from pathlib import Path
import hashlib
from functools import lru_cache


class VoiceCache:
    """Filesystem-backed cache for synthesised voice clips, with an optional
    bounded in-memory byte cache for fast repeat reads.

    The on-disk layout (``path_for``) is unchanged; ``store_bytes``/``read_bytes``
    add a small LRU of decoded audio so repeated playback of the same clip does
    not hit disk.
    """

    def __init__(self, root: str | Path = "cache/voice", memory_cache_size: int = 128) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._bytes_cache: dict[Path, bytes] = {}
        self._mem_max = max(1, memory_cache_size)

    @staticmethod
    @lru_cache(maxsize=256)
    def _digest(backend: str, text: str) -> str:
        return hashlib.sha1(f"{backend}:{text}".encode()).hexdigest()[:16]

    def path_for(self, text: str, backend: str, suffix: str = ".wav") -> Path:
        digest = self._digest(backend, text)
        return self.root / f"{backend}_{digest}{suffix}"

    def _cache_bytes(self, path: Path, data: bytes) -> None:
        self._bytes_cache.pop(path, None)
        self._bytes_cache[path] = data
        while len(self._bytes_cache) > self._mem_max:
            self._bytes_cache.pop(next(iter(self._bytes_cache)))

    def store_bytes(self, text: str, backend: str, data: bytes, suffix: str = ".wav") -> Path:
        path = self.path_for(text, backend, suffix)
        path.write_bytes(data)
        self._cache_bytes(path, data)
        return path

    def read_bytes(self, text: str, backend: str, suffix: str = ".wav") -> bytes | None:
        path = self.path_for(text, backend, suffix)
        # Serve from the in-memory cache first so a previously decoded clip is
        # available even if the on-disk file is gone (or to skip disk I/O).
        cached = self._bytes_cache.get(path)
        if cached is not None:
            return cached
        if path.exists():
            data = path.read_bytes()
            self._cache_bytes(path, data)
            return data
        return None
