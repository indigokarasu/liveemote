#!/usr/bin/env python3
"""Create tiny local demo character assets without storing binaries in git."""
from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

PNG_1X1_RGBA = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100ffff03000006000557bfab00000000"
    "49454e44ae426082"
)
EMOTE_STATES = [
    "neutral",
    "listening",
    "thinking",
    "happy",
    "concerned",
    "apologetic",
    "amused",
    "sad",
    "error_recovery",
]


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        for i in range(16000 // 5):
            sample = int(3000 * math.sin(2 * math.pi * 220 * i / 16000))
            wf.writeframes(struct.pack("<h", sample))


def create_sample_character(root: Path, overwrite: bool = False) -> None:
    canonical_dir = root / "canonical"
    emote_root = root / "emotes"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    profile = canonical_dir / "profile.yaml"
    if overwrite or not profile.exists():
        profile.write_text("character_id: indigo\nname: Indigo\n")

    canonical_png = canonical_dir / "canonical.png"
    if overwrite or not canonical_png.exists():
        canonical_png.write_bytes(PNG_1X1_RGBA)

    voice_reference = canonical_dir / "voice_reference.wav"
    if overwrite or not voice_reference.exists():
        write_wav(voice_reference)

    for state in EMOTE_STATES:
        path = emote_root / state / f"{state}_001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not path.exists():
            path.write_bytes(PNG_1X1_RGBA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", default="./character_input", help="Character folder to create/update")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing generated assets")
    args = parser.parse_args()
    create_sample_character(Path(args.character), args.overwrite)
    print(f"Sample character assets ready at {args.character}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
