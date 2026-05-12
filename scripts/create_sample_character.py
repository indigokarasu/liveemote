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
        profile.write_text("""character_id: indigo
name: Indigo
default_style_id: neutral
default_background_id: studio
styles:
  - id: neutral
    name: Neutral
    description: Default clean character presentation.
    default_background_id: studio
    workflow_tags: [general]
    voice:
      pace: 0.44
      warmth: 0.62
      intensity: 0.35
  - id: cyberpunk
    name: Cyberpunk
    description: High-contrast neon treatment intended for coding and technical workflows.
    default_background_id: cyberpunk-city
    workflow_tags: [coding, technical, automation]
    renderer_prompt: neon cyberpunk assistant, crisp high contrast lighting
    tags: [neon, coding]
    voice:
      pace: 0.5
      warmth: 0.5
      intensity: 0.48
      tags: [precise, focused]
  - id: cozy
    name: Cozy
    description: Warm, relaxed treatment intended for long-form writing and ideation.
    default_background_id: cozy-library
    workflow_tags: [writing, book, brainstorming]
    renderer_prompt: cozy warm character lighting, soft library atmosphere
    tags: [warm, writing]
    voice:
      pace: 0.38
      warmth: 0.82
      intensity: 0.28
      tags: [warm, gentle]
  - id: glitch
    name: Glitch
    description: Diagnostic visual treatment for editing, debugging, or system-recovery contexts.
    default_background_id: glitch-grid
    workflow_tags: [debugging, editing, system]
    renderer_prompt: controlled digital glitch effect, diagnostic overlay
    tags: [debug, glitch]
    voice:
      pace: 0.56
      warmth: 0.42
      intensity: 0.62
      tags: [diagnostic]
backgrounds:
  - id: studio
    name: Soft studio
    kind: gradient
    value: radial-gradient(circle,#374151,#030712)
    tags: [neutral]
  - id: cyberpunk-city
    name: Cyberpunk city
    kind: gradient
    value: linear-gradient(135deg,#111827,#312e81 45%,#ec4899)
    synced_style_id: cyberpunk
    tags: [coding, debug]
  - id: cozy-library
    name: Cozy library
    kind: gradient
    value: linear-gradient(135deg,#422006,#92400e 48%,#f59e0b)
    synced_style_id: cozy
    tags: [writing, warm]
  - id: glitch-grid
    name: Glitch grid
    kind: gradient
    value: linear-gradient(90deg,#020617,#164e63 50%,#7f1d1d)
    synced_style_id: glitch
    tags: [debug, system]
workflow_style_rules:
  - workflow: coding
    style_id: cyberpunk
    background_id: cyberpunk-city
    description: Use neon technical styling for coding tasks.
  - workflow: book_writing
    style_id: cozy
    background_id: cozy-library
    description: Use cozy styling for book-writing sessions.
  - workflow: debugging
    style_id: glitch
    background_id: glitch-grid
    description: Use glitch styling while editing or debugging the avatar system.
""")

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
