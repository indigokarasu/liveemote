from __future__ import annotations
from pathlib import Path
import yaml
from .asset_index import (
    BackgroundSpec,
    CharacterIndex,
    EmoteAsset,
    SUPPORTED_EMOTE_EXTS,
    SUPPORTED_TRAINING_IMAGE_EXTS,
    TrainingReference,
    VisualStyle,
    VoiceStyleSpec,
    WorkflowStyleRule,
)
from .emote_tagger import tags_for
from .validate import validate_character_folder

DEFAULT_STATES = [
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

DEFAULT_BACKGROUNDS = [
    BackgroundSpec(
        id="studio",
        name="Soft studio",
        value="radial-gradient(circle,#374151,#030712)",
        tags=["neutral"],
    ),
    BackgroundSpec(
        id="cyberpunk-city",
        name="Cyberpunk city",
        value="linear-gradient(135deg,#111827,#312e81 45%,#ec4899)",
        synced_style_id="cyberpunk",
        tags=["coding", "debug"],
    ),
    BackgroundSpec(
        id="cozy-library",
        name="Cozy library",
        value="linear-gradient(135deg,#422006,#92400e 48%,#f59e0b)",
        synced_style_id="cozy",
        tags=["writing", "warm"],
    ),
    BackgroundSpec(
        id="glitch-grid",
        name="Glitch grid",
        value="linear-gradient(90deg,#020617,#164e63 50%,#7f1d1d)",
        synced_style_id="glitch",
        tags=["debug", "system"],
    ),
]

DEFAULT_STYLES = [
    VisualStyle(
        id="neutral",
        name="Neutral",
        description="Default clean character presentation.",
        voice=VoiceStyleSpec(),
        default_background_id="studio",
        workflow_tags=["general"],
        tags=["default"],
    ),
    VisualStyle(
        id="cyberpunk",
        name="Cyberpunk",
        description="High-contrast neon treatment intended for coding and technical workflows.",
        voice=VoiceStyleSpec(
            pace=0.5, warmth=0.5, intensity=0.48, tags=["precise", "focused"]
        ),
        default_background_id="cyberpunk-city",
        workflow_tags=["coding", "technical", "automation"],
        renderer_prompt="neon cyberpunk assistant, crisp high contrast lighting",
        tags=["neon", "coding"],
    ),
    VisualStyle(
        id="cozy",
        name="Cozy",
        description="Warm, relaxed treatment intended for long-form writing and ideation.",
        voice=VoiceStyleSpec(
            pace=0.38, warmth=0.82, intensity=0.28, tags=["warm", "gentle"]
        ),
        default_background_id="cozy-library",
        workflow_tags=["writing", "book", "brainstorming"],
        renderer_prompt="cozy warm character lighting, soft library atmosphere",
        tags=["warm", "writing"],
    ),
    VisualStyle(
        id="glitch",
        name="Glitch",
        description="Diagnostic visual treatment for editing, debugging, or system-recovery contexts.",
        voice=VoiceStyleSpec(
            pace=0.56, warmth=0.42, intensity=0.62, tags=["diagnostic"]
        ),
        default_background_id="glitch-grid",
        workflow_tags=["debugging", "editing", "system"],
        renderer_prompt="controlled digital glitch effect, diagnostic overlay",
        tags=["debug", "glitch"],
    ),
]

DEFAULT_WORKFLOW_RULES = [
    WorkflowStyleRule(
        workflow="coding",
        style_id="cyberpunk",
        background_id="cyberpunk-city",
        description="Use neon technical styling for coding tasks.",
    ),
    WorkflowStyleRule(
        workflow="book_writing",
        style_id="cozy",
        background_id="cozy-library",
        description="Use cozy styling for book-writing sessions.",
    ),
    WorkflowStyleRule(
        workflow="debugging",
        style_id="glitch",
        background_id="glitch-grid",
        description="Use glitch styling while editing or debugging the avatar system.",
    ),
]


def _reference_weight(state: str) -> float:
    if state == "neutral":
        return 0.9
    if state in DEFAULT_STATES:
        return 0.75
    return 0.65


def _voice_style_from_dict(data: dict | None) -> VoiceStyleSpec:
    data = data or {}
    return VoiceStyleSpec(
        pace=float(data.get("pace", VoiceStyleSpec.pace)),
        warmth=float(data.get("warmth", VoiceStyleSpec.warmth)),
        intensity=float(data.get("intensity", VoiceStyleSpec.intensity)),
        backend=data.get("backend"),
        reference_audio=data.get("reference_audio"),
        elevenlabs_voice_id=data.get("elevenlabs_voice_id"),
        tags=list(data.get("tags", [])),
    )


def _styles_from_profile(profile: dict) -> list[VisualStyle]:
    styles = profile.get("styles")
    if not styles:
        return list(DEFAULT_STYLES)
    parsed: list[VisualStyle] = []
    for data in styles:
        parsed.append(
            VisualStyle(
                id=data["id"],
                name=data.get("name", data["id"].replace("_", " ").title()),
                description=data.get("description", ""),
                voice=_voice_style_from_dict(data.get("voice")),
                default_background_id=data.get("default_background_id"),
                workflow_tags=list(data.get("workflow_tags", [])),
                renderer_prompt=data.get("renderer_prompt"),
                tags=list(data.get("tags", [])),
            )
        )
    return parsed


def _backgrounds_from_profile(profile: dict) -> list[BackgroundSpec]:
    backgrounds = profile.get("backgrounds")
    if not backgrounds:
        return list(DEFAULT_BACKGROUNDS)
    return [
        BackgroundSpec(
            id=data["id"],
            name=data.get("name", data["id"].replace("_", " ").title()),
            kind=data.get("kind", "gradient"),
            value=data.get("value", "radial-gradient(circle,#374151,#030712)"),
            synced_style_id=data.get("synced_style_id"),
            tags=list(data.get("tags", [])),
        )
        for data in backgrounds
    ]


def _workflow_rules_from_profile(profile: dict) -> list[WorkflowStyleRule]:
    rules = profile.get("workflow_style_rules")
    if not rules:
        return list(DEFAULT_WORKFLOW_RULES)
    return [
        WorkflowStyleRule(
            workflow=data["workflow"],
            style_id=data["style_id"],
            background_id=data.get("background_id"),
            description=data.get("description", ""),
        )
        for data in rules
    ]


def build_asset_index(
    root: str | Path, character_id: str | None = None
) -> CharacterIndex:
    root = Path(root)
    validate_character_folder(root)
    profile_path = root / "canonical" / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text()) if profile_path.exists() else {}
    cid = (
        character_id or profile.get("character_id") or profile.get("name") or root.name
    )
    voice = root / "canonical" / "voice_reference.wav"
    eleven_file = root / "canonical" / "elevenlabs_voice_id.txt"
    eleven = eleven_file.read_text().strip() if eleven_file.exists() else None
    canonical_image = root / "canonical" / "canonical.png"
    styles = _styles_from_profile(profile)
    backgrounds = _backgrounds_from_profile(profile)
    default_style_id = profile.get("default_style_id") or (
        styles[0].id if styles else "neutral"
    )
    default_background_id = profile.get("default_background_id")
    if default_background_id is None:
        default_style = next(
            (style for style in styles if style.id == default_style_id), None
        )
        default_background_id = (
            default_style.default_background_id
            if default_style
            else (backgrounds[0].id if backgrounds else None)
        )
    index = CharacterIndex(
        character_id=str(cid),
        display_name=profile.get("name") or str(cid),
        canonical_image=str(canonical_image),
        voice_reference=str(voice) if voice.exists() else None,
        elevenlabs_voice_id=eleven,
        styles=styles,
        backgrounds=backgrounds,
        workflow_style_rules=_workflow_rules_from_profile(profile),
        default_style_id=default_style_id,
        default_background_id=default_background_id,
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
    video_exts = {".mp4", ".mov", ".webm"}
    if emote_root.exists():
        for state_dir in sorted(p for p in emote_root.iterdir() if p.is_dir()):
            state = state_dir.name
            expression_ref_idx = 1
            emote_idx = 1
            for asset in sorted(state_dir.iterdir()):
                suffix = asset.suffix.lower()
                if not asset.is_file() or suffix not in SUPPORTED_EMOTE_EXTS:
                    continue

                tags = tags_for(asset, state)
                is_video = suffix in video_exts
                index.emotes.append(
                    EmoteAsset(
                        id=f"{state}_{emote_idx:03d}",
                        path=str(asset),
                        state=state,
                        loopable=is_video,
                        duration_ms=2800 if is_video else None,
                        tags=tags,
                    )
                )
                emote_idx += 1

                if suffix in SUPPORTED_TRAINING_IMAGE_EXTS:
                    index.training_references.append(
                        TrainingReference(
                            id=f"{state}_expression_{expression_ref_idx:03d}",
                            path=str(asset),
                            role="expression_reference",
                            state=state,
                            weight=_reference_weight(state),
                            tags=tags + ["expression", state],
                        )
                    )
                    expression_ref_idx += 1
    return index
