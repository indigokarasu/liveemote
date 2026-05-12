from __future__ import annotations
from pathlib import Path
import yaml
from .asset_index import (
    BackgroundSpec,
    CharacterIndex,
    EmoteAsset,
    SUPPORTED_EMOTE_EXTS,
    SUPPORTED_TRAINING_IMAGE_EXTS,
    SUPPORTED_VIDEO_EMOTE_EXTS,
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


def _state_counters(index: CharacterIndex) -> dict[str, int]:
    counters: dict[str, int] = {}
    for emote in index.emotes:
        prefix = f"{emote.state}_"
        if emote.id.startswith(prefix) and emote.id[len(prefix) :].isdigit():
            counters[emote.state] = max(
                counters.get(emote.state, 0), int(emote.id[len(prefix) :])
            )
    return counters


def _next_emote_id(state: str, counters: dict[str, int], used_ids: set[str]) -> str:
    count = counters.get(state, 0)
    while True:
        count += 1
        emote_id = f"{state}_{count:03d}"
        if emote_id not in used_ids:
            counters[state] = count
            return emote_id


def _append_emote(
    index: CharacterIndex,
    path: Path,
    state: str,
    counters: dict[str, int],
    used_ids: set[str],
    expression_ref_counters: dict[str, int],
    *,
    emote_id: str | None = None,
    variant: str | None = None,
    intensity: float = 0.35,
    loopable: bool | None = None,
    duration_ms: int | None = None,
    priority: int = 0,
    tags: list[str] | None = None,
    training_reference: bool | None = None,
) -> None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EMOTE_EXTS:
        raise ValueError(f"Unsupported emote file type for {path}: {suffix}")
    if emote_id is None:
        emote_id = _next_emote_id(state, counters, used_ids)
    elif emote_id in used_ids:
        raise ValueError(f"Duplicate emote id in character profile: {emote_id}")

    base_tags = tags_for(path, state)
    if variant:
        base_tags.append(variant)
    merged_tags = list(dict.fromkeys([*base_tags, *(tags or [])]))
    is_video = suffix in SUPPORTED_VIDEO_EMOTE_EXTS
    index.emotes.append(
        EmoteAsset(
            id=emote_id,
            path=str(path),
            state=state,
            variant=variant,
            intensity=intensity,
            loopable=is_video if loopable is None else loopable,
            duration_ms=2800 if is_video and duration_ms is None else duration_ms,
            priority=priority,
            tags=merged_tags,
        )
    )
    used_ids.add(emote_id)

    should_add_training_reference = (
        suffix in SUPPORTED_TRAINING_IMAGE_EXTS
        if training_reference is None
        else training_reference
    )
    if should_add_training_reference and suffix in SUPPORTED_TRAINING_IMAGE_EXTS:
        expression_ref_counters[state] = expression_ref_counters.get(state, 0) + 1
        index.training_references.append(
            TrainingReference(
                id=f"{state}_expression_{expression_ref_counters[state]:03d}",
                path=str(path),
                role="expression_reference",
                state=state,
                weight=_reference_weight(state),
                tags=merged_tags + ["expression", state],
            )
        )


def _profile_emote_entries(profile: dict) -> list[dict]:
    entries: list[dict] = []
    for data in profile.get("emotes", []):
        variants = data.get("variants")
        if not variants:
            entries.append(data)
            continue

        shared = {key: value for key, value in data.items() if key != "variants"}
        for variant_data in variants:
            merged = {**shared, **variant_data}
            merged["state"] = variant_data.get("state", shared.get("state"))
            merged["variant"] = (
                variant_data.get("variant")
                or variant_data.get("name")
                or shared.get("variant")
            )
            entries.append(merged)
    return entries


def _resolve_profile_emote_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return path


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
    counters = _state_counters(index)
    used_ids = {emote.id for emote in index.emotes}
    expression_ref_counters: dict[str, int] = {}

    emote_root = root / "emotes"
    video_exts = {".mp4", ".mov", ".webm"}
    if emote_root.exists():
        for state_dir in sorted(p for p in emote_root.iterdir() if p.is_dir()):
            state = state_dir.name
            for asset in sorted(state_dir.iterdir()):
                if (
                    not asset.is_file()
                    or asset.suffix.lower() not in SUPPORTED_EMOTE_EXTS
                ):
                    continue
                _append_emote(
                    index,
                    asset,
                    state,
                    counters,
                    used_ids,
                    expression_ref_counters,
                )

    for data in _profile_emote_entries(profile):
        state = data.get("state")
        raw_path = data.get("path") or data.get("file")
        if not state or not raw_path:
            raise ValueError("Profile emotes require both 'state' and 'path'")
        asset = _resolve_profile_emote_path(root, raw_path)
        if not asset.is_file():
            raise ValueError(f"Profile emote file does not exist: {asset}")
        _append_emote(
            index,
            asset,
            str(state),
            counters,
            used_ids,
            expression_ref_counters,
            emote_id=data.get("id"),
            variant=data.get("variant"),
            intensity=float(data.get("intensity", 0.35)),
            loopable=data.get("loopable"),
            duration_ms=data.get("duration_ms"),
            priority=int(data.get("priority", 0)),
            tags=list(data.get("tags", [])),
            training_reference=data.get("training_reference"),
        )
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
