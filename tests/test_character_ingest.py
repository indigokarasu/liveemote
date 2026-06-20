from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.character.validate import validate_character_folder


def test_default_character_index_builds():
    warnings = validate_character_folder("character_input")
    assert isinstance(warnings, list)
    index = build_asset_index("character_input")
    assert index.character_id == "indigo"
    assert index.canonical_image.endswith("canonical.png")
    assert index.voice_reference and index.voice_reference.endswith(
        "voice_reference.wav"
    )
    assert any(e.state == "listening" for e in index.emotes)
    assert index.training_references[0].role == "identity_anchor"
    assert index.training_references[0].path.endswith("canonical.png")
    expr_refs = [ref for ref in index.training_references if ref.role == "expression_reference"]
    assert any(ref.state == "listening" for ref in expr_refs)
    listening_refs = [ref for ref in expr_refs if ref.state == "listening"]
    assert listening_refs[0].id == "listening_expression_001"


def test_character_styles_backgrounds_and_workflow_rules_load():
    index = build_asset_index("character_input")
    assert index.default_style_id == "neutral"
    assert index.find_style("cyberpunk").default_background_id == "cyberpunk-city"
    assert (
        index.find_style("cozy").voice.warmth > index.find_style("glitch").voice.warmth
    )
    assert index.find_background("glitch-grid").synced_style_id == "glitch"
    assert any(
        rule.workflow == "debugging" and rule.style_id == "glitch"
        for rule in index.workflow_style_rules
    )


def test_emote_ids_ignore_unsupported_files(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "canonical.png").write_bytes(b"png")
    emote_dir = tmp_path / "emotes" / "happy"
    emote_dir.mkdir(parents=True)
    (emote_dir / "01.png").write_bytes(b"png")
    (emote_dir / "notes.txt").write_text("not an emote")
    (emote_dir / "02.webp").write_bytes(b"webp")

    index = build_asset_index(tmp_path)

    assert [emote.id for emote in index.emotes] == ["happy_001", "happy_002"]


def test_profile_emotes_support_external_paths_and_variants(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "canonical.png").write_bytes(b"png")
    extras = tmp_path / "extras"
    extras.mkdir()
    (extras / "wave.png").write_bytes(b"png")
    (extras / "nod.mp4").write_bytes(b"video")
    outside = tmp_path.parent / "outside_wink.webp"
    outside.write_bytes(b"webp")
    (canonical / "profile.yaml").write_text(
        """
character_id: manifest
emotes:
  - state: greeting
    variants:
      - name: wave
        path: extras/wave.png
        priority: 5
        intensity: 0.2
        tags: [intro]
      - name: nod
        path: extras/nod.mp4
        duration_ms: 1200
        loopable: true
        tags: [ack]
  - id: wink_custom
    state: wink
    variant: playful
    path: ../outside_wink.webp
    priority: 3
    tags: [playful]
"""
    )

    index = build_asset_index(tmp_path)

    assert index.find_emote("greeting").variant == "wave"
    assert index.find_emote("greeting", variant="nod").duration_ms == 1200
    assert index.find_emote("wink").id == "wink_custom"
    assert index.find_emote("wink").path.endswith("outside_wink.webp")
    assert [emote.variant for emote in index.emotes_for("greeting")] == ["wave", "nod"]
    assert index.find_emote("greeting", tags={"intro"}).variant == "wave"
    expr_refs = [ref for ref in index.training_references if ref.role == "expression_reference"]
    assert any(ref.state == "wink" for ref in expr_refs)
    assert not any(
        ref.path.endswith("nod.mp4") for ref in expr_refs
    )


def test_profile_emote_duplicate_ids_are_rejected(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "canonical.png").write_bytes(b"png")
    emotes = tmp_path / "emotes" / "happy"
    emotes.mkdir(parents=True)
    (emotes / "01.png").write_bytes(b"png")
    extra = tmp_path / "extra.png"
    extra.write_bytes(b"png")
    (canonical / "profile.yaml").write_text(
        """
emotes:
  - id: happy_001
    state: happy
    path: extra.png
"""
    )

    try:
        build_asset_index(tmp_path)
    except ValueError as exc:
        assert "Duplicate emote id" in str(exc)
    else:
        raise AssertionError("Expected duplicate profile emote id to be rejected")
