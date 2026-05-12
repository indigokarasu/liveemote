from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.character.validate import validate_character_folder


def test_default_character_index_builds():
    warnings = validate_character_folder("character_input")
    assert isinstance(warnings, list)
    index = build_asset_index("character_input")
    assert index.character_id == "indigo"
    assert index.canonical_image.endswith("canonical.png")
    assert index.voice_reference and index.voice_reference.endswith("voice_reference.wav")
    assert any(e.state == "listening" for e in index.emotes)
    assert index.training_references[0].role == "identity_anchor"
    assert index.training_references[0].path.endswith("canonical.png")
    assert any(ref.state == "listening" for ref in index.expression_references())
    assert index.expression_references("listening")[0].id == "listening_expression_001"
