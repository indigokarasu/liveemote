from pathlib import Path

from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.renderer.deeplivecam_adapter import DeepLiveCamAdapter
from scripts.create_sample_character import PNG_1X1_RGBA


def test_deeplivecam_canonical_only_character_activates_replacement(tmp_path):
    character = tmp_path / "canonical_only"
    canonical_dir = character / "canonical"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "canonical.png").write_bytes(PNG_1X1_RGBA)

    index = build_asset_index(character)
    assert [ref for ref in index.training_references if ref.role == "expression_reference"] == []

    adapter = DeepLiveCamAdapter(enabled=True)
    adapter.load_character(index)
    caps = adapter.capabilities()

    assert caps["replacement_active"] is True
    assert caps["source_reference_role"] == "identity_anchor"
    assert caps["source_reference_id"] == "identity_anchor_001"
    assert Path(caps["source_image_path"]).name == "canonical.png"
    assert caps["error"] is None
