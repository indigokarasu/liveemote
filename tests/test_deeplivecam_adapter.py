from pathlib import Path

from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.renderer.deeplivecam_adapter import DeepLiveCamAdapter
from scripts.create_sample_character import PNG_1X1_RGBA


def test_deeplivecam_degrades_when_backend_absent(tmp_path):
    """The canonical-only character resolves a source face, but with no backend
    binary / models present (the expected CI state) the adapter degrades to a
    transparent passthrough rather than claiming an active swap."""
    character = tmp_path / "canonical_only"
    canonical_dir = character / "canonical"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "canonical.png").write_bytes(PNG_1X1_RGBA)

    index = build_asset_index(character)
    assert [ref for ref in index.training_references if ref.role == "expression_reference"] == []

    adapter = DeepLiveCamAdapter(enabled=True)
    adapter.load_character(index)
    caps = adapter.capabilities()

    # Source face is still resolved from the canonical image.
    assert caps["source_reference_role"] == "identity_anchor"
    assert caps["source_reference_id"] == "identity_anchor_001"
    assert Path(caps["source_image_path"]).name == "canonical.png"
    assert caps["source_image_present"] is True
    # But the backend runtime package is not importable in CI (no cv2 /
    # onnxruntime), so the adapter detects this at activation and degrades to a
    # transparent passthrough rather than spawning a doomed GUI process.
    assert caps["degraded"] is True
    assert caps["passthrough"] is True
    assert caps["online"] is False
    assert caps["replacement_active"] is False
    # Models may be present on disk, but the runtime is not usable -> error set.
    assert caps["error"] is not None


def test_deeplivecam_missing_source_face_reports_absent(tmp_path):
    """When the character has no resolvable source face, that fact is reported
    honestly and the adapter does not crash."""
    character = tmp_path / "no_canonical"
    character.mkdir(parents=True)
    index = build_asset_index(character)

    adapter = DeepLiveCamAdapter(enabled=True)
    adapter.load_character(index)
    caps = adapter.capabilities()

    assert caps["source_image_present"] is False
    assert caps["source_image_path"] is None
    assert caps["degraded"] is True
