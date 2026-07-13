from pathlib import Path

from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.renderer.deeplivecam_adapter import DeepLiveCamAdapter
from hermes_avatar.renderer.facefusion_adapter import FaceSwapAdapter
from scripts.create_sample_character import PNG_1X1_RGBA


def _build_canonical_only_character(tmp_path: Path):
    character = tmp_path / "canonical_only"
    canonical_dir = character / "canonical"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "canonical.png").write_bytes(PNG_1X1_RGBA)
    return build_asset_index(character)


def test_deeplivecam_canonical_only_degrades_gracefully_without_backend(tmp_path):
    """When no face-swap backend/models are present, the adapter must NOT fake
    success: it reports an honest ``replacement_active=False`` with a truthful
    error while still resolving the source face for when a backend is present."""
    index = _build_canonical_only_character(tmp_path)
    # Sanity: build_asset_index synthesizes an identity anchor from canonical.png.
    anchors = [ref for ref in index.training_references if ref.role == "identity_anchor"]
    assert anchors, "expected an identity_anchor reference from canonical.png"

    adapter = DeepLiveCamAdapter(enabled=True)
    adapter.load_character(index)
    caps = adapter.capabilities()

    # Honest degradation: no backend/models in this environment.
    assert caps["replacement_active"] is False
    assert caps["backend_available"] is False
    assert caps["models_available"] is False
    assert "backend" in (caps["error"] or "").lower()

    # Source selection still works so the swap is ready when a backend appears.
    assert caps["source_reference_role"] == "identity_anchor"
    assert caps["source_reference_id"] == "identity_anchor_001"
    assert Path(caps["source_image_path"]).name == "canonical.png"
    assert caps["error"] is not None


def test_deeplivecam_backward_compatible_signature_and_subclass():
    """DeepLiveCamAdapter keeps its historical ``(enabled, vendor_dir)``
    constructor and is a genuine FaceSwapAdapter subclass."""
    adapter = DeepLiveCamAdapter(enabled=True, vendor_dir="vendor/Deep-Live-Cam")
    assert isinstance(adapter, FaceSwapAdapter)
    assert issubclass(DeepLiveCamAdapter, FaceSwapAdapter)
    assert adapter.enabled is True
    assert adapter.vendor_dir == Path("vendor/Deep-Live-Cam")
    assert adapter.backend_label == "deeplivecam"
