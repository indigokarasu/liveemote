from pathlib import Path

from scripts.setup_deeplivecam_models import missing_models, selected_models


def test_missing_models_reports_absent_or_empty_files(tmp_path: Path):
    models = selected_models()
    (tmp_path / models[0].name).write_bytes(b"not empty")
    (tmp_path / models[1].name).touch()

    missing = missing_models(tmp_path, models)

    assert [model.name for model in missing] == [models[1].name]


def test_selected_models_can_include_legacy_files():
    model_names = [model.name for model in selected_models(include_legacy=True)]

    assert "inswapper_128_fp16.onnx" in model_names
    assert "GFPGANv1.4.onnx" in model_names
    assert "inswapper_128.onnx" in model_names
    assert "GFPGANv1.4.pth" in model_names
