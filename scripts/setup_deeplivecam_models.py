#!/usr/bin/env python3
"""Download or verify Deep-Live-Cam model files.

The upstream Deep-Live-Cam repository keeps large model weights outside of git.
This helper places the recommended Hugging Face model files where the upstream
runtime expects them: ``vendor/Deep-Live-Cam/models``.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "vendor" / "Deep-Live-Cam" / "models"
HF_RESOLVE_BASE = "https://huggingface.co/hacksider/deep-live-cam/resolve/main"


@dataclass(frozen=True)
class ModelFile:
    """A Deep-Live-Cam model file and its canonical download URL."""

    name: str
    url: str
    description: str


RECOMMENDED_MODELS: tuple[ModelFile, ...] = (
    ModelFile(
        name="inswapper_128_fp16.onnx",
        url=f"{HF_RESOLVE_BASE}/inswapper_128_fp16.onnx",
        description="Face swapper model used by current Deep-Live-Cam builds.",
    ),
    ModelFile(
        name="GFPGANv1.4.onnx",
        url=f"{HF_RESOLVE_BASE}/GFPGANv1.4.onnx",
        description="Face enhancer model used by current Deep-Live-Cam builds.",
    ),
)

LEGACY_MODELS: tuple[ModelFile, ...] = (
    ModelFile(
        name="inswapper_128.onnx",
        url=f"{HF_RESOLVE_BASE}/inswapper_128.onnx",
        description="Legacy full-precision face swapper model.",
    ),
    ModelFile(
        name="GFPGANv1.4.pth",
        url=f"{HF_RESOLVE_BASE}/GFPGANv1.4.pth",
        description="Legacy PyTorch face enhancer model.",
    ),
)


def selected_models(include_legacy: bool = False) -> tuple[ModelFile, ...]:
    """Return the model files this setup invocation should manage."""

    if include_legacy:
        return RECOMMENDED_MODELS + LEGACY_MODELS
    return RECOMMENDED_MODELS


def missing_models(models_dir: Path, models: tuple[ModelFile, ...]) -> list[ModelFile]:
    """Return required model files that are absent or empty."""

    missing: list[ModelFile] = []
    for model in models:
        path = models_dir / model.name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(model)
    return missing


def download_model(model: ModelFile, destination: Path) -> None:
    """Download one model file atomically enough for interrupted retries."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    print(f"Downloading {model.name} -> {destination}")
    try:
        with urllib.request.urlopen(model.url) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except urllib.error.URLError as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"failed to download {model.name} from {model.url}: {exc}") from exc

    temporary.replace(destination)


def ensure_models(models_dir: Path, models: tuple[ModelFile, ...], check_only: bool = False) -> list[ModelFile]:
    """Verify required models exist, downloading missing files when requested."""

    missing = missing_models(models_dir, models)
    if not missing:
        return []
    if check_only:
        return missing

    for model in missing:
        download_model(model, models_dir / model.name)
    return missing_models(models_dir, models)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory that should contain Deep-Live-Cam model files (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify model placement; do not download missing files.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also manage legacy inswapper_128.onnx and GFPGANv1.4.pth files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = selected_models(include_legacy=args.include_legacy)
    if not args.models_dir.parent.exists():
        print(
            f"Deep-Live-Cam checkout not found at {args.models_dir.parent}. "
            "Run `make setup` or clone the upstream repository first.",
            file=sys.stderr,
        )
        return 1

    try:
        unresolved = ensure_models(args.models_dir, models, check_only=args.check_only)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if unresolved:
        print(f"Deep-Live-Cam models missing from {args.models_dir}:", file=sys.stderr)
        for model in unresolved:
            print(f"  - {model.name}: {model.url}", file=sys.stderr)
        if args.check_only:
            print("Run `python scripts/setup_deeplivecam_models.py` to download them.", file=sys.stderr)
        return 1

    print(f"Deep-Live-Cam models are present in {args.models_dir}:")
    for model in models:
        print(f"  - {model.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
