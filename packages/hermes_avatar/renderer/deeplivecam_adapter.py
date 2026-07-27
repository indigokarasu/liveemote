from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import (
    BackgroundSpec,
    CharacterIndex,
    TrainingReference,
    VisualStyle,
)
from .base import Renderer
from .facefusion_adapter import FaceSwapAdapter, FaceSwapConfig

logger = __import__("logging").getLogger(__name__)


class DeepLiveCamAdapter(FaceSwapAdapter):
    """Backward-compatible face-swap adapter targeting the vendored Deep-Live-Cam runtime.

    This is a thin specialization of :class:`FaceSwapAdapter` that pins the
    backend to ``deeplivecam`` and the default vendor directory to
    ``vendor/Deep-Live-Cam``. Deep-Live-Cam only needs one source face image to
    start a replacement session, so the canonical character image is a valid
    source even when a character has no emote/expression references.

    The class is preserved (name, constructor signature, and capability keys) so
    existing call sites and tests keep working, while the real swap pipeline,
    backend lifecycle management, and graceful degradation now live in
    :class:`FaceSwapAdapter`.
    """

    def __init__(
        self,
        enabled: bool = False,
        vendor_dir: str = "vendor/Deep-Live-Cam",
        config: FaceSwapConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            backend="deeplivecam",
            enabled=enabled,
            vendor_dir=vendor_dir,
            **kwargs,
        )

    # The Renderer abstract methods, capability surface, and graceful
    # degradation are all inherited from FaceSwapAdapter. We re-expose the
    # base class explicitly for clarity / static analysis.
    load_character = FaceSwapAdapter.load_character
    set_theme = FaceSwapAdapter.set_theme
    set_behavior = FaceSwapAdapter.set_behavior
    speak = FaceSwapAdapter.speak
    interrupt = FaceSwapAdapter.interrupt
    capabilities = FaceSwapAdapter.capabilities
    health = FaceSwapAdapter.health
