from __future__ import annotations

class VirtualCameraOutput:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import pyvirtualcam  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install pyvirtualcam and a platform virtual camera device to enable virtualcam output") from exc
