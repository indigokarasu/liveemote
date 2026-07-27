"""LiveEmote server-side perception.

The avatar reads the human on webcam as a signal (focus + energy), not as a
face-reenactment source. This package exposes the trackers that turn webcam
frames into the affect signals the runtime consumes.
"""

from .mediapipe_tracker import (
    FaceSignals,
    MediaPipeFaceTracker,
    NullFaceTracker,
    build_tracker,
    persist_canvas_jpeg,
)

__all__ = [
    "FaceSignals",
    "MediaPipeFaceTracker",
    "NullFaceTracker",
    "build_tracker",
    "persist_canvas_jpeg",
]
