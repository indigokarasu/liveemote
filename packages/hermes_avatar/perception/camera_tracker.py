from __future__ import annotations
import time
from .gaze_estimator import estimate_gaze

class CameraTracker:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
    def frames(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python for webcam capture") from exc
        cap = cv2.VideoCapture(self.camera_index)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            # Lightweight fallback: assume centered face until MediaPipe detector is wired.
            center = (0.5, 0.45)
            gaze = estimate_gaze(center)
            yield {"type": "perception.frame", "timestamp_ms": int(time.time()*1000), "face_detected": True, "face_center": center, "head_yaw": gaze["head_yaw"], "head_pitch": gaze["head_pitch"], "expression": {"smile": 0.0, "frown": 0.0, "brow_raise": 0.0, "eye_open": 0.7}}
        cap.release()
