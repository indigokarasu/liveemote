from __future__ import annotations
from hermes_avatar.affect.smoothing import clamp

def estimate_gaze(face_center: tuple[float, float] | None, max_yaw: float = 12, max_pitch: float = 8) -> dict:
    if not face_center:
        return {"gaze_direction": "away", "head_yaw": 0.0, "head_pitch": 0.0}
    x, y = face_center
    return {"gaze_direction": "toward_user" if abs(x - 0.5) < 0.22 and abs(y - 0.5) < 0.22 else "away", "head_yaw": clamp((0.5 - x) * max_yaw * 2, -max_yaw, max_yaw), "head_pitch": clamp((y - 0.5) * max_pitch * 2, -max_pitch, max_pitch)}
