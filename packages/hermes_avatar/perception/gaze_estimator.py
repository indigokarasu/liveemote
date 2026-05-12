from __future__ import annotations

from hermes_avatar.affect.smoothing import clamp


def estimate_gaze(face_center: tuple[float, float] | None, max_yaw: float = 12, max_pitch: float = 8, eye_vector: tuple[float, float] | None = None) -> dict:
    if not face_center:
        return {"gaze_direction": "away", "head_yaw": 0.0, "head_pitch": 0.0, "gaze_confidence": 0.0}
    x, y = face_center
    eye_x, eye_y = eye_vector or (0.0, 0.0)
    yaw = clamp((0.5 - x) * max_yaw * 2 + eye_x * max_yaw, -max_yaw, max_yaw)
    pitch = clamp((y - 0.5) * max_pitch * 2 + eye_y * max_pitch, -max_pitch, max_pitch)
    centered = abs(x - 0.5) < 0.22 and abs(y - 0.5) < 0.22 and abs(eye_x) < 0.35 and abs(eye_y) < 0.35
    confidence = clamp(1.0 - (abs(x - 0.5) + abs(y - 0.5)) * 1.4, 0.15, 0.95)
    return {"gaze_direction": "toward_user" if centered else "away", "head_yaw": yaw, "head_pitch": pitch, "gaze_confidence": confidence}
