"""Server-side webcam perception.

This module converts raw webcam frames into the perceptually grounded affect
signals the LiveEmote avatar actually animates from:

    * **focus / attention** — eye aspect ratio (EAR), gaze centrality, blink
      cadence (face-mesh landmark topology);
    * **energy / arousal** — short-window variance of head pose and mouth-open
      amplitude;
    * **valence** — smile curvature (mouth-width vs upper-lip height) and brow
      distance;
    * **tension** — brow-furrow distance and jaw tension proxies.

MediaPipe Face Landmarker (tasks API) is the chosen tracker because:

    1. it is already an optional extra in ``pyproject.toml`` ([perception]);
    2. it ships a 478-landmark face topology that produces stable
       EAR / head-pose signals;
    3. it is optional — if it or its model file isn't available,
       :class:`NullFaceTracker` provides a no-op fallback so the package
       still imports and the demo stays runnable.

The tracker is intentionally **stateless across frames** — it returns a small
JSON-serialisable payload that the :class:`AffectRuntime` consumes via its
existing ``perception.frame`` event contract. Smoothing lives in the runtime
(EMA), not the tracker, so a noisy frame can't dominate downstream policy
decisions. The browser continues to publish its own VAD events for speech
energy/rate, which the runtime merges.

The sensor frames themselves are NEVER replayed — only the signals. The avatar's
own face image and emotes drive the visual output; the user's webcam never
appears in the avatar.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceSignals:
    """Affect-relevant signals derived from one or more frames."""

    face_detected: bool = False
    attention: float = 0.0          # 0…1  (gaze centrality + ear)
    valence: float = 0.0            # -1…1 (smile proxy)
    arousal: float = 0.0            # 0…1  (pose variance, mouth-open)
    tension: float = 0.0            # 0…1  (brow-furrow distance)
    dominant_expression: str = "neutral"
    gaze_direction: str = "unknown"
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_pose_variance: float = 0.0
    eye_aspect_ratio: float = 0.0
    smile_ratio: float = 0.0
    blink_rate: float = 0.0
    mouth_open: float = 0.0
    emotion_confidence: float = 0.0
    gaze_confidence: float = 0.0
    last_updated_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_detected": self.face_detected,
            "attention": self.attention,
            "valence": self.valence,
            "arousal": self.arousal,
            "tension": self.tension,
            "dominant_expression": self.dominant_expression,
            "gaze_direction": self.gaze_direction,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "gaze_confidence": self.gaze_confidence,
            "emotion_confidence": self.emotion_confidence,
            "expression": {
                "smile": self.smile_ratio,
                "frown": self.tension,
                "brow_raise": max(0.0, 0.6 - self.tension),
                "eye_open": self.eye_aspect_ratio,
                "mouth_open": self.mouth_open,
            },
            "head_pose_variance": self.head_pose_variance,
            "blink_rate": self.blink_rate,
            "timestamp_ms": self.last_updated_ms,
        }


@dataclass
class NullFaceTracker:
    """Fallback tracker; produces empty signals so the runtime stays runnable."""

    def __init__(self) -> None:
        self.last_signals: FaceSignals = FaceSignals()

    def process_frame(self, *_args: Any, **_kwargs: Any) -> FaceSignals:
        timestamp_ms = _kwargs.get("timestamp_ms", 0) if _kwargs else 0
        if not isinstance(timestamp_ms, int):
            try:
                timestamp_ms = int(timestamp_ms)
            except (TypeError, ValueError):
                timestamp_ms = 0
        return FaceSignals(last_updated_ms=timestamp_ms)

    def process_bgr(self, _bgr_frame: Any = None, timestamp_ms: int = 0) -> FaceSignals:
        """No-op BGR path for API parity with MediaPipeFaceTracker."""
        return FaceSignals(last_updated_ms=timestamp_ms)

    def is_available(self) -> bool:
        return False

    def kind(self) -> str:
        return "null"


# Default search paths for the FaceLandmarker model file.
_DEFAULT_MODEL_PATHS = [
    os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", ""),
    "/tmp/face_landmarker.task",
    str(Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"),
    "face_landmarker.task",
]


def _resolve_model_path(explicit: str | None) -> str | None:
    """Return the first existing model path, or *explicit* if truthy."""
    if explicit:
        return explicit
    for cand in _DEFAULT_MODEL_PATHS:
        if cand and Path(cand).exists():
            return cand
    return None


class MediaPipeFaceTracker:
    """MediaPipe Face Landmarker with rolling buffers for energy (variance).

    Uses the modern ``mediapipe.tasks.vision.FaceLandmarker`` (tasks API).
    Falls back to :class:`NullFaceTracker` if mediapipe / opencv / the model
    file are missing. The wrapper never raises into the demo's routing layer.
    """

    def __init__(
        self,
        window: int = 30,
        max_faces: int = 1,
        model_path: str | None = None,
    ) -> None:
        self.window = max(1, window)
        self._yaw_buffer: list[float] = []
        self._pitch_buffer: list[float] = []
        self._ear_buffer: list[float] = []
        self._blink_window: list[float] = []
        self.last_signals = FaceSignals()
        self._impl, self._backend, self._image_module = self._construct_impl(
            max_faces, model_path
        )

    @staticmethod
    def _construct_impl(
        max_faces: int, model_path: str | None
    ) -> tuple[Any, str, Any | None]:
        """Try to build a FaceLandmarker (tasks API)."""
        try:
            import cv2  # noqa: F401
            from mediapipe.tasks.python.core import base_options as base_options_mod
            from mediapipe.tasks.python.vision import face_landmarker
            from mediapipe.tasks.python.vision.core import (
                image as image_mod,
                vision_task_running_mode as running_mode_mod,
            )
        except Exception as exc:
            log.info("mediapipe/opencv not available; face tracker disabled (%s)", exc)
            return None, "unavailable", None

        resolved = _resolve_model_path(model_path)
        if not resolved:
            log.info(
                "FaceLandmarker model not found (paths: %s); tracker disabled",
                _DEFAULT_MODEL_PATHS,
            )
            return None, "model_missing", None

        try:
            opts = face_landmarker.FaceLandmarkerOptions(
                base_options=base_options_mod.BaseOptions(
                    model_asset_path=resolved
                ),
                running_mode=running_mode_mod.VisionTaskRunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=0.4,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            landmarker = face_landmarker.FaceLandmarker.create_from_options(opts)
            return landmarker, "mediapipe", image_mod
        except Exception as exc:
            log.warning("mediapipe FaceLandmarker failed to initialise: %s", exc)
            return None, "init_failed", None

    def is_available(self) -> bool:
        return self._backend == "mediapipe" and self._impl is not None

    def kind(self) -> str:
        return self._backend

    # --- Public API ------------------------------------------------------

    def process_frame(
        self, jpeg_b64: str | bytes | None, timestamp_ms: int
    ) -> FaceSignals:
        """Decode a JPEG frame and emit affect signals.

        *None* / empty input → empty signals.
        """
        if not jpeg_b64 or not self.is_available():
            return self._empty_signals(timestamp_ms=timestamp_ms)

        try:
            import cv2
            import numpy as np
        except Exception:
            return self._empty_signals(timestamp_ms=timestamp_ms)

        try:
            img_b = self._decode_jpeg(jpeg_b64)
        except Exception as exc:
            log.debug("jpeg decode failed for perception frame: %s", exc)
            return self._empty_signals(timestamp_ms=timestamp_ms)
        if img_b is None:
            return self._empty_signals(timestamp_ms=timestamp_ms)

        return self._process_bgr(img_b, timestamp_ms)

    def process_bgr(self, bgr_frame: Any, timestamp_ms: int) -> FaceSignals:
        """Accept an OpenCV BGR frame (numpy array)."""
        if not self.is_available() or bgr_frame is None:
            return self._empty_signals(timestamp_ms=timestamp_ms)
        return self._process_bgr(bgr_frame, timestamp_ms)

    # --- Internal: frame → signals -------------------------------------

    def _process_bgr(self, bgr_frame: Any, timestamp_ms: int) -> FaceSignals:
        try:
            mp_image = self._image_module.Image(
                image_format=self._image_module.ImageFormat.SRGB,
                data=bgr_frame,
            )
        except Exception as exc:
            log.debug("mp.Image creation failed: %s", exc)
            return self._empty_signals(timestamp_ms=timestamp_ms)

        try:
            result = self._impl.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            log.debug("mediapipe detect_for_video failed: %s", exc)
            return self._empty_signals(timestamp_ms=timestamp_ms)

        signals = self._signals_from_results(result, timestamp_ms=timestamp_ms)
        self.last_signals = signals
        return signals

    # --- Internal: frame decode ----------------------------------------

    @staticmethod
    def _decode_jpeg(jpeg_b64: str | bytes) -> Any | None:
        if isinstance(jpeg_b64, str):
            payload = jpeg_b64.split(",", 1)[-1]
            try:
                raw = base64.b64decode(payload)
            except Exception:
                return None
        else:
            raw = bytes(jpeg_b64)
        try:
            import cv2
            import numpy as np
        except Exception:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == 0:
            return None
        try:
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    # --- Internal: signals from landmarks -------------------------------

    def _signals_from_results(
        self, result: Any, timestamp_ms: int
    ) -> FaceSignals:
        try:
            lm_sets = result.face_landmarks or []
        except Exception:
            lm_sets = []
        if not lm_sets:
            return self._empty_signals(timestamp_ms=timestamp_ms)

        # FaceLandmarker result.face_landmarks[i] is a list of NormalizedLandmark
        # objects (478 landmarks, same topology as the legacy 468-face + 10 iris).
        landmarks = lm_sets[0]

        # Compute signals
        left_ear = self._eye_aspect_ratio(landmarks, side="left")
        right_ear = self._eye_aspect_ratio(landmarks, side="right")
        ear = (left_ear + right_ear) / 2
        smile = self._smile_proxy(landmarks)
        brow_furrow = self._brow_furrow_distance(landmarks)
        mouth_open = self._mouth_open(landmarks)
        head_yaw, head_pitch = self._head_pose_proxy(landmarks)

        # Roll buffers for energy / blink
        self._yaw_buffer.append(head_yaw)
        self._pitch_buffer.append(head_pitch)
        self._ear_buffer.append(ear)
        if len(self._yaw_buffer) > self.window:
            self._yaw_buffer.pop(0)
            self._pitch_buffer.pop(0)
            self._ear_buffer.pop(0)
        head_var = self._variance(self._yaw_buffer + self._pitch_buffer)
        blink_rate = self._blink_rate(ear)

        # Translate to UserAffectState-aligned signals
        attention = self._attention_from(ear, head_yaw, head_pitch)
        valence = self._clamp(smile - 0.45, -1.0, 1.0)
        arousal = self._clamp(
            0.4 * blink_rate + 0.6 * head_var + 0.4 * mouth_open, 0.0, 1.0
        )
        tension = self._clamp(brow_furrow, 0.0, 1.0)
        dominant_expression = self._dominant_expression(valence, arousal, tension)
        gaze = self._gaze_label(head_yaw, head_pitch)
        gaze_confidence = 0.85 if ear > 0.18 else 0.4
        emotion_confidence = self._clamp(0.4 + 0.6 * attention, 0.0, 1.0)

        return FaceSignals(
            face_detected=True,
            attention=attention,
            valence=valence,
            arousal=arousal,
            tension=tension,
            dominant_expression=dominant_expression,
            gaze_direction=gaze,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_pose_variance=head_var,
            eye_aspect_ratio=ear,
            smile_ratio=smile,
            blink_rate=blink_rate,
            mouth_open=mouth_open,
            emotion_confidence=emotion_confidence,
            gaze_confidence=gaze_confidence,
            last_updated_ms=timestamp_ms,
        )

    def _empty_signals(self, timestamp_ms: int) -> FaceSignals:
        return FaceSignals(
            face_detected=False,
            attention=0.0,
            emotion_confidence=0.0,
            gaze_confidence=0.0,
            last_updated_ms=timestamp_ms,
        )

    # --- Geometry helpers -----------------------------------------------

    # Landmark indices from the MediaPipe face mesh topology (identical
    # between legacy solutions API and tasks FaceLandmarker).
    #
    # Left eye:     33 (outer corner), 133 (inner corner),
    #                159 (upper mid), 145 (lower mid),
    #                158 (upper-inner), 144 (lower-inner)
    # Right eye:    362 (outer corner), 263 (inner corner),
    #                386 (upper mid), 374 (lower mid),
    #                385 (upper-inner), 373 (lower-inner)
    # Mouth:        61 (left corner), 291 (right corner),
    #                13 (upper lip), 14 (lower lip)
    # Eyebrows:     107 (left inner brow), 336 (right inner brow)
    # Nose/pose:    1 (nose tip), 234 (left ear), 454 (right ear)

    @staticmethod
    def _eye_aspect_ratio(landmarks: Any, side: str) -> float:
        """Eye Aspect Ratio from vertical eyelid distance / horizontal width."""
        try:
            if side == "left":
                p_outer = landmarks[33]
                p_inner = landmarks[133]
                p_up = landmarks[159]
                p_up2 = landmarks[158]
                p_down = landmarks[145]
                p_down2 = landmarks[144]
            else:
                p_outer = landmarks[362]
                p_inner = landmarks[263]
                p_up = landmarks[386]
                p_up2 = landmarks[385]
                p_down = landmarks[374]
                p_down2 = landmarks[373]
            horiz = max(abs(p_outer.x - p_inner.x), 1e-6)
            vert_upper = abs(p_up.y - p_down.y)
            vert_upper2 = abs(p_up2.y - p_down2.y)
            vert = (vert_upper + vert_upper2) / 2.0
            return max(0.0, min(0.6, vert / (2.0 * horiz)))
        except Exception:
            return 0.0

    @staticmethod
    def _smile_proxy(landmarks: Any) -> float:
        try:
            left = landmarks[61]
            right = landmarks[291]
            width = abs(left.x - right.x)
            upper = landmarks[13]
            lower = landmarks[14]
            depth = max(abs(upper.y - lower.y), 1e-6)
            return max(0.0, min(1.0, width / (depth * 4.0)))
        except Exception:
            return 0.0

    @staticmethod
    def _brow_furrow_distance(landmarks: Any) -> float:
        try:
            inner_l = landmarks[107]
            inner_r = landmarks[336]
            dist = abs(inner_l.x - inner_r.x)
            # Smaller distance → more furrowed (tension).
            return max(0.0, min(1.0, 0.45 - dist))
        except Exception:
            return 0.0

    @staticmethod
    def _mouth_open(landmarks: Any) -> float:
        try:
            upper = landmarks[13]
            lower = landmarks[14]
            return max(0.0, min(1.0, abs(upper.y - lower.y) * 4.0))
        except Exception:
            return 0.0

    @staticmethod
    def _head_pose_proxy(landmarks: Any) -> tuple[float, float]:
        """Cheap proxy without solvePnP: nose offset vs symmetric ears."""
        try:
            nose = landmarks[1]
            left_ear = landmarks[234]
            right_ear = landmarks[454]
            centre_x = (left_ear.x + right_ear.x) / 2.0
            yaw = (nose.x - centre_x) * 60.0
            centre_y = (left_ear.y + right_ear.y) / 2.0
            pitch = -(nose.y - centre_y) * 50.0
            return max(-1.0, min(1.0, yaw)), max(-1.0, min(1.0, pitch))
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _variance(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return min(1.0, sum((v - mean) ** 2 for v in values) / len(values))

    def _blink_rate(self, ear: float) -> float:
        if ear < 0.18:
            self._blink_window.append(1.0)
        else:
            self._blink_window.append(0.0)
        if len(self._blink_window) > self.window:
            self._blink_window.pop(0)
        return self._variance(self._blink_window) if self._blink_window else 0.0

    @staticmethod
    def _attention_from(ear: float, yaw: float, pitch: float) -> float:
        if ear <= 0.05:
            return 0.0
        eye_component = min(1.0, ear / 0.32)
        centered = max(0.0, 1.0 - (abs(yaw) + abs(pitch)) / 0.4)
        return max(0.0, min(1.0, 0.6 * eye_component + 0.4 * centered))

    @staticmethod
    def _dominant_expression(
        valence: float, arousal: float, tension: float
    ) -> str:
        if tension > 0.6:
            return "frustrated"
        if valence > 0.4 and arousal > 0.4:
            return "happy"
        if valence < -0.3 and arousal < 0.4:
            return "sad"
        if arousal < 0.1:
            return "tired"
        if tension > 0.3:
            return "frustrated"
        return "neutral"

    @staticmethod
    def _gaze_label(yaw: float, pitch: float) -> str:
        if abs(yaw) > 0.3 or abs(pitch) > 0.3:
            return "away"
        return "toward_user"

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))


def build_tracker(
    prefer: str = "mediapipe",
) -> MediaPipeFaceTracker | NullFaceTracker:
    """Public factory; returns the best available tracker."""
    prefer = (prefer or "mediapipe").lower().strip()
    if prefer == "mediapipe":
        return MediaPipeFaceTracker()
    return NullFaceTracker()


def persist_canvas_jpeg(
    jpeg_b64: str | bytes, dest: Path
) -> Path | None:
    """Helper for tests/devs: persist a base64 JPEG to disk for inspection."""
    if isinstance(jpeg_b64, str):
        payload = jpeg_b64.split(",", 1)[-1]
        try:
            raw = base64.b64decode(payload)
        except Exception:
            return None
    elif isinstance(jpeg_b64, (bytes, bytearray)):
        raw = bytes(jpeg_b64)
    else:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return dest
