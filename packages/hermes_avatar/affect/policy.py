from __future__ import annotations
import time
from hermes_avatar.config.schema import AppConfig, load_config
from .state import UserAffectState, ConversationState, AvatarBehaviorState
from .smoothing import ema, clamp, ExpressionLatch, reaction_delay
from .listening_policy import listening_behavior
from .speaking_policy import speaking_behavior
from .mirror_policy import mirrored_affect
from .reflect_policy import reflected_affect
from .interruption_policy import interruption_risk

class AffectRuntime:
    def __init__(self, config: AppConfig | None = None, emote_lookup=None) -> None:
        self.config = config or load_config()
        self.user = UserAffectState()
        self.conversation = ConversationState()
        self.avatar = AvatarBehaviorState()
        self.mode = self.config.behavior.default_mode
        self.hermes_tags: dict | None = None
        self.expression_latch = ExpressionLatch(dwell_ms=self.config.affect.min_emote_dwell_ms)
        self.emote_lookup = emote_lookup or (lambda state: None)
        self._last_tick_ms = self._now()
        self._last_speaking_ms = 0

    def _now(self) -> int:
        return int(time.time() * 1000)

    def consume(self, event) -> AvatarBehaviorState:
        etype = event.get("type") if isinstance(event, dict) else getattr(event, "type", "")
        data = event if isinstance(event, dict) else event.model_dump()
        if etype == "perception.frame":
            self._update_face(data)
        elif etype == "audio.vad":
            self._update_audio(data)
        elif etype == "hermes.response":
            self.hermes_tags = data.get("tags", {})
            self.conversation.turn_state = "assistant_thinking"
        return self.tick(data.get("timestamp_ms") or self._now())

    def _dominant_expression(self, expr: dict) -> tuple[str, float]:
        smile, frown = expr.get("smile", 0.0), expr.get("frown", 0.0)
        brow, eye = expr.get("brow_raise", 0.0), expr.get("eye_open", 0.5)
        if frown > 0.55 and brow > 0.25:
            return "frustrated", frown
        if frown > 0.45:
            return "sad", frown
        if smile > 0.45:
            return "happy", smile
        if eye < 0.25:
            return "tired", 1 - eye
        return "neutral", 0.3

    def _update_face(self, data: dict) -> None:
        a = self.config.affect.smoothing.face_alpha
        self.user.face_detected = bool(data.get("face_detected"))
        self.user.head_yaw = clamp(ema(self.user.head_yaw, float(data.get("head_yaw", 0)), a), -self.config.gaze.max_yaw_deg, self.config.gaze.max_yaw_deg)
        self.user.head_pitch = clamp(ema(self.user.head_pitch, float(data.get("head_pitch", 0)), a), -self.config.gaze.max_pitch_deg, self.config.gaze.max_pitch_deg)
        center = data.get("face_center") or (0.5, 0.5)
        centered = abs(center[0] - 0.5) < 0.22 and abs(center[1] - 0.5) < 0.22
        self.user.gaze_direction = "toward_user" if self.user.face_detected and centered else "away"
        target_attention = 0.9 if self.user.face_detected and centered else 0.35 if self.user.face_detected else 0.0
        self.user.attention = ema(self.user.attention, target_attention, a)
        dominant, conf = self._dominant_expression(data.get("expression", {}))
        self.user.dominant_expression = self.expression_latch.update(dominant, conf, int(data.get("timestamp_ms", self._now())))
        self.user.valence = ema(self.user.valence, 0.5 if dominant == "happy" else -0.4 if dominant in {"sad", "frustrated"} else 0.0, self.config.affect.smoothing.affect_alpha)
        self.user.tension = ema(self.user.tension, 0.7 if dominant == "frustrated" else 0.25, self.config.affect.smoothing.affect_alpha)
        self.user.last_updated_ms = int(data.get("timestamp_ms", self._now()))

    def _update_audio(self, data: dict) -> None:
        a = self.config.affect.smoothing.audio_alpha
        speaking = bool(data.get("speaking"))
        self.user.speaking = speaking
        self.user.speech_energy = ema(self.user.speech_energy, float(data.get("energy", 0)), a)
        self.user.speech_rate = ema(self.user.speech_rate, float(data.get("speech_rate", 0)), a)
        now = int(data.get("timestamp_ms", self._now()))
        if speaking:
            self._last_speaking_ms = now
            self.conversation.turn_state = "user_speaking"
            self.conversation.silence_ms = 0
        else:
            self.conversation.silence_ms = now - self._last_speaking_ms if self._last_speaking_ms else 0
            if self.conversation.turn_state == "user_speaking" and self.conversation.silence_ms > 500:
                self.conversation.turn_state = "assistant_thinking"
        self.user.last_updated_ms = now

    def tick(self, timestamp_ms: int | None = None) -> AvatarBehaviorState:
        now = timestamp_ms or self._now()
        dt = max(0, now - self._last_tick_ms)
        self._last_tick_ms = now
        if self.conversation.turn_state == "user_speaking":
            self.conversation.user_turn_ms += dt
        elif self.conversation.turn_state == "assistant_speaking":
            self.conversation.assistant_turn_ms += dt
        self.conversation.tension = self.user.tension
        self.conversation.interruption_risk = interruption_risk(self.user, self.conversation)
        if self.conversation.turn_state == "assistant_speaking":
            self.avatar = speaking_behavior(self.user, self.hermes_tags, self.emote_lookup("speaking_optional"))
        elif self.user.speaking:
            self.avatar = listening_behavior(self.user, self.conversation, self.emote_lookup("listening"))
        elif self.conversation.turn_state == "assistant_thinking":
            affect, intensity = (mirrored_affect(self.user) if self.mode == "mirror" else reflected_affect(self.user))
            self.avatar = AvatarBehaviorState(mode="thinking", affect=affect, gaze_target=self.user.gaze_direction, emote_id=self.emote_lookup("thinking"), intensity=intensity, delay_ms=reaction_delay(self.mode, self.config))
        else:
            affect, intensity = (mirrored_affect(self.user) if self.mode == "mirror" else reflected_affect(self.user))
            self.avatar = AvatarBehaviorState(mode="idle", affect=affect, gaze_target=self.user.gaze_direction if self.user.face_detected else "soft_forward", emote_id=self.emote_lookup("neutral"), intensity=intensity, mirror_strength=self.config.behavior.mirroring_strength if self.mode == "mirror" else 0.0, delay_ms=reaction_delay(self.mode, self.config))
        return self.avatar

    def set_mode(self, mode: str) -> None:
        if mode not in {"mirror", "reflect"}:
            raise ValueError("mode must be mirror or reflect")
        self.mode = mode
