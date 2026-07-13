from __future__ import annotations
import logging
import time
from hermes_avatar.config.schema import AppConfig, load_config
from .state import (
    UserAffectState,
    ConversationState,
    AvatarBehaviorState,
    fill_behavior_state,
    acquire_behavior_state,
    release_behavior_state,
)
from hermes_avatar.observability.tracing import get_trace_id

logger = logging.getLogger(__name__)

# Bounded ring buffer of recent tick records exposed via the debug endpoint.
HISTORY_LIMIT = 30
from .smoothing import ema, clamp, ExpressionLatch, reaction_delay
from .listening_policy import listening_behavior
from .speaking_policy import speaking_behavior
from .mirror_policy import mirrored_affect
from .reflect_policy import reflected_affect
from .interruption_policy import interruption_risk

_AFFECT_NONE = "neutral"


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
        # Bounded, most-recent-last history of tick outputs for the
        # debug/visualization endpoint. Each entry carries the active trace id
        # so a developer can correlate a tick with the request that caused it.
        self.history: list[dict] = []

        # Cache frequently-read config scalars so the per-frame hot path avoids
        # repeated nested attribute lookups (self.config.affect.smoothing.*).
        self._face_alpha = self.config.affect.smoothing.face_alpha
        self._audio_alpha = self.config.affect.smoothing.audio_alpha
        self._affect_alpha = self.config.affect.smoothing.affect_alpha
        self._max_yaw = self.config.gaze.max_yaw_deg
        self._max_pitch = self.config.gaze.max_pitch_deg
        self._mirroring_strength = self.config.behavior.mirroring_strength
        self._reaction_delay = self.config.affect.reaction_delay_ms

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
        a = self._face_alpha
        max_yaw = self._max_yaw
        max_pitch = self._max_pitch
        aff = self._affect_alpha
        self.user.face_detected = bool(data.get("face_detected"))
        self.user.head_yaw = clamp(ema(self.user.head_yaw, float(data.get("head_yaw", 0)), a), -max_yaw, max_yaw)
        self.user.head_pitch = clamp(ema(self.user.head_pitch, float(data.get("head_pitch", 0)), a), -max_pitch, max_pitch)
        center = data.get("face_center") or (0.5, 0.5)
        centered = abs(center[0] - 0.5) < 0.22 and abs(center[1] - 0.5) < 0.22
        self.user.gaze_direction = "toward_user" if self.user.face_detected and centered else "away"
        self.user.gaze_confidence = ema(self.user.gaze_confidence, float(data.get("gaze_confidence", 0.85 if self.user.face_detected else 0.0)), a)
        target_attention = 0.9 if self.user.face_detected and centered else 0.35 if self.user.face_detected else 0.0
        self.user.attention = ema(self.user.attention, target_attention, a)
        dominant, conf = self._dominant_expression(data.get("expression", {}))
        conf = max(conf, float(data.get("emotion_confidence", 0.0)))
        self.user.emotion_confidence = ema(self.user.emotion_confidence, conf, a)
        self.user.dominant_expression = self.expression_latch.update(dominant, conf, int(data.get("timestamp_ms", self._now())))
        expression_arousal = (
            0.65
            if dominant in {"happy", "frustrated"}
            else 0.25
            if dominant == "sad"
            else 0.1
            if dominant == "tired"
            else 0.2
        )
        valence_target = 0.5 if dominant == "happy" else -0.4 if dominant in {"sad", "frustrated"} else 0.0
        tension_target = 0.7 if dominant == "frustrated" else 0.25
        self.user.valence = ema(self.user.valence, valence_target, aff)
        self.user.tension = ema(self.user.tension, tension_target, aff)
        self.user.arousal = ema(self.user.arousal, expression_arousal, aff)
        self.user.last_updated_ms = int(data.get("timestamp_ms", self._now()))

    def _update_audio(self, data: dict) -> None:
        a = self._audio_alpha
        aff = self._affect_alpha
        speaking = bool(data.get("speaking"))
        self.user.speaking = speaking
        self.user.speech_energy = ema(self.user.speech_energy, float(data.get("energy", 0)), a)
        self.user.speech_rate = ema(self.user.speech_rate, float(data.get("speech_rate", 0)), a)
        vocal_arousal = clamp(
            (self.user.speech_energy * 0.65) + (self.user.speech_rate * 0.35),
            0.0,
            1.0,
        )
        self.user.arousal = ema(self.user.arousal, vocal_arousal, aff)
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

        # Acquire a pooled AvatarBehaviorState and populate it in place (perf-4).
        out = acquire_behavior_state()
        if self.conversation.turn_state == "assistant_speaking":
            speaking_behavior(self.user, self.hermes_tags, self.emote_lookup("speaking_optional"), out=out)
            out.full_body_pose = "presenting"
        elif self.user.speaking:
            listening_behavior(self.user, self.conversation, self.emote_lookup("listening"), out=out)
            out.full_body_pose = "attentive_lean"
        elif self.conversation.turn_state == "assistant_thinking":
            affect, intensity = (mirrored_affect(self.user) if self.mode == "mirror" else reflected_affect(self.user))
            fill_behavior_state(
                out,
                mode="thinking",
                affect=affect,
                gaze_target=self.user.gaze_direction,
                emote_id=self.emote_lookup("thinking"),
                intensity=intensity,
                delay_ms=reaction_delay(self.mode, self._reaction_delay),
                full_body_pose="thinking_shift",
            )
        else:
            affect, intensity = (mirrored_affect(self.user) if self.mode == "mirror" else reflected_affect(self.user))
            fill_behavior_state(
                out,
                mode="idle",
                affect=affect,
                gaze_target=self.user.gaze_direction if self.user.face_detected else "soft_forward",
                emote_id=self.emote_lookup("neutral"),
                intensity=intensity,
                mirror_strength=self._mirroring_strength if self.mode == "mirror" else 0.0,
                delay_ms=reaction_delay(self.mode, self._reaction_delay),
            )

        # Hand the previous frame's object back to the pool. We release *before*
        # the next acquire in the following tick, so a caller holding this
        # returned reference is never mutated underneath.
        prev = self.avatar
        self.avatar = out
        release_behavior_state(prev)

        self._record_history(now)
        return self.avatar

    def _record_history(self, ts_ms: int) -> None:
        """Append a compact, trace-correlated record to the bounded history."""
        trace_id = get_trace_id()
        self.history.append({
            "ts_ms": ts_ms,
            "trace_id": trace_id,
            "mode": self.mode,
            "turn_state": self.conversation.turn_state,
            "avatar": self.avatar.to_dict(),
        })
        if len(self.history) > HISTORY_LIMIT:
            del self.history[0 : len(self.history) - HISTORY_LIMIT]
        logger.debug(
            "affect tick",
            extra={
                "trace_id": trace_id,
                "audit": {
                    "event": "affect.tick",
                    "mode": self.mode,
                    "turn_state": self.conversation.turn_state,
                    "affect": self.avatar.affect,
                },
            },
        )

    def set_mode(self, mode: str) -> None:
        if mode not in {"mirror", "reflect"}:
            raise ValueError("mode must be mirror or reflect")
        self.mode = mode
