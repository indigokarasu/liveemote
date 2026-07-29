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

        # Prefer the tracker's direct gaze/attention signals when available
        # (MediaPipeFaceTracker produces these). Fall back to the old browser-
        # side face_center heuristic when they are absent.
        tracker_gaze = data.get("gaze_direction")
        tracker_attention = data.get("attention")
        if tracker_gaze and isinstance(tracker_attention, (int, float)):
            self.user.gaze_direction = str(tracker_gaze)
            self.user.gaze_confidence = ema(self.user.gaze_confidence, float(data.get("gaze_confidence", 0.85 if self.user.face_detected else 0.0)), a)
            self.user.attention = ema(self.user.attention, float(tracker_attention), a)
        else:
            center = data.get("face_center") or (0.5, 0.5)
            centered = abs(center[0] - 0.5) < 0.22 and abs(center[1] - 0.5) < 0.22
            self.user.gaze_direction = "toward_user" if self.user.face_detected and centered else "away"
            self.user.gaze_confidence = ema(self.user.gaze_confidence, float(data.get("gaze_confidence", 0.85 if self.user.face_detected else 0.0)), a)
            target_attention = 0.9 if self.user.face_detected and centered else 0.35 if self.user.face_detected else 0.0
            self.user.attention = ema(self.user.attention, target_attention, a)

        # Tracker produces direct dominant_expression, valence, arousal, tension.
        # Prefer them; fall back to heuristic derivation from expression map.
        tracker_expr = data.get("dominant_expression")
        tracker_valence = data.get("valence")
        tracker_arousal = data.get("arousal")
        tracker_tension = data.get("tension")
        if tracker_expr and isinstance(tracker_valence, (int, float)) and isinstance(tracker_arousal, (int, float)):
            dominant = str(tracker_expr)
            conf = max(float(data.get("emotion_confidence", 0.0)), 0.5)
            expression_arousal = float(tracker_arousal)
        else:
            dominant, conf = self._dominant_expression(data.get("expression", {}))
            conf = max(conf, float(data.get("emotion_confidence", 0.0)))
            expression_arousal = (
                0.65 if dominant in {"happy", "frustrated"}
                else 0.25 if dominant == "sad"
                else 0.1 if dominant == "tired"
                else 0.2
            )
        self.user.emotion_confidence = ema(self.user.emotion_confidence, conf, a)
        self.user.dominant_expression = self.expression_latch.update(dominant, conf, int(data.get("timestamp_ms", self._now())))
        self.user.valence = ema(
            self.user.valence,
            float(tracker_valence) if isinstance(tracker_valence, (int, float))
            else (0.5 if dominant == "happy" else -0.4 if dominant in {"sad", "frustrated"} else 0.0),
            self.config.affect.smoothing.affect_alpha,
        )
        self.user.tension = ema(
            self.user.tension,
            float(tracker_tension) if isinstance(tracker_tension, (int, float))
            else (0.7 if dominant == "frustrated" else 0.25),
            self.config.affect.smoothing.affect_alpha,
        )
        self.user.arousal = ema(self.user.arousal, expression_arousal, self.config.affect.smoothing.affect_alpha)
        self.user.last_updated_ms = int(data.get("timestamp_ms", self._now()))

    def _update_audio(self, data: dict) -> None:
        a = self.config.affect.smoothing.audio_alpha
        speaking = bool(data.get("speaking"))
        self.user.speaking = speaking
        self.user.speech_energy = ema(self.user.speech_energy, float(data.get("energy", 0)), a)
        self.user.speech_rate = ema(self.user.speech_rate, float(data.get("speech_rate", 0)), a)
        vocal_arousal = clamp(
            (self.user.speech_energy * 0.65) + (self.user.speech_rate * 0.35),
            0.0,
            1.0,
        )
        self.user.arousal = ema(self.user.arousal, vocal_arousal, self.config.affect.smoothing.affect_alpha)
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

    def tick(
        self,
        timestamp_ms: int | None = None,
        *,
        accumulate_dt: bool = True,
    ) -> AvatarBehaviorState:
        """Recompute ``self.avatar`` for the current moment and return it.

        ``accumulate_dt`` (default ``True``) gates whether this tick is allowed
        to advance the conversation-turn timers (``user_turn_ms`` /
        ``assistant_turn_ms``) and reset ``_last_tick_ms``. The default
        ``True`` keeps the historical contract for event-driven callers
        (``consume``, ``speak_test``, ``set_policy_mode``, ``trigger``). Pass
        ``accumulate_dt=False`` for bookkeeping-only ticks that must NOT
        reflect artifact drift into the conversation-state payload - this is
        what :meth:`DemoOrchestrator.status` does on every 1.5 s browser
        poll, so the conversation timers stay event-driven instead of
        compounding on every heartbeat. The staleness override below still
        fires under ``accumulate_dt=False`` because it reads
        ``self.user.last_updated_ms`` (event-driven timestamp), not
        ``_last_tick_ms``.
        """
        now = timestamp_ms or self._now()
        if accumulate_dt:
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
            self.avatar.full_body_pose = "presenting"
        elif self.user.speaking:
            self.avatar = listening_behavior(self.user, self.conversation, self.emote_lookup("listening"))
            self.avatar.full_body_pose = "attentive_lean"
        elif self.conversation.turn_state == "assistant_thinking":
            affect, intensity = (mirrored_affect(self.user) if self.mode == "mirror" else reflected_affect(self.user))
            self.avatar = AvatarBehaviorState(mode="thinking", affect=affect, gaze_target=self.user.gaze_direction, emote_id=self.emote_lookup("thinking"), intensity=intensity, delay_ms=reaction_delay(self.mode, self.config.affect.reaction_delay_ms), full_body_pose="thinking_shift")
        else:
            affect, intensity = (mirrored_affect(self.user) if self.mode == "mirror" else reflected_affect(self.user))
            self.avatar = AvatarBehaviorState(mode="idle", affect=affect, gaze_target=self.user.gaze_direction if self.user.face_detected else "soft_forward", emote_id=self.emote_lookup("neutral"), intensity=intensity, mirror_strength=self.config.behavior.mirroring_strength if self.mode == "mirror" else 0.0, delay_ms=reaction_delay(self.mode, self.config.affect.reaction_delay_ms))
        # Ambient-recovery override --------------------------------------
        # When the perception stream has gone silent for longer than the
        # configured ambient threshold, snap the avatar into ``mode="recovering"``
        # so the browser's CSS ambient loop fires (see
        # ``apps/demo_server/static/demo.css`` ``.mode-recovering``). Without this,
        # the avatar would simply freeze on whatever ``mode``/``intensity`` it
        # happened to be emitting when the user's webcam disconnected - the
        # exact "pausing or freezing on its last drawn state" feel the user
        # wants to avoid.
        #
        # Three deliberate invariants:
        #   * ``self.user.last_updated_ms > 0`` keeps us out of the ambient
        #     path on first boot (no perception event yet = not a loss, just
        #     silence pending). Without this guard a cold-started server
        #     would flip to ambient before its first ever frame landed.
        #   * ``self.config.affect.ambient_after_ms > 0`` is the configured
        #     opt-out (``AFFECT__AMBIENT_AFTER_MS=0`` keeps the legacy
        #     freeze-on-last behaviour).
        #   * ``mirror_strength = 0.0`` and ``emote_id = self.emote_lookup("neutral")``
        #     preserve the project's headline signal-leakage invariant: the
        #     ambient branch never carries webcam-derived state into the
        #     avatar.
        if (
            self.config.affect.ambient_after_ms > 0
            and self.user.last_updated_ms > 0
            and (now - self.user.last_updated_ms) > self.config.affect.ambient_after_ms
        ):
            self.avatar.mode = "recovering"
            # Damp the existing intensity down so the CSS keyframes drive the
            # visible motion regardless of the last user-driven amplitude.
            self.avatar.intensity = min(self.avatar.intensity, 0.10)
            self.avatar.gaze_target = "soft_forward"
            self.avatar.mirror_strength = 0.0
            neutral = self.emote_lookup("neutral")
            if neutral is not None:
                self.avatar.emote_id = neutral
            self.avatar.full_body_pose = "standing_idle"
        return self.avatar

    def set_mode(self, mode: str) -> None:
        if mode not in {"mirror", "reflect"}:
            raise ValueError("mode must be mirror or reflect")
        self.mode = mode
