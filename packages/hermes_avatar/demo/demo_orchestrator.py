from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import logging
import time

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import BackgroundSpec, CharacterIndex, VisualStyle
from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.config.schema import AppConfig, load_config, reload_config
from hermes_avatar.demo.meeting_join import MeetingJoinService
from hermes_avatar.protocol.agent_bridge import AgentBridge
from hermes_avatar.renderer.deeplivecam_adapter import DeepLiveCamAdapter
from hermes_avatar.renderer.facefusion_adapter import FaceSwapAdapter
from hermes_avatar.renderer.livetalking_adapter import LiveTalkingAdapter
from hermes_avatar.voice.base import VoiceBackend, VoiceStyle
from hermes_avatar.voice.elevenlabs_adapter import ElevenLabsAdapter
from hermes_avatar.voice.luxtts_adapter import LuxTTSAdapter
from hermes_avatar.voice.noop_adapter import NoopVoiceAdapter
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics for orchestrator operations
AFFECT_TICKS = Counter(
    'demo_affect_runtime_ticks_total',
    'Total number of affect runtime ticks'
)

EVENT_PROCESSING_TIME = Histogram(
    'demo_event_processing_seconds',
    'Time taken to process events'
)

AGENT_RESPONSE_TIME = Histogram(
    'demo_agent_response_seconds',
    'Time taken for agent to generate response'
)

CHARACTER_LOADS = Counter(
    'demo_character_loads_total',
    'Total number of character loads'
)

STYLE_CHANGES = Counter(
    'demo_style_changes_total',
    'Total number of style changes'
)

BACKGROUND_CHANGES = Counter(
    'demo_background_changes_total',
    'Total number of background changes'
)

WORKFLOW_EXECUTIONS = Counter(
    'demo_workflow_executions_total',
    'Total number of workflow executions',
    ['workflow']
)

MEETING_JOINS = Counter(
    'demo_meeting_joins_total',
    'Total number of meeting joins'
)

MEETING_LEAVES = Counter(
    'demo_meeting_leaves_total',
    'Total number of meeting leaves'
)

INTERRUPTS = Counter(
    'demo_interrupts_total',
    'Total number of interruptions'
)

RESETS = Counter(
    'demo_resets_total',
    'Total number of resets'
)

def discover_character_catalog(
    character: str | Path,
) -> tuple[dict[str, Path], dict[str, CharacterIndex]]:
    root = Path(character)
    roots: list[Path]
    if (root / "canonical").is_dir():
        roots = [root]
    elif root.exists():
        roots = sorted(path for path in root.iterdir() if (path / "canonical").is_dir())
    else:
        roots = []

    character_roots: dict[str, Path] = {}
    character_catalog: dict[str, CharacterIndex] = {}
    for candidate in roots:
        index = build_asset_index(candidate)
        character_roots[index.character_id] = candidate
        character_catalog[index.character_id] = index
    return character_roots, character_catalog

class DemoOrchestrator:
    def __init__(
        self,
        character: str,
        renderer: str = "livetalking",
        voice_backend: str = "luxtts",
        agent_mode: str = "fake",
        config: AppConfig | None = None,
        agent_url: str | None = None,
        agent_harness: str = "generic",
    ) -> None:
        self.config = config or load_config()
        self.character_roots, self.character_catalog = discover_character_catalog(character)
        if not self.character_roots:
            index = build_asset_index(character)
            self.character_roots = {index.character_id: Path(character)}
            self.character_catalog = {index.character_id: index}
        self.index = next(iter(self.character_catalog.values()))
        self.active_style_id = self.index.default_style_id
        self.active_background_id = self.index.default_background_id
        self.sync_background_to_style = True
        self.runtime = self._new_runtime()
        self.agent_mode = agent_mode
        self.agent_harness = agent_harness
        self.agent = AgentBridge(agent_mode, agent_url or self.config.agent.url, agent_harness)
        self.hermes = self.agent  # Backward-compatible alias for older status/UI naming.
        # Face-swap renderers (FaceFusion or Deep-Live-Cam) are handled by the
        # same backend-agnostic adapter; the ``renderer`` string selects which
        # backend binary to drive. LiveTalking remains the default working
        # renderer when neither is chosen. The adapter is gated by
        # ``config.faceswap.enabled`` (defaults False), so naming the renderer
        # does NOT auto-start the model. Opt in via
        # ``FACESWAP__ENABLED=true`` in the environment.
        if renderer in ("facefusion", "deeplivecam"):
            self.renderer = FaceSwapAdapter(
                config=self.config.faceswap,
                backend=renderer,
                enabled=self.config.faceswap.enabled,
            )
        else:
            self.renderer = LiveTalkingAdapter(self.config.renderer.livetalking_url)
        self.renderer.load_character(self.index)
        self._notify_renderer_theme()
        self.voice_backend_name = voice_backend
        self.voice = self._voice_backend(voice_backend)
        self.last_response_text = ""
        self.meeting = MeetingJoinService(self.renderer)

    def _voice_backend(self, backend: str) -> VoiceBackend:
        normalized = (backend or "none").lower().replace("_", "-")
        if normalized in {"none", "off", "disabled", "silent", "no-tts"}:
            return NoopVoiceAdapter()
        if normalized == "elevenlabs":
            return ElevenLabsAdapter(cache_dir=self.config.voice.cache_dir)
        return LuxTTSAdapter(device=self.config.voice.device, cache_dir=self.config.voice.cache_dir)

    def _new_runtime(self) -> AffectRuntime:
        def lookup(state: str) -> str | None:
            emote = self.index.find_emote(state)
            return emote.id if emote else None

        return AffectRuntime(self.config, emote_lookup=lookup)

    def active_style(self) -> VisualStyle | None:
        return self.index.find_style(self.active_style_id)

    def active_background(self) -> BackgroundSpec | None:
        return self.index.find_background(self.active_background_id)

    def _notify_renderer_theme(self) -> None:
        set_theme = getattr(self.renderer, "set_theme", None)
        if callable(set_theme):
            style = self.active_style()
            background = self.active_background()
            if style is not None and background is not None:
                set_theme(self.index, style, background)

    def _neutral_avatar_state(self) -> AvatarBehaviorState:
        neutral_emote = self.index.find_emote("neutral")
        return AvatarBehaviorState(
            mode="idle",
            affect="neutral",
            gaze_target="toward_user",
            emote_id=neutral_emote.id if neutral_emote else None,
        )

    def _reset_runtime_for_character(self) -> None:
        self.runtime = self._new_runtime()
        self.runtime.avatar = self._neutral_avatar_state()

    def status(self) -> dict:
        # Heartbeat tick: pulse the affect runtime so the staleness clock
        # advances even when perception events stop arriving. The browser
        # polls /api/status every 1.5 s, so this is the heartbeat that
        # converts ``now_ms - user.last_updated_ms > ambient_after_ms`` into
        # an ambient-recovery ``mode="recovering"`` state. Without this the
        # runtime would only tick on explicit events and the CSS ambient
        # fallback would never trigger after signal loss.
        # We deliberately do NOT call AFFECT_TICKS.inc() here: the heartbeat
        # pulse is bookkeeping, not a real affect transition, and Prometheus
        # tick counters should reflect visible avatar state changes only.
        # We also pass ``accumulate_dt=False`` so the 1.5 s heartbeat does
        # NOT advance ``self.conversation.user_turn_ms`` /
        # ``assistant_turn_ms`` (event-driven only) nor reset ``_last_tick_ms``
        # (which would shrink the dt observed by the next real event).
        # The staleness override below STILL fires because it reads
        # ``self.user.last_updated_ms`` (an event-driven timestamp), not
        # ``_last_tick_ms``.
        self.runtime.tick(int(time.time() * 1000), accumulate_dt=False)
        return {
            "user": self.runtime.user.to_dict(),
            "conversation": self.runtime.conversation.to_dict(),
            "avatar": self.runtime.avatar.to_dict(),
            "mode_policy": self.runtime.mode,
            "agent_response_text": self.last_response_text,
            "hermes_response_text": self.last_response_text,
            "character_id": self.index.character_id,
            "character_name": self.index.display_name or self.index.character_id,
            "characters": self.character_options(),
            "styles": [asdict(style) for style in self.index.styles],
            "backgrounds": [asdict(background) for background in self.index.backgrounds],
            "workflow_style_rules": [asdict(rule) for rule in self.index.workflow_style_rules],
            "active_style_id": self.active_style_id,
            "active_background_id": self.active_background_id,
            "sync_background_to_style": self.sync_background_to_style,
            "active_style": asdict(self.active_style()) if self.active_style() is not None else None,
            "active_background": asdict(self.active_background()) if self.active_background() is not None else None,
            "capabilities": self.capabilities(),
            "meeting": self.meeting.status(),
        }

    def character_options(self) -> list[dict]:
        return [
            {
                "id": index.character_id,
                "name": index.display_name or index.character_id,
                "path": str(self.character_roots[index.character_id]),
                "emote_count": len(index.emotes),
            }
            for index in self.character_catalog.values()
        ]

    def capabilities(self) -> dict:
        renderer_caps = self.renderer.capabilities() if hasattr(self.renderer, "capabilities") else {"backend": type(self.renderer).__name__}
        voice_caps = self.voice.capability_status() if hasattr(self.voice, "capability_status") else {"backend": type(self.voice).__name__}
        agent_caps = self.agent.capability_status() if hasattr(self.agent, "capability_status") else {"backend": type(self.agent).__name__}
        return {
            "renderer": renderer_caps,
            "voice": voice_caps,
            "agent": agent_caps,
            "mobile_layout": True,
            "multi_character_switching": True,
            "cloud_manifest_available": True,
        }

    def apply_event(self, event: dict) -> dict:
        start_time = time.time()
        behavior = self.runtime.consume(event)
        self.renderer.set_behavior(behavior)
        EVENT_PROCESSING_TIME.observe(time.time() - start_time)
        return self.status()

    async def speak_test(self, text: str) -> dict:
        self.runtime.conversation.turn_state = "assistant_thinking"
        start_time = time.time()
        response = await self.agent.generate_response(text, self.runtime.user)
        AGENT_RESPONSE_TIME.observe(time.time() - start_time)
        self.last_response_text = response.text
        self.runtime.hermes_tags = response.tags
        if not response.text:
            behavior = self.runtime.tick(int(time.time() * 1000))
            AFFECT_TICKS.inc()
            self.renderer.set_behavior(behavior)
            self.runtime.conversation.turn_state = "idle"
            return {**self.status(), "speech": None, "agent_response": asdict(response)}

        self.runtime.conversation.turn_state = "assistant_speaking"
        behavior = self.runtime.tick(int(time.time() * 1000))
        AFFECT_TICKS.inc()
        response_voice = response.tags.get("voice", {}) if isinstance(response.tags.get("voice", {}), dict) else {}
        style = self.active_style()
        style_voice = asdict(style.voice) if style is not None else {}
        merged_voice = {**style_voice, **response_voice}
        speech = self.voice.synthesize(
            response.text,
            VoiceStyle(**{k: v for k, v in merged_voice.items() if k in {"pace", "warmth", "intensity"}}),
            self.index.voice_reference,
        )
        self.renderer.speak(speech.audio_path, response.text, behavior)
        self.runtime.conversation.turn_state = "idle"
        self.runtime.avatar = self._neutral_avatar_state()
        return {**self.status(), "speech": speech.__dict__}

    def set_policy_mode(self, mode: str) -> dict:
        self.runtime.set_mode(mode)
        self.runtime.tick(int(time.time() * 1000))
        AFFECT_TICKS.inc()
        return self.status()

    def set_character(self, character_id: str) -> dict:
        if character_id not in self.character_roots:
            raise ValueError(f"Unknown character: {character_id}")
        self.index = self.character_catalog[character_id]
        self.active_style_id = self.index.default_style_id
        self.active_background_id = self.index.default_background_id
        self.sync_background_to_style = True
        self.renderer.load_character(self.index)
        self._reset_runtime_for_character()
        self._notify_renderer_theme()
        CHARACTER_LOADS.inc()
        return self.status()

    def set_style(self, style_id: str, sync_background: bool = True) -> dict:
        style = self.index.find_style(style_id)
        if style is None:
            raise ValueError(f"Unknown style for {self.index.character_id}: {style_id}")
        self.active_style_id = style.id
        self.sync_background_to_style = sync_background
        if sync_background and style.default_background_id:
            self.active_background_id = style.default_background_id
        self._notify_renderer_theme()
        STYLE_CHANGES.inc()
        return self.status()

    def set_background(self, background_id: str, sync_background: bool = False) -> dict:
        background = self.index.find_background(background_id)
        if background is None:
            raise ValueError(f"Unknown background for {self.index.character_id}: {background_id}")
        self.active_background_id = background.id
        self.sync_background_to_style = sync_background
        self._notify_renderer_theme()
        BACKGROUND_CHANGES.inc()
        return self.status()

    def apply_workflow(self, workflow: str) -> dict:
        rule = next((rule for rule in self.index.workflow_style_rules if rule.workflow == workflow), None)
        if rule is None:
            raise ValueError(f"Unknown workflow for {self.index.character_id}: {workflow}")
        self.active_style_id = rule.style_id
        if rule.background_id:
            self.active_background_id = rule.background_id
        elif self.sync_background_to_style:
            style = self.active_style()
            if style is not None and style.default_background_id:
                self.active_background_id = style.default_background_id
        self._notify_renderer_theme()
        WORKFLOW_EXECUTIONS.labels(workflow=workflow).inc()
        return self.status()

    def trigger(self, state: str) -> dict:
        if state == "interrupt":
            self.renderer.interrupt()
            self.runtime.conversation.turn_state = "interrupted"
            INTERRUPTS.inc()
        elif state == "reset":
            self.runtime.conversation.turn_state = "idle"
            RESETS.inc()
        elif state in {"listening", "thinking"}:
            self.runtime.conversation.turn_state = "user_speaking" if state == "listening" else "assistant_thinking"
        self.runtime.tick(int(time.time() * 1000))
        AFFECT_TICKS.inc()
        return self.status()

    def join_meeting(self, meeting_url: str, display_name: str | None = None) -> dict:
        logger.info("meeting join requested", extra={"audit": {"event": "meeting.join", "url": meeting_url}})
        meeting = self.meeting.join(meeting_url, display_name)
        MEETING_JOINS.inc()
        return {**self.status(), "meeting": meeting}

    def leave_meeting(self) -> dict:
        logger.info("meeting leave requested", extra={"audit": {"event": "meeting.leave"}})
        meeting = self.meeting.leave()
        MEETING_LEAVES.inc()
        return {**self.status(), "meeting": meeting}

    def reload_config(self) -> dict:
        """Reload configuration from defaults.yaml and environment variables, and update dependent components."""
        new_config = reload_config(self.config)
        # Track what changed to know what to recreate
        config_changed = False
        voice_changed = False
        renderer_changed = False
        agent_changed = False

        # Compare relevant sections
        if self.config.affect != new_config.affect:
            config_changed = True
        if self.config.gaze != new_config.gaze:
            config_changed = True
        if self.config.behavior != new_config.behavior:
            config_changed = True
        if self.config.agent != new_config.agent:
            agent_changed = True
            config_changed = True
        if self.config.renderer != new_config.renderer:
            renderer_changed = True
            config_changed = True
        if self.config.voice != new_config.voice:
            voice_changed = True
            config_changed = True

        # Update the config
        self.config = new_config

        # Recreate voice backend if voice config changed
        if voice_changed:
            self.voice = self._voice_backend(self.voice_backend_name)

        # Recreate renderer if renderer config changed
        if renderer_changed:
            # Face-swap adapters (FaceFusion / Deep-Live-Cam) are both
            # FaceSwapAdapter instances; recreate preserving the backend + enabled
            # flag from the freshly reloaded config.
            if isinstance(self.renderer, FaceSwapAdapter):
                self.renderer = FaceSwapAdapter(
                    config=self.config.faceswap,
                    backend=self.renderer.config.backend,
                    enabled=self.config.faceswap.enabled,
                )
            else:
                # For LiveTalkingAdapter, we just create a new one with the URL
                self.renderer = LiveTalkingAdapter(self.config.renderer.livetalking_url)
            self.renderer.load_character(self.index)
            self._notify_renderer_theme()

        # Recreate agent if agent config changed
        if agent_changed:
            self.agent = AgentBridge(self.agent_mode, self.config.agent.url, self.agent_harness)
            self.hermes = self.agent  # Update the alias

        # Recreate the runtime (which depends on config) but preserve the current state
        # We'll create a new runtime and then copy over the state from the old runtime
        old_user = self.runtime.user
        old_conversation = self.runtime.conversation
        old_avatar = self.runtime.avatar
        old_mode = self.runtime.mode
        old_hermes_tags = self.runtime.hermes_tags
        self.runtime = self._new_runtime()
        # Restore the state
        self.runtime.user = old_user
        self.runtime.conversation = old_conversation
        self.runtime.avatar = old_avatar
        self.runtime.mode = old_mode
        self.runtime.hermes_tags = old_hermes_tags
        # Note: The expression_latch is recreated in _new_runtime, so it uses the new config's min_emote_dwell_ms.
        # The _last_tick_ms and _last_speaking_ms are reset to 0 in _new_runtime, but we might want to preserve the last tick time?
        # For simplicity, we'll reset them. The affect runtime will continue from the current time.

        logger.info(
            "orchestrator config reloaded",
            extra={
                "audit": {
                    "event": "orchestrator.config_reloaded",
                    "config_changed": config_changed,
                    "voice_changed": voice_changed,
                    "renderer_changed": renderer_changed,
                    "agent_changed": agent_changed,
                }
            },
        )
        return self.status()