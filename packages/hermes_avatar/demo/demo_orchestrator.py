from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import BackgroundSpec, VisualStyle
from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.config.schema import AppConfig, load_config
from hermes_avatar.demo.meeting_join import MeetingJoinService
from hermes_avatar.protocol.agent_bridge import AgentBridge
from hermes_avatar.renderer.deeplivecam_adapter import DeepLiveCamAdapter
from hermes_avatar.renderer.livetalking_adapter import LiveTalkingAdapter
from hermes_avatar.voice.base import VoiceStyle
from hermes_avatar.voice.elevenlabs_adapter import ElevenLabsAdapter
from hermes_avatar.voice.luxtts_adapter import LuxTTSAdapter
from hermes_avatar.voice.moss_adapter import MossTTSAdapter
from hermes_avatar.voice.noop_adapter import NoopVoiceAdapter


def discover_character_roots(character: str | Path) -> dict[str, Path]:
    root = Path(character)
    roots: list[Path]
    if (root / "canonical").is_dir():
        roots = [root]
    else:
        roots = sorted(path for path in root.iterdir() if (path / "canonical").is_dir()) if root.exists() else []

    catalog: dict[str, Path] = {}
    for candidate in roots:
        index = build_asset_index(candidate)
        catalog[index.character_id] = candidate
    return catalog


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
        self.character_roots = discover_character_roots(character)
        if not self.character_roots:
            self.character_roots = {build_asset_index(character).character_id: Path(character)}
        self.character_catalog = {cid: build_asset_index(path) for cid, path in self.character_roots.items()}
        self.index = next(iter(self.character_catalog.values()))
        self.active_style_id = self.index.default_style_id
        self.active_background_id = self.index.default_background_id
        self.sync_background_to_style = True
        self.runtime = self._new_runtime()
        self.hermes = HermesBridge(hermes_mode, self.config.hermes.url)
        self.renderer = DeepLiveCamAdapter() if renderer == "deeplivecam" else LiveTalkingAdapter(self.config.renderer.livetalking_url)
        self.renderer.load_character(self.index)
        self._notify_renderer_theme()
        self.voice_backend_name = voice_backend
        self.voice = self._voice_backend(voice_backend)
        self.last_response_text = ""
        self.meeting = MeetingJoinService(self.renderer)

    def _new_runtime(self) -> AffectRuntime:
        return AffectRuntime(self.config, emote_lookup=lambda state: (self.index.find_emote(state).id if self.index.find_emote(state) else None))

    def _voice_backend(self, backend: str):
        normalized = (backend or "none").lower().replace("_", "-")
        if normalized in {"none", "off", "disabled", "silent", "no-tts"}:
            return NoopVoiceAdapter()
        if normalized == "elevenlabs":
            return ElevenLabsAdapter(cache_dir=self.config.voice.cache_dir)
        if normalized == "moss":
            return MossTTSAdapter()
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
            set_theme(self.index, self.active_style(), self.active_background())

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
            "active_style": asdict(self.active_style()) if self.active_style() else None,
            "active_background": asdict(self.active_background()) if self.active_background() else None,
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
        return {
            "renderer": renderer_caps,
            "voice": voice_caps,
            "mobile_layout": True,
            "multi_character_switching": True,
            "cloud_manifest_available": True,
        }

    def apply_event(self, event: dict) -> dict:
        behavior = self.runtime.consume(event)
        self.renderer.set_behavior(behavior)
        return self.status()

    async def speak_test(self, text: str) -> dict:
        self.runtime.conversation.turn_state = "assistant_thinking"
        response = await self.agent.generate_response(text, self.runtime.user)
        self.last_response_text = response.text
        self.runtime.hermes_tags = response.tags
        if not response.text:
            behavior = self.runtime.tick(int(time.time() * 1000))
            self.renderer.set_behavior(behavior)
            self.runtime.conversation.turn_state = "idle"
            return {**self.status(), "speech": None, "agent_response": asdict(response)}

        self.runtime.conversation.turn_state = "assistant_speaking"
        behavior = self.runtime.tick(int(time.time() * 1000))
        response_voice = response.tags.get("voice", {}) if isinstance(response.tags.get("voice", {}), dict) else {}
        style = self.active_style()
        style_voice = asdict(style.voice) if style else {}
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
        return self.status()

    def set_background(self, background_id: str, sync_background: bool = False) -> dict:
        background = self.index.find_background(background_id)
        if background is None:
            raise ValueError(f"Unknown background for {self.index.character_id}: {background_id}")
        self.active_background_id = background.id
        self.sync_background_to_style = sync_background
        self._notify_renderer_theme()
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
            if style and style.default_background_id:
                self.active_background_id = style.default_background_id
        self._notify_renderer_theme()
        return self.status()

    def trigger(self, state: str) -> dict:
        if state == "interrupt":
            self.renderer.interrupt()
            self.runtime.conversation.turn_state = "interrupted"
        elif state == "reset":
            self.runtime.conversation.turn_state = "idle"
        elif state in {"listening", "thinking"}:
            self.runtime.conversation.turn_state = "user_speaking" if state == "listening" else "assistant_thinking"
        self.runtime.tick(int(time.time() * 1000))
        return self.status()

    def join_meeting(self, meeting_url: str, display_name: str | None = None) -> dict:
        meeting = self.meeting.join(meeting_url, display_name)
        return {**self.status(), "meeting": meeting}

    def leave_meeting(self) -> dict:
        meeting = self.meeting.leave()
        return {**self.status(), "meeting": meeting}

    def safe_audio_roots(self) -> list[Path]:
        return [Path(self.config.voice.cache_dir).resolve()]

    def select_character(self, character_path: str) -> dict:
        selected = build_asset_index(character_path)
        self.character_roots[selected.character_id] = Path(character_path)
        self.character_catalog[selected.character_id] = selected
        return self.set_character(selected.character_id)
