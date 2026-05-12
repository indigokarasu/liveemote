from __future__ import annotations
import time
from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.config.schema import AppConfig, load_config
from hermes_avatar.protocol.hermes_bridge import HermesBridge
from hermes_avatar.renderer.livetalking_adapter import LiveTalkingAdapter
from hermes_avatar.renderer.deeplivecam_adapter import DeepLiveCamAdapter
from hermes_avatar.voice.base import VoiceStyle
from hermes_avatar.voice.luxtts_adapter import LuxTTSAdapter
from hermes_avatar.voice.elevenlabs_adapter import ElevenLabsAdapter
from hermes_avatar.voice.moss_adapter import MossTTSAdapter

class DemoOrchestrator:
    def __init__(self, character: str, renderer: str = "livetalking", voice_backend: str = "luxtts", hermes_mode: str = "fake", config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.index = build_asset_index(character)
        lookup = lambda state: (self.index.find_emote(state).id if self.index.find_emote(state) else None)
        self.runtime = AffectRuntime(self.config, emote_lookup=lookup)
        self.hermes = HermesBridge(hermes_mode, self.config.hermes.url)
        self.renderer = DeepLiveCamAdapter() if renderer == "deeplivecam" else LiveTalkingAdapter(self.config.renderer.livetalking_url)
        self.renderer.load_character(self.index)
        self.voice = self._voice_backend(voice_backend)
        self.last_response_text = ""

    def _voice_backend(self, backend: str):
        if backend == "elevenlabs":
            return ElevenLabsAdapter(cache_dir=self.config.voice.cache_dir)
        if backend == "moss":
            return MossTTSAdapter()
        return LuxTTSAdapter(device=self.config.voice.device, cache_dir=self.config.voice.cache_dir)

    def status(self) -> dict:
        return {"user": self.runtime.user.to_dict(), "conversation": self.runtime.conversation.to_dict(), "avatar": self.runtime.avatar.to_dict(), "mode_policy": self.runtime.mode, "hermes_response_text": self.last_response_text, "character_id": self.index.character_id}

    def apply_event(self, event: dict) -> dict:
        behavior = self.runtime.consume(event)
        self.renderer.set_behavior(behavior)
        return self.status()

    async def speak_test(self, text: str) -> dict:
        self.runtime.conversation.turn_state = "assistant_thinking"
        response = await self.hermes.generate_response(text, self.runtime.user)
        self.last_response_text = response.text
        self.runtime.hermes_tags = response.tags
        self.runtime.conversation.turn_state = "assistant_speaking"
        behavior = self.runtime.tick(int(time.time()*1000))
        style_data = response.tags.get("voice", {}) if isinstance(response.tags.get("voice", {}), dict) else {}
        speech = self.voice.synthesize(response.text, VoiceStyle(**{k: v for k, v in style_data.items() if k in {"pace", "warmth", "intensity"}}), self.index.voice_reference)
        self.renderer.speak(speech.audio_path, response.text, behavior)
        self.runtime.conversation.turn_state = "idle"
        self.runtime.avatar = AvatarBehaviorState(mode="idle", affect="neutral", gaze_target="toward_user", emote_id=(self.index.find_emote("neutral").id if self.index.find_emote("neutral") else None))
        return {**self.status(), "speech": speech.__dict__}

    def set_policy_mode(self, mode: str) -> dict:
        self.runtime.set_mode(mode)
        self.runtime.tick(int(time.time()*1000))
        return self.status()

    def trigger(self, state: str) -> dict:
        if state == "interrupt":
            self.renderer.interrupt(); self.runtime.conversation.turn_state = "interrupted"
        elif state == "reset":
            self.runtime.conversation.turn_state = "idle"
        elif state in {"listening", "thinking"}:
            self.runtime.conversation.turn_state = "user_speaking" if state == "listening" else "assistant_thinking"
        self.runtime.tick(int(time.time()*1000))
        return self.status()
