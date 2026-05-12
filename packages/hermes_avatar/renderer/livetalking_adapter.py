from __future__ import annotations
import subprocess
from pathlib import Path
import httpx
from .base import Renderer
from hermes_avatar.character.asset_index import CharacterIndex
from hermes_avatar.affect.state import AvatarBehaviorState

class LiveTalkingAdapter(Renderer):
    def __init__(self, base_url: str = "http://127.0.0.1:8010", vendor_dir: str = "vendor/LiveTalking") -> None:
        self.base_url = base_url.rstrip("/")
        self.vendor_dir = Path(vendor_dir)
        self.character_index: CharacterIndex | None = None
        self.process: subprocess.Popen | None = None
        self.last_behavior: AvatarBehaviorState | None = None

    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index
        self._post("/avatar/character", character_index.to_dict(), optional=True)

    def set_idle_emote(self, emote_id: str) -> None:
        self._post("/avatar/emote", {"emote_id": emote_id}, optional=True)

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        self.last_behavior = behavior
        self._post("/avatar/behavior", behavior.to_dict(), optional=True)

    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None:
        self.set_behavior(behavior)
        self._post("/avatar/speak", {"audio_path": audio_path, "text": text, "behavior": behavior.to_dict()}, optional=True)

    def interrupt(self) -> None:
        self._post("/avatar/interrupt", {}, optional=True)

    def start_webrtc(self) -> None:
        self._post("/avatar/start_webrtc", {}, optional=True)

    def start_virtualcam(self) -> None:
        self._post("/avatar/start_virtualcam", {}, optional=True)

    def _post(self, path: str, payload: dict, optional: bool = False) -> dict:
        try:
            with httpx.Client(timeout=1.5) as client:
                r = client.post(f"{self.base_url}{path}", json=payload)
                r.raise_for_status()
                return r.json() if r.content else {}
        except Exception:
            if optional:
                return {"offline": True}
            raise
