from __future__ import annotations

import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from .base import Renderer
from hermes_avatar.character.asset_index import BackgroundSpec, CharacterIndex, VisualStyle
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import CharacterIndex


class LiveTalkingAdapter(Renderer):
    """HTTP adapter for LiveTalking-compatible avatar runtimes.

    The adapter now exposes a contract-first status surface: every optional endpoint
    is tracked, health is probed, and unsupported calls return structured capability
    information instead of disappearing into silent no-ops.
    """

    ENDPOINTS = {
        "health": ("GET", "/health"),
        "character": ("POST", "/avatar/character"),
        "emote": ("POST", "/avatar/emote"),
        "behavior": ("POST", "/avatar/behavior"),
        "speak": ("POST", "/avatar/speak"),
        "interrupt": ("POST", "/avatar/interrupt"),
        "webrtc": ("POST", "/avatar/start_webrtc"),
        "virtualcam": ("POST", "/avatar/start_virtualcam"),
        "join_meeting": ("POST", "/avatar/join_meeting"),
        "leave_meeting": ("POST", "/avatar/leave_meeting"),
    }

    def __init__(self, base_url: str = "http://127.0.0.1:8010", vendor_dir: str = "vendor/LiveTalking") -> None:
        self.base_url = base_url.rstrip("/")
        self.vendor_dir = Path(vendor_dir)
        self.character_index: CharacterIndex | None = None
        self.process: subprocess.Popen | None = None
        self.last_behavior: AvatarBehaviorState | None = None
        self.endpoint_status: dict[str, dict[str, Any]] = {}
        self.last_latency_ms: int | None = None

    def capabilities(self) -> dict:
        online = self._request("health", {}, optional=True).get("ok", False)
        return {
            "base_url": self.base_url,
            "vendor_dir_exists": self.vendor_dir.exists(),
            "online": online,
            "endpoint_status": self.endpoint_status,
            "last_latency_ms": self.last_latency_ms,
        }
        self.active_style: VisualStyle | None = None
        self.active_background: BackgroundSpec | None = None

    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index
        self._request("character", character_index.to_dict(), optional=True)

    def set_idle_emote(self, emote_id: str) -> None:
        self._request("emote", {"emote_id": emote_id}, optional=True)

    def set_theme(self, character_index: CharacterIndex, style: VisualStyle | None, background: BackgroundSpec | None) -> None:
        self.character_index = character_index
        self.active_style = style
        self.active_background = background
        self._post(
            "/avatar/theme",
            {
                "character_id": character_index.character_id,
                "style": asdict(style) if style else None,
                "background": asdict(background) if background else None,
            },
            optional=True,
        )

    def set_behavior(self, behavior: AvatarBehaviorState) -> None:
        self.last_behavior = behavior
        self._request("behavior", behavior.to_dict(), optional=True)

    def speak(self, audio_path: str, text: str, behavior: AvatarBehaviorState) -> None:
        self.set_behavior(behavior)
        self._request("speak", {"audio_path": audio_path, "text": text, "behavior": behavior.to_dict()}, optional=True)

    def interrupt(self) -> None:
        self._request("interrupt", {}, optional=True)

    def start_webrtc(self) -> None:
        self._request("webrtc", {}, optional=True)

    def start_virtualcam(self) -> None:
        self._request("virtualcam", {}, optional=True)

    def join_meeting(self, meeting_url: str, display_name: str = "Hermes Avatar") -> dict:
        return self._request("join_meeting", {"meeting_url": meeting_url, "display_name": display_name}, optional=True)

    def leave_meeting(self) -> dict:
        return self._request("leave_meeting", {}, optional=True)

    def _request(self, endpoint: str, payload: dict, optional: bool = False) -> dict:
        method, path = self.ENDPOINTS[endpoint]
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=1.5) as client:
                if method == "GET":
                    r = client.get(f"{self.base_url}{path}")
                else:
                    r = client.post(f"{self.base_url}{path}", json=payload)
                elapsed = int((time.perf_counter() - started) * 1000)
                self.last_latency_ms = elapsed
                self.endpoint_status[endpoint] = {"supported": True, "status_code": r.status_code, "latency_ms": elapsed}
                r.raise_for_status()
                data = r.json() if r.content else {}
                return {"ok": True, **data}
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            self.endpoint_status[endpoint] = {"supported": False, "latency_ms": elapsed, "error": str(exc)}
            if optional:
                return {"ok": False, "offline": True, "endpoint": endpoint, "error": str(exc)}
            raise
