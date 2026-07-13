from __future__ import annotations

import logging
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from .base import Renderer
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.asset_index import BackgroundSpec, CharacterIndex, VisualStyle

logger = logging.getLogger(__name__)


class RendererUnavailableError(RuntimeError):
    """Raised when a required (non-optional) renderer call cannot be served because the
    circuit breaker is OPEN. Callers can catch this specifically to distinguish a
    renderer-outage from other request failures."""


class LiveTalkingAdapter(Renderer):
    """HTTP adapter for LiveTalking-compatible avatar runtimes.

    The adapter exposes a contract-first status surface: every optional endpoint is
    tracked, health is probed, and unsupported calls return structured capability
    information instead of disappearing into silent no-ops.
    """

    ENDPOINTS = {
        "health": ("GET", "/health"),
        "character": ("POST", "/avatar/character"),
        "theme": ("POST", "/avatar/theme"),
        "emote": ("POST", "/avatar/emote"),
        "behavior": ("POST", "/avatar/behavior"),
        "speak": ("POST", "/avatar/speak"),
        "interrupt": ("POST", "/avatar/interrupt"),
        "webrtc": ("POST", "/avatar/start_webrtc"),
        "virtualcam": ("POST", "/avatar/start_virtualcam"),
        "join_meeting": ("POST", "/avatar/join_meeting"),
        "leave_meeting": ("POST", "/avatar/leave_meeting"),
    }

    def __init__(self, base_url: str = "http://127.0.0.1:8010", vendor_dir: str = "vendor/LiveTalking", timeout: float | None = None, connect_timeout: float | None = None, config: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.vendor_dir = Path(vendor_dir)
        self.character_index: CharacterIndex | None = None
        self.process: subprocess.Popen | None = None
        self.last_behavior: AvatarBehaviorState | None = None
        self.endpoint_status: dict[str, dict[str, Any]] = {}
        self.last_latency_ms: int | None = None

        # Resolve request / connect timeouts. An explicitly passed value wins; otherwise
        # a passed config (renderer section) is consulted; finally fall back to defaults.
        if timeout is None:
            timeout = getattr(getattr(config, "renderer", None), "request_timeout", None)
        if connect_timeout is None:
            connect_timeout = getattr(getattr(config, "renderer", None), "connect_timeout", None)
        self.request_timeout: float = float(timeout) if timeout else 1.5
        self.connect_timeout: float = float(connect_timeout) if connect_timeout else 1.0

        # Circuit breaker state. The breaker trips to "open" after
        # ``cb_failure_threshold`` consecutive failures and refuses work for
        # ``cb_timeout`` seconds (then probes once in "half-open") so a dead
        # renderer is not hammered and the demo degrades gracefully.
        self.cb_failure_count = 0
        self.cb_last_failure_time: float | None = None
        self.cb_state = "closed"  # closed (healthy) -> open (tripped) -> half-open (probing)
        self.cb_failure_threshold = 5  # consecutive failures before tripping
        self.cb_timeout = 60  # seconds the breaker stays open before probing

        # Retry configuration: exponential backoff (base_delay * 2**attempt)
        # capped at max_delay, with +/- jitter_factor proportional dithering to
        # avoid synchronized retry storms across many concurrent requests.
        self.max_retries = 3
        self.base_delay = 0.5  # seconds, first retry delay
        self.max_delay = 4.0  # seconds, ceiling on retry delay
        self.jitter_factor = 0.1  # +/- 10% randomization on each delay

        self.active_style: VisualStyle | None = None
        self.active_background: BackgroundSpec | None = None

    def capabilities(self) -> dict:
        online = self._request("health", {}, optional=True).get("ok", False)
        return {
            "base_url": self.base_url,
            "vendor_dir_exists": self.vendor_dir.exists(),
            "online": online,
            "endpoint_status": self.endpoint_status,
            "last_latency_ms": self.last_latency_ms,
            "circuit_breaker": {
                "state": self.cb_state,
                "failure_count": self.cb_failure_count,
                "last_failure_time": self.cb_last_failure_time,
            },
        }

    def load_character(self, character_index: CharacterIndex) -> None:
        self.character_index = character_index
        self._request("character", character_index.to_dict(), optional=True)

    def set_idle_emote(self, emote_id: str) -> None:
        self._request("emote", {"emote_id": emote_id}, optional=True)

    def set_theme(self, character_index: CharacterIndex, style: VisualStyle | None, background: BackgroundSpec | None) -> None:
        self.character_index = character_index
        self.active_style = style
        self.active_background = background
        self._request(
            "theme",
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

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Determine if an exception is retryable with exponential backoff."""
        # Don't retry if circuit breaker is open (handled elsewhere)
        if self.cb_state == "open":
            return False
            
        # HTTP status codes that are retryable
        if hasattr(exc, 'response') and getattr(exc, 'response', None) is not None:
            status_code = exc.response.status_code
            # Retry on 5xx server errors and 429 rate limiting
            if 500 <= status_code < 600 or status_code == 429:
                return True
        
        # Network-related errors that are typically transient
        error_str = str(exc).lower()
        retryable_errors = [
            'connection',
            'timeout',
            'timeout',
            'network',
            'temporary',
            'temporarily',
            'unavailable',
            'service unavailable',
            'gateway timeout',
            'bad gateway',
        ]
        
        return any(err in error_str for err in retryable_errors)

    def _request(self, endpoint: str, payload: dict, optional: bool = False) -> dict:
        # Circuit breaker logic
        if self.cb_state == "open":
            if self.cb_last_failure_time and (time.time() - self.cb_last_failure_time) > self.cb_timeout:
                self.cb_state = "half-open"
                logger.info(
                    "renderer circuit breaker probing (half-open)",
                    extra={"audit": {"event": "renderer.cb_half_open", "endpoint": endpoint}},
                )
            else:
                # Circuit is open and timeout not elapsed, fail fast
                if optional:
                    return {"ok": False, "offline": True, "endpoint": endpoint, "error": "circuit breaker open"}
                else:
                    raise RendererUnavailableError("Renderer circuit breaker is open")

        method, path = self.ENDPOINTS[endpoint]
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=httpx.Timeout(connect=self.connect_timeout, read=self.request_timeout, write=self.request_timeout, pool=self.request_timeout)) as client:
                    if method == "GET":
                        r = client.get(f"{self.base_url}{path}")
                    else:
                        r = client.post(f"{self.base_url}{path}", json=payload)
                    elapsed = int((time.perf_counter() - started) * 1000)
                    self.last_latency_ms = elapsed
                    self.endpoint_status[endpoint] = {
                        "supported": True,
                        "status_code": r.status_code,
                        "latency_ms": elapsed,
                    }
                    r.raise_for_status()
                    data = r.json() if r.content else {}
                    # On success, reset circuit breaker if half-open or closed
                    if self.cb_state == "half-open":
                        self.cb_state = "closed"
                        self.cb_failure_count = 0
                        self.cb_last_failure_time = None
                        logger.info(
                            "renderer circuit breaker recovered (closed)",
                            extra={"audit": {"event": "renderer.cb_recovered", "endpoint": endpoint}},
                        )
                    return {"ok": True, **data}
            except Exception as exc:
                elapsed = int((time.perf_counter() - started) * 1000)
                self.endpoint_status[endpoint] = {
                    "supported": False,
                    "latency_ms": elapsed,
                    "error": str(exc),
                }
                last_exception = exc
                
                # If this is the last attempt, don't retry
                if attempt == self.max_retries:
                    break
                
                # Check if this is a transient error worth retrying
                if not self._is_retryable_error(exc):
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = delay * self.jitter_factor * (2 * random.random() - 1)  # +/- jitter%
                delay += jitter
                delay = max(0, delay)  # Ensure non-negative
                
                time.sleep(delay)
        
        # Update circuit breaker on failure
        self.cb_failure_count += 1
        self.cb_last_failure_time = time.time()
        if self.cb_failure_count >= self.cb_failure_threshold:
            self.cb_state = "open"
            logger.warning(
                "renderer circuit breaker tripped (open)",
                extra={
                    "audit": {
                        "event": "renderer.cb_open",
                        "endpoint": endpoint,
                        "failure_count": self.cb_failure_count,
                        "error": str(last_exception),
                    }
                },
            )
        
        if optional:
            return {"ok": False, "offline": True, "endpoint": endpoint, "error": str(last_exception)}
        raise last_exception