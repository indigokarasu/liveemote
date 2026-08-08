"""``VoiceLoopClient`` — browser voice ⇄ speech-to-speech sidecar relay.

The avatar's spoken conversation runs through the Hugging Face
``speech-to-speech`` pipeline, which ships an OpenAI-Realtime-compatible
WebSocket server (``ws://host:port/v1/realtime``). That pipeline lives in its
own process (``sidecar/voice_loop/app.py``) because it needs torch.

This client, which runs in the main LiveEmote server, mirrors the
``FaceFusionSidecarDaemon`` pattern:

* :meth:`capability_status` / :meth:`health` probe the sidecar's HTTP control
  plane (non-raising, cached) so ``/api/status`` and ``/api/health`` can
  report the voice loop without hammering it.
* :meth:`relay` is a full-duplex WebSocket proxy between a browser connection
  and the pipeline. It snoops the OpenAI Realtime event stream to drive the
  avatar's affect runtime:

  - ``conversation.item.input_audio_transcription.completed`` → the user's
    spoken turn (avatar shows engaged listening)
  - ``response.audio.delta`` → the avatar's voice started (avatar animates)
  - ``response.output_audio.done`` / ``response.done`` → voice finished

All failure modes degrade gracefully: pipeline down → the browser gets an
``error`` event and the connection closes; the avatar keeps running on the
perception pipeline exactly as before.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_URL = "http://127.0.0.1:8766"
DEFAULT_PIPELINE_WS_URL = "ws://127.0.0.1:8765/v1/realtime"
HEALTH_TIMEOUT_S = 1.5
# /api/status is polled at ~20 Hz by the browser; never probe the control
# plane more than once per TTL so the sidecar isn't load-bearing for the
# status loop.
PROBE_CACHE_TTL_S = 3.0


@dataclass
class VoiceLoopState:
    enabled: bool = False
    reachable: bool = False
    degraded: bool = False
    reason: str | None = None
    pipeline: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "degraded": False, "reachable": False, "reason": "voice loop disabled"}
        return {
            "enabled": True,
            "reachable": self.reachable,
            "degraded": self.degraded,
            "reason": self.reason,
            "pipeline": self.pipeline,
        }


async def _maybe_call(fn: Callable[..., Any] | None, *args: Any) -> None:
    """Invoke a sync or async callback without letting it break the relay."""
    if fn is None:
        return
    try:
        result = fn(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("voice-loop callback failed: %s", exc)


class VoiceLoopClient:
    """Client for the speech-to-speech sidecar control plane + relay."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        control_url: str | None = None,
        pipeline_ws_url: str | None = None,
        api_key: str | None = None,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
    ) -> None:
        self.enabled = bool(
            enabled
            if enabled is not None
            else os.getenv("VOICE_LOOP__ENABLED", "0").lower() in ("1", "true", "yes")
        )
        self.control_url = control_url or os.getenv("VOICE_LOOP__CONTROL_URL", DEFAULT_CONTROL_URL)
        self.pipeline_ws_url = pipeline_ws_url or os.getenv(
            "VOICE_LOOP__PIPELINE_WS_URL", DEFAULT_PIPELINE_WS_URL
        )
        self.api_key = api_key or os.getenv("VOICE_LOOP__API_KEY") or None
        self.health_timeout_s = health_timeout_s
        self._headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.last_error: str | None = None
        # Cached probe state (guarded by a lock; /api/status calls are sync).
        self._probe_lock = threading.Lock()
        self._probe_ts = 0.0
        self._probe_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------ health
    async def _probe_control(self) -> dict[str, Any]:
        """One HTTP probe of the sidecar control plane. Never raises."""
        try:
            async with httpx.AsyncClient(timeout=self.health_timeout_s) as client:
                resp = await client.get(f"{self.control_url}/health", headers=self._headers)
            if resp.status_code == 200:
                return {"reachable": True, "pipeline": resp.json()}
            return {"reachable": False, "reason": f"control plane HTTP {resp.status_code}"}
        except Exception as exc:
            self.last_error = str(exc)
            return {"reachable": False, "reason": f"control plane unreachable: {exc}"}

    async def health(self) -> VoiceLoopState:
        if not self.enabled:
            return VoiceLoopState(enabled=False)
        probe = await self._probe_control()
        reachable = bool(probe.get("reachable"))
        pipeline = probe.get("pipeline") or {}
        pipeline_running = bool(pipeline.get("pipeline_running", reachable))
        degraded = not reachable or not pipeline_running or bool(pipeline.get("last_error"))
        return VoiceLoopState(
            enabled=True,
            reachable=reachable,
            degraded=degraded,
            reason=probe.get("reason") or pipeline.get("last_error"),
            pipeline=pipeline,
        )

    def capability_status(self) -> dict[str, Any]:
        """Synchronous, cached, non-raising probe suitable for /api/status.

        Uses ``asyncio.run`` when no event loop is running (the normal
        "/api/status" request path inside uvicorn). Under ``pytest-asyncio``
        the cached probe is bypassed and returns a fresh fallback so the
        test event loop is never starved.
        """
        if not self.enabled:
            return {"enabled": False, "degraded": False, "reachable": False, "reason": "voice loop disabled"}
        now = time.monotonic()
        with self._probe_lock:
            if self._probe_cache is not None and (now - self._probe_ts) <= PROBE_CACHE_TTL_S:
                return dict(self._probe_cache)
            # Cache miss — avoid ``asyncio.run`` when an event loop is
            # already active (e.g. test context). The caller in the
            # orchestrator also calls ``health()`` directly.
            try:
                asyncio.get_running_loop()
                # Loop is running; the caller is testing or in an async
                # context that doesn't want sync-wait. Return a fresh
                # non-cached degraded marker that won't poison the cache.
                return {"enabled": True, "degraded": True, "reachable": False, "reason": "probe skipped (loop running)"}
            except RuntimeError:
                state = asyncio.run(self.health())
                self._probe_cache = state.to_dict()
                self._probe_ts = now
            return dict(self._probe_cache)

    # ----------------------------------------------------------------- control
    async def start(self) -> dict[str, Any]:
        """Ask the sidecar to (re)start the pipeline."""
        try:
            async with httpx.AsyncClient(timeout=self.health_timeout_s) as client:
                resp = await client.post(f"{self.control_url}/start", headers=self._headers)
            return {"ok": resp.status_code == 200, "detail": resp.text[:300]}
        except Exception as exc:
            return {"ok": False, "detail": str(exc)}

    async def stop(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.health_timeout_s) as client:
                resp = await client.post(f"{self.control_url}/stop", headers=self._headers)
            return {"ok": resp.status_code == 200, "detail": resp.text[:300]}
        except Exception as exc:
            return {"ok": False, "detail": str(exc)}

    # ------------------------------------------------------------------- relay
    async def relay(
        self,
        browser_ws: Any,
        *,
        callbacks: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """Full-duplex proxy between a browser WebSocket and the pipeline.

        ``browser_ws`` must expose ``receive_json`` / ``send_json`` / ``close``
        (a FastAPI ``WebSocket`` or a stand-in with the same surface).

        Callbacks (all optional): ``on_user_transcript(text)``,
        ``on_assistant_start()``, ``on_assistant_end()``,
        ``on_assistant_text(text)``.
        """
        callbacks = callbacks or {}
        import websockets  # lazily; uvicorn[standard] ships it

        try:
            pipeline_ws = await websockets.connect(
                self.pipeline_ws_url, open_timeout=5.0, max_size=8 * 1024 * 1024
            )
        except Exception as exc:
            self.last_error = str(exc)
            try:
                await browser_ws.send_json(
                    {"type": "error", "error": {"message": f"voice loop pipeline unreachable: {exc}"}}
                )
                await browser_ws.close()
            except Exception:
                pass
            return

        speaking = {"value": False}

        async def pump_pipeline_to_browser() -> None:
            try:
                async for raw in pipeline_ws:
                    try:
                        ev = json.loads(raw)
                    except (ValueError, TypeError):
                        ev = {"type": "raw", "payload": raw[:512]}
                    await self._handle_server_event(
                        ev, speaking=speaking, callbacks=callbacks
                    )
                    try:
                        await browser_ws.send_json(ev)
                    except Exception:
                        return
            except Exception as exc:
                logger.info("pipeline\u2192browser pump ended: %s", exc)

        pump_task = asyncio.create_task(pump_pipeline_to_browser())
        try:
            while True:
                msg = await browser_ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") in ("close", "control.close"):
                    break
                await pipeline_ws.send(json.dumps(msg))
        except Exception:
            pass  # browser disconnected or sent non-JSON
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await pipeline_ws.close()
            except Exception:
                pass

    async def _handle_server_event(
        self,
        ev: dict[str, Any],
        *,
        speaking: dict[str, bool],
        callbacks: dict[str, Callable[..., Any]],
    ) -> None:
        """Snoop the OpenAI Realtime event stream for affect-relevant signals.

        Extracted as a method so tests can drive the snooping without a real
        pipeline or browser socket.
        """
        etype = ev.get("type", "")

        if etype == "conversation.item.input_audio_transcription.completed":
            text = ev.get("transcript") or ev.get("text") or ""
            if text.strip():
                await _maybe_call(callbacks.get("on_user_transcript"), text.strip())

        elif etype == "response.audio.delta":
            if not speaking["value"]:
                speaking["value"] = True
                await _maybe_call(callbacks.get("on_assistant_start"))

        elif etype in ("response.output_audio.done", "response.done", "response.output_text.done"):
            if speaking["value"]:
                speaking["value"] = False
                await _maybe_call(callbacks.get("on_assistant_end"))

        elif etype == "response.output_text.delta":
            delta = ev.get("delta") or ev.get("text") or ""
            if delta.strip():
                await _maybe_call(callbacks.get("on_assistant_text"), delta.strip())

        elif etype == "conversation.item.created":
            item = ev.get("item") or {}
            if item.get("role") == "assistant":
                content = item.get("content") or []
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") in ("output_text", "text")
                ]
                text = " ".join(parts).strip()
                if text:
                    await _maybe_call(callbacks.get("on_assistant_text"), text)