"""``sidecar/voice_loop/app.py`` — supervisor for the Hugging Face speech-to-speech pipeline.

The Heavy lifting for a live voice conversation (Silero VAD -> STT -> LLM ->
TTS) is done by the `speech-to-speech` package, which ships an OpenAI-Realtime-
compatible WebSocket server. It pulls in torch/transformers/nltk and can take
minutes to install, so it MUST NOT live inside the main LiveEmote process.

This daemon follows the existing sidecar pattern (``sidecar/app.py``):

* It runs the pipeline as a **managed subprocess** and exposes a small HTTP
  control plane (health / start / stop / config) on :8766.
* The pipeline itself listens on :8765 for browser audio at
  ``ws://<host>:<port>/v1/realtime`` (OpenAI Realtime GA protocol).
* The main LiveEmote server never imports torch. Its ``VoiceLoopClient``
  probes this control plane for health and relays the browser WebSocket to
  the pipeline socket, snooping transcripts so the avatar can animate while
  the voice conversation happens.

Run (from a venv that has ``pip install speech-to-speech``):

    python -m sidecar.voice_loop.app --host 0.0.0.0 --port 8766

Configuration is entirely env-var driven so it flows through the same overlay
mechanism as the rest of the project:

* ``VOICE_LOOP__AUTOSTART``            (default "1") spawn the pipeline on boot
* ``VOICE_LOOP__PIPELINE_WS_HOST``     (default "127.0.0.1")
* ``VOICE_LOOP__PIPELINE_WS_PORT``     (default "8765")
* ``VOICE_LOOP__STT__MODE``            (default "parakeet-tdt"; also whisper, faster-whisper, mlx-audio-whisper, whisper-mlx, paraformer)
* ``VOICE_LOOP__LLM__MODE``            (default "chat-completions"; also responses-api, mlx-lm, transformers)
* ``VOICE_LOOP__LLM__BASE_URL``        (default: $OPENAI_COMPATIBLE_BASE_URL) -> --responses_api_base_url
* ``VOICE_LOOP__LLM__MODEL``           (default: $OPENAI_COMPATIBLE_MODEL)    -> --model_name
* ``VOICE_LOOP__LLM__API_KEY``         (default: $OPENAI_COMPATIBLE_API_KEY)  -> --responses_api_api_key
* ``VOICE_LOOP__TTS__MODE``            (default "qwen3"; also kokoro, pocket, chatTTS, facebookMMS)
* ``VOICE_LOOP__NUM_PIPELINES``        (default "1")
* ``VOICE_LOOP__LOG_TAIL``             (default "40") lines of pipeline log kept
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger("sidecar.voice_loop")


# -----------------------------------------------------------------------------
# Config — env-var overlay with the main process's OpenAI-compatible env vars
# as the LLM defaults so the avatar's existing LLM key works out of the box.
# -----------------------------------------------------------------------------
def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


@dataclass
class Config:
    autostart: bool = field(
        default_factory=lambda: _env("VOICE_LOOP__AUTOSTART", "1") in ("1", "true", "yes")
    )
    ws_host: str = field(default_factory=lambda: _env("VOICE_LOOP__PIPELINE_WS_HOST", "127.0.0.1") or "127.0.0.1")
    ws_port: int = field(default_factory=lambda: int(_env("VOICE_LOOP__PIPELINE_WS_PORT", "8765") or "8765"))
    stt_mode: str = field(default_factory=lambda: _env("VOICE_LOOP__STT__MODE", "parakeet-tdt") or "parakeet-tdt")
    llm_mode: str = field(default_factory=lambda: _env("VOICE_LOOP__LLM__MODE", "chat-completions") or "chat-completions")
    llm_base_url: str | None = field(
        default_factory=lambda: _env("VOICE_LOOP__LLM__BASE_URL", _env("OPENAI_COMPATIBLE_BASE_URL"))
    )
    llm_model: str | None = field(
        default_factory=lambda: _env("VOICE_LOOP__LLM__MODEL", _env("OPENAI_COMPATIBLE_MODEL"))
    )
    llm_api_key: str | None = field(
        default_factory=lambda: _env("VOICE_LOOP__LLM__API_KEY", _env("OPENAI_COMPATIBLE_API_KEY"))
    )
    tts_mode: str = field(default_factory=lambda: _env("VOICE_LOOP__TTS__MODE", "qwen3") or "qwen3")
    num_pipelines: int = field(default_factory=lambda: int(_env("VOICE_LOOP__NUM_PIPELINES", "1") or "1"))
    log_tail: int = field(default_factory=lambda: int(_env("VOICE_LOOP__LOG_TAIL", "40") or "40"))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ws_url": f"ws://{self.ws_host}:{self.ws_port}/v1/realtime",
            "stt": self.stt_mode,
            "llm_mode": self.llm_mode,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_api_key_configured": bool(self.llm_api_key),
            "tts": self.tts_mode,
            "num_pipelines": self.num_pipelines,
            "autostart": self.autostart,
        }
        return d


def build_pipeline_args(cfg: Config) -> list[str]:
    """Translate the config into ``speech-to-speech serve`` CLI arguments.

    The upstream CLI (0.1.x) is a flat ``serve`` subcommand — the old
    ``--module.*`` / ``--mode realtime`` spelling is deprecated. The
    ``serve`` command hosts the OpenAI-Realtime WebSocket server directly;
    both LLM API backends (chat-completions and responses-api) share the
    ``--responses_api_*`` connection flags.
    """
    args = [
        "--host", cfg.ws_host,
        "--port", str(cfg.ws_port),
        "--stt", cfg.stt_mode,
        "--llm_backend", cfg.llm_mode,
        "--tts", cfg.tts_mode,
        "--num_pipelines", str(cfg.num_pipelines),
        "--enable_live_transcription",
        "--log_level", "info",
    ]
    if cfg.llm_base_url:
        args += ["--responses_api_base_url", cfg.llm_base_url]
    if cfg.llm_model:
        args += ["--model_name", cfg.llm_model]
    if cfg.llm_api_key:
        args += ["--responses_api_api_key", cfg.llm_api_key]
    return args


# -----------------------------------------------------------------------------
# Process manager — spawns the pipeline, captures a rolling log tail, and
# tracks exit state so /health can say why it is down.
# -----------------------------------------------------------------------------
class PipelineManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.proc: asyncio.subprocess.Process | None = None
        self.log_tail: deque[str] = deque(maxlen=max(cfg.log_tail, 5))
        self.last_error: str | None = None
        self.last_exit_code: int | None = None
        self._readers: list[asyncio.Task] = []

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    def _tail(self) -> list[str]:
        return list(self.log_tail)

    async def start(self) -> dict[str, Any]:
        if self.running:
            return self.status()
        args = [sys.executable, "-m", "speech_to_speech.s2s_pipeline", "serve", *build_pipeline_args(self.cfg)]
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        except Exception as exc:
            self.last_error = f"spawn failed: {exc}"
            self.proc = None
            return self.status()

        async def _drain() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            async for line in self.proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                self.log_tail.append(text)
            self.last_exit_code = self.proc.returncode
            if self.proc.returncode not in (0, None):
                self.last_error = f"pipeline exited with code {self.proc.returncode}"
            logger.info("voice-loop pipeline exited (code=%s)", self.proc.returncode)

        self._readers = [asyncio.create_task(_drain())]
        self.last_error = None
        return self.status()

    async def stop(self) -> dict[str, Any]:
        if not self.running:
            self.last_error = None
            return self.status()
        assert self.proc is not None
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=8)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        for t in self._readers:
            t.cancel()
        self._readers = []
        self.proc = None
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "pipeline_running": self.running,
            "pipeline_exit_code": self.last_exit_code,
            "last_error": self.last_error,
            "log_tail": self._tail()[-8:],
        }


# -----------------------------------------------------------------------------
# Realtime WS reachability probe — a successful HTTP 101 upgrade (even if the
# server closes immediately without a session) proves the pipeline is serving.
# -----------------------------------------------------------------------------
async def ws_reachable(host: str, port: int, timeout_s: float = 3.0) -> bool:
    try:
        import websockets  # lazily; the pipeline venv always ships it
        url = f"ws://{host}:{port}/v1/realtime"
        async with websockets.connect(url, open_timeout=timeout_s, close_timeout=1.0) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except Exception:
                pass  # any frame (or close) means the server answered
            return True
    except Exception:
        return False


_cfg = Config()
_manager = PipelineManager(_cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _cfg.autostart and not _manager.running:
        try:
            await _manager.start()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("autostart failed: %s", exc)
    yield
    if _manager.running:
        try:
            await _manager.stop()
        except Exception:  # pragma: no cover
            pass


app = FastAPI(title="LiveEmote Voice-Loop Sidecar", version="0.1.0", lifespan=lifespan)


async def _health_body() -> dict[str, Any]:
    reachable = False
    if _manager.running:
        try:
            reachable = await ws_reachable(_cfg.ws_host, _cfg.ws_port)
        except Exception:
            reachable = False
    return {
        "status": "ok" if (reachable and not _manager.last_error) else "degraded",
        "pipeline_running": _manager.running,
        "ws_reachable": reachable,
        "ws_url": f"ws://{_cfg.ws_host}:{_cfg.ws_port}/v1/realtime",
        "last_error": _manager.last_error,
        "log_tail": _manager._tail()[-5:],
    }


@app.get("/health")
async def health() -> JSONResponse:
    body = await _health_body()
    return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)


@app.get("/status")
async def status() -> JSONResponse:
    body = await _health_body()
    return JSONResponse({**body, "config": _cfg.to_dict()})


@app.get("/config")
async def config() -> JSONResponse:
    return JSONResponse(_cfg.to_dict())


@app.post("/start")
async def start() -> JSONResponse:
    body = await _manager.start()
    return JSONResponse(body, status_code=200 if body.get("pipeline_running") else 503)


@app.post("/stop")
async def stop() -> JSONResponse:
    body = await _manager.stop()
    return JSONResponse(body)


def main() -> None:
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="Voice-loop sidecar supervisor")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    logging.basicConfig(level=os.getenv("SIDECAR_LOG_LEVEL", "INFO"))
    logger.info(
        "voice-loop sidecar on %s:%d (pipeline -> ws://%s:%d/v1/realtime)",
        args.host, args.port, _cfg.ws_host, _cfg.ws_port,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
