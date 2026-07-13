from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from typing import Any
from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.demo.meeting_join import MeetingJoinError
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

from hermes_avatar.observability.tracing import ensure_trace_id, get_trace_id

logger = logging.getLogger("hermes_avatar.demo.routes")

# Prometheus metrics
REQUEST_COUNT = Counter(
    'demo_server_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'demo_server_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_CHARACTERS = Gauge(
    'demo_server_active_characters',
    'Number of actively loaded characters'
)

SPEECH_REQUESTS = Counter(
    'demo_server_speech_requests_total',
    'Total number of speech synthesis requests',
    ['voice_backend', 'result']
)

CHARACTER_CHANGES = Counter(
    'demo_server_character_changes_total',
    'Total number of character changes'
)

STYLE_CHANGES = Counter(
    'demo_server_style_changes_total',
    'Total number of style changes'
)

BACKGROUND_CHANGES = Counter(
    'demo_server_background_changes_total',
    'Total number of background changes'
)

WORKFLOW_EXECUTIONS = Counter(
    'demo_server_workflow_executions_total',
    'Total number of workflow executions',
    ['workflow']
)

AGENT_RESPONSE_TIME = Histogram(
    'demo_server_agent_response_seconds',
    'Time taken for agent to generate response'
)

class SpeakRequest(BaseModel):
    text: str = "Test line"

class ModeRequest(BaseModel):
    mode: str

class CharacterRequest(BaseModel):
    character_id: str

class StyleRequest(BaseModel):
    style_id: str
    sync_background: bool = True

class BackgroundRequest(BaseModel):
    background_id: str
    sync_background: bool = False

class WorkflowRequest(BaseModel):
    workflow: str

class EventRequest(BaseModel):
    event: dict

class MeetingJoinRequest(BaseModel):
    meeting_url: str
    display_name: str | None = None

class CharacterSelectRequest(BaseModel):
    character_path: str

class ConfigReloadRequest(BaseModel):
    # Empty body for now, but we can extend if needed
    pass

def build_router(static_dir: str) -> APIRouter:
    async def trace_dependency(request: Request, response: Response) -> str:
        # Establish a per-request trace id once, so every downstream log line
        # (and the affect runtime tick that runs within this context) shares it.
        # Declaring ``response`` lets us stamp the id on the outgoing response
        # for end-to-end correlation without router-level middleware (which
        # APIRouter does not support).
        trace_id = ensure_trace_id()
        response.headers["X-Trace-Id"] = trace_id
        logger.info(
            "request received",
            extra={
                "trace_id": trace_id,
                "audit": {
                    "event": "http.request",
                    "method": request.method,
                    "path": request.url.path,
                },
            },
        )
        return trace_id

    router = APIRouter(dependencies=[Depends(trace_dependency)])

    @router.get("/")
    def index() -> FileResponse:
        return FileResponse(f"{static_dir}/index.html")

    @router.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @router.get("/api/audio")
    def audio(path: str, request: Request) -> Response:
        audio_path = Path(path).resolve()
        roots = [Path(request.app.state.orchestrator.config.voice.cache_dir).resolve()]
        if audio_path.suffix.lower() != ".wav":
            raise HTTPException(status_code=404, detail="Audio not found")
        if not any(audio_path.is_relative_to(root) for root in roots):
            raise HTTPException(status_code=403, detail="Audio path is outside the voice cache")
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio not found")
        return FileResponse(str(audio_path), media_type="audio/wav")

    @router.get("/api/status")
    def status(request: Request) -> dict:
        # Update active characters gauge
        orchestrator = request.app.state.orchestrator
        ACTIVE_CHARACTERS.set(len(orchestrator.character_roots))
        return orchestrator.status()

    @router.post("/api/event")
    def event(payload: EventRequest, request: Request) -> dict:
        return request.app.state.orchestrator.apply_event(payload.event)

    @router.post("/api/speak")
    async def speak(payload: SpeakRequest, request: Request) -> dict:
        orchestrator = request.app.state.orchestrator
        start_time = time.time()
        try:
            result = await orchestrator.speak_test(payload.text)
            SPEECH_REQUESTS.labels(voice_backend=orchestrator.voice_backend_name, result="success").inc()
            AGENT_RESPONSE_TIME.observe(time.time() - start_time)
            return result
        except Exception as e:
            SPEECH_REQUESTS.labels(voice_backend=orchestrator.voice_backend_name, result="error").inc()
            raise

    @router.post("/api/mode")
    def mode(payload: ModeRequest, request: Request) -> dict:
        return request.app.state.orchestrator.set_policy_mode(payload.mode)

    @router.post("/api/character")
    def character(payload: CharacterRequest, request: Request) -> dict:
        try:
            result = request.app.state.orchestrator.set_character(payload.character_id)
            CHARACTER_CHANGES.inc()
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/style")
    def style(payload: StyleRequest, request: Request) -> dict:
        try:
            result = request.app.state.orchestrator.set_style(payload.style_id, payload.sync_background)
            STYLE_CHANGES.inc()
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/background")
    def background(payload: BackgroundRequest, request: Request) -> dict:
        try:
            result = request.app.state.orchestrator.set_background(payload.background_id, payload.sync_background)
            BACKGROUND_CHANGES.inc()
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/workflow")
    def workflow(payload: WorkflowRequest, request: Request) -> dict:
        try:
            result = request.app.state.orchestrator.apply_workflow(payload.workflow)
            WORKFLOW_EXECUTIONS.labels(workflow=payload.workflow).inc()
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/trigger/{state}")
    def trigger(state: str, request: Request) -> dict:
        return request.app.state.orchestrator.trigger(state)

    @router.post("/api/character/select")
    def select_character(payload: CharacterSelectRequest, request: Request) -> dict:
        orchestrator = request.app.state.orchestrator
        selected = build_asset_index(payload.character_path)
        orchestrator.character_roots[selected.character_id] = Path(payload.character_path)
        orchestrator.character_catalog[selected.character_id] = selected
        return orchestrator.set_character(selected.character_id)

    @router.post("/api/meeting/join")
    def join_meeting(payload: MeetingJoinRequest, request: Request) -> dict:
        try:
            return request.app.state.orchestrator.join_meeting(payload.meeting_url, payload.display_name)
        except MeetingJoinError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/meeting/leave")
    def leave_meeting(request: Request) -> dict:
        return request.app.state.orchestrator.leave_meeting()

    @router.post("/api/config/reload")
    def reload_config(request: Request) -> dict:
        orchestrator = request.app.state.orchestrator
        return orchestrator.reload_config()

    @router.get("/health")
    @router.get("/api/health")
    def health(request: Request) -> dict:
        """Fast, never-raising aggregate health endpoint.

        Returns {status, components} where status is one of ok|degraded|error and
        each component reports {status, detail}. Every probe is wrapped so a single
        failing component degrades the overall status instead of 500-ing.
        """
        orchestrator = request.app.state.orchestrator
        components: dict[str, Any] = {}
        overall = "ok"

        def mark(status_val: str) -> None:
            nonlocal overall
            if status_val == "error":
                overall = "error"
            elif status_val == "degraded" and overall != "error":
                overall = "degraded"

        # --- config (loaded + hardware profile) ---
        try:
            cfg = orchestrator.config
            components["config"] = {
                "status": "ok",
                "detail": {
                    "loaded": True,
                    "hardware_profile": getattr(cfg, "hardware_profile", None),
                    "affect_update_hz": cfg.affect.update_hz,
                    "renderer_url": cfg.renderer.livetalking_url,
                },
            }
        except Exception as exc:
            components["config"] = {"status": "degraded", "detail": {"error": str(exc)}}
        mark(components["config"]["status"])

        # --- renderer (capabilities probe; also informs external reachability) ---
        renderer_caps: dict[str, Any] = {}
        try:
            renderer = orchestrator.renderer
            renderer_caps = renderer.capabilities() if hasattr(renderer, "capabilities") else {}
            online = bool(renderer_caps.get("online", False))
            components["renderer"] = {
                "status": "ok" if online else "degraded",
                "detail": {"online": online, "circuit_breaker": renderer_caps.get("circuit_breaker")},
            }
        except Exception as exc:
            components["renderer"] = {"status": "degraded", "detail": {"error": str(exc)}}
        mark(components["renderer"]["status"])

        # --- voice backend ---
        try:
            components["voice_backend"] = {
                "status": "ok",
                "detail": {"name": orchestrator.voice_backend_name},
            }
        except Exception as exc:
            components["voice_backend"] = {"status": "degraded", "detail": {"error": str(exc)}}
        mark(components["voice_backend"]["status"])

        # --- character catalog ---
        try:
            catalog = orchestrator.character_catalog
            components["character_catalog"] = {
                "status": "ok",
                "detail": {"count": len(catalog), "ids": list(catalog.keys())},
            }
        except Exception as exc:
            components["character_catalog"] = {"status": "degraded", "detail": {"error": str(exc)}}
        mark(components["character_catalog"]["status"])

        # --- external LiveTalking reachability (best-effort, reuses renderer probe) ---
        try:
            renderer = orchestrator.renderer
            url = getattr(renderer, "base_url", None)
            reachable = bool(renderer_caps.get("online", False))
            components["livetalking_reachability"] = {
                "status": "ok" if reachable else "degraded",
                "detail": {"url": url, "reachable": reachable},
            }
        except Exception as exc:
            components["livetalking_reachability"] = {
                "status": "degraded",
                "detail": {"error": str(exc)},
            }
        mark(components["livetalking_reachability"]["status"])

        return {"status": overall, "components": components}

    @router.get("/debug/affect")
    def debug_affect(request: Request) -> dict:
        """Expose the live internal affect state for development/visualization.

        Guarded by the ``debug`` config flag so it is NEVER served in
        production by default. Returns the user / conversation / avatar state,
        the active policy mode, the last agent tags, and a bounded history of
        recent ticks (each correlated with the trace id that caused it).
        """
        orchestrator = request.app.state.orchestrator
        if not getattr(orchestrator.config, "debug", False):
            raise HTTPException(status_code=404, detail="Debug endpoint disabled")
        runtime = orchestrator.runtime
        return {
            "trace_id": get_trace_id(),
            "user": runtime.user.to_dict(),
            "conversation": runtime.conversation.to_dict(),
            "avatar": runtime.avatar.to_dict(),
            "mode": runtime.mode,
            "hermes_tags": runtime.hermes_tags,
            "recent_history": runtime.history,
        }

    return router