from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from hermes_avatar.demo.meeting_join import MeetingJoinError

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

def build_router(static_dir: str) -> APIRouter:
    router = APIRouter()
    @router.get("/")
    def index():
        return FileResponse(f"{static_dir}/index.html")
    @router.get("/api/audio")
    def audio(path: str, request: Request):
        audio_path = Path(path).resolve()
        roots = request.app.state.orchestrator.safe_audio_roots()
        if not audio_path.exists() or audio_path.suffix.lower() != ".wav":
            raise HTTPException(status_code=404, detail="Audio not found")
        if not any(audio_path.is_relative_to(root) for root in roots):
            raise HTTPException(status_code=403, detail="Audio path is outside the voice cache")
        return FileResponse(str(audio_path), media_type="audio/wav")
    @router.get("/api/status")
    def status(request: Request):
        return request.app.state.orchestrator.status()
    @router.post("/api/event")
    def event(payload: EventRequest, request: Request):
        return request.app.state.orchestrator.apply_event(payload.event)
    @router.post("/api/speak")
    async def speak(payload: SpeakRequest, request: Request):
        return await request.app.state.orchestrator.speak_test(payload.text)
    @router.post("/api/mode")
    def mode(payload: ModeRequest, request: Request):
        return request.app.state.orchestrator.set_policy_mode(payload.mode)
    @router.post("/api/character")
    def character(payload: CharacterRequest, request: Request):
        try:
            return request.app.state.orchestrator.set_character(payload.character_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    @router.post("/api/style")
    def style(payload: StyleRequest, request: Request):
        try:
            return request.app.state.orchestrator.set_style(payload.style_id, payload.sync_background)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    @router.post("/api/background")
    def background(payload: BackgroundRequest, request: Request):
        try:
            return request.app.state.orchestrator.set_background(payload.background_id, payload.sync_background)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    @router.post("/api/workflow")
    def workflow(payload: WorkflowRequest, request: Request):
        try:
            return request.app.state.orchestrator.apply_workflow(payload.workflow)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    @router.post("/api/trigger/{state}")
    def trigger(state: str, request: Request):
        return request.app.state.orchestrator.trigger(state)
    @router.post("/api/character/select")
    def select_character(payload: CharacterSelectRequest, request: Request):
        return request.app.state.orchestrator.select_character(payload.character_path)
    @router.post("/api/meeting/join")
    def join_meeting(payload: MeetingJoinRequest, request: Request):
        try:
            return request.app.state.orchestrator.join_meeting(payload.meeting_url, payload.display_name)
        except MeetingJoinError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    @router.post("/api/meeting/leave")
    def leave_meeting(request: Request):
        return request.app.state.orchestrator.leave_meeting()
    return router
