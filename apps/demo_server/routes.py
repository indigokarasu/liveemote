from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from hermes_avatar.demo.meeting_join import MeetingJoinError

class SpeakRequest(BaseModel):
    text: str = "Test line"

class ModeRequest(BaseModel):
    mode: str

class EventRequest(BaseModel):
    event: dict

class MeetingJoinRequest(BaseModel):
    meeting_url: str
    display_name: str | None = None

def build_router(static_dir: str) -> APIRouter:
    router = APIRouter()
    @router.get("/")
    def index():
        return FileResponse(f"{static_dir}/index.html")
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
    @router.post("/api/trigger/{state}")
    def trigger(state: str, request: Request):
        return request.app.state.orchestrator.trigger(state)
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
