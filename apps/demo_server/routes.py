"""HTTP routes for the LiveEmote demo.

LiveEmote is a real AI avatar keyed off the user's webcam affect. The avatar
has its own identity (canonical image + emotes) and its own autonomous motion;
the webcam is a signal source (focus + energy), never a face-reenactment
target. This module exposes the routes the front-end needs to:

* query the avatar + character state (``/api/status``);
* push perception frames from the webcam (``/api/perception/video``) so the
  server-side tracker can compute rich affect signals;
* push generic affect events from older sensor paths (``/api/event``);
* speak / trigger / drive the avatar over HTTP and websocket;
* serve character assets (``/api/character/asset``) for the browser avatar
  to render the avatar's own face and emotes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from hermes_avatar.character.ingest import build_asset_index


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


class PerceptionFrameRequest(BaseModel):
    """A single base64-encoded webcam frame for server-side perception."""

    image: str | None = None
    timestamp_ms: int | None = None


def build_router(static_dir: str) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def index():
        return FileResponse(f"{static_dir}/index.html")

    # ---- Audio playback ------------------------------------------------

    @router.get("/api/audio")
    def audio(path: str, request: Request):
        audio_path = Path(path).resolve()
        roots = [Path(request.app.state.orchestrator.config.voice.cache_dir).resolve()]
        if audio_path.suffix.lower() != ".wav":
            raise HTTPException(status_code=404, detail="Audio not found")
        if not any(audio_path.is_relative_to(root) for root in roots):
            raise HTTPException(status_code=403, detail="Audio path is outside the voice cache")
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio not found")
        return FileResponse(str(audio_path), media_type="audio/wav")

    # ---- Character asset serving --------------------------------------

    @router.get("/api/character/asset")
    def character_asset(path: str, request: Request):
        orchestrator = request.app.state.orchestrator
        if not path:
            raise HTTPException(status_code=400, detail="asset path required")
        asset = Path(path).resolve()
        allowed_roots = [Path(root).resolve() for root in orchestrator.character_roots.values()]
        try:
            canonical_root = (Path(list(orchestrator.character_roots.values())[0]) / "canonical").resolve()
            allowed_roots.append(canonical_root)
        except Exception:
            pass
        if not any(asset.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="asset path is outside character roots")
        if not asset.exists() or not asset.is_file():
            raise HTTPException(status_code=404, detail="character asset not found")
        suffix = asset.suffix.lower()
        media = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
        }.get(suffix, "application/octet-stream")
        return FileResponse(str(asset), media_type=media)

    # ---- Server-side perception --------------------------------------

    @router.post("/api/perception/video")
    async def perception_video(payload: PerceptionFrameRequest, request: Request):
        orchestrator = request.app.state.orchestrator
        tracker = getattr(orchestrator, "tracker", None)
        timestamp_ms = payload.timestamp_ms or 0
        if not tracker or not tracker.is_available():
            return {
                "tracker": (tracker.kind() if tracker else "null"),
                "available": False,
                "signals": {},
                "message": "Server-side perception tracker unavailable; install [perception] extras.",
            }
        signals = tracker.process_frame(payload.image, timestamp_ms=timestamp_ms)
        # Push through the runtime so policy is updated.
        orchestrator.apply_event({"type": "perception.frame", **signals.to_dict()})
        return {
            "tracker": tracker.kind(),
            "available": True,
            "signals": signals.to_dict(),
        }

    @router.get("/api/perception/info")
    def perception_info(request: Request):
        orchestrator = request.app.state.orchestrator
        tracker = getattr(orchestrator, "tracker", None)
        return {
            "available": bool(tracker and tracker.is_available()),
            "backend": tracker.kind() if tracker else "null",
            "no_face_reenactment": True,
            "no_face_swap": True,
            "drives": "AvatarBehaviorState via focus + energy signals; webcam is signal source only.",
        }

    # ---- Core status / event ingest / control ------------------------

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
        orchestrator = request.app.state.orchestrator
        selected = build_asset_index(payload.character_path)
        orchestrator.character_roots[selected.character_id] = Path(payload.character_path)
        orchestrator.character_catalog[selected.character_id] = selected
        return orchestrator.set_character(selected.character_id)

    @router.post("/api/meeting/join")
    def join_meeting(payload: MeetingJoinRequest, request: Request):
        try:
            return request.app.state.orchestrator.join_meeting(payload.meeting_url, payload.display_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/meeting/leave")
    def leave_meeting(request: Request):
        return request.app.state.orchestrator.leave_meeting()

    return router
