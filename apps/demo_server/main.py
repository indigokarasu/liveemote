from __future__ import annotations
import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routes import build_router
from .websocket_api import websocket_endpoint
from hermes_avatar.demo.demo_orchestrator import DemoOrchestrator

def create_app(args=None) -> FastAPI:
    args = args or argparse.Namespace(character="./character_input", renderer="livetalking", voice_backend="luxtts", transport="webrtc", agent_mode="fake", agent_url=None, agent_harness="generic", hermes_mode=None)
    app = FastAPI(title="Hermes Live Avatar Demo")
    static = Path(__file__).with_name("static")
    agent_mode = getattr(args, "agent_mode", None) or getattr(args, "hermes_mode", None) or "fake"
    app.state.orchestrator = DemoOrchestrator(args.character, args.renderer, args.voice_backend, agent_mode, agent_url=getattr(args, "agent_url", None), agent_harness=getattr(args, "agent_harness", "generic"))
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    app.include_router(build_router(str(static)))
    app.websocket("/ws")(websocket_endpoint)
    return app

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--character", default="./character_input")
    p.add_argument("--renderer", default="livetalking", choices=["livetalking", "deeplivecam"])
    p.add_argument("--voice-backend", default="luxtts", choices=["luxtts", "elevenlabs", "moss", "none"])
    p.add_argument("--transport", default="webrtc", choices=["webrtc", "virtualcam", "browser"])
    p.add_argument("--agent-mode", default=None, choices=["fake", "external", "offline", "none", "openclaw", "hermes", "deerflow"])
    p.add_argument("--agent-url", default=None)
    p.add_argument("--agent-harness", default="generic")
    p.add_argument("--hermes-mode", default=None, choices=["fake", "external", "offline", "none"], help="Backward-compatible alias for --agent-mode")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port)
