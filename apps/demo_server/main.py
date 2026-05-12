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
    args = args or argparse.Namespace(character="./character_input", renderer="livetalking", voice_backend="luxtts", transport="webrtc", hermes_mode="fake")
    app = FastAPI(title="Hermes Live Avatar Demo")
    static = Path(__file__).with_name("static")
    app.state.orchestrator = DemoOrchestrator(args.character, args.renderer, args.voice_backend, args.hermes_mode)
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    app.include_router(build_router(str(static)))
    app.websocket("/ws")(websocket_endpoint)
    return app

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--character", default="./character_input")
    p.add_argument("--renderer", default="livetalking", choices=["livetalking", "deeplivecam"])
    p.add_argument("--voice-backend", default="luxtts", choices=["luxtts", "elevenlabs", "moss"])
    p.add_argument("--transport", default="webrtc", choices=["webrtc", "virtualcam", "browser"])
    p.add_argument("--hermes-mode", default="fake", choices=["fake", "external"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port)
