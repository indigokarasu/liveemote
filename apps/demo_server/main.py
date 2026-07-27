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
    app = FastAPI(title="Hermes Live Avatar Demo")
    static = Path(__file__).with_name("static")
    agent_mode = getattr(args, "agent_mode", None) or getattr(args, "hermes_mode", None) or "fake"
    perception_tracker = getattr(args, "perception_tracker", "mediapipe")
    app.state.orchestrator = DemoOrchestrator(
        args.character,
        args.renderer,
        args.voice_backend,
        agent_mode,
        agent_url=getattr(args, "agent_url", None),
        agent_harness=getattr(args, "agent_harness", "generic"),
        perception_tracker=perception_tracker,
    )
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    app.include_router(build_router(str(static)))
    app.websocket("/ws")(websocket_endpoint)
    return app

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--character", default="./character_input")
    p.add_argument(
        "--renderer",
        default="web",
        choices=["web", "livetalking", "deeplivecam"],
        help=(
            "web = autonomous, browser-animated avatar (default); "
            "livetalking/deeplivecam are opt-in face-reenactment / face-swap tools."
        ),
    )
    p.add_argument("--voice-backend", default="luxtts", choices=["luxtts", "elevenlabs", "moss", "none"])
    p.add_argument(
        "--agent-mode",
        default=None,
        choices=[
            "fake",
            "external",
            "offline",
            "none",
            "openclaw",
            "hermes",
            "deerflow",
            "openai",
            "openai_compatible",
        ],
        help=(
            "openai/openai_compatible = call an OpenAI-compatible chat endpoint "
            "(set OPENAI_COMPATIBLE_API_KEY and friends); fake/offline = "
            "no LLM (mirroring/reflect continues to work)."
        ),
    )
    p.add_argument("--agent-url", default=None)
    p.add_argument("--agent-harness", default="generic")
    p.add_argument(
        "--hermes-mode",
        default=None,
        choices=["fake", "external", "offline", "none"],
        help="Backward-compatible alias for --agent-mode",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--transport",
        default="webrtc",
        choices=["webrtc", "virtualcam"],
        help="Transport to start after the app boots (LiveTalking only).",
    )
    p.add_argument(
        "--perception-tracker",
        default="mediapipe",
        choices=["mediapipe", "null"],
        help="mediapipe = real face-mesh focus/energy tracker; null = no op (fallback).",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    renderer = app.state.orchestrator.renderer
    transport = getattr(args, "transport", "webrtc")
    if transport == "virtualcam":
        renderer.start_virtualcam()
    else:
        renderer.start_webrtc()
    uvicorn.run(app, host=args.host, port=args.port)
