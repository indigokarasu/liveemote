from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)
from fastapi.staticfiles import StaticFiles
from .routes import build_router, REQUEST_COUNT, REQUEST_LATENCY
from .websocket_api import websocket_endpoint
from hermes_avatar.demo.demo_orchestrator import DemoOrchestrator

def create_app(args=None) -> FastAPI:
    app = FastAPI(title="Hermes Live Avatar Demo")
    
    # Add middleware for collecting HTTP metrics
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(process_time)
        
        return response
    
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
    p.add_argument("--voice-backend", default="luxtts", choices=["luxtts", "elevenlabs", "fishaudio", "moss", "none"])
    p.add_argument("--agent-mode", default=None, choices=["fake", "external", "offline", "none", "openclaw", "hermes", "deerflow"])
    p.add_argument("--agent-url", default=None)
    p.add_argument("--agent-harness", default="generic")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    from hermes_avatar.util import configure_logging

    configure_logging()
    app = create_app(args)

    # Diagnostic: surface degraded backends at startup. Graceful passthrough is
    # the intended headless/CI behaviour (never crash), but operators should
    # know when the renderer is offline so they don't mistake passthrough for
    # full function.
    orchestrator = getattr(app.state, "orchestrator", None)
    if orchestrator is not None:
        caps = (
            orchestrator.renderer.capabilities()
            if hasattr(orchestrator.renderer, "capabilities")
            else {}
        )
        if not caps.get("online", True):
            from hermes_avatar.util.audit import audit_event, KIND_STARTUP_DEGRADED
            audit_event(
                "startup.renderer",
                KIND_STARTUP_DEGRADED,
                level=logging.WARNING,
                backend=caps.get("backend"),
            )

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=30,
        ws_max_size=8 * 1024 * 1024,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        log_level="info",
    )
    logger.info("starting demo server on ws://%s:%d", args.host, args.port)
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        logger.info("demo server stopped")