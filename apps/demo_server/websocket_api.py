from __future__ import annotations
from fastapi import WebSocket

async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    app = ws.app
    await ws.send_json(app.state.orchestrator.status())
    while True:
        message = await ws.receive_json()
        if message.get("type") == "control.speak":
            status = await app.state.orchestrator.speak_test(message.get("text", "Test line"))
        elif message.get("type") == "control.mode":
            status = app.state.orchestrator.set_policy_mode(message.get("mode", "reflect"))
        else:
            status = app.state.orchestrator.apply_event(message)
        await ws.send_json(status)
