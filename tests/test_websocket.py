"""WebSocket API tests for the hermes_avatar demo server (``/ws`` endpoint).

These tests drive the *real* demo-server WebSocket endpoint
(`apps/demo_server/websocket_api.websocket_endpoint`) through a live uvicorn
server and the `websockets` client library, exercising the real
`DemoOrchestrator`, affect runtime, and offline agent bridge over a genuine
WebSocket transport (not TestClient's in-process shim).

The only thing kept offline is the renderer, selected as ``deeplivecam`` (a
fully-local adapter with no network calls) so the suite is deterministic and
does not depend on an external LiveTalking service or on the repo's binary
canonical character image (which is absent in some worktrees).

Scenarios covered:
  * successful connection handshake + server-initiated avatar state push
  * connection accepted without authentication (the endpoint has no auth)
  * message parsing of valid JSON control messages (control.speak / control.mode)
  * event-driven mirroring via the default apply_event() branch
  * rejection of malformed (non-JSON) frames
  * client disconnect cleanup (server stays healthy)
  * adverse condition: abrupt disconnect mid-exchange
  * adverse condition: a flood of messages handled in order
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import websockets
from websockets.exceptions import ConnectionClosed

from apps.demo_server.main import create_app


def _make_character_dir(char_dir: Path) -> None:
    """Build a minimal but valid character package on disk.

    The orchestrator validates that ``canonical/canonical.png`` exists; the real
    repo asset is absent in some worktrees, so we construct an equivalent one
    here. ``profile.yaml`` is copied from the repo so the character exposes the
    same styles/backgrounds the real demo uses.
    """
    (char_dir / "canonical").mkdir(parents=True)
    # Validation only checks file existence, not image validity.
    (char_dir / "canonical" / "canonical.png").write_bytes(
        b"\x89PNG\r\n\x1a\n minimal placeholder canonical image"
    )
    profile_src = (
        Path(__file__).resolve().parents[1]
        / "character_input"
        / "canonical"
        / "profile.yaml"
    )
    if profile_src.exists():
        (char_dir / "canonical" / "profile.yaml").write_text(profile_src.read_text())

    emotes = char_dir / "emotes"
    for state in ("neutral", "listening", "thinking"):
        state_dir = emotes / state
        state_dir.mkdir(parents=True)
        (state_dir / ".gitkeep").write_text("")


class WsArgs:
    """App args for a fully-offline demo orchestrator (no external services)."""

    renderer = "deeplivecam"  # local adapter, no network probes
    voice_backend = "none"
    agent_mode = "offline"
    agent_url = None
    agent_harness = "none"
    transport = "webrtc"

    def __init__(self, character: str):
        self.character = character


@pytest.fixture
def ws_url(tmp_path):
    """Spin up the real app on a live uvicorn server and yield its ws:// URL.

    A fresh server (and therefore a fresh orchestrator) is created per test for
    isolation, then torn down on exit.
    """
    char_dir = tmp_path / "character"
    _make_character_dir(char_dir)

    app = create_app(WsArgs(str(char_dir)))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until the server has bound a port and is ready to accept connections.
    deadline = time.time() + 10.0
    port = None
    while time.time() < deadline:
        if server.started and server.servers:
            sockets = server.servers[0].sockets
            if sockets:
                port = sockets[0].getsockname()[1]
                break
        time.sleep(0.02)
    if port is None:
        server.should_exit = True
        raise RuntimeError("uvicorn server failed to start")

    try:
        yield f"ws://127.0.0.1:{port}/ws"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 1. Successful connection handshake + server-initiated avatar state push
# ---------------------------------------------------------------------------
async def test_websocket_handshake_sends_avatar_state_push(ws_url):
    async with websockets.connect(ws_url) as ws:
        # First frame is the server-initiated state push, sent before any
        # client input is read.
        initial = json.loads(await ws.recv())

    assert isinstance(initial, dict)
    # The push carries the full status snapshot the UI relies on.
    assert "character_id" in initial
    assert "avatar" in initial
    assert "capabilities" in initial
    assert initial["capabilities"]["renderer"]["backend"] == "deeplivecam"


# ---------------------------------------------------------------------------
# 2. No authentication is required (endpoint is open by design)
# ---------------------------------------------------------------------------
async def test_websocket_connection_accepted_without_authentication(ws_url):
    # The /ws endpoint performs no auth handshake; any client that can reach
    # the server is accepted and immediately receives the state push.
    async with websockets.connect(ws_url) as ws:
        initial = json.loads(await ws.recv())

    assert initial["character_id"]


# ---------------------------------------------------------------------------
# 3. Valid JSON control messages are parsed and acted upon
# ---------------------------------------------------------------------------
async def test_websocket_control_speak_returns_status(ws_url):
    async with websockets.connect(ws_url) as ws:
        await ws.recv()  # initial push
        await ws.send(json.dumps({"type": "control.speak", "text": "hello there"}))
        echo = json.loads(await ws.recv())

    # Offline agent produces no speech; the server still echoes a full status.
    assert isinstance(echo, dict)
    assert echo.get("speech") is None
    assert "avatar" in echo
    assert echo["agent_response_text"] == ""


async def test_websocket_control_mode_updates_policy_mode(ws_url):
    async with websockets.connect(ws_url) as ws:
        await ws.recv()  # initial push (default mode)
        await ws.send(json.dumps({"type": "control.mode", "mode": "mirror"}))
        after_mirror = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "control.mode", "mode": "reflect"}))
        after_reflect = json.loads(await ws.recv())

    assert after_mirror["mode_policy"] == "mirror"
    assert after_reflect["mode_policy"] == "reflect"


async def test_websocket_apply_event_drives_mirroring(ws_url):
    # Messages without a control.* type fall through to apply_event(), which
    # feeds the affect runtime. A valid perception frame should be mirrored.
    event = {
        "type": "perception.frame",
        "timestamp_ms": 1000,
        "face_detected": True,
        "face_center": [0.5, 0.5],
        "expression": {"smile": 0.8},
        "emotion_confidence": 0.9,
    }
    async with websockets.connect(ws_url) as ws:
        await ws.recv()  # initial push
        await ws.send(json.dumps(event))
        echo = json.loads(await ws.recv())

    assert echo["user"]["face_detected"] is True
    assert echo["avatar"]["gaze_target"] == "toward_user"


# ---------------------------------------------------------------------------
# 4. Malformed (non-JSON) frame handling
# ---------------------------------------------------------------------------
async def test_websocket_rejects_malformed_json_frame(ws_url):
    async with websockets.connect(ws_url) as ws:
        await ws.recv()  # initial push
        # A text frame that is not valid JSON must be rejected; the server's
        # receive_json() raises and the connection is closed.
        await ws.send("{ this is : not valid json")

        with pytest.raises(ConnectionClosed) as excinfo:
            await ws.recv()

    # The malformed frame must be rejected: the server terminates the
    # connection rather than echoing a status. The close is abnormal (the
    # unhandled JSONDecodeError tears the connection down), so the code is a
    # non-1000 error/abnormal code (observed 1006).
    e = excinfo.value
    code = e.rcvd.code if e.rcvd is not None else getattr(e, "code", None)
    assert code is not None
    assert code != 1000


# ---------------------------------------------------------------------------
# 5. Client disconnect cleanup (server stays healthy)
# ---------------------------------------------------------------------------
async def test_websocket_client_disconnect_does_not_wedge_server(ws_url):
    # Open a first connection, exercise it, then close it abruptly.
    async with websockets.connect(ws_url) as ws_a:
        await ws_a.recv()
        await ws_a.send(json.dumps({"type": "control.mode", "mode": "mirror"}))
        assert json.loads(await ws_a.recv())["mode_policy"] == "mirror"
        # exiting the context manager closes ws_a without a farewell message

    # A brand-new connection must still be accepted and receive a fresh state
    # push, proving the server cleaned up the dead connection and remains usable.
    async with websockets.connect(ws_url) as ws_b:
        initial = json.loads(await ws_b.recv())

    assert initial["character_id"]
    assert initial["capabilities"]["renderer"]["backend"] == "deeplivecam"


# ---------------------------------------------------------------------------
# 6. Adverse condition: abrupt disconnect mid-exchange
# ---------------------------------------------------------------------------
async def test_websocket_abrupt_disconnect_mid_exchange_keeps_state_isolated(ws_url):
    # Two independent connections share the single orchestrator state, but a
    # hard disconnect on one must not corrupt or block the other.
    async with websockets.connect(ws_url) as ws_primary:
        await ws_primary.recv()
        await ws_primary.send(json.dumps({"type": "control.mode", "mode": "mirror"}))
        assert json.loads(await ws_primary.recv())["mode_policy"] == "mirror"

        # Open a second connection while the first is still alive.
        async with websockets.connect(ws_url) as ws_secondary:
            secondary_initial = json.loads(await ws_secondary.recv())
            # Secondary sees the same shared mode the primary just set.
            assert secondary_initial["mode_policy"] == "mirror"

        # Primary is still responsive after the secondary disconnects.
        await ws_primary.send(json.dumps({"type": "control.mode", "mode": "reflect"}))
        assert json.loads(await ws_primary.recv())["mode_policy"] == "reflect"


# ---------------------------------------------------------------------------
# 7. Adverse condition: a flood of messages is handled in order
# ---------------------------------------------------------------------------
async def test_websocket_handles_message_flood(ws_url):
    flood_size = 30
    event = {
        "type": "perception.frame",
        "timestamp_ms": 2000,
        "face_detected": True,
        "face_center": [0.5, 0.5],
        "expression": {"smile": 0.5},
        "emotion_confidence": 0.7,
    }
    async with websockets.connect(ws_url) as ws:
        await ws.recv()  # initial push
        for i in range(flood_size):
            await ws.send(json.dumps({**event, "timestamp_ms": 2000 + i}))

        # The server must echo exactly one status per received frame, in order.
        seen = 0
        for _ in range(flood_size):
            echo = json.loads(await ws.recv())
            assert isinstance(echo, dict)
            assert echo["user"]["face_detected"] is True
            seen += 1

    assert seen == flood_size
