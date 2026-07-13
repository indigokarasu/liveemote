"""Load / stress tests for concurrent WebSocket users on the demo server.

These exercise the *real* demo server (a live uvicorn process) under
concurrency, using the fully-offline ``deeplivecam`` renderer so the suite is
deterministic and makes no external network calls. They are marked
``performance`` and therefore excluded from the default CI run; execute them
explicitly with ``pytest -m performance`` (or ``pytest tests/test_performance.py``).

Scenarios
--------
* **concurrency ceiling** -- many clients connect at once; all are accepted and
  each receives the server-initiated avatar-state push.
* **message throughput under load** -- many clients exchange messages
  simultaneously; every frame is echoed, and per-message latency stays bounded.
* **sustained operation** -- clients stream messages for a few seconds while a
  fresh probe connection verifies the server stays responsive (no degradation).
* **burst cleanup** -- a burst of connections opens and closes; a new connection
  immediately after must still be accepted (no resource leak / wedge).

All thresholds are chosen to be comfortably satisfiable on a single CPU-core
uvicorn loop driving the offline renderer, while still catching regressions
where the server drops, stalls, or wedges under concurrent load.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import websockets

from apps.demo_server.main import create_app


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
CONCURRENCY_N = 25          # simultaneous connections for the ceiling test
THROUGHPUT_CLIENTS = 10      # clients exchanging messages at once
THROUGHPUT_MSGS = 15         # messages per client in the throughput test
SUSTAINED_CLIENTS = 5        # long-lived clients for the responsiveness test
SUSTAINED_SECONDS = 2.5      # how long the sustained burst runs
BURST_N = 30                 # connections in the burst-cleanup test

# Latency ceilings (seconds). Generous enough for a CPU-only loop under load,
# tight enough to fail if the server stalls.
PER_MSG_LATENCY_CEIL = 2.0
PROBE_LATENCY_CEIL = 1.0
CONNECT_SETTLE_CEIL = 5.0
THROUGHPUT_FLOOR = 20.0      # messages/sec, aggregate, under load


def _make_character_dir(char_dir: Path) -> None:
    (char_dir / "canonical").mkdir(parents=True)
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

    renderer = "deeplivecam"
    voice_backend = "none"
    agent_mode = "offline"
    agent_url = None
    agent_harness = "none"
    transport = "webrtc"

    def __init__(self, character: str):
        self.character = character


@pytest.fixture
def live_ws_url(tmp_path):
    """Spin up the real app on a live uvicorn server and yield its ws:// URL.

    A fresh server (and therefore a fresh orchestrator) is created per test for
    isolation, then torn down on exit. Mirrors the harness in test_websocket.py.
    """
    char_dir = tmp_path / "character"
    _make_character_dir(char_dir)

    app = create_app(WsArgs(str(char_dir)))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

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
# Helpers
# ---------------------------------------------------------------------------
async def _connect_and_get_push(url: str, timeout: float = CONNECT_SETTLE_CEIL):
    ws = await websockets.connect(url)
    initial = json.loads(await asyncio.wait_for(ws.recv(), timeout))
    return ws, initial


async def _latency_client(url: str, msgs: int):
    """Connect, read the push, then exchange ``msgs`` frames, returning stats.

    Returns a dict with the connection result, the number of echoed frames, and
    the list of per-message latencies (send -> echo) in seconds.
    """
    t0 = time.monotonic()
    ws, initial = await _connect_and_get_push(url)
    connect_ok = isinstance(initial, dict) and "character_id" in initial
    latencies: list[float] = []
    received = 0
    try:
        event = {
            "type": "perception.frame",
            "timestamp_ms": 1000,
            "face_detected": True,
            "face_center": [0.5, 0.5],
            "expression": {"smile": 0.5},
            "emotion_confidence": 0.7,
        }
        for i in range(msgs):
            send_ts = time.monotonic()
            await ws.send(json.dumps({**event, "timestamp_ms": 1000 + i}))
            echo = json.loads(await asyncio.wait_for(ws.recv(), PER_MSG_LATENCY_CEIL))
            latencies.append(time.monotonic() - send_ts)
            if isinstance(echo, dict):
                received += 1
    finally:
        await ws.close()
    return {
        "connect_ok": connect_ok,
        "received": received,
        "expected": msgs,
        "latencies": latencies,
        "wall": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# 1. Concurrency ceiling: many clients connect at once
# ---------------------------------------------------------------------------
@pytest.mark.performance
async def test_concurrent_connections_all_accepted(live_ws_url):
    results = await asyncio.gather(
        *[_connect_and_get_push(live_ws_url) for _ in range(CONCURRENCY_N)]
    )
    try:
        assert len(results) == CONCURRENCY_N
        for ws, initial in results:
            assert isinstance(initial, dict)
            assert "character_id" in initial
            assert "avatar" in initial
            assert initial["capabilities"]["renderer"]["backend"] == "deeplivecam"
    finally:
        for ws, _ in results:
            await ws.close()


# ---------------------------------------------------------------------------
# 2. Throughput + bounded latency under concurrent message pressure
# ---------------------------------------------------------------------------
@pytest.mark.performance
async def test_concurrent_message_throughput_and_latency(live_ws_url):
    clients = [
        _latency_client(live_ws_url, THROUGHPUT_MSGS)
        for _ in range(THROUGHPUT_CLIENTS)
    ]
    stats = await asyncio.gather(*clients)

    total_expected = THROUGHPUT_CLIENTS * THROUGHPUT_MSGS
    total_received = sum(s["received"] for s in stats)
    all_connected = all(s["connect_ok"] for s in stats)

    # Every client connected and every frame was echoed back.
    assert all_connected, "a client failed to connect under load"
    assert total_received == total_expected, (
        f"echo loss under load: got {total_received}/{total_expected}"
    )

    # Latency must stay bounded even with everyone talking at once.
    all_latencies = [l for s in stats for l in s["latencies"]]
    assert all_latencies, "no latency samples recorded"
    p95 = sorted(all_latencies)[int(0.95 * (len(all_latencies) - 1))]
    assert p95 <= PER_MSG_LATENCY_CEIL, f"p95 latency {p95:.3f}s exceeds {PER_MSG_LATENCY_CEIL}s"

    # Aggregate throughput floor.
    total_wall = max(s["wall"] for s in stats)
    throughput = total_received / total_wall if total_wall > 0 else 0.0
    assert throughput >= THROUGHPUT_FLOOR, (
        f"aggregate throughput {throughput:.1f} msg/s below floor {THROUGHPUT_FLOOR}"
    )


# ---------------------------------------------------------------------------
# 3. Sustained operation: server stays responsive to a fresh probe
# ---------------------------------------------------------------------------
@pytest.mark.performance
async def test_server_stays_responsive_under_sustained_load(live_ws_url):
    stop = asyncio.Event()

    async def _streamer():
        ws, _ = await _connect_and_get_push(live_ws_url)
        try:
            event = {
                "type": "perception.frame",
                "timestamp_ms": 1,
                "face_detected": True,
                "face_center": [0.5, 0.5],
                "expression": {"smile": 0.4},
                "emotion_confidence": 0.6,
            }
            i = 0
            while not stop.is_set():
                await ws.send(json.dumps({**event, "timestamp_ms": 1 + i}))
                await asyncio.wait_for(ws.recv(), PER_MSG_LATENCY_CEIL)
                i += 1
        except Exception:
            pass
        finally:
            await ws.close()

    streamers = [asyncio.create_task(_streamer()) for _ in range(SUSTAINED_CLIENTS)]
    try:
        # Let the sustained burst run for a bit.
        await asyncio.sleep(SUSTAINED_SECONDS * 0.5)

        # A fresh probe must still be accepted and respond quickly mid-burst.
        probe_start = time.monotonic()
        ws, initial = await _connect_and_get_push(live_ws_url, PROBE_LATENCY_CEIL)
        probe_latency = time.monotonic() - probe_start
        await ws.close()
        assert isinstance(initial, dict) and "character_id" in initial
        assert probe_latency <= PROBE_LATENCY_CEIL, (
            f"probe connect latency {probe_latency:.3f}s exceeds {PROBE_LATENCY_CEIL}s"
        )

        # Let it keep running, then probe again to confirm no late degradation.
        await asyncio.sleep(SUSTAINED_SECONDS * 0.5)
        ws2, initial2 = await _connect_and_get_push(live_ws_url, PROBE_LATENCY_CEIL)
        await ws2.close()
        assert isinstance(initial2, dict) and "character_id" in initial2
    finally:
        stop.set()
        for task in streamers:
            task.cancel()
        await asyncio.gather(*streamers, return_exceptions=True)


# ---------------------------------------------------------------------------
# 4. Burst cleanup: connections open + close, then a new one still works
# ---------------------------------------------------------------------------
@pytest.mark.performance
async def test_burst_connections_clean_up(live_ws_url):
    # Open a burst of connections, each doing a quick push-read then closing.
    async def _burst_client():
        ws, initial = await _connect_and_get_push(live_ws_url)
        ok = isinstance(initial, dict) and "character_id" in initial
        await ws.close()
        return ok

    results = await asyncio.gather(*[_burst_client() for _ in range(BURST_N)])
    assert all(results), "a burst connection was not accepted/cleaned up"

    # Immediately after the burst, a brand-new connection must still succeed --
    # proving the server reclaimed the dead connections and is not wedged.
    ws, initial = await _connect_and_get_push(live_ws_url, CONNECT_SETTLE_CEIL)
    try:
        assert isinstance(initial, dict)
        assert initial["character_id"]
        assert initial["capabilities"]["renderer"]["backend"] == "deeplivecam"
    finally:
        await ws.close()
