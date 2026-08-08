"""Tests for ``hermes_avatar.voice.voice_loop.VoiceLoopClient``.

The voice loop talks to the speech-to-speech sidecar over HTTP (control plane)
and WebSocket (OpenAI Realtime relay). These tests exercise the client's
degraded-passthrough contract and the event snooping that drives the avatar,
without requiring the heavy pipeline to be installed.
"""
from __future__ import annotations

import asyncio

from hermes_avatar.voice.voice_loop import VoiceLoopClient


def test_disabled_client_reports_disabled():
    client = VoiceLoopClient(enabled=False)
    assert client.enabled is False
    state = asyncio.run(client.health())
    assert state.enabled is False
    assert state.degraded is False
    status = client.capability_status()
    assert status["enabled"] is False
    assert status["degraded"] is False


async def test_health_degraded_when_control_plane_down():
    # Port 1 refuses connections instantly; the probe must degrade, not raise.
    client = VoiceLoopClient(enabled=True, control_url="http://127.0.0.1:1", health_timeout_s=0.4)
    state = await client.health()
    assert state.enabled is True
    assert state.degraded is True
    assert state.reason


def test_capability_status_cached_and_non_raising():
    client = VoiceLoopClient(enabled=True, control_url="http://127.0.0.1:1", health_timeout_s=0.4)
    first = client.capability_status()
    second = client.capability_status()
    # Cache hit: same values, no second network attempt.
    assert first == second
    assert first["enabled"] is True
    assert first["degraded"] is True


async def test_relay_snoops_user_transcript():
    client = VoiceLoopClient(enabled=True)
    seen: list[str] = []
    await client._handle_server_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "  hello there  "},
        speaking={"value": False},
        callbacks={"on_user_transcript": seen.append},
    )
    assert seen == ["hello there"]


async def test_relay_audio_delta_toggles_speaking_once():
    client = VoiceLoopClient(enabled=True)
    started: list[str] = []
    ended: list[str] = []
    speaking = {"value": False}
    callbacks = {"on_assistant_start": lambda: started.append("started"), "on_assistant_end": lambda: ended.append("ended")}

    await client._handle_server_event({"type": "response.audio.delta", "audio": "AA=="}, speaking=speaking, callbacks=callbacks)
    assert speaking["value"] is True
    assert len(started) == 1

    # A second delta mid-stream must not re-fire the start callback.
    await client._handle_server_event({"type": "response.audio.delta", "audio": "AA=="}, speaking=speaking, callbacks=callbacks)
    assert len(started) == 1

    await client._handle_server_event({"type": "response.output_audio.done"}, speaking=speaking, callbacks=callbacks)
    assert speaking["value"] is False
    assert len(ended) == 1


async def test_relay_response_done_ends_speaking():
    client = VoiceLoopClient(enabled=True)
    ended: list[str] = []
    speaking = {"value": True}
    await client._handle_server_event({"type": "response.done"}, speaking=speaking, callbacks={"on_assistant_end": lambda: ended.append("ended")})
    assert speaking["value"] is False
    assert len(ended) == 1


async def test_relay_assistant_text_delta():
    client = VoiceLoopClient(enabled=True)
    seen: list[str] = []
    await client._handle_server_event(
        {"type": "response.output_text.delta", "delta": "Hi there"},
        speaking={"value": False},
        callbacks={"on_assistant_text": seen.append},
    )
    assert seen == ["Hi there"]


async def test_relay_unknown_event_ignored_without_error():
    client = VoiceLoopClient(enabled=True)
    seen: list[str] = []
    await client._handle_server_event(
        {"type": "session.created", "session": {"id": "s1"}},
        speaking={"value": False},
        callbacks={"on_assistant_start": seen.append},
    )
    assert seen == []
