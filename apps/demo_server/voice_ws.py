"""``/ws/voice`` — browser voice ⇄ speech-to-speech sidecar relay endpoint.

The browser dials this WebSocket and speaks the OpenAI Realtime protocol
(``input_audio_buffer.append`` with 16 kHz PCM16 base64). The orchestrator's
``VoiceLoopClient`` proxies the stream to the sidecar pipeline and snoops the
events so the avatar animates while it listens and replies.

Mirrors the existing ``websocket_api.py`` style: accept, read the orchestrator
off ``ws.app``, let the client handle the session lifecycle.
"""
from __future__ import annotations

from fastapi import WebSocket

from hermes_avatar.voice.voice_loop import VoiceLoopClient


async def voice_websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    app = ws.app
    orchestrator = app.state.orchestrator
    client: VoiceLoopClient = orchestrator.voice_loop

    if not client.enabled:
        await ws.send_json(
            {"type": "error", "error": {"message": "voice loop disabled — start the server with --voice-loop"}}
        )
        await ws.close()
        return

    await client.relay(
        ws,
        callbacks={
            "on_user_transcript": orchestrator.on_voice_user_transcript,
            "on_assistant_start": orchestrator.on_voice_speaking_start,
            "on_assistant_end": orchestrator.on_voice_speaking_end,
            "on_assistant_text": orchestrator.on_voice_assistant_text,
        },
    )
