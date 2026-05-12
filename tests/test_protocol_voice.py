from pathlib import Path
from hermes_avatar.protocol.events import parse_event, AudioVADEvent
from hermes_avatar.voice.luxtts_adapter import LuxTTSAdapter
from hermes_avatar.voice.base import VoiceStyle


def test_parse_vad_event():
    event = parse_event({"type": "audio.vad", "timestamp_ms": 1, "speaking": True, "energy": 0.5, "speech_rate": 0.2})
    assert isinstance(event, AudioVADEvent)
    assert event.speaking is True


def test_luxtts_fallback_writes_wav(tmp_path):
    adapter = LuxTTSAdapter(cache_dir=tmp_path)
    speech = adapter.synthesize("hello avatar", VoiceStyle())
    assert Path(speech.audio_path).exists()
    assert speech.backend == "luxtts"
