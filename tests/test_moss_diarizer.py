"""Tests for ``hermes_avatar.voice.moss_diarizer``.

Covers the self-contained transcript parser (no 3.12-only MOSS package
required in-process) and the sidecar client's degraded-passthrough contract.
"""
from __future__ import annotations

import httpx
import pytest

from hermes_avatar.voice.moss_diarizer import DiarizationSegment, MossDiarizer, parse_moss_transcript


def test_parse_moss_transcript_segments():
    text = (
        "[0.48][S01]Welcome everyone[1.66]"
        "[12.26][S02]The pipeline is ready[13.81]"
        "[14.36][S01]Great, include the diarization results in the report[18.76]"
    )
    segs = parse_moss_transcript(text)
    assert len(segs) == 3
    assert segs[0] == DiarizationSegment(start=0.48, end=1.66, speaker="S01", text="Welcome everyone")
    assert segs[1] == DiarizationSegment(start=12.26, end=13.81, speaker="S02", text="The pipeline is ready")
    assert segs[2].speaker == "S01"
    assert segs[2].end == 18.76
    assert segs[2].text == "Great, include the diarization results in the report"


def test_parse_moss_transcript_last_segment_without_end():
    segs = parse_moss_transcript("[0.48][S01]Welcome everyone[1.66][12.26][S02]The pipeline is ready")
    assert len(segs) == 2
    assert segs[1].end is None
    assert segs[1].text == "The pipeline is ready"


def test_parse_moss_transcript_empty_or_none():
    assert parse_moss_transcript("") == []
    assert parse_moss_transcript(None) == []
    assert parse_moss_transcript("no segments here") == []


async def test_diarize_unavailable_on_connection_error():
    client = MossDiarizer(sidecar_url="http://127.0.0.1:1", health_timeout_s=0.4)
    result = await client.diarize_async(b"RIFF....")
    assert result.available is False
    assert result.segments == []
    assert result.reason


def test_health_degraded_when_sidecar_down():
    client = MossDiarizer(sidecar_url="http://127.0.0.1:1", health_timeout_s=0.4)
    status = client.capability_status()
    assert status["reachable"] is False
    assert status["degraded"] is True


async def test_diarize_parses_server_response(monkeypatch):
    client = MossDiarizer(sidecar_url="http://sidecar.test", health_timeout_s=0.4)
    payload = {
        "available": True,
        "segments": [{"start": 0.5, "end": 1.2, "speaker": "S01", "text": "hi"}],
        "text": "[0.5][S01]hi[1.2]",
        "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "elapsed_sec": 1.5,
    }

    class FakeResponse:
        status_code = 200

        def json(self):
            return payload

    async def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await client.diarize_async(b"RIFF....")
    assert result.available is True
    assert len(result.segments) == 1
    assert result.segments[0].text == "hi"
    assert result.segments[0].speaker == "S01"
    assert result.model == "OpenMOSS-Team/MOSS-Transcribe-Diarize"


async def test_diarize_reports_sidecar_error(monkeypatch):
    client = MossDiarizer(sidecar_url="http://sidecar.test", health_timeout_s=0.4)

    class FakeErrorResponse:
        status_code = 503

        def json(self):
            return {"available": False, "reason": "model not loaded"}

    async def fake_post(*args, **kwargs):
        return FakeErrorResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await client.diarize_async(b"RIFF....")
    assert result.available is False
    assert "503" in result.reason
