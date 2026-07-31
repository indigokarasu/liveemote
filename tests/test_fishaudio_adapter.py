"""Tests for FishAudioAdapter — follows the same parametrized pattern as
test_luxtts_resilience.py and test_openai_adapter_resilience.py."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest import mock

import pytest

# Ensure the packages dir is on sys.path (mirrors conftest.py).
_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent / "packages"
import sys

if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from hermes_avatar.voice.fishaudio_adapter import FishAudioAdapter
from hermes_avatar.voice.base import VoiceStyle
from hermes_avatar.util.audit import reset as audit_reset, snapshot as audit_snapshot


@pytest.fixture(autouse=True)
def _clean_audit():
    audit_reset()
    yield
    audit_reset()


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    """FishAudioAdapter with a fake API key but pointed at a temp cache dir."""
    monkeypatch.setenv("FISH_API_KEY", "test-key-12345")
    monkeypatch.setenv("FISH_REFERENCE_ID", "test-voice-id")
    return FishAudioAdapter(
        api_key="test-key-12345",
        reference_id="test-voice-id",
        cache_dir=str(tmp_path / "cache_voice"),
    )


class TestCapabilityStatus:
    def test_reports_configured_when_key_present(self, adapter):
        caps = adapter.capability_status()
        assert caps["backend"] == "fishaudio"
        assert caps["configured"] is True
        assert caps["model"] == "s2.1-pro"
        assert caps["reference_id"] == "test-voice-id"

    def test_reports_unconfigured_when_key_absent(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FISH_API_KEY", raising=False)
        a = FishAudioAdapter(api_key=None, cache_dir=str(tmp_path / "cache_voice"))
        assert a.capability_status()["configured"] is False

    def test_exposes_circuit_breaker_snapshot(self, adapter):
        snap = adapter.capability_status()["circuit_breaker"]
        assert snap["state"] == "closed"
        assert snap["failure_count"] == 0
        assert snap["name"] == "fishaudio"

    def test_exposes_audit_snapshot(self, adapter):
        snap = adapter.capability_status()["audit"]
        assert snap["events_total"] == 0


class TestSynthesizeErrors:
    def test_raises_without_api_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FISH_API_KEY", raising=False)
        a = FishAudioAdapter(api_key=None, cache_dir=str(tmp_path / "cache_voice"))
        with pytest.raises(RuntimeError, match="FISH_API_KEY"):
            a.synthesize("hello", VoiceStyle())

    def test_breaker_open_raises(self, adapter):
        # Manually force the breaker open.
        for _ in range(5):
            adapter.cb.record_failure()
        assert adapter.cb.state == "open"
        with pytest.raises(RuntimeError, match="circuit breaker is open"):
            adapter.synthesize("hello", VoiceStyle())


class TestSynthesizeSuccess:
    def test_caches_result_and_returns_synthesized_speech(self, adapter):
        """Mock httpx so we don't hit the real Fish Audio API.

        The adapter should:
        - POST to /v1/tts
        - Write the response bytes to the cache
        - Return a SynthesizedSpeech with backend='fishaudio'
        - Record a breaker success
        - The second call with the same text hits the cache (no HTTP call).
        """
        fake_audio = b"RIFF....WAVEfake audio data...."

        with mock.patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_resp = mock.MagicMock()
            mock_resp.content = fake_audio
            mock_resp.raise_for_status = mock.MagicMock()
            mock_client.post.return_value = mock_resp

            result = adapter.synthesize("Hello, world!", VoiceStyle(pace=0.5, intensity=0.3))

        assert result.text == "Hello, world!"
        assert result.backend == "fishaudio"
        assert result.engine == "s2.1-pro"
        assert os.path.isfile(result.audio_path)
        assert Path(result.audio_path).read_bytes() == fake_audio
        assert adapter.cb.state == "closed"
        assert adapter.cb.snapshot()["failure_count"] == 0

        # Second call — cache hit, no HTTP call.
        mock_client_cls.reset_mock()
        with mock.patch("httpx.Client") as mock2:
            result2 = adapter.synthesize("Hello, world!", VoiceStyle())
            mock2.assert_not_called()
        assert result2.audio_path == result.audio_path

    def test_includes_reference_audio_when_provided(self, adapter, tmp_path):
        """When reference_audio is passed, it should be base64-encoded and
        included in the payload as ``references``."""
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"fake-ref-audio")

        with mock.patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_resp = mock.MagicMock()
            mock_resp.content = b"RIFF....WAVEaudio"
            mock_resp.raise_for_status = mock.MagicMock()
            mock_client.post.return_value = mock_resp

            adapter.synthesize("test", VoiceStyle(), reference_audio=str(ref_audio))

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs[1]["json"]
            assert "references" in payload
            assert len(payload["references"]) == 1
            assert "audio" in payload["references"][0]
            assert "reference_id" not in payload  # per-call ref overrides persistent


class TestRetryAndBreaker:
    def test_retries_on_transient_error(self, adapter):
        """503 → retry → success: breaker should stay closed."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                resp = mock.MagicMock()
                resp.status_code = 503
                resp.raise_for_status.side_effect = Exception("503 Service Unavailable")
                raise Exception("503 Service Unavailable")
            resp = mock.MagicMock()
            resp.content = b"RIFF....WAVEok"
            resp.raise_for_status = mock.MagicMock()
            return resp

        with mock.patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.side_effect = side_effect

            result = adapter.synthesize("retry test", VoiceStyle())

        assert call_count[0] == 3
        assert result.backend == "fishaudio"
        assert adapter.cb.state == "closed"

    def test_non_retryable_error_trips_breaker(self, adapter):
        """401 (auth error) should NOT retry — fail immediately and count
        toward the breaker."""
        with mock.patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.side_effect = Exception("401 Unauthorized")

            with pytest.raises(Exception, match="401"):
                adapter.synthesize("bad auth", VoiceStyle())

        assert adapter.cb.snapshot()["failure_count"] >= 1


class TestAuditIntegration:
    def test_emits_audit_on_synthesis_failure(self, adapter):
        """After all retries are exhausted, audit should record a
        vendor_fallback event for voice.fishaudio."""
        with mock.patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.side_effect = Exception("timeout")

            with pytest.raises(Exception):
                adapter.synthesize("timeout text", VoiceStyle())

        snap = audit_snapshot("voice.fishaudio")
        assert snap["events_total"] >= 1
        assert snap["last_event_kind"] == "vendor_fallback"
