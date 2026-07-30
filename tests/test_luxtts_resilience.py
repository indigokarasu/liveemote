"""Resilience tests for the LuxTTSAdapter circuit breaker.

These tests lock down the protection against cascading 120 s vendor hangs
that previously made ``/api/speak`` unusable during a LuxTTS outage.
Without the breaker, every failed utterance re-spawned the vendor
subprocess and re-ran the 120 s blocking timeout (N5 in-flight calls =
10 minutes of stalled UI). With the breaker, two consecutive failures
trip it ``OPEN`` for 60 s and ``synthesize()`` short-circuits to the
parametric fallback so the user still hears audio.

Test surface:

* success keeps the breaker CLOSED,
* two consecutive failures trip it OPEN (and the engine reports
  ``local-parametric`` fallback each time),
* an OPEN breaker short-circuits the vendor call entirely — never
  invokes ``_run_vendor_command``, never spawns ``subprocess.run``,
* after the open window elapses, a single successful probe closes the
  breaker again (HALF_OPEN -> CLOSED),
* ``capability_status()`` exposes the breaker snapshot so the demo
  server's ``/api/health`` surface can report it alongside the renderer
  breaker.
"""

from __future__ import annotations

import subprocess
import time
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_avatar.voice.luxtts_adapter import LuxTTSAdapter
from hermes_avatar.voice.base import VoiceStyle
from hermes_avatar.util import CircuitBreaker, OPEN, CLOSED


def _stub_vendor_writes_wav(text: str, output: Path, reference_audio) -> None:
    """Mock ``_run_vendor_command`` that materialises a valid empty WAV.

    The real implementation drives the vendor subprocess; for tests we
    side-step the subprocess entirely but still produce a file
    ``_wav_duration_ms()`` can open so the returned ``SynthesizedSpeech``
    has a real ``audio_path``.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    """A ``LuxTTSAdapter`` wired with ``LUXTTS_COMMAND`` set + a short-timeout breaker.

    Replaces the production breaker with one whose ``open_timeout`` is
    just long enough that a single ``time.sleep(0.08)`` advances past
    the open window in the half-open recovery test, avoiding any
    time-mocking globally.
    """
    monkeypatch.setenv("LUXTTS_COMMAND", "true")
    a = LuxTTSAdapter(
        vendor_dir=str(tmp_path / "vendor"),
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
    )
    a.cb = CircuitBreaker(
        failure_threshold=2,
        open_timeout=0.05,
        name="luxtts-test",
    )
    return a


def test_success_keeps_breaker_closed(adapter):
    voice_style = VoiceStyle()
    with patch.object(
        adapter, "_run_vendor_command", side_effect=_stub_vendor_writes_wav
    ) as mr:
        out = adapter.synthesize("hello world", voice_style)
    mr.assert_called_once()
    assert adapter.cb.state == CLOSED
    assert adapter.cb.snapshot()["failure_count"] == 0
    assert adapter.last_error is None
    assert out.engine == "luxtts-vendor"
    assert out.audio_path and Path(out.audio_path).exists()


def test_two_failures_trip_breaker_open(adapter):
    voice_style = VoiceStyle()
    side_effect = subprocess.TimeoutExpired(cmd="true", timeout=120)
    # Two distinct texts → two distinct cache paths, both miss the cache
    with patch.object(adapter, "_run_vendor_command", side_effect=side_effect):
        adapter.synthesize("trips once", voice_style)
        adapter.synthesize("trips twice", voice_style)
    assert adapter.cb.state == OPEN
    snap = adapter.cb.snapshot()
    assert snap["failure_count"] >= 2
    assert adapter.last_engine == "local-parametric"
    assert adapter.last_error is not None
    # ``subprocess.TimeoutExpired.__str__()`` produces "Command 'X' timed
    # out after N seconds" (with a space). Match the actual wording rather
    # than ""timeout"" so the assertion survives future stdlib tweaks.
    assert "timed out" in adapter.last_error.lower()


def test_open_short_circuits_subprocess(adapter):
    voice_style = VoiceStyle()
    # Trip the breaker independently of synthesize to keep the test cheap
    adapter.cb.record_failure()
    adapter.cb.record_failure()
    assert adapter.cb.state == OPEN
    adapter.last_engine = None
    adapter.last_error = None

    with patch.object(adapter, "_run_vendor_command") as mr_runner:
        with patch("subprocess.run") as mr_subprocess:
            out = adapter.synthesize("while open", voice_style)
    # Vendor call surface was never invoked at all
    mr_runner.assert_not_called()
    mr_subprocess.assert_not_called()
    # Parametric fallback still produced audio
    assert out.engine == "local-parametric"
    assert out.audio_path and Path(out.audio_path).exists()
    # And we surfaced WHY we skipped the vendor in ``last_error``
    assert adapter.last_error and "breaker" in adapter.last_error.lower()


def test_half_open_recovery_closes_breaker(adapter):
    voice_style = VoiceStyle()
    # Trip the breaker via real synthesize calls so the failure_count
    # is realistic for the half-open probe math.
    with patch.object(
        adapter,
        "_run_vendor_command",
        side_effect=subprocess.TimeoutExpired(cmd="true", timeout=120),
    ):
        adapter.synthesize("trip alpha", voice_style)
        adapter.synthesize("trip beta", voice_style)
    assert adapter.cb.state == OPEN

    # Open window elapses (``open_timeout=0.05`` in the fixture)
    time.sleep(0.08)

    # Probe call — vendor succeeds; breaker should transition to CLOSED.
    with patch.object(
        adapter, "_run_vendor_command", side_effect=_stub_vendor_writes_wav
    ) as mr:
        out = adapter.synthesize("probe", voice_style)
    mr.assert_called_once()
    assert out.engine == "luxtts-vendor"
    assert adapter.cb.state == CLOSED
    assert adapter.cb.snapshot()["failure_count"] == 0
    assert adapter.last_error is None


def test_capability_status_includes_breaker_snapshot(adapter):
    status = adapter.capability_status()
    assert "circuit_breaker" in status
    snap = status["circuit_breaker"]
    assert snap["name"] == "luxtts-test"
    # Fresh adapter: CLOSED
    assert snap["state"] == CLOSED

    # After a single failure, count reflects it WITHOUT tripping
    adapter.cb.record_failure()
    tripped = adapter.capability_status()
    assert tripped["circuit_breaker"]["failure_count"] >= 1
    # And state is still CLOSED (below the threshold of 2)
    assert tripped["circuit_breaker"]["state"] == CLOSED
