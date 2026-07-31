"""Tests for Wav2LipAdapter — parametrized across passthrough, breaker, retry,
and audit integration.  No GPU / models required."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent / "packages"
import sys

if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from hermes_avatar.renderer.wav2lip_adapter import Wav2LipAdapter
from hermes_avatar.util.audit import reset as audit_reset, snapshot as audit_snapshot


@pytest.fixture(autouse=True)
def _clean_audit():
    audit_reset()
    yield
    audit_reset()


@pytest.fixture
def vendor_dir(tmp_path):
    """Create a minimal vendor dir with a fake ``run_wav2lip.py`` entrypoint.

    The entrypoint writes the output file and exits 0 so the subprocess path
    can be exercised without real models.
    """
    vd = tmp_path / "Wav2Lip"
    vd.mkdir()
    (vd / "checkpoints").mkdir()
    entry = vd / "run_wav2lip.py"
    entry.write_text(
        'import argparse, sys, pathlib\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--face")\n'
        'p.add_argument("--audio")\n'
        'p.add_argument("--outfile")\n'
        'p.add_argument("--checkpoint_path")\n'
        'p.add_argument("--device", default="cpu")\n'
        'p.add_argument("--pads", nargs=4, type=int, default=[0,10,0,0])\n'
        'p.add_argument("--nosmooth", action="store_true")\n'
        'p.add_argument("--resize_factor", type=int, default=1)\n'
        'p.add_argument("--box", nargs=4, type=int, default=None)\n'
        'p.add_argument("--face_detection_batch_size", type=int, default=16)\n'
        'args = p.parse_args()\n'
        'pathlib.Path(args.outfile).write_text("fake-lip-synced-frame")\n'
        'sys.exit(0)\n'
    )
    return str(vd)


@pytest.fixture
def adapter(vendor_dir):
    """Wav2LipAdapter pointing at the fake vendor dir, enabled."""
    return Wav2LipAdapter(
        vendor_dir=vendor_dir,
        model_path=f"{vendor_dir}/checkpoints/wav2lip_gan.pth",
        device="cpu",
        enabled=True,
        timeout_seconds=5,
    )


class TestPassthroughMode:
    def test_disabled_adapter_is_passthrough(self, vendor_dir):
        a = Wav2LipAdapter(vendor_dir=vendor_dir, enabled=False)
        assert a.capability_status()["passthrough"] is True
        result = a.synthesize_lip_sync("audio.wav", "face.png")
        assert result["passthrough"] is True
        assert result["video_path"] == "face.png"

    def test_missing_entrypoint_is_passthrough(self, tmp_path):
        vd = tmp_path / "empty"
        vd.mkdir()
        a = Wav2LipAdapter(vendor_dir=str(vd), enabled=True)
        assert a.capability_status()["passthrough"] is True
        result = a.synthesize_lip_sync("a.wav", "f.png")
        assert result["passthrough"] is True

    def test_passthrough_call_is_counted(self, adapter):
        adapter.enabled = False
        adapter.synthesize_lip_sync("a.wav", "f.png")
        assert adapter.passthrough_calls == 1
        assert adapter.total_calls == 1


class TestCapabilityStatus:
    def test_reports_backend_and_device(self, adapter):
        caps = adapter.capability_status()
        assert caps["backend"] == "wav2lip"
        assert caps["device"] == "cpu"
        assert caps["enabled"] is True
        assert caps["passthrough"] is False

    def test_exposes_circuit_breaker(self, adapter):
        caps = adapter.capability_status()
        cb = caps["circuit_breaker"]
        assert cb["state"] == "closed"
        assert cb["name"] == "wav2lip"

    def test_exposes_audit_snapshot(self, adapter):
        caps = adapter.capability_status()
        assert caps["audit"]["events_total"] == 0


class TestSynthesizeSuccess:
    def test_runs_entrypoint_and_returns_video_path(self, adapter, tmp_path):
        audio = tmp_path / "speech.wav"
        audio.write_bytes(b"fake-audio")
        face = tmp_path / "frame.png"
        face.write_bytes(b"fake-frame")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = adapter.synthesize_lip_sync(str(audio), str(face), str(out_dir))

        assert result["passthrough"] is False
        assert "wav2lip_" in result["video_path"]
        assert result["latency_ms"] >= 0
        assert Path(result["video_path"]).exists()
        assert "fake-lip-synced-frame" in Path(result["video_path"]).read_text()

    def test_records_success_on_breaker(self, adapter, tmp_path):
        audio = tmp_path / "s.wav"
        audio.write_bytes(b"x")
        face = tmp_path / "f.png"
        face.write_bytes(b"x")
        adapter.synthesize_lip_sync(str(audio), str(face), str(tmp_path))
        assert adapter.cb.state == "closed"
        assert adapter.cb.snapshot()["failure_count"] == 0


class TestRetryAndBreaker:
    def test_retries_on_subprocess_timeout(self, adapter, tmp_path):
        """Simulate subprocess timeout → retry → success.

        Replace subprocess.run with a mock that fails twice then succeeds.
        """
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        face = tmp_path / "f.png"
        face.write_bytes(b"x")

        call_count = [0]

        orig_run = subprocess.run

        def fake_run(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("connection timeout during subprocess call")
            # On third attempt, succeed by writing the output file.
            outfile = None
            for i, arg in enumerate(cmd):
                if arg == "--outfile" and i + 1 < len(cmd):
                    outfile = cmd[i + 1]
            if outfile:
                Path(outfile).write_text("ok-after-retry")
            return mock.MagicMock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            result = adapter.synthesize_lip_sync(str(audio), str(face), str(tmp_path))

        assert call_count[0] == 3
        assert result["passthrough"] is False

    def test_breaker_open_returns_passthrough(self, adapter, tmp_path):
        """When the breaker is open, calls should passthrough."""
        for _ in range(5):
            adapter.cb.record_failure()
        assert adapter.cb.state == "open"

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        face = tmp_path / "f.png"
        face.write_bytes(b"x")

        result = adapter.synthesize_lip_sync(str(audio), str(face))
        assert result["passthrough"] is True
        assert "circuit breaker open" in result.get("reason", "")


class TestAuditIntegration:
    def test_emits_audit_on_exhausted_retries(self, adapter, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        face = tmp_path / "f.png"
        face.write_bytes(b"x")

        with mock.patch("subprocess.run", side_effect=RuntimeError("boom")):
            result = adapter.synthesize_lip_sync(str(audio), str(face), str(tmp_path))

        assert result["passthrough"] is True
        snap = audit_snapshot("renderer.wav2lip")
        assert snap["events_total"] >= 1
        assert snap["last_event_kind"] == "retry_exhausted"
