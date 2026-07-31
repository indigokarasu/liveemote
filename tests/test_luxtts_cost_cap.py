"""Tests for the LuxTTSAdapter rolling-second cost-cap budget.

The LuxtTS subprocess vendor can take ~30-120 seconds per call (blocking
``subprocess.run`` with ``timeout=120``). The breaker (commit ``ff83c2f``)
already protects against per-call FAILURE storms; the cost-cap (this
commit) is the orthogonal protection against per-call RESOURCE pressure
during a thundering-herd of ``/api/speak`` calls when OpenAI +
ElevenLabs + LiveTalking are all down.

The two protections cooperate:

* breaker CLOSED + budget room -> subprocess runs.
* breaker CLOSED + budget exhausted -> fallback (no subprocess, no
  breaker record_failure), audit event ``voice.luxtts.cost_cap_exceeded``.
* breaker OPEN -> fallback (existing policy), orthogonal to budget.
* timeout / vendor exception -> budget STILL consumes the elapsed
  seconds (thinker-validated -- a hung subprocess is real resource
  pressure, not a hypothetical one).

Test surface uses a fake ``now_fn`` so the 60-second rolling window can
be advanced instantly without ``time.sleep``.
"""

from __future__ import annotations

import subprocess
import time
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_avatar.voice.luxtts_adapter import LuxTTSAdapter, CostWindow
from hermes_avatar.voice.base import VoiceStyle


# -- tiny utilities ---------------------------------------------------------


def _stub_vendor_writes_wav(text: str, output: Path, reference_audio) -> None:
    """Mock ``_run_vendor_command`` -- materialises a valid empty WAV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)


def _stub_vendor_slow(seconds: float):
    """Mock ``_run_vendor_command`` -- sleeps ``seconds`` then writes a WAV."""
    def _impl(text: str, output: Path, reference_audio) -> None:
        time.sleep(seconds)
        _stub_vendor_writes_wav(text, output, reference_audio)
    return _impl


class _FakeClock:
    """A monotonic clock stand-in for advancing the 60s rolling window."""

    def __init__(self, t0: float = 1000.0):
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


# -- CostWindow direct tests -----------------------------------------------


def test_cost_window_floor_forces_cap_to_at_least_30s():
    cw = CostWindow(cap_seconds=5, window_seconds=60)
    assert cw.cap_seconds == 30.0  # 30s floor


def test_cost_window_would_block_after_consume():
    cw = CostWindow(cap_seconds=60, window_seconds=60, now_fn=_FakeClock())
    cw.record(seconds_used=40, now=1000.0)
    assert cw.would_block(sample_seconds=10) is False
    assert cw.would_block(sample_seconds=21) is True


def test_cost_window_rolls_off_old_entries():
    clock = _FakeClock()
    cw = CostWindow(cap_seconds=60, window_seconds=60, now_fn=clock)
    cw.record(seconds_used=60, now=1000.0)  # full cap used
    clock.advance(61)
    # Old entry has rolled off -> budget restored
    snap = cw.snapshot()
    assert snap["used_seconds"] == 0
    assert cw.would_block(sample_seconds=60) is False


def test_cost_window_record_clamps_to_window():
    """A single record cannot exceed ``window_seconds`` (prevents absurd peaks)."""
    cw = CostWindow(cap_seconds=60, window_seconds=60, now_fn=_FakeClock())
    cw.record(seconds_used=10_000, now=1000.0)
    assert cw.snapshot()["used_seconds"] == 60  # clamped to window


# -- adapter-level cost-cap tests -----------------------------------------


@pytest.fixture(autouse=True)
def _reset_audit_between_tests():
    """Each test starts with a clean audit counter cache.

    Without this, ``events_total`` accumulates across tests in the same
    module run, which would mask regressions in the new code (each test
    should observe its own emissions only).
    """
    from hermes_avatar.util.audit import reset as audit_reset
    audit_reset()  # clear every name before the test
    yield
    audit_reset()  # and after, so siblings in adjacent test runs are clean


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    """A ``LuxTTSAdapter`` with ``LUXTTS_COMMAND`` set + a tight breakable cost-cap.

    Uses ``cost_cap_seconds=120`` so the cost-cap logic is exercised with
    a single synthetic call (the adapter's own _vendor_max_seconds=120 so
    the would-block check uses that as the expected sample).
    """
    monkeypatch.setenv("LUXTTS_COMMAND", "true")
    monkeypatch.delenv("LUXTTS_COST_CAP_SECONDS_PER_MINUTE", raising=False)
    a = LuxTTSAdapter(
        vendor_dir=str(tmp_path / "vendor"),
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
        cost_cap_seconds=120,  # small enough that any meaningful call exceeds it
        cost_cap_window=60,
    )
    # Replace the cost-cap clock with a fake so tests don't sleep.
    from hermes_avatar.voice.luxtts_adapter import CostWindow
    fake = _FakeClock()
    a.cost_cap = CostWindow(cap_seconds=120, window_seconds=60, now_fn=fake)
    return a


def test_capability_status_exposes_cost_cap_snapshot(adapter):
    snap = adapter.capability_status()
    assert "cost_cap" in snap
    cc = snap["cost_cap"]
    # Fixture pins cost_cap_seconds=120 (above the 30s HARD_FLOOR_SECONDS
    # hard-coded into CostWindow).
    assert cc["cap_seconds"] == 120.0
    assert cc["window_seconds"] == 60.0
    assert cc["used_seconds"] == 0.0
    assert cc["remaining_seconds"] == 120.0
    assert cc["calls_blocked"] == 0


def test_budget_hit_emits_cost_cap_exceeded_audit_and_falls_back(adapter):
    """A would-block call avoids subprocess + emits audit + doesn't trip the breaker."""
    from hermes_avatar.util.audit import snapshot, KIND_COST_CAP_EXCEEDED
    # Pre-fill the budget to the cap so the next call would_block.
    adapter.cost_cap.record(seconds_used=10, now=adapter.cost_cap._now_fn())
    assert adapter.cost_cap.would_block(sample_seconds=120) is True

    breaker_failures_before = adapter.cb.snapshot()["failure_count"]
    with patch.object(adapter, "_run_vendor_command") as mr_runner:
        with patch("subprocess.run") as mr_subprocess:
            out = adapter.synthesize("budget hit", VoiceStyle())

    # Subprocess + vendor wrapper NEVER invoked
    mr_runner.assert_not_called()
    mr_subprocess.assert_not_called()
    # Parametric fallback produced audio
    assert out.engine == "local-parametric"
    # Breaker failure count UNCHANGED (orthogonality preserved)
    assert adapter.cb.snapshot()["failure_count"] == breaker_failures_before
    # Audit event recorded under the right name + kind
    s = snapshot("voice.luxtts")
    assert s["events_total"] == 1
    assert s["last_event_kind"] == KIND_COST_CAP_EXCEEDED
    # And ``calls_blocked`` incremented on the snapshot
    assert adapter.cost_cap.snapshot()["calls_blocked"] == 1


def test_budget_under_limit_runs_vendor_normally(adapter):
    """Fresh budget -> subprocess runs + audit may or may not fire (success path)."""
    # Budget is empty (used=0, cap=10). A single slow vendor call (1s)
    # is under the sample threshold (120s) easily.
    with patch.object(
        adapter, "_run_vendor_command", side_effect=_stub_vendor_slow(0.05)
    ) as mr:
        out = adapter.synthesize("fresh budget", VoiceStyle())
    mr.assert_called_once()
    assert out.engine == "luxtts-vendor"
    assert adapter.cb.state == "closed"
    # Budget consumed ~0.05s
    snap = adapter.cost_cap.snapshot()
    assert snap["used_seconds"] >= 0.0
    assert snap["used_seconds"] < 1.0  # well under the 10s cap


def test_timeout_still_consumes_budget(adapter):
    """A vendor timeout drains the budget anyway (hung subprocess = real cost)."""
    with patch.object(
        adapter,
        "_run_vendor_command",
        side_effect=subprocess.TimeoutExpired(cmd="true", timeout=120),
    ):
        out = adapter.synthesize("timeout 1", VoiceStyle())
    # Even on exception, the budget was recorded with the full ~120s sample
    # (cost_cap.record caps at window=60, so used_seconds <= 60, not 120)
    snap = adapter.cost_cap.snapshot()
    assert snap["used_seconds"] > 0.0
    # Breaker has recorded the failure
    assert adapter.cb.snapshot()["failure_count"] >= 1
    # Fallback audio still produced
    assert out.engine == "local-parametric"


def test_breaker_open_takes_precedence_over_cost_cap(adapter):
    """When breaker is OPEN we never even consult the budget -- first-match wins."""
    from hermes_avatar.util.audit import snapshot, KIND_TRIP, KIND_VENDOR_FALLBACK
    # Trip the breaker via record_failure * threshold. Adapter has failure_threshold=2.
    adapter.cb.record_failure()
    adapter.cb.record_failure()
    assert adapter.cb.state == "open"

    # The cost-cap is empty, so it WOULD have permitted the call -- but the
    # breaker-OPEN branch returns first, so no cost-cap audit is emitted.
    with patch.object(adapter, "_run_vendor_command") as mr_runner:
        with patch("subprocess.run") as mr_subprocess:
            out = adapter.synthesize("breaker open", VoiceStyle())

    mr_runner.assert_not_called()
    mr_subprocess.assert_not_called()
    assert out.engine == "local-parametric"
    # voice.luxtts emits ONE audit event (vendor_fallback with
    # reason=circuit_breaker_open) to surface the fallback to operators
    # via /api/health.  The cost-cap path is NOT taken because breaker-open
    # is checked first.
    s = snapshot("voice.luxtts")
    assert s["events_total"] == 1
    assert s["last_event_kind"] == KIND_VENDOR_FALLBACK


def test_rolling_window_refills_budget(adapter):
    """After 60s of fake clock advance, previously consumed budget becomes available."""
    fake = _FakeClock()
    from hermes_avatar.voice.luxtts_adapter import CostWindow
    adapter.cost_cap = CostWindow(cap_seconds=120, window_seconds=60, now_fn=fake)
    # Saturate the budget
    adapter.cost_cap.record(seconds_used=10, now=fake())
    assert adapter.cost_cap.would_block(sample_seconds=120) is True

    # Roll the clock forward past the window
    fake.advance(61)
    snap = adapter.cost_cap.snapshot()
    assert snap["used_seconds"] == 0
    assert snap["remaining_seconds"] == 120

    # And a fresh vendor call now succeeds
    with patch.object(
        adapter, "_run_vendor_command", side_effect=_stub_vendor_writes_wav
    ) as mr:
        out = adapter.synthesize("after refill", VoiceStyle())
    mr.assert_called_once()
    assert out.engine == "luxtts-vendor"


def test_cost_cap_default_unbounded_when_env_unset(monkeypatch, tmp_path):
    """Without the env var, the default cap is high enough that ordinary
    workloads never trip the cost cap (240s/min = 2 in-flight x 120s)."""
    monkeypatch.setenv("LUXTTS_COMMAND", "")
    monkeypatch.delenv("LUXTTS_COST_CAP_SECONDS_PER_MINUTE", raising=False)
    a = LuxTTSAdapter(
        vendor_dir=str(tmp_path / "vendor"),
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
    )
    snap = a.capability_status()["cost_cap"]
    assert snap["cap_seconds"] == 240.0  # the documented default
