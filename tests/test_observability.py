from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.config.schema import AppConfig, load_config, reload_config
from hermes_avatar.observability.tracing import (
    ensure_trace_id,
    get_trace_id,
    new_trace_id,
    set_trace_id,
)
from apps.demo_server.routes import build_router


# ---------------------------------------------------------------------------
# Minimal orchestrator stand-in: the debug endpoint only reads
# ``app.state.orchestrator.config`` (an AppConfig) and ``.runtime`` (an
# AffectRuntime), so a tiny object satisfies it without spinning up the full
# DemoOrchestrator / character discovery.
# ---------------------------------------------------------------------------
class _Orchestrator:
    def __init__(self, debug: bool = False) -> None:
        self.config = AppConfig(debug=debug)
        self.runtime = AffectRuntime()


def _make_client(debug: bool = False) -> TestClient:
    app = FastAPI()
    orch = _Orchestrator(debug=debug)
    # Exercise the pipeline so the runtime history is populated.
    orch.runtime.tick(1000)
    orch.runtime.tick(1100)
    app.state.orchestrator = orch
    app.include_router(build_router("./apps/demo_server/static"))
    return TestClient(app)


def _audit(record: logging.LogRecord) -> dict:
    return getattr(record, "audit", {})


# ===========================================================================
# obs-2: distributed tracing (contextvar trace id propagation)
# ===========================================================================
def test_trace_id_helpers_roundtrip():
    set_trace_id(None)
    assert get_trace_id() is None
    rid = new_trace_id()
    assert isinstance(rid, str) and len(rid) == 32
    set_trace_id(rid)
    assert get_trace_id() == rid


def test_ensure_trace_id_creates_and_reuses():
    set_trace_id(None)
    first = ensure_trace_id()
    second = ensure_trace_id()
    assert first == second
    assert get_trace_id() == first


def test_request_middleware_sets_trace_header():
    client = _make_client(debug=True)
    response = client.get("/debug/affect")
    assert response.status_code == 200
    # Every HTTP response should carry the request trace id.
    assert "X-Trace-Id" in response.headers
    assert len(response.headers["X-Trace-Id"]) == 32

def test_affect_tick_records_trace_id_in_history():
    set_trace_id(None)
    rt = AffectRuntime()
    trace = ensure_trace_id()
    rt.tick(1000)
    rt.tick(1100)
    assert len(rt.history) == 2
    for entry in rt.history:
        assert entry["trace_id"] == trace
        assert "avatar" in entry and "mode" in entry


def test_trace_id_scoped_to_context():
    import contextvars

    set_trace_id("outer")
    captured: dict = {}

    def child():
        captured["before"] = get_trace_id()
        set_trace_id("inner")
        captured["inner"] = get_trace_id()

    ctx = contextvars.copy_context()
    ctx.run(child)
    # Child's change must not leak into the outer context.
    assert captured["before"] == "outer"
    assert captured["inner"] == "inner"
    assert get_trace_id() == "outer"


# ===========================================================================
# obs-4: debug endpoint for affect state visualization (guarded by config.debug)
# ===========================================================================
def test_debug_affect_disabled_by_default_is_404():
    client = _make_client()
    response = client.get("/debug/affect")
    assert response.status_code == 404


def test_debug_affect_enabled_exoses_state_and_history():
    client = _make_client(debug=True)

    response = client.get("/debug/affect")
    assert response.status_code == 200
    body = response.json()
    for key in ("user", "conversation", "avatar", "mode", "recent_history"):
        assert key in body
    assert isinstance(body["recent_history"], list)
    assert body["recent_history"]


# ===========================================================================
# obs-5: structured audit logging on config load / reload
# ===========================================================================
def test_load_config_audit_log_records_env_overrides(caplog, monkeypatch):
    monkeypatch.setenv("AFFECT__UPDATE_HZ", "60")
    try:
        with caplog.at_level(logging.INFO, logger="hermes_avatar.config"):
            cfg = load_config()
        assert cfg.affect.update_hz == 60
        records = [
            r for r in caplog.records if _audit(r).get("event") == "config.loaded"
        ]
        assert records, "expected a config.loaded audit record"
        audit = _audit(records[-1])
        assert "affect.update_hz" in audit["env_overrides"]
        assert audit["hardware_profile"] is not None
    finally:
        monkeypatch.undo()


def test_reload_config_emits_loaded_and_reloaded_audit(caplog, monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_HW_AWARE", "true")
    with caplog.at_level(logging.INFO, logger="hermes_avatar.config"):
        # Pass a baseline so a no-op reload reports zero changed keys.
        reload_config(load_config())
    events = [_audit(r).get("event") for r in caplog.records]
    assert "config.loaded" in events
    assert "config.reloaded" in events

    reloaded = [
        r for r in caplog.records if _audit(r).get("event") == "config.reloaded"
    ][-1]
    # No functional change between identical configs -> changed_keys empty.
    assert _audit(reloaded)["changed_keys"] == []


def test_reload_config_reports_changed_keys(caplog, monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_HW_AWARE", "true")
    previous = load_config()
    monkeypatch.setenv("AFFECT__UPDATE_HZ", "99")
    with caplog.at_level(logging.INFO, logger="hermes_avatar.config"):
        reload_config(previous)
    reloaded = [
        r for r in caplog.records if _audit(r).get("event") == "config.reloaded"
    ][-1]
    assert "affect.update_hz" in _audit(reloaded)["changed_keys"]


def test_appconfig_debug_default_false():
    assert AppConfig().debug is False
