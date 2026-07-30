"""Resilience-surface contract for the demo_server ``/api/health`` endpoint.

The full ``/api/health`` handler at ``apps/demo_server/routes.py`` reports
the live state of every external dependency that we now wrap in the shared
``CircuitBreaker`` + retry-with-jitter primitives. This single test pins the
exposed surface so future refactors of any individual adapter cannot silently
drop its breaker / retry counters from operator-facing health views.

The 3+1 contract:

* ``voice_backend.detail.circuit_breaker`` -- LuxtTSAdapter (subprocess vendor)
* ``renderer.detail.circuit_breaker`` -- LiveTalkingAdapter (HTTP renderer;
  also referred to as the "vision_renderer" in operator dashboards)
* ``protocol_agent.detail.circuit_breaker`` -- OpenAI-compatible adapter
* ``protocol_agent.detail.retry`` -- retry-config + per-request/cumulative
  retry counters on the OpenAI-compatible adapter

Each subcomponent has a ``name`` field identifying the underlying primitive,
so a flip from ``"rendering:renderrer"`` to ``"live-talking"`` (or similar)
in a future refactor surfaces here as a test failure rather than silently.
"""

from fastapi.testclient import TestClient

from apps.demo_server.main import create_app


class HealthArgs:
    # Defaults that match tests/test_demo_app.py ``Args`` so we don't drag in
    # a separate fixture; overrides here exercise the openai-compatible
    # mode so the protocol_agent component is fully populated with breaker
    # + retry surfaces.
    character = "./character_input"
    renderer = "livetalking"
    voice_backend = "luxtts"
    transport = "webrtc"
    hermes_mode = "fake"
    agent_mode = "openai-compatible"
    agent_url = None
    agent_harness = "generic"
    perception_tracker = "null"


def test_health_exposes_voice_backend_renderer_and_protocol_agent_breakers():
    """Single contract lock: /api/health surfaces all 3 breakers + retry counter."""
    client = TestClient(create_app(HealthArgs()))
    response = client.get("/api/health")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] in {"ok", "degraded"}, body  # never 500

    components = body["components"]

    # --- voice_backend.circuit_breaker (LuxtTSAdapter) ---
    assert "voice_backend" in components, "voice_backend component must be present"
    vb = components["voice_backend"]["detail"]
    assert vb["name"] == "luxtts", vb
    assert vb.get("circuit_breaker") is not None, vb
    vb_cb = vb["circuit_breaker"]
    assert vb_cb["name"] == "luxtts", vb_cb
    assert vb_cb["state"] in {"closed", "open", "half-open"}, vb_cb

    # --- renderer (vision_renderer).circuit_breaker (LiveTalkingAdapter) ---
    assert "renderer" in components, "renderer component must be present"
    renderer = components["renderer"]["detail"]
    assert renderer.get("circuit_breaker") is not None, renderer
    r_cb = renderer["circuit_breaker"]
    assert r_cb["name"] == "renderer", r_cb
    assert r_cb["state"] in {"closed", "open", "half-open"}, r_cb

    # --- NEW protocol_agent component (OpenAICompatibleAdapter) ---
    assert "protocol_agent" in components, (
        "protocol_agent component must be present (item 2.2 retry surface)"
    )
    pa = components["protocol_agent"]["detail"]
    assert pa["mode"] == "openai-compatible", pa
    assert pa.get("circuit_breaker") is not None, pa
    pa_cb = pa["circuit_breaker"]
    assert pa_cb["name"] == "openai", pa_cb
    assert pa_cb["state"] in {"closed", "open", "half-open"}, pa_cb

    # --- AND the protocol_agent.retry subcomponent ---
    assert pa.get("retry") is not None, pa
    retry = pa["retry"]
    assert retry["max_retries"] == 2, retry
    assert retry["base_delay"] == 0.5, retry
    assert retry["max_delay"] == 4.0, retry
    assert retry["jitter_factor"] == 0.1, retry
    # Fresh adapter last/total counters are zero.
    assert retry["last_retry_count"] == 0, retry
    assert retry["total_retries"] == 0, retry
    # And the human-readable classification string is present.
    assert "compute_backoff_delay" in retry["retry_via"], retry
    assert "is_retryable_error" in retry["retry_via"], retry
