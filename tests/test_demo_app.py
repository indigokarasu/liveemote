from fastapi.testclient import TestClient
from apps.demo_server.main import create_app

class Args:
    character = "./character_input"
    renderer = "livetalking"
    voice_backend = "luxtts"
    transport = "webrtc"
    hermes_mode = "fake"


def test_demo_status_endpoint():
    client = TestClient(create_app(Args()))
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["character_id"] == "indigo"


def test_demo_status_exposes_capabilities_and_characters():
    client = TestClient(create_app(Args()))
    payload = client.get("/api/status").json()
    assert payload["capabilities"]["multi_character_switching"] is True
    assert payload["capabilities"]["mobile_layout"] is True
    assert payload["characters"]


def test_audio_route_rejects_paths_outside_voice_cache():
    client = TestClient(create_app(Args()))
    response = client.get("/api/audio", params={"path": "/tmp/not-cache.wav"})
    assert response.status_code in {403, 404}
def test_demo_can_switch_style_background_and_workflow():
    client = TestClient(create_app(Args()))
    style_response = client.post("/api/style", json={"style_id": "cyberpunk", "sync_background": True})
    assert style_response.status_code == 200
    assert style_response.json()["active_style_id"] == "cyberpunk"
    assert style_response.json()["active_background_id"] == "cyberpunk-city"

    background_response = client.post("/api/background", json={"background_id": "studio"})
    assert background_response.status_code == 200
    assert background_response.json()["active_background_id"] == "studio"
    assert background_response.json()["sync_background_to_style"] is False

    workflow_response = client.post("/api/workflow", json={"workflow": "debugging"})
    assert workflow_response.status_code == 200
    assert workflow_response.json()["active_style_id"] == "glitch"
    assert workflow_response.json()["active_background_id"] == "glitch-grid"

class OfflineArgs:
    character = "./character_input"
    renderer = "livetalking"
    voice_backend = "none"
    transport = "webrtc"
    agent_mode = "offline"
    agent_url = None
    agent_harness = "none"


def test_demo_runs_without_agent_or_voice_and_still_mirrors():
    client = TestClient(create_app(OfflineArgs()))
    status = client.get("/api/status").json()
    assert status["capabilities"]["agent"]["available"] is False
    assert status["capabilities"]["voice"]["backend"] == "none"

    event_response = client.post(
        "/api/event",
        json={
            "event": {
                "type": "perception.frame",
                "timestamp_ms": 1000,
                "face_detected": True,
                "face_center": [0.5, 0.5],
                "expression": {"smile": 0.8},
                "emotion_confidence": 0.9,
            }
        },
    )
    assert event_response.status_code == 200
    payload = event_response.json()
    assert payload["user"]["face_detected"] is True
    assert payload["avatar"]["gaze_target"] == "toward_user"


def test_offline_speak_test_does_not_synthesize_speech():
    client = TestClient(create_app(OfflineArgs()))
    response = client.post("/api/speak", json={"text": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["speech"] is None
    assert payload["agent_response_text"] == ""


class DeepLiveCamArgs:
    character = "./character_input"
    renderer = "deeplivecam"
    voice_backend = "none"
    transport = "webrtc"
    agent_mode = "offline"
    agent_url = None
    agent_harness = "none"


def test_deeplivecam_renderer_uses_canonical_face_source():
    client = TestClient(create_app(DeepLiveCamArgs()))
    payload = client.get("/api/status").json()
    renderer = payload["capabilities"]["renderer"]

    assert renderer["backend"] == "deeplivecam"
    assert renderer["enabled"] is True
    assert renderer["replacement_active"] is True
    assert renderer["source_reference_role"] == "identity_anchor"
    assert renderer["source_image_path"].endswith("canonical/canonical.png")
    assert renderer["error"] is None
