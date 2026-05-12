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
