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
    assert payload["capabilities"]["multi_character_scene"] is True
    assert payload["capabilities"]["mobile_layout"] is True
    assert payload["characters"]
