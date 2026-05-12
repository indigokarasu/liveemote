import pytest
from fastapi.testclient import TestClient

from apps.demo_server.main import create_app
from hermes_avatar.demo.meeting_join import MeetingJoinError, MeetingJoinService, normalize_google_meet_url


class Args:
    character = "./character_input"
    renderer = "livetalking"
    voice_backend = "luxtts"
    transport = "webrtc"
    hermes_mode = "fake"


class FakeRenderer:
    def join_meeting(self, meeting_url, display_name):
        return {"available": True, "joined": True, "meeting_url": meeting_url, "display_name": display_name}

    def leave_meeting(self):
        return {"left": True}


def test_normalize_google_meet_url_accepts_code_without_scheme():
    assert normalize_google_meet_url("meet.google.com/abc-defg-hij") == "https://meet.google.com/abc-defg-hij"


@pytest.mark.parametrize("url", ["", "https://example.com/abc-defg-hij", "http://meet.google.com/abc-defg-hij", "https://meet.google.com/not-a-code"])
def test_normalize_google_meet_url_rejects_unsupported_urls(url):
    with pytest.raises(MeetingJoinError):
        normalize_google_meet_url(url)


def test_meeting_join_service_reports_renderer_latency():
    service = MeetingJoinService(FakeRenderer())
    status = service.join("https://meet.google.com/abc-defg-hij", "Test Avatar")
    assert status["status"] == "joined"
    assert status["display_name"] == "Test Avatar"
    assert status["estimated_join_latency_ms"] is not None
    assert status["renderer"]["joined"] is True


def test_meeting_join_endpoint_validates_google_meet_url():
    client = TestClient(create_app(Args()))
    response = client.post("/api/meeting/join", json={"meeting_url": "https://example.com/nope"})
    assert response.status_code == 400
    assert "meet.google.com" in response.json()["detail"]
