from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

MEET_CODE_RE = re.compile(r"^/[a-z]{3}-[a-z]{4}-[a-z]{3}/?$")


@dataclass
class MeetingSession:
    url: str = ""
    display_name: str = "Hermes Avatar"
    status: str = "idle"
    detail: str = "No meeting joined yet."
    started_at_ms: int | None = None
    joined_at_ms: int | None = None
    estimated_join_latency_ms: int | None = None
    renderer: dict | None = None
    browser_pid: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MeetingJoinError(ValueError):
    pass


class MeetingJoinService:
    """Coordinates the test-mode path for joining a Google Meet over WebRTC.

    The preferred path delegates to the renderer/LiveTalking process so the same
    avatar WebRTC media pipeline used in production is exercised. As a local
    fallback for development, this service can open a Chromium-family browser
    with media permissions pre-approved, leaving the Meet lobby/join button to
    the user when Google requires account or host approval.
    """

    def __init__(self, renderer, display_name: str = "Hermes Avatar") -> None:
        self.renderer = renderer
        self.session = MeetingSession(display_name=display_name)
        self.browser_process: subprocess.Popen | None = None

    def status(self) -> dict:
        if self.browser_process and self.browser_process.poll() is not None:
            if self.session.status in {"launching", "joined", "waiting_for_admission"}:
                self.session.status = "closed"
                self.session.detail = "The local browser session has exited."
        return self.session.to_dict()

    def join(self, url: str, display_name: str | None = None) -> dict:
        clean_url = normalize_google_meet_url(url)
        if display_name:
            self.session.display_name = display_name.strip() or self.session.display_name
        started = int(time.time() * 1000)
        self.session = MeetingSession(
            url=clean_url,
            display_name=self.session.display_name,
            status="launching",
            detail="Starting avatar WebRTC meeting join...",
            started_at_ms=started,
        )

        renderer_result = self._join_via_renderer(clean_url)
        self.session.renderer = renderer_result
        if renderer_result.get("joined"):
            joined = int(time.time() * 1000)
            self.session.status = "joined"
            self.session.joined_at_ms = joined
            self.session.estimated_join_latency_ms = joined - started
            self.session.detail = "Renderer reported that the avatar joined the Google Meet WebRTC session."
            return self.status()

        browser_result = self._launch_browser(clean_url)
        if browser_result.get("launched"):
            joined = int(time.time() * 1000)
            self.session.status = "waiting_for_admission"
            self.session.joined_at_ms = joined
            self.session.estimated_join_latency_ms = joined - started
            self.session.browser_pid = browser_result.get("pid")
            self.session.detail = (
                "Opened Google Meet in a WebRTC browser session with camera/mic permissions enabled. "
                "Complete any Google lobby, sign-in, or host-admission prompts in that browser."
            )
        else:
            self.session.status = "renderer_unavailable"
            self.session.detail = (
                "No compatible LiveTalking /avatar/join_meeting endpoint or Chromium-family browser was available. "
                "Start LiveTalking with meeting-join support, or install Chrome/Chromium on this machine."
            )
        return self.status()

    def leave(self) -> dict:
        renderer_result = {}
        leave_meeting = getattr(self.renderer, "leave_meeting", None)
        if callable(leave_meeting):
            renderer_result = leave_meeting()
        if self.browser_process and self.browser_process.poll() is None:
            self.browser_process.terminate()
        self.session.status = "left"
        self.session.detail = "Requested the avatar to leave the meeting."
        self.session.renderer = renderer_result or self.session.renderer
        return self.status()

    def _join_via_renderer(self, url: str) -> dict:
        join_meeting = getattr(self.renderer, "join_meeting", None)
        if not callable(join_meeting):
            return {"available": False, "joined": False, "reason": "renderer has no join_meeting hook"}
        result = join_meeting(url, self.session.display_name)
        if not isinstance(result, dict):
            return {"available": True, "joined": False, "raw": str(result)}
        return result

    def _launch_browser(self, url: str) -> dict:
        binary = find_chromium_binary()
        if not binary:
            return {"launched": False, "reason": "no chromium binary found"}
        args = [
            binary,
            "--new-window",
            "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-features=HardwareMediaKeyHandling,MediaSessionService",
            url,
        ]
        self.browser_process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"launched": True, "pid": self.browser_process.pid, "binary": binary}


def normalize_google_meet_url(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        raise MeetingJoinError("Paste a Google Meet URL before joining.")
    if not candidate.startswith(("https://", "http://")):
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme != "https":
        raise MeetingJoinError("Google Meet URLs must use https://.")
    if parsed.netloc.lower() != "meet.google.com":
        raise MeetingJoinError("Only meet.google.com links are supported in test mode.")
    if not MEET_CODE_RE.match(parsed.path):
        raise MeetingJoinError("Use a Google Meet link like https://meet.google.com/abc-defg-hij.")
    return f"https://meet.google.com{parsed.path.rstrip('/')}"


def find_chromium_binary() -> str | None:
    candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
    ]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None
