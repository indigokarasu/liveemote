"""Mount ``sidecar.app:app`` under ``httpx.ASGITransport`` and exercise the
real FastAPI dep machinery.

Why this file exists separately from :mod:`tests.test_facefusion_sidecar`:

* That contract suite mocks ``httpx.Client`` at the DAEMON level, so it
  proves the daemon sends the right ``Authorization: Bearer tok`` header.
  It never proved the SERVER reads it. ``_bearer_dep`` was once written
  as ``async def _bearer_dep(authorization: str | None = None)`` — a
  FastAPI dep signature that treats unannotated ``str`` as a query
  parameter, so the real ``Authorization`` header never reached the dep
  and the gate silently 401'd on every request. MockTransport tests did
  not catch that.

* This file boots the actual FastAPI app via ``ASGITransport`` with
  ``python-multipart`` installed and a stubbed ``FaceFusionRunner``, then
  asserts the bearer dep correctly accepts / rejects / passes headers,
  the /health JSON shape, and that the multipart route parser is reached
  on success. Because the FastAPI dep tree is the genuine production
  code, every signature/annotation regression is caught here.

Async tests use ``asyncio_mode = "auto"`` (from pyproject.toml), so no
``@pytest.mark.asyncio`` decorator is needed.
"""
from __future__ import annotations

import os

# sidecar/app.py reads these at import time, so they MUST be set BEFORE the
# `import sidecar.app` below. pytest collects and imports the module fresh
# for each test session, so doing it at module top is correct.
os.environ.setdefault("FACESWAP__SIDECAR__AUTH_REQUIRED", "true")
os.environ.setdefault("FACESWAP__SIDECAR__API_KEY", "smoketest-secret")
os.environ.setdefault("FACESWAP__SIDECAR__VENDOR_DIR", "/home/daytona/codebase/vendor/FaceFusion")

import pytest
from httpx import ASGITransport, AsyncClient

import sidecar.app as sidecar_app  # noqa: E402


# ---------------------------------------------------------------------------
# Runner stub — bypass the vendored FaceFusion import path entirely. The
# runner's constructor is lazy (no FF import at __init__), but warmup() WILL
# try to import facefusion — we never call warmup() here, just build a
# HealthSnapshot and return it. Same runner shape, no ONNX.
# ---------------------------------------------------------------------------
class _StubRunner:
    vendor_present = True
    health = sidecar_app.FaceFusionRunner(
        vendor_dir=os.environ["FACESWAP__SIDECAR__VENDOR_DIR"]
    ).health
    # Force degraded — exercises the 503 path while keeping everything lazy.
    health.healthy = False
    health.last_error = "test stub: facefusion python-3.11 not exercised here"
    health.vendor_dir = os.environ["FACESWAP__SIDECAR__VENDOR_DIR"]

    def warmup(self):
        return self.health

    def extract_face(self, image):
        return None

    def swap(self, frame, *, source_face, target_face=None, intensity=1.0):
        return frame


sidecar_app._runner = _StubRunner()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ASGI client fixture
# ---------------------------------------------------------------------------
@pytest.fixture
async def client():
    transport = ASGITransport(app=sidecar_app.app)
    async with AsyncClient(transport=transport, base_url="http://sidecar.local") as c:
        yield c


# ---------------------------------------------------------------------------
# Bearer auth matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("auth_header", "expected_status"),
    [
        (None, 401),                              # no header → 401 missing bearer
        ("Token smoketest-secret", 401),          # wrong scheme → 401 missing bearer
        ("Bearer WRONG", 403),                    # wrong token  → 403 invalid bearer
        ("Bearer smoketest-secret", 503),         # valid token → reaches the route → 503 degraded
    ],
)
async def test_bearer_auth_matrix(client, auth_header, expected_status):
    headers = {"Authorization": auth_header} if auth_header else {}
    r = await client.get("/health", headers=headers)
    assert r.status_code == expected_status, (r.status_code, r.text)
    if expected_status == 401:
        # 401 must include WWW-Authenticate for clients to self-correct.
        assert r.headers.get("www-authenticate") == "Bearer"


async def test_health_returns_documented_payload_after_auth(client):
    r = await client.get(
        "/health", headers={"Authorization": "Bearer smoketest-secret"}
    )
    assert r.status_code == 503, r.text
    body = r.json()
    for key in (
        "status",
        "vendor_present",
        "vendor_dir",
        "face_analyser_loaded",
        "face_swapper_loaded",
        "swap_count",
    ):
        assert key in body, f"missing key: {key}"
    assert body["vendor_present"] is True
    assert body["status"] == "degraded"


async def test_swap_reaches_route_parser_after_auth(client):
    """With a valid Bearer, a request without multipart files must reach
    FastAPI's body parser and fail with 422 Unprocessable Entity. If auth
    were broken, this would 401 instead. The 422 contract is what proves
    the dep tree succeeded."""
    r = await client.post(
        "/api/v1/swap", headers={"Authorization": "Bearer smoketest-secret"}
    )
    assert r.status_code == 422, (r.status_code, r.text)


async def test_auth_disabled_skips_bearer_check(monkeypatch):
    """When ``FACESWAP__SIDECAR__AUTH_REQUIRED`` is false, the gate must
    be skipped regardless of header. The dep function reads the module
    attr at request time, so monkeypatch overrides work without a
    reload (which would break pytest's fixture finalizer accounting).
    """
    monkeypatch.setattr(sidecar_app, "AUTH_REQUIRED", False)
    monkeypatch.setattr(sidecar_app, "API_KEY", "")
    transport = ASGITransport(app=sidecar_app.app)
    async with AsyncClient(transport=transport, base_url="http://sidecar.local") as c:
        r = await c.get("/health")  # NO Authorization header
        # Reaches the route handler — degraded stub → 503 (not 401).
        assert r.status_code == 503, (r.status_code, r.text)
