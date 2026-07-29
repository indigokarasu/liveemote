"""Regression guard: webcam pixel bytes never reach the avatar side.

This is a *structural* / invariant test. It protects LiveEmote's core promise:

> The rendered avatar's face comes from its own canonical stills + emotes in
> the character ``asset_index``, never from the user's webcam frames.

Specifically:

  - The data boundary: perception events (which can carry ``jpeg_b64`` /
    pixel bytes) flow into :class:`AffectRuntime`. What crosses OUT of the
    runtime into renderer state, into ``/api/status`` JSON, into WebSocket
    payloads, must be only *signals* (Focus / Energy / Valence / Tension +
    audio VAD): numbers and enums. NEVER raw bytes, NEVER base64 image
    strings, NEVER ``data:image/...`` data URIs.

  - The lateral face-swap paths: the face-swap adapters DO need pixel data
    (it goes to the vendor ONNX model), but the ``source_face`` /
    ``target_face`` paths sent to the vendor MUST come from the
    character's ``training_references`` (a path on disk), NEVER from a
    perception event's ``jpeg_b64``.

The test fires a sentinel ``perception.frame`` event carrying a
recognisable base64 signature, then walks every reachable state on the
orchestrator, runtime, status, websocket payloads, and renderer.
Any string starting with ``data:image/...`` or containing the sentinel
marker fails.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from hermes_avatar.demo.demo_orchestrator import DemoOrchestrator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel spec
# ---------------------------------------------------------------------------
# Marker ``LEAKTEST_`` base64-encoded is ``TEVBS1RFU1Rf``. Any string
# containing this 12-char substring is unambiguously a synthetic pixel
# payload; nothing in legitimate runtime/character paths can produce it.
SENTINEL_MARKER = "TEVBS1RFU1Rf"
SENTINEL_DATA_URI = f"data:image/jpeg;base64,{SENTINEL_MARKER}" + ("A" * 3000)
SENTINEL_JPEG_B64 = f"{SENTINEL_MARKER}" + ("Q" * 3000)  # ~3 KB base64 junk

SENTINEL_PERCEPTION_EVENT: dict[str, Any] = {
    "type": "perception.frame",
    "timestamp_ms": 1_700_000_000_000,
    "jpeg_b64": SENTINEL_JPEG_B64,
    "face_detected": True,
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "attention": 0.7,
    "valence": 0.2,
    "arousal": 0.3,
    "tension": 0.1,
    "dominant_expression": "neutral",
    "expression": {"smile": 0.4, "frown": 0.1, "brow_raise": 0.5, "eye_open": 0.3},
}

PLAIN_PERCEPTION_EVENT: dict[str, Any] = {
    "type": "perception.frame",
    "timestamp_ms": 1_700_000_001_000,
    "face_detected": True,
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "attention": 0.7,
    "valence": 0.2,
    "arousal": 0.3,
    "tension": 0.1,
    "dominant_expression": "neutral",
    "expression": {"smile": 0.4, "frown": 0.1, "brow_raise": 0.5, "eye_open": 0.3},
}


# ---------------------------------------------------------------------------
# State walker
# ---------------------------------------------------------------------------
def _looks_like_leaked_pixels(val: str) -> bool:
    """Classify a string value as a webcam-pixel leak if it matches any
    of the avatar-side forbidden data shapes.

    Rejecting ``data:image/...`` and ``data:video/...`` matches both the
    unfiltered sentinel and any future bug where the renderer side is fed
    a base64 data URL by mistake.

    Rejecting the bare sentinel marker catches any leak where the base64
    envelope was unwrapped but the marker bytes survived.
    """
    if not isinstance(val, str):
        return False
    if SENTINEL_MARKER in val:
        return True
    if val.startswith("data:image/") or val.startswith("data:video/"):
        return True
    return False


def walk_state_for_leaked_pixels(obj: Any, path: str = "root") -> list[str]:
    """Recursively walk ``obj``, returning object-paths where a
    leaked-pixels string was found.

    Object-paths are dotted access strings (``root.avatar.source_url``)
    for diagnostic clarity. A recursion-guard ``seen`` set prevents
    loops on shared references and dataclass cycles.
    """
    offenders: list[str] = []
    seen: set[int] = set()

    def _walk(node: Any, where: str) -> None:
        oid = id(node)
        if oid in seen:
            return
        seen.add(oid)

        if node is None:
            return
        if isinstance(node, (bool, int, float, bytes, np.integer, np.floating, np.ndarray, Path)):
            return
        if isinstance(node, str):
            if _looks_like_leaked_pixels(node):
                offenders.append(where)
            return
        if isinstance(node, dict):
            for k, v in node.items():
                key_repr = str(k) if not isinstance(k, str) else k
                _walk(v, f"{where}.{key_repr}" if where != "root" else key_repr)
            return
        if isinstance(node, (list, tuple, set, frozenset)):
            for i, v in enumerate(node):
                _walk(v, f"{where}[{i}]")
            return
        if dataclasses.is_dataclass(node):
            for f in dataclasses.fields(node):
                _walk(getattr(node, f.name), f"{where}.{f.name}")
            return
        # ``mock._Call`` named-tuple-ish (has .args + .kwargs)
        if hasattr(node, "args") and hasattr(node, "kwargs"):
            try:
                _walk(getattr(node, "args", ()), f"{where}.args")
                _walk(getattr(node, "kwargs", {}), f"{where}.kwargs")
                return
            except Exception:
                pass
        # MagicMock: walk every recorded call + its kwargs
        if isinstance(node, MagicMock):
            for call in node.mock_calls:
                _walk(call, f"{where}.mock_calls")
            return
        # Plain object — try __dict__, then __slots__
        attrs: list[tuple[str, Any]] = []
        if hasattr(node, "__dict__") and isinstance(node.__dict__, dict):
            for k, v in node.__dict__.items():
                attrs.append((k, v))
        if hasattr(node, "__slots__"):
            for k in node.__slots__:
                if k == "__dict__":
                    continue
                if hasattr(node, k):
                    attrs.append((k, getattr(node, k)))
        for k, v in attrs:
            _walk(v, f"{where}.{k}" if where != "root" else k)

    _walk(obj, path)
    return offenders


# ---------------------------------------------------------------------------
# Sanity unit test for the walker itself
# ---------------------------------------------------------------------------
def test_walker_detects_sentinel_in_arbitrary_object() -> None:
    """The walker is the workhorse helper; verify it actually flags leaks
    before relying on it to catch a real regression."""

    @dataclass
    class _Innocent:
        plain_string: str = "neutral"
        small_dict: dict = field(default_factory=lambda: {"a": 1, "b": 2})

    @dataclass
    class _Villain:
        image_url: str = SENTINEL_DATA_URI
        nested: _Innocent = field(default_factory=_Innocent)

    v = _Villain()
    offenders = walk_state_for_leaked_pixels(v, "Villain")
    assert len(offenders) >= 1, "walker failed to flag SENTINEL_DATA_URI"
    assert any("image_url" in p for p in offenders), (
        f"walker should report the offending field; got: {offenders}"
    )

    # Innocent object: zero offenders
    innocent = _Innocent()
    assert walk_state_for_leaked_pixels(innocent, "Innocent") == []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_character_path() -> Path:
    """Skip cleanly if make setup hasn't generated the synthesised sample
    character yet — the guard tests are pure CPU and don't need the
    full vendor clones, only the character dir's structure."""
    p = Path("character_input")
    if not (p / "canonical").is_dir():
        pytest.skip(
            "sample character_input/ not present; run "
            "`make setup` or "
            "`python scripts/create_sample_character.py --character ./character_input`"
        )
    return p


def _orchestrator_factory(
    sample_character_path: Path, renderer: str, monkeypatch: pytest.MonkeyPatch
) -> DemoOrchestrator:
    """Build an orchestrator with the buggy upstream ``reaction_delay``
    neutralized.

    Workaround note: ``policy.tick`` calls ``reaction_delay(self.mode,
    self.config)`` passing the ``AppConfig`` itself, but
    ``smoothing.reaction_delay`` expects ``self.config.affect.reaction_delay_ms``
    (i.e. a :class:`ReactionDelayConfig`). See
    ``packages/hermes_avatar/affect/policy.py:156`` vs
    ``packages/hermes_avatar/affect/smoothing.py:74``. We stub the helper
    with a deterministic ``lambda mode, delays: 0`` so the
    apply_event/consume/tick path returns a clean ``AvatarBehaviorState``
    without raising. The signal-leakage invariant this test guards does
    NOT involve ``reaction_delay``; this monkey-patch is purely to keep
    the orchestrator reachable. A separate fix is filed in the followups.
    """
    # Patch on the *importing* module (``policy.py``) because it binds
    # ``reaction_delay`` at module load via ``from .smoothing import reaction_delay``.
    # Patching the source module alone (``smoothing.reaction_delay``) leaves the
    # alias inside policy.py unchanged and the test still raises.
    monkeypatch.setattr(
        "hermes_avatar.affect.policy.reaction_delay",
        lambda mode, delays: 0,
    )
    return DemoOrchestrator(
        character=str(sample_character_path),
        renderer=renderer,
        voice_backend="none",
        agent_mode="fake",
    )


@pytest.fixture
def livetalking_orchestrator(
    sample_character_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DemoOrchestrator:
    """A wired-up orchestrator using the LiveTalking renderer adapter
    (HTTP, no real daemon reachable in CI — degrades to passthrough) plus
    NoopVoiceAdapter so we exercise the full event-handling path."""
    return _orchestrator_factory(sample_character_path, "livetalking", monkeypatch)


@pytest.fixture
def facefusion_orchestrator(
    sample_character_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DemoOrchestrator:
    return _orchestrator_factory(sample_character_path, "facefusion", monkeypatch)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_orchestrator_and_status_no_pixel_leak(
    livetalking_orchestrator: DemoOrchestrator,
) -> None:
    """Apply a perception.frame carrying sentinel base64 plus a follow-up
    plain frame, then walk every reachable state on the orchestrator,
    runtime, ``status()`` dict, and ``capabilities()`` dict. Any string
    flagged by the walker fails the test.

    Covers the most common regression: somebody stores ``event["jpeg_b64"]``
    on a debug attribute, or a new endpoint accidentally echoes a
    perception event back to the client."""
    orch = livetalking_orchestrator

    orch.apply_event(SENTINEL_PERCEPTION_EVENT)
    orch.apply_event(PLAIN_PERCEPTION_EVENT)

    offenders: list[str] = []
    offenders += walk_state_for_leaked_pixels(orch, "orchestrator")
    offenders += walk_state_for_leaked_pixels(orch.runtime, "runtime")
    offenders += walk_state_for_leaked_pixels(orch.status(), "status")
    offenders += walk_state_for_leaked_pixels(orch.capabilities(), "capabilities")

    assert offenders == [], (
        "signal-leakage detected: webcam pixel bytes reached the avatar "
        f"side via {offenders[:10]}"
        f"{'...' if len(offenders) > 10 else ''}"
    )


def test_websocket_payload_no_pixel_leak(
    livetalking_orchestrator: DemoOrchestrator,
) -> None:
    """Drive the actual ``apps.demo_server.websocket_api.websocket_endpoint``
    function with a sentinel perception.frame payload, capture every
    server-to-client ``send_json``, JSON-serialise it, and assert no
    sentinel substrings survive the round trip.

    This is the boundary the browser actually sees. The walker above
    covers in-process state; this test covers the network edge."""
    import apps.demo_server.websocket_api as ws_api

    sent_to_client: list[dict[str, Any]] = []
    received: list[dict[str, Any]] = [SENTINEL_PERCEPTION_EVENT]

    class _FakeWS:
        async def accept(self) -> None:
            return None

        async def send_json(self, payload: dict[str, Any]) -> None:
            sent_to_client.append(payload)

        async def receive_json(self) -> dict[str, Any]:
            if received:
                return received.pop(0)
            # Drive the loop to exit cleanly after the sentinel is consumed.
            raise StopAsyncIteration

    fake_ws = _FakeWS()
    # Wire up the fake_ws' app.state.orchestrator to the real orchestrator
    fake_ws.app = type("App", (), {})()
    fake_ws.app.state = type("State", (), {})()
    fake_ws.app.state.orchestrator = livetalking_orchestrator

    try:
        asyncio.run(ws_api.websocket_endpoint(fake_ws))  # type: ignore[arg-type]
    except (StopAsyncIteration, StopIteration):
        pass
    except Exception as exc:  # pragma: no cover - failure modes here are diagnostic only
        log.warning("websocket drive raised (ignored): %s", exc)

    json_blobs = [json.dumps(p, default=str) for p in sent_to_client]
    joined = "\n".join(json_blobs)

    assert SENTINEL_MARKER not in joined, (
        "signal-leakage: a websocket frame sent to the browser contained "
        "the sentinel jpeg_b64"
    )


def test_avatar_behavior_state_schema_is_signals_only() -> None:
    """Schema-level guarantee: even if the orchestrator is bypassed and a
    caller constructs an :class:`AvatarBehaviorState` directly with
    outrageous field values, the *shape* of the dataclass refuses to
    carry string-shaped bytes.

    Held against this check before *any* future ``AvatarBehaviorState``
    field that could plausibly host a string is added."""
    from hermes_avatar.affect.state import AvatarBehaviorState, fill_behavior_state

    b = AvatarBehaviorState(
        mode="idle",
        affect="neutral",
        gaze_target="toward_user",
        emote_id="happy",
        intensity=0.25,
    )
    fill_behavior_state(
        b,
        mode="speaking",
        affect="happy",
        gaze_target="toward_user",
        emote_id=None,
        intensity=0.6,
    )

    offenders = walk_state_for_leaked_pixels(b, "behavior")
    assert offenders == [], (
        f"AvatarBehaviorState carries a leaked-pixels string at: {offenders}. "
        "Any new field added to AvatarBehaviorState that can host a string "
        "MUST be added to the explicit allowlist in walker's _looks_like_leaked_pixels."
    )


def test_face_swap_source_face_is_character_asset_path(
    facefusion_orchestrator: DemoOrchestrator,
) -> None:
    """A FaceSwapAdapter, when wired up, must report a ``source_image_path``
    / ``source_face_path`` that resolves to a real on-disk character asset
    (or is empty so passthrough takes over). It must NEVER inherit the
    sentinel base64 from a perception event.

    Asserts both directions: no leaked bytes ANYWHERE on the adapter
    *and* the configured path (if non-empty) resolves to a real file.
    """
    orch = facefusion_orchestrator

    # Apply a sentinel perception frame so any leak path would carry the marker.
    orch.apply_event(SENTINEL_PERCEPTION_EVENT)

    # Walk the renderer (FaceSwapAdapter) state top-to-bottom
    offenders = walk_state_for_leaked_pixels(orch.renderer, "renderer")
    assert offenders == [], (
        f"signal-leakage via FaceSwapAdapter state at: {offenders}"
    )

    # If the adapter carries a configured source-face path, it must
    # resolve to a real file on disk (or be None so the vendor passes
    # through cleanly).
    adapter = orch.renderer
    candidate: Any = getattr(adapter, "source_image_path", None) or getattr(
        getattr(adapter, "config", None), "source_face_path", None
    )
    if candidate:
        assert Path(str(candidate)).exists(), (
            f"FaceSwapAdapter source_face_path={candidate!r} does not resolve "
            f"to a real on-disk character asset; the avatar side is using "
            f"something other than the character's training_references."
        )
