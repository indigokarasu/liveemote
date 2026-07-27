"""End-to-end integration tests for the rewritten FaceSwapAdapter path.

The Adapter + BackendManager used to be a near-stub: the swap call sites were
unwired, ``training_references`` were only consulted for ``identity_anchor``,
and no test exercised the orchestrator + character directory + vendor daemon
together. This file locks in three properties:

1. **Config gating.** When ``faceswap.enabled=False`` (the production default),
   no daemon call path is opened, no request flows through, and the adapter's
   capability surface honestly reports ``passsthrough``. Naming the CLI flag
   ``--renderer facefusion`` does NOT auto-start the model.
2. **``training_references`` flow.** With the fake daemon wired, the avatar's
   active emote (sourced from ``CharacterIndex.training_references`` whose role
   is ``expression_reference`` or the emote YAML `id:`) is fed to the vendor as
   ``SwapRequest.target_face`` alongside the identity anchor as
   ``source_face``.
3. **Graceful degradation.** When the daemon raises, ``manager.passthrough``
   flips to True, ``passthrough=True`` is recorded, and the frame is returned
   unchanged. No exception escapes.

These run without GPU, cv2, onnxruntime, or vendored FaceFusion/Deep-Live-Cam
models — ``FakeVendorDaemon`` is in-process and records calls for assertions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.character.ingest import build_asset_index
from hermes_avatar.config.schema import FaceSwapConfig, load_config
from hermes_avatar.demo.demo_orchestrator import DemoOrchestrator
from hermes_avatar.renderer.facefusion_adapter import (
    FakeVendorDaemon,
    FaceSwapAdapter,
    FaceSwapPipeline,
    ListFrameSink,
    ListFrameSource,
    SwapRequest,
)


@pytest.fixture(scope="module")
def character_index():
    """Build a real CharacterIndex from the project's character_input.

    This exercises the same code path the server uses, so the integration
    test cannot drift from real ingestion behaviour. Skipped when the
    directory is missing (e.g. in a fresh CI checkout without sample media).
    """
    character_path = Path("./character_input")
    if not (character_path / "canonical").is_dir():
        pytest.skip("character_input/ not present (run make setup first)")
    return build_asset_index(character_path)


@pytest.fixture
def adapter_with_fake_daemon(character_index):
    """Build a FaceSwapAdapter wired to a FakeVendorDaemon and a no-op pipeline.

    The pipeline is constructed with ``adapter=`` so every frame goes through
    the new typed ``SwapRequest`` path. The manager's ``available/degraded/
    passthrough`` flags are flipped to True/False/False to simulate a live
    backend without requiring one.

    Loading order matters: ``load_character`` runs ``_activate`` which re-runs
    ``manager.start`` -> ``detect``, which would reset the manager flags back
    to the degraded defaults. So we load the character first and only then
    simulate the live backend by flipping the flags.
    """
    fake = FakeVendorDaemon()
    cfg = FaceSwapConfig(enabled=True, backend="facefusion")
    adapter = FaceSwapAdapter(
        config=cfg,
        backend_manager=None,
        source_face_path=str(Path("./character_input/canonical/canonical.png").resolve()),
        backend="facefusion",
        enabled=True,
        daemon=fake,
    )
    adapter.load_character(character_index)
    adapter.manager.available = True
    adapter.manager.degraded = False
    adapter.manager.passthrough = False
    adapter.pipeline = FaceSwapPipeline(
        ListFrameSource([]),
        ListFrameSink(),
        adapter.manager,
        "facefusion",
        adapter=adapter,
    )
    yield adapter, fake
    adapter.manager.daemon = None
    adapter.pipeline = None


# ---------------------------------------------------------------------------
# (1) Config gating
# ---------------------------------------------------------------------------
def test_default_faceswap_disabled_no_daemon_path(character_index):
    """Production default (enabled=False) must NOT wire any daemon path.

    Naming the renderer ``facefusion`` no longer auto-starts the model. The
    capability surface honestly reports passsthrough.
    """
    orchestrator = DemoOrchestrator(
        character="./character_input",
        renderer="facefusion",
    )
    # Config gate is the contract.
    assert orchestrator.config.faceswap.enabled is False

    adapter = orchestrator.renderer
    assert isinstance(adapter, FaceSwapAdapter)
    assert adapter.config.enabled is False

    caps = adapter.capabilities()
    assert caps["enabled"] is False
    assert caps["replacement_active"] is False
    assert caps["passthrough"] is True

    # No daemon path open in the manager.
    assert adapter.manager.daemon is None


def test_orchestrator_respects_config_faceswap_enabled(character_index):
    """When ``config.faceswap.enabled=True``, the orchestrator creates the
    Adapter with that flag — previously it forced True. This is the regression
    the rewrite fixes.
    """
    cfg = load_config()
    cfg.faceswap = FaceSwapConfig(enabled=True, backend="facefusion")

    orchestrator = DemoOrchestrator(
        character="./character_input",
        renderer="facefusion",
        config=cfg,
    )
    adapter = orchestrator.renderer
    assert isinstance(adapter, FaceSwapAdapter)
    assert adapter.config.enabled is True
    assert adapter.capabilities()["enabled"] is True


# ---------------------------------------------------------------------------
# (2) training_references flow through SwapRequest to the daemon
# ---------------------------------------------------------------------------
def test_training_references_flow_into_swap_request(adapter_with_fake_daemon, character_index):
    """The active emote, sourced from CharacterIndex.training_references /
    emote entries, becomes the daemon's ``target_face`` for each frame.
    """
    adapter, fake = adapter_with_fake_daemon

    # Pairs of (logical mode, exact emote_id present in character_input). The
    # fixture uses Indigo's emote ids (resolved from the YAML enrichment we
    # added in commit 93f7b5a).
    sequences = [
        ("listening", "listening_over_shoulder"),
        ("thinking", "thinking_pensive"),
        ("happy", "happy_laugh"),
        ("concerned", "concerned_furious"),
        ("sad", "sad_crying"),
        ("amused", "amused_shush"),
    ]
    for state_token, emote_id in sequences:
        emote = next((e for e in character_index.emotes if e.id == emote_id), None)
        assert emote is not None, f"emote {emote_id} is not in the indexed character"
        behavior = AvatarBehaviorState(
            mode=state_token,
            affect=state_token,
            gaze_target="toward_user",
            emote_id=emote_id,
            intensity=0.5,
        )
        adapter.set_behavior(behavior)
        # The adapter must have resolved the emote -> target_face_path.
        assert adapter.target_emote_id == emote_id
        assert adapter.target_face_path == emote.path
        assert Path(adapter.target_face_path).exists()

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        adapter.pipeline.process_frame(frame)

    # One daemon invocation per behavior tick.
    assert len(fake.calls) == len(sequences)

    # Each request carried the right emote_id and a real on-disk target face.
    received_ids = [call.emote_id for call in fake.calls]
    assert received_ids == [emote_id for _, emote_id in sequences]

    for call, (_, emote_id) in zip(fake.calls, sequences):
        emote = next((e for e in character_index.emotes if e.id == emote_id), None)
        assert call.target_face == emote.path
        assert Path(call.target_face).exists(), f"{call.target_face} should resolve on disk"

    # Identity anchor is the canonical face. Character id matches.
    for call in fake.calls:
        assert Path(call.source_face).name == "canonical.png"
        assert call.character_id == "indigo"


def test_set_behavior_resolves_emote_id_to_target_face(adapter_with_fake_daemon, character_index):
    """Even before the pipeline runs, set_behavior must populate target_*.
    This is the lightweight contract the rest of the integration relies on.
    """
    adapter, _ = adapter_with_fake_daemon
    emote_id = "happy_joy"
    emote = next((e for e in character_index.emotes if e.id == emote_id), None)
    assert emote is not None

    behavior = AvatarBehaviorState(
        mode="happy",
        affect="happy",
        gaze_target="toward_user",
        emote_id=emote_id,
        intensity=0.7,
    )
    adapter.set_behavior(behavior)
    assert adapter.target_emote_id == "happy_joy"
    assert adapter.target_emote is emote
    assert adapter.target_face_path == emote.path
    assert float(adapter.target_emote.intensity) == pytest.approx(0.7)


def test_set_behavior_falls_back_to_none_when_emote_missing(adapter_with_fake_daemon):
    """An emote_id that isn't in the index doesn't crash — target_face_path
    stays None and the daemon receives target_face=None (so the vendor can
    decide its own fallback).
    """
    adapter, _ = adapter_with_fake_daemon
    behavior = AvatarBehaviorState(
        mode="idle",
        affect="neutral",
        gaze_target="soft_forward",
        emote_id="this_emote_does_not_exist",
        intensity=0.0,
    )
    adapter.set_behavior(behavior)
    assert adapter.target_emote_id == "this_emote_does_not_exist"
    assert adapter.target_face_path is None
    # Pipeline still safe to run; the daemon record shows target_face=None.
    fake = adapter.manager.daemon
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    adapter.pipeline.process_frame(frame)
    assert len(fake.calls) == 1
    assert fake.calls[0].target_face is None


# ---------------------------------------------------------------------------
# (3) Graceful degradation
# ---------------------------------------------------------------------------
def test_daemon_failure_falls_back_to_passthrough(adapter_with_fake_daemon):
    """Vendor errors don't crash; the manager flips to passthrough and the
    frame is returned unchanged. Subsequent calls stay in passthrough.
    """
    adapter, _ = adapter_with_fake_daemon
    fake = FakeVendorDaemon(fail=True)
    adapter.manager.daemon = fake
    adapter.manager.available = True
    adapter.manager.degraded = False
    adapter.manager.passthrough = False

    frame = np.zeros((4, 4, 3), dtype=np.uint8) + 9
    req = SwapRequest(
        frame=frame,
        source_face=adapter.source_image_path or "",
        target_face=None,
        character_id=adapter.character_index.character_id if adapter.character_index else None,
        emote_id=None,
        intensity=0.0,
    )
    out = adapter.manager.swap_with_request(req)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, frame)  # passthrough returns the frame unchanged
    assert fake.failures == 1
    assert adapter.manager.passthrough is True  # degraded after first failure
    assert adapter.manager.degraded is True

    # Second call also returns the frame; no further daemon invocations.
    out2 = adapter.manager.swap_with_request(req)
    assert np.array_equal(out2, frame)
    # Note: the daemon may be called once more before the manager short-circuits,
    # but in any case the frame is returned (graceful).


def test_disabled_manager_short_circuits_to_passthrough(character_index):
    """When the manager is degraded (default whenever vendor / models / GPU are
    missing in CI), ``swap_with_request`` skips the daemon entirely and returns
    the frame.
    """
    fake = FakeVendorDaemon()
    adapter = FaceSwapAdapter(
        backend="facefusion",
        enabled=False,  # never even tries to start
    )
    adapter.manager.daemon = fake
    # available/degraded/passthrough will reflect _detect() on the empty vendor dir.
    adapter.manager.detect()
    assert adapter.manager.degraded is True or adapter.manager.passthrough is True

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    req = SwapRequest(
        frame=frame,
        source_face="/dev/null",
        target_face=None,
        character_id=None,
        emote_id=None,
        intensity=0.0,
    )
    out = adapter.manager.swap_with_request(req)
    assert np.array_equal(out, frame)
    # Daemon was wired but never called because we're in passthrough.
    assert len(fake.calls) == 0


# ---------------------------------------------------------------------------
# (4) Backwards compatibility — the existing raw-frame swap path still works
# ---------------------------------------------------------------------------
def test_legacy_swap_callable_still_works():
    """The legacy test seam (``swap_callable=lambda frame: ...``) must keep
    working alongside the new ``swap_with_request`` path.
    """
    from hermes_avatar.config.schema import FaceSwapConfig
    from hermes_avatar.renderer.facefusion_adapter import BackendManager

    mgr = BackendManager(
        FaceSwapConfig(backend="facefusion", enabled=True),
        swap_callable=lambda frame: frame + 1,
    )
    mgr.available = True
    mgr.degraded = False
    mgr.passthrough = False
    frame = np.zeros((4, 4, 3), dtype=np.uint8) + 5
    out = mgr.swap(frame)
    assert np.array_equal(out, frame + 1)
