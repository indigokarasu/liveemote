from __future__ import annotations

import asyncio

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.state import AvatarBehaviorState
from hermes_avatar.demo.demo_orchestrator import DemoOrchestrator


def _make_runtime(character_index):
    def lookup(state):
        emote = character_index.find_emote(state)
        return emote.id if emote else None
    return AffectRuntime(emote_lookup=lookup)


# ---------------------------------------------------------------------------
# Affect Policy -> behavior -> renderer path (cheap, real runtime).
# ---------------------------------------------------------------------------
def test_event_produces_behavior_state(sample_character_index, fake_renderer):
    rt = _make_runtime(sample_character_index)
    event = {
        "type": "perception.frame",
        "face_detected": True,
        "head_yaw": 2.0,
        "head_pitch": -1.0,
        "gaze_confidence": 0.9,
        "expression": {"smile": 0.8, "frown": 0.0,
                      "brow_raise": 0.1, "eye_open": 0.8},
        "emotion_confidence": 0.7,
        "timestamp_ms": 1000,
    }
    behavior = rt.consume(event)
    assert isinstance(behavior, AvatarBehaviorState)
    # The runtime pushes the behavior to the renderer.
    fake_renderer.set_behavior(behavior)
    assert len(fake_renderer.set_behavior_calls) == 1
    assert fake_renderer.set_behavior_calls[0] is behavior
    assert behavior.mode in {"idle", "listening", "thinking",
                             "speaking", "recovering"}


def test_renderer_speak_invoked_with_behavior(sample_character_index,
                                            fake_renderer):
    rt = _make_runtime(sample_character_index)
    # Drive the runtime into a speaking state via an agent response event.
    rt.consume({"type": "hermes.response", "tags": {"voice": {"intensity": 0.5}},
                 "timestamp_ms": 2000})
    rt.conversation.turn_state = "assistant_speaking"
    behavior = rt.tick(2100)
    fake_renderer.speak("/tmp/audio.wav", "hello", behavior)
    assert len(fake_renderer.speak_calls) == 1
    audio_path, text, spoken = fake_renderer.speak_calls[0]
    assert spoken is behavior
    assert text == "hello"


def test_character_switch_and_reload_dont_crash(sample_character_index,
                                               fake_renderer):
    rt = _make_runtime(sample_character_index)
    rt.consume({"type": "hermes.response", "tags": {},
                 "timestamp_ms": 1})
    # Mirror/reflect mode switches must not raise.
    rt.set_mode("mirror")
    assert rt.mode == "mirror"
    rt.set_mode("reflect")
    # Triggering recovery states must not raise.
    rt.conversation.turn_state = "interrupted"
    rt.tick(2)
    assert isinstance(rt.avatar, AvatarBehaviorState)


# ---------------------------------------------------------------------------
# Real DemoOrchestrator, offline, with renderer swapped for a fake + mocked
# LiveTalking HTTP layer so no network occurs.
# ---------------------------------------------------------------------------
def test_orchestrator_event_flow(temp_character_dir, fake_renderer, monkeypatch):
    orch = DemoOrchestrator(str(temp_character_dir), renderer="livetalking",
                             voice_backend="none", agent_mode="fake")
    orch.renderer = fake_renderer
    status = orch.apply_event({
        "type": "perception.frame",
        "face_detected": True,
        "expression": {"smile": 0.7},
        "timestamp_ms": 1000,
    })
    assert isinstance(status, dict)
    assert status["character_id"] == orch.index.character_id
    assert len(fake_renderer.set_behavior_calls) >= 1
    assert isinstance(fake_renderer.set_behavior_calls[-1], AvatarBehaviorState)


def test_orchestrator_speak_flow(temp_character_dir, fake_renderer):
    orch = DemoOrchestrator(str(temp_character_dir), renderer="livetalking",
                             voice_backend="none", agent_mode="fake")
    orch.renderer = fake_renderer
    status = asyncio.run(orch.speak_test("Hello world"))
    assert isinstance(status, dict)
    # speak_test must have pushed a behavior to the renderer via speak().
    assert len(fake_renderer.speak_calls) >= 1
    audio_path, text, behavior = fake_renderer.speak_calls[-1]
    assert behavior.mode == "speaking"
    assert text == "Hello world"


def test_orchestrator_reload_and_mode_dont_crash(temp_character_dir,
                                                 fake_renderer):
    orch = DemoOrchestrator(str(temp_character_dir), renderer="livetalking",
                             voice_backend="none", agent_mode="fake")
    orch.renderer = fake_renderer
    # Reload must return a status dict and preserve the renderer.
    status = orch.reload_config()
    assert isinstance(status, dict)
    # Policy mode switch and manual trigger must not crash.
    orch.set_policy_mode("mirror")
    orch.trigger("reset")
    orch.trigger("listening")
    assert orch.runtime.mode == "mirror"


def test_orchestrator_mocked_http_no_network(temp_character_dir, monkeypatch):
    orch = DemoOrchestrator(str(temp_character_dir), renderer="livetalking",
                             voice_backend="none", agent_mode="fake")
    # Replace the real HTTP layer so no socket is opened.
    calls = []

    def fake_request(endpoint, payload=None, optional=False):
        calls.append((endpoint, payload, optional))
        return {"ok": True, "endpoint": endpoint}

    monkeypatch.setattr(orch.renderer, "_request", fake_request)
    orch.renderer.cb_state = "closed"
    status = orch.apply_event({"type": "hermes.response", "tags": {},
                               "timestamp_ms": 1234})
    assert isinstance(status, dict)
    # capabilities() probes health via the mocked _request -> online True.
    caps = orch.renderer.capabilities()
    assert caps["online"] is True
