from __future__ import annotations

import time

import numpy as np
import pytest

from hermes_avatar.affect.policy import AffectRuntime
from hermes_avatar.affect.smoothing import clamp, clamp_vector, ema, ema_vector
from hermes_avatar.affect.state import (
    AvatarBehaviorState,
    acquire_behavior_state,
    release_behavior_state,
    reset_behavior_state_pool,
)
from hermes_avatar.character.asset_index import EmoteAsset
from hermes_avatar.config.schema import load_config
from hermes_avatar.voice.voice_cache import VoiceCache


# ---------------------------------------------------------------------------
# perf-1: tick() stays correct and fast after config-scalar caching
# ---------------------------------------------------------------------------
def test_runtime_caches_config_scalars():
    rt = AffectRuntime(emote_lookup=lambda s: f"{s}_001")
    assert rt._face_alpha == rt.config.affect.smoothing.face_alpha
    assert rt._audio_alpha == rt.config.affect.smoothing.audio_alpha
    assert rt._affect_alpha == rt.config.affect.smoothing.affect_alpha
    assert rt._max_yaw == rt.config.gaze.max_yaw_deg
    assert rt._mirroring_strength == rt.config.behavior.mirroring_strength
    assert rt._reaction_delay is rt.config.affect.reaction_delay_ms


def test_tick_behavior_unchanged_across_modes():
    rt = AffectRuntime(emote_lookup=lambda s: f"{s}_001")
    # user_speaking -> listening
    rt.consume({"type": "audio.vad", "timestamp_ms": 1000, "speaking": True,
                "energy": 0.7, "speech_rate": 0.5})
    b = rt.tick(1016)
    assert b.mode == "listening"
    assert b.gaze_target == "toward_user"
    # assistant_thinking -> thinking (stop "user speaking" first)
    rt.user.speaking = False
    rt.conversation.turn_state = "assistant_thinking"
    rt.user.dominant_expression = "happy"
    rt.user.face_detected = True
    b2 = rt.tick(1032)
    assert b2.mode == "thinking"
    assert 0.0 <= b2.intensity <= 1.0
    assert b2.full_body_pose == "thinking_shift"
    # idle -> idle
    rt.conversation.turn_state = "idle"
    b3 = rt.tick(1048)
    assert b3.mode == "idle"
    assert 0.0 <= b3.intensity <= 1.0
    assert b3.to_dict()["mode"] == "idle"


@pytest.mark.performance
def test_tick_latency_with_pooling():
    rt = AffectRuntime(emote_lookup=lambda s: f"{s}_001")
    N = 4000
    rt.tick(0)  # warmup
    t0 = time.perf_counter()
    for i in range(N):
        rt.tick(i)
    elapsed = time.perf_counter() - t0
    assert elapsed < 4.0, f"{elapsed:.3f}s for {N} ticks"


# ---------------------------------------------------------------------------
# perf-2: numpy vector EMA / clamp helpers match scalar math
# ---------------------------------------------------------------------------
def test_ema_vector_matches_scalar_loop():
    prev = [0.1, 0.2, 0.3, 0.4]
    cur = [0.9, 0.1, 0.5, 0.0]
    alpha = 0.3
    expected = [ema(p, c, alpha) for p, c in zip(prev, cur)]
    got = ema_vector(prev, cur, alpha)
    assert isinstance(got, np.ndarray)
    assert np.allclose(got, expected)


def test_ema_vector_broadcast_alpha():
    prev = np.array([0.0, 0.5, 1.0])
    cur = np.array([1.0, 1.0, 1.0])
    got = ema_vector(prev, cur, 0.25)
    expected = prev + 0.25 * (cur - prev)
    assert np.allclose(got, expected)


def test_ema_scalar_none_returns_current():
    assert ema(None, 0.42, 0.3) == 0.42


def test_clamp_vector_matches_scalar():
    vals = [-0.5, 0.3, 1.2, 0.8]
    got = clamp_vector(vals, 0.0, 1.0)
    expected = [clamp(v, 0.0, 1.0) for v in vals]
    assert np.allclose(got, expected)


@pytest.mark.performance
def test_numpy_ema_faster_than_scalar_loop_for_vectors():
    n = 256
    prev = np.zeros(n)
    cur = np.full(n, 0.3)
    alpha = 0.3
    # Establish both produce the same result first (real-logic check).
    vec = ema_vector(prev, cur, alpha)
    scalar = np.array([ema(p, c, alpha) for p, c in zip(prev, cur)])
    assert np.allclose(vec, scalar)
    # numpy must be meaningfully faster for this vector size.
    reps = 5000
    t0 = time.perf_counter()
    for _ in range(reps):
        ema_vector(prev, cur, alpha)
    t_vec = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(reps):
        [ema(p, c, alpha) for p, c in zip(prev, cur)]
    t_scalar = time.perf_counter() - t1
    assert t_vec < t_scalar, f"numpy {t_vec:.3f}s not faster than scalar {t_scalar:.3f}s"


# ---------------------------------------------------------------------------
# perf-3: caching of expensive lookups
# ---------------------------------------------------------------------------
def test_asset_index_emotes_for_is_memoized(sample_character_index):
    idx = sample_character_index
    first = idx.emotes_for("neutral")
    second = idx.emotes_for("neutral")
    assert first is second  # same cached list object
    assert [e.id for e in first] == ["neutral_001"]
    # distinct query args are cached separately
    assert idx.emotes_for("thinking") is idx.emotes_for("thinking")


def test_asset_index_emotes_for_tags_query_cached(sample_character_index):
    idx = sample_character_index
    idx.emotes.append(EmoteAsset(id="listening_wave", path="/tmp/lw.png",
                                 state="listening", tags=["wave"]))
    a = idx.emotes_for("listening", tags={"wave"})
    b = idx.emotes_for("listening", tags={"wave"})
    assert a is b
    assert [e.id for e in a] == ["listening_wave"]


def test_asset_index_find_style_and_background_cached(sample_character_index):
    idx = sample_character_index
    s1 = idx.find_style("cyberpunk")
    s2 = idx.find_style("cyberpunk")
    assert s1 is s2
    assert s1.id == "cyberpunk"
    bg1 = idx.find_background("studio")
    bg2 = idx.find_background("studio")
    assert bg1 is bg2
    assert bg1.id == "studio"
    assert idx.find_style(None) is None
    assert idx.find_background(None) is None


def test_voice_cache_store_and_read_roundtrip(tmp_path):
    cache = VoiceCache(root=tmp_path / "vc", memory_cache_size=4)
    data = b"RIFF....fake-wav-bytes"
    path = cache.store_bytes("hello world", "luxtts", data)
    assert path.exists()
    assert cache.read_bytes("hello world", "luxtts") == data
    # Same path is returned for identical inputs.
    assert cache.path_for("hello world", "luxtts") == path


def test_voice_cache_in_memory_hit_survives_missing_file(tmp_path):
    cache = VoiceCache(root=tmp_path / "vc", memory_cache_size=4)
    data = b"audio-bytes"
    cache.store_bytes("repeat me", "luxtts", data)
    # Delete on-disk copy to prove the in-memory cache serves the read.
    cache.path_for("repeat me", "luxtts").unlink()
    assert cache.read_bytes("repeat me", "luxtts") == data


def test_voice_cache_memory_cache_is_bounded(tmp_path):
    cache = VoiceCache(root=tmp_path / "vc", memory_cache_size=2)
    for i in range(10):
        cache.store_bytes(f"t{i}", "luxtts", bytes([i]))
    assert len(cache._bytes_cache) <= 2


# ---------------------------------------------------------------------------
# perf-4: object pooling for AvatarBehaviorState
# ---------------------------------------------------------------------------
def test_pool_reuses_released_objects():
    reset_behavior_state_pool()
    o1 = acquire_behavior_state()
    release_behavior_state(o1)
    o2 = acquire_behavior_state()
    assert o2 is o1  # recycled, no new allocation


def test_behavior_state_reset_clears_fields():
    o = acquire_behavior_state()
    o.mode = "speaking"
    o.affect = "focused"
    o.intensity = 0.9
    o.lip_sync_enabled = True
    o.reset()
    assert o.mode == "idle"
    assert o.affect == "neutral"
    assert o.intensity == 0.25
    assert o.lip_sync_enabled is False
    assert o.full_body_pose == "standing_idle"


def test_tick_uses_pool_without_corrupting_prior_frame():
    reset_behavior_state_pool()
    rt = AffectRuntime(emote_lookup=lambda s: f"{s}_001")
    # Capture a returned state, then tick again. The pool must not mutate the
    # captured object (it is released only after the next acquire).
    captured = rt.tick(0)
    captured_snapshot = captured.to_dict()
    rt.tick(1)
    assert captured.to_dict() == captured_snapshot


def test_tick_recycles_pooled_instances():
    reset_behavior_state_pool()
    rt = AffectRuntime(emote_lookup=lambda s: f"{s}_001")
    a = rt.tick(0)
    b = rt.tick(1)
    c = rt.tick(2)
    # Only one object is ever in flight, so the pool recycles it.
    assert c is a
    assert isinstance(b, AvatarBehaviorState)
