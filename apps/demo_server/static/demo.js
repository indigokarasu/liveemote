/**
 * Hermes Avatar — Realism Client (demo.js)
 *
 * Six realism improvements driven by the server's AvatarBehaviorState:
 *   1. Emotional inertia — crossfade between .affect-* classes
 *   2. Idle micro-movements — pure CSS (no JS needed; see demo.css)
 *   3. Breathing entrainment — maps server breath_rate_hz → CSS --breath-duration
 *   4. Head tilt mirroring — maps server head_yaw/head_pitch → CSS custom props
 *   5. Gaze triangle — reads server gaze_point → toggles .gaze-* class
 *   6. Anticipatory micro-expressions — adds .pre-speech before speech start
 *
 * Also includes the perception-frame capture loop that POSTs webcam
 * frames to /api/perception/video at ~320 ms intervals.
 */

(function () {
  "use strict";

  // ─── Configuration ──────────────────────────────────────────────────
  const POLL_INTERVAL_MS = 50;          // status poll rate (~20 Hz)
  const PERCEPTION_INTERVAL_MS = 320;   // webcam frame capture rate
  const STATUS_ENDPOINT = "/api/status";
  const PERCEPTION_ENDPOINT = "/api/perception/video";

  // ─── DOM refs ───────────────────────────────────────────────────────
  const avatarContainer = document.getElementById("avatar-container");
  const avatarEmote = document.getElementById("avatar-emote");

  // ─── Client-side state ──────────────────────────────────────────────
  let prevBehavior = null;            // cached AvatarBehaviorState
  let perceptionTimer = null;         // setInterval handle for webcam loop
  let statusTimer = null;             // setInterval handle for status poll
  let videoStream = null;             // MediaStream from getUserMedia
  let videoEl = null;                 // hidden <video> for frame capture
  let canvasEl = null;                // hidden <canvas> for JPEG encode
  let preSpeechTimeout = null;        // timeout for anticipatory micro-flash

  // ─── Gaze triangle client-side timer ────────────────────────────────
  let gazeDwellRemaining = 0;
  let gazePoint = "eyes";
  let lastGazeTick = performance.now();

  // =====================================================================
  //  PERCEPTION FRAME CAPTURE (webcam → server)
  // =====================================================================

  async function startPerceptionCapture() {
    try {
      videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 180, facingMode: "user" },
        audio: false,
      });
    } catch (err) {
      console.warn("[hermes] webcam unavailable — perception disabled:", err.message);
      return;
    }

    // Hidden video element for frame capture
    videoEl = document.createElement("video");
    videoEl.srcObject = videoStream;
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    videoEl.muted = true;
    videoEl.width = 320;
    videoEl.height = 180;
    videoEl.style.display = "none";
    document.body.appendChild(videoEl);

    // Hidden canvas for JPEG encoding
    canvasEl = document.createElement("canvas");
    canvasEl.width = 320;
    canvasEl.height = 180;
    canvasEl.style.display = "none";
    document.body.appendChild(canvasEl);

    await videoEl.play();

    perceptionTimer = setInterval(captureAndSendFrame, PERCEPTION_INTERVAL_MS);
  }

  function captureAndSendFrame() {
    if (!videoEl || !canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, 320, 180);
    const jpegDataUrl = canvasEl.toDataURL("image/jpeg", 0.75);

    fetch(PERCEPTION_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: jpegDataUrl,
        timestamp_ms: Date.now(),
      }),
    }).catch(() => {
      // Silently ignore; server may not have the route yet
    });
  }

  function stopPerceptionCapture() {
    if (perceptionTimer) clearInterval(perceptionTimer);
    if (videoStream) {
      videoStream.getTracks().forEach((t) => t.stop());
      videoStream = null;
    }
    if (videoEl) { videoEl.remove(); videoEl = null; }
    if (canvasEl) { canvasEl.remove(); canvasEl = null; }
  }

  // =====================================================================
  //  STATUS POLL (server state → DOM)
  // =====================================================================

  async function startStatusPoll() {
    statusTimer = setInterval(pollStatus, POLL_INTERVAL_MS);
  }

  async function pollStatus() {
    try {
      const resp = await fetch(STATUS_ENDPOINT);
      if (!resp.ok) return;
      const data = await resp.json();
      const behavior = data.behavior || data;
      applyBehavior(behavior);
      prevBehavior = behavior;
    } catch {
      // Silently ignore transient failures
    }
  }

  /** Apply an AvatarBehaviorState to the DOM. */
  function applyBehavior(behavior) {
    if (!avatarContainer || !avatarEmote) return;

    // ── 1. Emotional inertia: crossfade between affect classes ────────
    applyAffectCrossfade(behavior);

    // ── 3. Breathing entrainment ──────────────────────────────────────
    applyBreathing(behavior.breath_rate_hz);

    // ── 4. Head tilt mirroring ────────────────────────────────────────
    applyHeadTilt(behavior.head_yaw, behavior.head_pitch);

    // ── 5. Gaze triangle ──────────────────────────────────────────────
    applyGazePoint(behavior.gaze_point, behavior.cognitive_mode);

    // ── 6. Anticipatory micro-expressions ─────────────────────────────
    applyPreSpeech(behavior);

    // ── Intensity scaling ─────────────────────────────────────────────
    applyIntensity(behavior.intensity);

    // ── Mode class ────────────────────────────────────────────────────
    avatarEmote.className = avatarEmote.className
      .replace(/\bmode-\S+/g, "")
      .trim();
    avatarEmote.classList.add(`mode-${behavior.mode || "reflect"}`);
  }

  // --------------------------------------------------------------------
  //  1. Emotional inertia: crossfade between affect classes
  // --------------------------------------------------------------------
  function applyAffectCrossfade(behavior) {
    const prevAffect = prevBehavior ? prevBehavior.target_affect : "neutral";
    const currentAffect = behavior.target_affect || "neutral";
    const progress = behavior.transition_progress ?? 1.0;
    const fadeMs = behavior.affect_fade_ms || 600;

    // Remove all existing affect classes
    const affectClassPattern = /\baffect-\S+/g;
    avatarEmote.className = avatarEmote.className.replace(affectClassPattern, "").trim();

    if (progress < 1.0 && prevAffect !== currentAffect) {
      // Mid-transition: add transitioning class for CSS crossfade
      avatarEmote.classList.add("affect-transitioning");
      // Set CSS transition duration to match server's recommended fade
      avatarEmote.style.transitionDuration = `${fadeMs}ms`;
    } else {
      avatarEmote.classList.remove("affect-transitioning");
      avatarEmote.style.transitionDuration = "";
    }

    // Apply current affect class
    avatarEmote.classList.add(`affect-${currentAffect}`);
  }

  // --------------------------------------------------------------------
  //  3. Breathing entrainment: breath_rate_hz → CSS --breath-duration
  // --------------------------------------------------------------------
  function applyBreathing(breathRateHz) {
    if (breathRateHz == null) return;
    // Convert Hz to seconds per cycle
    const durationSec = 1.0 / Math.max(breathRateHz, 0.1);
    document.documentElement.style.setProperty("--breath-duration", `${durationSec}s`);

    // Set breath intensity class for amplitude tuning
    const body = document.querySelector(".avatar-body");
    if (!body) return;

    body.className = body.className.replace(/\bbreath-\S+/g, "").trim();
    if (breathRateHz <= 0.2)       body.classList.add("breath-calm");
    else if (breathRateHz <= 0.28) body.classList.add("breath-neutral");
    else if (breathRateHz <= 0.38) body.classList.add("breath-alert");
    else                           body.classList.add("breath-excited");
  }

  // --------------------------------------------------------------------
  //  4. Head tilt mirroring: head_yaw/head_pitch → CSS custom props
  // --------------------------------------------------------------------
  function applyHeadTilt(headYaw, headPitch) {
    // MediaPipe head yaw/pitch are in degrees; clamp to reasonable range
    const yaw = Math.max(-15, Math.min(15, headYaw || 0));
    const pitch = Math.max(-10, Math.min(10, headPitch || 0));

    // Map to avatar's coordinate space (mirrored horizontally, dampened vertically)
    document.documentElement.style.setProperty("--head-yaw", `${-yaw * 0.6}deg`);
    document.documentElement.style.setProperty("--head-pitch", `${pitch * 0.4}deg`);
  }

  // --------------------------------------------------------------------
  //  5. Gaze triangle: gaze_point → .gaze-* class
  // --------------------------------------------------------------------
  function applyGazePoint(serverGazePoint, cognitiveMode) {
    // Remove all gaze classes
    avatarContainer.className = avatarContainer.className
      .replace(/\bgaze-\S+/g, "")
      .trim();

    // Apply server-suggested gaze point
    const point = serverGazePoint || "soft_forward";
    avatarContainer.classList.add(`gaze-${point}`);
  }

  // --------------------------------------------------------------------
  //  6. Anticipatory micro-expressions: pre-speech flash
  // --------------------------------------------------------------------
  function applyPreSpeech(behavior) {
    const wasSpeaking = prevBehavior ? prevBehavior.is_speaking : false;
    const isSpeaking = behavior.is_speaking === true;

    if (isSpeaking && !wasSpeaking) {
      // Speech just started: show pre-speech flash for ~120 ms
      avatarEmote.classList.add("pre-speech");
      if (preSpeechTimeout) clearTimeout(preSpeechTimeout);
      preSpeechTimeout = setTimeout(() => {
        avatarEmote.classList.remove("pre-speech");
        avatarEmote.classList.add("mode-speaking");
      }, 120);
    } else if (!isSpeaking && wasSpeaking) {
      avatarEmote.classList.remove("mode-speaking", "pre-speech");
    }
  }

  // --------------------------------------------------------------------
  //  Intensity scaling
  // --------------------------------------------------------------------
  function applyIntensity(intensity) {
    const clamped = Math.max(0, Math.min(1, intensity || 0.5));
    document.documentElement.style.setProperty("--intensity-scale", clamped.toString());

    avatarEmote.className = avatarEmote.className
      .replace(/\bintensity-\S+/g, "")
      .trim();
    if (clamped <= 0.4)      avatarEmote.classList.add("intensity-low");
    else if (clamped <= 0.7) avatarEmote.classList.add("intensity-mid");
    else                     avatarEmote.classList.add("intensity-high");
  }

  // =====================================================================
  //  INIT
  // =====================================================================

  function init() {
    startPerceptionCapture();
    startStatusPoll();
  }

  // Wait for DOM readiness
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Expose for debugging
  window.__hermesRealism = {
    getPrevBehavior: () => prevBehavior,
    stop: () => {
      stopPerceptionCapture();
      if (statusTimer) clearInterval(statusTimer);
    },
  };
})();
