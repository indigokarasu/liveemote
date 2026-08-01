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
 * frames to /api/perception/video at ~320 ms intervals, and populates
 * the on-screen indicator cards (affect bar, breath circle, gaze dot,
 * head-tilt readouts, telemetry pills).
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
  const webcamPreview = document.getElementById("webcam-preview");
  const webcamNote = document.getElementById("webcam-note");

  // Indicator card DOM refs
  const affectLabel = document.getElementById("affect-label");
  const affectBarFill = document.getElementById("affect-bar-fill");
  const affectTransitionLabel = document.getElementById("affect-transition-label");
  const breathRateDisplay = document.getElementById("breath-rate-display");
  const gazeDot = document.getElementById("gaze-dot");
  const gazeLabel = document.getElementById("gaze-label");
  const tiltYawVal = document.getElementById("tilt-yaw-val");
  const tiltYawFill = document.getElementById("tilt-yaw-fill");
  const tiltPitchVal = document.getElementById("tilt-pitch-val");
  const tiltPitchFill = document.getElementById("tilt-pitch-fill");

  // Telemetry pill refs
  const telemMode = document.getElementById("telem-mode");
  const telemCognitive = document.getElementById("telem-cognitive");
  const telemSpeaking = document.getElementById("telem-speaking");
  const telemIntensity = document.getElementById("telem-intensity");
  const telemFace = document.getElementById("telem-face");
  const telemEmote = document.getElementById("telem-emote");
  const telemMirror = document.getElementById("telem-mirror");

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
      if (webcamNote) {
        webcamNote.textContent = "Webcam unavailable. Perception pipeline inactive.";
        webcamNote.style.color = "#ef4444";
      }
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

    // Wire the visible webcam preview (shares the same MediaStream)
    if (webcamPreview) {
      webcamPreview.srcObject = videoStream;
      webcamPreview.classList.add("active");
      if (webcamNote) {
        webcamNote.textContent = "Frames stay local. Downsampled JPEGs feed the server-side MediaPipe tracker. The avatar never displays your face.";
        webcamNote.style.color = "";
      }
    }

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
    if (webcamPreview) {
      webcamPreview.classList.remove("active");
    }
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

    // ── Update all indicator cards ────────────────────────────────────
    updateIndicators(behavior);
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
  //  INDICATOR CARD UPDATES
  // =====================================================================

  function updateIndicators(behavior) {
    // ── Affect indicator (1) ──────────────────────────────────────────
    if (affectLabel) {
      const affect = behavior.target_affect || "neutral";
      affectLabel.textContent = affect.replace(/_/g, " ");
    }
    if (affectBarFill) {
      const intensity = behavior.intensity ?? 0.5;
      affectBarFill.style.width = `${intensity * 100}%`;
      // Color gradient based on affect category
      const aff = (behavior.target_affect || "");
      if (/warm|smile|joy|happy/i.test(aff))
        affectBarFill.style.background = "#fbbf24";
      else if (/sad|consol|concern|ground/i.test(aff))
        affectBarFill.style.background = "#818cf8";
      else if (/angry|validat|patien/i.test(aff))
        affectBarFill.style.background = "#34d399";
      else if (/calm|neutral/i.test(aff))
        affectBarFill.style.background = "#a78bfa";
      else
        affectBarFill.style.background = "#818cf8";
    }
    if (affectTransitionLabel) {
      const prog = behavior.transition_progress ?? 1.0;
      if (prog >= 1.0)
        affectTransitionLabel.textContent = "settled";
      else
        affectTransitionLabel.textContent = `crossfading ${Math.round(prog * 100)}%`;
    }

    // ── Breath indicator (3) ──────────────────────────────────────────
    if (breathRateDisplay) {
      const hz = behavior.breath_rate_hz ?? 0.25;
      const bpm = Math.round(hz * 60);
      breathRateDisplay.textContent = `${bpm} bpm`;
    }

    // ── Gaze indicator (5) ────────────────────────────────────────────
    if (gazeDot) {
      const gp = behavior.gaze_point || "soft_forward";
      const offsets = {
        eyes: "translate(-6px, -4px)",
        mouth: "translate(0px, 6px)",
        away: "translate(8px, -6px)",
        soft_forward: "translate(0px, 0px)",
      };
      gazeDot.style.transform = offsets[gp] || offsets.soft_forward;
    }
    if (gazeLabel) {
      gazeLabel.textContent = (behavior.gaze_point || "soft_forward").replace(/_/g, " ");
    }

    // ── Head-tilt indicator (4) ───────────────────────────────────────
    if (tiltYawVal && tiltYawFill) {
      const yaw = Math.max(-15, Math.min(15, behavior.head_yaw || 0));
      tiltYawVal.textContent = `${yaw.toFixed(1)}°`;
      tiltYawFill.style.marginLeft = `${((yaw + 15) / 30) * 60}%`;
    }
    if (tiltPitchVal && tiltPitchFill) {
      const pitch = Math.max(-10, Math.min(10, behavior.head_pitch || 0));
      tiltPitchVal.textContent = `${pitch.toFixed(1)}°`;
      tiltPitchFill.style.marginLeft = `${((pitch + 10) / 20) * 60}%`;
    }

    // ── Telemetry pills ───────────────────────────────────────────────
    if (telemMode) telemMode.textContent = behavior.mode || "-";
    if (telemCognitive) telemCognitive.textContent = behavior.cognitive_mode || "-";
    if (telemSpeaking) telemSpeaking.textContent = behavior.is_speaking ? "yes" : "no";
    if (telemIntensity) telemIntensity.textContent = ((behavior.intensity ?? 0.5) * 100).toFixed(0) + "%";
    if (telemFace) telemFace.textContent = prevBehavior ? (prevBehavior.target_affect || "-") : "-";
    if (telemEmote) telemEmote.textContent = behavior.emote_id || behavior.target_affect || "-";
    if (telemMirror) telemMirror.textContent = ((behavior.mirror_strength ?? 0) * 100).toFixed(0) + "%";
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
