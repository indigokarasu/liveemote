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
  const avatarNote = document.getElementById("avatar-note");
  const avatarFallback = document.getElementById("avatar-fallback");

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
    }).then((resp) => {
      if (!resp.ok && webcamNote) {
        webcamNote.textContent = `Perception endpoint unavailable (${resp.status}). Avatar is in recovery mode.`;
        webcamNote.style.color = "#f59e0b";
      }
    }).catch(() => {
      if (webcamNote) {
        webcamNote.textContent = "Perception connection lost. Avatar is in recovery mode.";
        webcamNote.style.color = "#f59e0b";
      }
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
      // /api/status deliberately namespaces live behavior under `avatar`.
      // Keep the fallback for older deployments, but never read the wrapper
      // object as if it were AvatarBehaviorState.
      const behavior = data.avatar || data.behavior || data;
      applyAvatarVisual(data.capabilities?.renderer?.avatar_visual);
      applyBehavior(behavior);
      applyVoiceLoopStatus(data.voice_loop);
      applyMeetingStatus(data.voice_loop?.diarization);
      // On refresh, re-render the last diarized turns the server remembers.
      if (data.voice_loop?.last_diarized_turns?.length) {
        renderMeetingTurns(data.voice_loop.last_diarized_turns);
      }
      prevBehavior = behavior;
    } catch {
      if (avatarNote) {
        avatarNote.textContent = "Avatar telemetry unavailable; showing the safe synthetic fallback.";
        avatarNote.style.color = "#f59e0b";
      }
    }
  }

  /** Apply the renderer's real character asset selection to the browser avatar. */
  function applyAvatarVisual(visual) {
    if (!visual || !avatarEmote) return;
    const imageUrl = visual.active_emote_url || visual.canonical_url;
    let image = avatarEmote.querySelector(".avatar-image");
    if (imageUrl) {
      if (!image) {
        image = document.createElement("img");
        image.className = "avatar-image";
        image.alt = "Synthetic avatar character";
        image.loading = "eager";
        avatarEmote.querySelector(".avatar-face")?.prepend(image);
      }
      if (image.src !== new URL(imageUrl, window.location.href).href) image.src = imageUrl;
      image.hidden = false;
      if (avatarFallback) avatarFallback.hidden = true;
      if (avatarNote) {
        avatarNote.textContent = `Live character asset: ${visual.active_emote?.id || "canonical"}. Webcam frames are not rendered here.`;
        avatarNote.style.color = "";
      }
    } else if (avatarFallback) {
      avatarFallback.hidden = false;
      if (avatarNote) {
        avatarNote.textContent = "Character asset unavailable; synthetic SVG fallback is active.";
        avatarNote.style.color = "#f59e0b";
      }
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
  //  VOICE LOOP (browser mic → server → speech-to-speech sidecar)
  // =====================================================================
  // Speaks the OpenAI Realtime GA protocol to the server's /ws/voice relay:
  //   - sends  input_audio_buffer.append  (PCM16 16 kHz mono base64, ~40 ms)
  //   - receives response.audio.delta (played back) + transcript events
  // The avatar animates server-side while this conversation happens.

  const VOICE_WS_URL = "/ws/voice";
  const VOICE_TARGET_RATE = 16000;
  const VOICE_CHUNK_SAMPLES = 640; // 40 ms at 16 kHz

  const voiceToggle = document.getElementById("voice-toggle");
  const voiceStatusChip = document.getElementById("voice-status");
  const voiceTranscriptEl = document.getElementById("voice-transcript");
  const voiceMeterFill = document.getElementById("voice-meter-fill");
  const voiceNoteEl = document.getElementById("voice-note");

  let voice = {
    active: false,
    ws: null,
    audioCtx: null,
    micStream: null,
    scriptNode: null,
    analyser: null,
    sendBuffer: new Float32Array(0),
    pcmQueue: [],
    playing: false,
    speaking: false,
    assistantBuf: "",
  };

  function setVoiceStatus(label, tone) {
    if (!voiceStatusChip) return;
    voiceStatusChip.textContent = label;
    voiceStatusChip.className = "voice-status-chip" + (tone ? " " + tone : "");
  }

  function setVoiceTranscript(who, text) {
    if (!voiceTranscriptEl) return;
    if (!text) { voiceTranscriptEl.textContent = "—"; return; }
    const span = document.createElement("span");
    span.className = "who";
    span.textContent = who === "you" ? "you" : "avatar";
    voiceTranscriptEl.innerHTML = "";
    voiceTranscriptEl.appendChild(span);
    voiceTranscriptEl.appendChild(document.createTextNode(text));
  }

  // ── PCM helpers ────────────────────────────────────────────────────
  function float32ToInt16(f32) {
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return i16;
  }

  function int16ToBase64(i16) {
    const bytes = new Uint8Array(i16.buffer);
    let bin = "";
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(bin);
  }

  function base64ToFloat32(b64) {
    const bin = atob(b64);
    const i16 = new Int16Array(bin.length / 2);
    for (let i = 0; i < i16.length; i++) {
      i16[i] = (bin.charCodeAt(i * 2) & 0xff) | (bin.charCodeAt(i * 2 + 1) << 8);
    }
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 0x8000;
    return f32;
  }

  // ── Mic capture → 16 kHz PCM16 chunks ──────────────────────────────
  function onMicAudio(e) {
    if (!voice.ws || voice.ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const ratio = voice.audioCtx.sampleRate / VOICE_TARGET_RATE;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      let acc = 0, n = 0;
      const start = Math.floor(i * ratio);
      const end = Math.min(input.length, Math.floor((i + 1) * ratio));
      for (let j = start; j < end; j++) { acc += input[j]; n++; }
      out[i] = n ? acc / n : 0;
    }
    // Mic level meter
    let rms = 0;
    for (let i = 0; i < out.length; i++) rms += out[i] * out[i];
    rms = Math.sqrt(rms / Math.max(1, out.length));
    if (voiceMeterFill) voiceMeterFill.style.width = Math.min(100, rms * 220) + "%";

    // Accumulate into fixed 640-sample chunks and send
    const combined = new Float32Array(voice.sendBuffer.length + out.length);
    combined.set(voice.sendBuffer, 0);
    combined.set(out, voice.sendBuffer.length);
    voice.sendBuffer = combined;
    while (voice.sendBuffer.length >= VOICE_CHUNK_SAMPLES) {
      const chunk = voice.sendBuffer.slice(0, VOICE_CHUNK_SAMPLES);
      voice.sendBuffer = voice.sendBuffer.slice(VOICE_CHUNK_SAMPLES);
      if (voice.ws && voice.ws.readyState === WebSocket.OPEN) {
        voice.ws.send(JSON.stringify({
          type: "input_audio_buffer.append",
          audio: int16ToBase64(float32ToInt16(chunk)),
        }));
      }
    }
  }

  // ── Playback queue (response.audio.delta → speakers) ───────────────
  function schedulePlayback() {
    if (voice.playing || voice.pcmQueue.length === 0 || !voice.audioCtx) return;
    voice.playing = true;
    const chunk = voice.pcmQueue.shift();
    const buffer = voice.audioCtx.createBuffer(1, chunk.length, VOICE_TARGET_RATE);
    buffer.copyToChannel(chunk, 0);
    const source = voice.audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(voice.audioCtx.destination);
    source.onended = () => { voice.playing = false; schedulePlayback(); };
    source.start();
  }

  // ── WebSocket session ──────────────────────────────────────────────
  function sendSessionConfig() {
    if (!voice.ws || voice.ws.readyState !== WebSocket.OPEN) return;
    voice.ws.send(JSON.stringify({
      type: "session.update",
      session: {
        output_modalities: ["audio"],
        turn_detection: { type: "server_vad", interrupt_response: true },
        instructions: "You are the voice of a warm, observant AI avatar. Keep replies short, under 40 words. Respond naturally to what the user says.",
      },
    }));
    // Startup greeting so the voice is immediately audible.
    voice.ws.send(JSON.stringify({
      type: "conversation.item.create",
      item: { type: "message", role: "user", content: [{ type: "input_text", text: "[Start the conversation with a one-line friendly greeting.]" }] },
    }));
    voice.ws.send(JSON.stringify({ type: "response.create" }));
  }

  function handleVoiceEvent(raw) {
    let ev;
    try { ev = JSON.parse(raw); } catch { return; }
    const type = ev.type || "";
    if (type === "conversation.item.input_audio_transcription.completed") {
      const text = (ev.transcript || ev.text || "").trim();
      if (text) setVoiceTranscript("you", text);
    } else if (type === "response.audio.delta") {
      if (!voice.speaking) {
        voice.speaking = true;
        setVoiceStatus("avatar speaking…", "busy");
      }
      if (ev.audio) {
        voice.pcmQueue.push(base64ToFloat32(ev.audio));
        schedulePlayback();
      }
    } else if (type === "response.output_text.delta") {
      const delta = ev.delta || ev.text || "";
      if (delta) {
        voice.assistantBuf += delta;
        setVoiceTranscript("avatar", voice.assistantBuf);
      }
    } else if (type === "response.output_audio.done" || type === "response.done") {
      if (voice.speaking) {
        voice.speaking = false;
        setVoiceStatus("connected — talk to the avatar", "ok");
      }
    } else if (type === "input_audio_buffer.speech_started") {
      setVoiceStatus("hearing you…", "busy");
    } else if (type === "input_audio_buffer.speech_stopped") {
      setVoiceStatus("thinking…", "busy");
    } else if (type === "error") {
      const msg = ev.error && ev.error.message ? ev.error.message : "voice loop error";
      setVoiceStatus("error", "err");
      if (voiceNoteEl) voiceNoteEl.textContent = msg;
    }
  }

  function openVoiceSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    let ws;
    try {
      ws = new WebSocket(proto + "//" + location.host + VOICE_WS_URL);
    } catch (err) {
      setVoiceStatus("connect failed", "err");
      return;
    }
    voice.ws = ws;
    setVoiceStatus("connecting…", "busy");
    ws.onopen = () => { setVoiceStatus("connected", "ok"); sendSessionConfig(); };
    ws.onmessage = (e) => handleVoiceEvent(e.data);
    ws.onerror = () => setVoiceStatus("socket error", "err");
    ws.onclose = () => { stopVoice(); };
  }

  async function startVoice() {
    try {
      voice.micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      setVoiceStatus("mic blocked", "err");
      if (voiceNoteEl) voiceNoteEl.textContent = "Microphone unavailable: " + err.message;
      stopVoice();
      return;
    }
    try {
      voice.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = voice.audioCtx.createMediaStreamSource(voice.micStream);
      voice.scriptNode = voice.audioCtx.createScriptProcessor(1024, 1, 1);
      voice.analyser = voice.audioCtx.createAnalyser();
      voice.analyser.fftSize = 512;
      const silent = voice.audioCtx.createGain();
      silent.gain.value = 0;
      src.connect(voice.scriptNode);
      voice.scriptNode.connect(silent);
      silent.connect(voice.audioCtx.destination);
      voice.scriptNode.connect(voice.analyser);
      voice.scriptNode.onaudioprocess = onMicAudio;
    } catch (err) {
      setVoiceStatus("audio setup failed", "err");
      if (voiceNoteEl) voiceNoteEl.textContent = "Audio setup failed: " + err.message;
      stopVoice();
      return;
    }
    openVoiceSocket();
  }

  function stopVoice() {
    voice.active = false;
    voice.pcmQueue = [];
    voice.playing = false;
    voice.speaking = false;
    voice.assistantBuf = "";
    if (voice.ws) { try { voice.ws.close(); } catch { /* noop */ } voice.ws = null; }
    if (voice.scriptNode) { try { voice.scriptNode.disconnect(); } catch { /* noop */ } voice.scriptNode = null; }
    if (voice.analyser) { try { voice.analyser.disconnect(); } catch { /* noop */ } voice.analyser = null; }
    if (voice.micStream) { voice.micStream.getTracks().forEach((t) => t.stop()); voice.micStream = null; }
    if (voice.audioCtx) { try { voice.audioCtx.close(); } catch { /* noop */ } voice.audioCtx = null; }
    if (voiceMeterFill) voiceMeterFill.style.width = "0%";
    if (voiceToggle) { voiceToggle.textContent = "Enable voice"; voiceToggle.classList.remove("active"); }
    setVoiceStatus("off", "off");
  }

  async function toggleVoice() {
    if (voice.active) { stopVoice(); return; }
    voice.active = true;
    if (voiceToggle) { voiceToggle.textContent = "Disable voice"; voiceToggle.classList.add("active"); }
    setVoiceStatus("requesting mic…", "busy");
    await startVoice();
  }

  /** Update the voice chip/toggle from the server's voice_loop status. */
  function applyVoiceLoopStatus(vl) {
    if (!voiceStatusChip || !vl) return;
    if (voice.active) return; // a live session owns the chip
    if (!vl.enabled) {
      setVoiceStatus("server: voice off (--voice-loop)", "off");
      if (voiceToggle) voiceToggle.disabled = true;
    } else if (vl.degraded || !vl.reachable) {
      setVoiceStatus("sidecar offline", "err");
      if (voiceToggle) voiceToggle.disabled = true;
      if (voiceNoteEl) {
        voiceNoteEl.textContent = "Voice sidecar unreachable — run sidecar/voice_loop/app.py (pip install speech-to-speech).";
      }
    } else {
      setVoiceStatus("ready — enable voice", "ok");
      if (voiceToggle) voiceToggle.disabled = false;
    }
  }

  function bindVoiceControls() {
    if (voiceToggle) voiceToggle.addEventListener("click", toggleVoice);
  }

  // =====================================================================
  //  MEETING TRANSCRIPT (MOSS diarization sidecar → /api/transcribe)
  // =====================================================================
  // Records a short clip or uploads an audio file to the MOSS
  // Transcribe-Diarize sidecar, which returns per-speaker turns with
  // timestamps. Rendered as a transcript list with speaker color chips.

  const TRANSCRIBE_ENDPOINT = "/api/transcribe";
  const MAX_RECORD_MS = 5 * 60 * 1000; // safety cap on clip length

  const meetingRecord = document.getElementById("meeting-record");
  const meetingUpload = document.getElementById("meeting-upload");
  const meetingFile = document.getElementById("meeting-file");
  const meetingStatusChip = document.getElementById("meeting-status");
  const meetingListEl = document.getElementById("meeting-list");
  const meetingTimerEl = document.getElementById("meeting-timer");
  const meetingNoteEl = document.getElementById("meeting-note");

  let meeting = {
    recording: false,
    recorder: null,
    chunks: [],
    micStream: null,
    startedAt: 0,
    timerHandle: null,
  };

  function setMeetingStatus(label, tone) {
    if (!meetingStatusChip) return;
    meetingStatusChip.textContent = label;
    meetingStatusChip.className = "voice-status-chip" + (tone ? " " + tone : "");
  }

  /** Reflect the MOSS sidecar availability (from /api/status). */
  function applyMeetingStatus(dz) {
    if (!meetingStatusChip || !dz) return;
    if (meeting.recording) return; // a live recording owns the chip
    const reachable = dz.reachable === true;
    const degraded = dz.degraded === true || !reachable;
    const canTranscribe = reachable && !degraded;
    if (!reachable) {
      setMeetingStatus("sidecar offline", "err");
      if (meetingNoteEl) {
        meetingNoteEl.textContent = "MOSS sidecar unreachable — run: python sidecar/moss_daemon.py (Python 3.12 venv).";
      }
    } else if (degraded) {
      setMeetingStatus("degraded", "err");
      if (meetingNoteEl && dz.reason) meetingNoteEl.textContent = "MOSS sidecar degraded: " + dz.reason;
    } else {
      setMeetingStatus("ready — diarize", "ok");
      if (meetingNoteEl) {
        meetingNoteEl.textContent = "Records a clip or uploads an audio file; the MOSS sidecar returns per-speaker turns with timestamps in one pass. Audio is normalized to 16 kHz mono by the sidecar (ffmpeg).";
      }
    }
    if (meetingRecord) meetingRecord.disabled = !canTranscribe;
    if (meetingUpload) meetingUpload.disabled = !canTranscribe;
  }

  // ── Rendering ────────────────────────────────────────────────────────
  function fmtClock(sec) {
    if (sec == null || isNaN(sec)) return "–:––";
    const s = Math.max(0, Math.floor(sec));
    const m = Math.floor(s / 60);
    return `${m}:${String(s % 60).padStart(2, "0")}`;
  }

  function speakerClass(speaker) {
    const m = /(\d+)/.exec(speaker || "");
    const n = m ? parseInt(m[1], 10) : 0;
    if (n >= 1 && n <= 3) return "s" + n;
    return "sx";
  }

  /** Render diarized turns [{start,end,speaker,text}] into the list. */
  function renderMeetingTurns(turns) {
    if (!meetingListEl) return;
    const segs = Array.isArray(turns) ? turns.filter((t) => t && t.text) : [];
    meetingListEl.innerHTML = "";
    if (!segs.length) {
      const li = document.createElement("li");
      li.className = "meeting-empty";
      li.textContent = "No diarized turns returned.";
      meetingListEl.appendChild(li);
      return;
    }
    for (const seg of segs) {
      const li = document.createElement("li");
      li.className = "meeting-turn";

      const ts = document.createElement("span");
      ts.className = "m-ts";
      ts.textContent = fmtClock(seg.start);
      li.appendChild(ts);

      const sp = document.createElement("span");
      sp.className = "m-speaker " + speakerClass(seg.speaker);
      sp.textContent = (seg.speaker || "S?").replace("S0", "S");
      li.appendChild(sp);

      const txt = document.createElement("span");
      txt.className = "m-text";
      txt.textContent = seg.text;
      li.appendChild(txt);

      meetingListEl.appendChild(li);
    }
  }

  // ── Upload + transcribe ─────────────────────────────────────────────
  async function transcribeAudio(blob, filename) {
    setMeetingStatus("transcribing…", "busy");
    const form = new FormData();
    form.append("audio", blob, filename || "recording.webm");
    try {
      const resp = await fetch(TRANSCRIBE_ENDPOINT, { method: "POST", body: form });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.available === false) {
        setMeetingStatus("transcribe failed", "err");
        if (meetingNoteEl) {
          meetingNoteEl.textContent = (data && data.reason) || ("Server responded " + resp.status + ".");
        }
        return;
      }
      renderMeetingTurns(data.segments || []);
      setMeetingStatus("done — " + (data.segments ? data.segments.length : 0) + " turns", "ok");
      if (meetingNoteEl && data.elapsed_sec != null) {
        meetingNoteEl.textContent = `Diarized in ${data.elapsed_sec.toFixed(1)}s (${data.model || "MOSS"}).`;
      }
    } catch (err) {
      setMeetingStatus("request failed", "err");
      if (meetingNoteEl) meetingNoteEl.textContent = "Transcribe request failed: " + err.message;
    }
  }

  // ── Recording (MediaRecorder → blob → upload) ───────────────────────
  function updateMeetingTimer() {
    if (!meetingTimerEl) return;
    const elapsed = Date.now() - meeting.startedAt;
    meetingTimerEl.textContent = fmtClock(elapsed / 1000);
    if (elapsed >= MAX_RECORD_MS) stopMeetingRecording();
  }

  async function startMeetingRecording() {
    try {
      meeting.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setMeetingStatus("mic blocked", "err");
      if (meetingNoteEl) meetingNoteEl.textContent = "Microphone unavailable: " + err.message;
      return;
    }
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((t) =>
      window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)
    ) || "";
    meeting.chunks = [];
    try {
      meeting.recorder = mime ? new MediaRecorder(meeting.micStream, { mimeType: mime }) : new MediaRecorder(meeting.micStream);
    } catch (err) {
      setMeetingStatus("recorder failed", "err");
      if (meetingNoteEl) meetingNoteEl.textContent = "MediaRecorder failed: " + err.message;
      meeting.micStream.getTracks().forEach((t) => t.stop());
      meeting.micStream = null;
      return;
    }
    meeting.recording = true;
    meeting.startedAt = Date.now();
    if (meetingRecord) {
      meetingRecord.textContent = "Stop";
      meetingRecord.classList.add("active");
    }
    if (meetingTimerEl) { meetingTimerEl.classList.add("on"); meetingTimerEl.textContent = "0:00"; }
    setMeetingStatus("recording…", "busy");
    meeting.recorder.ondataavailable = (e) => { if (e.data && e.data.size) meeting.chunks.push(e.data); };
    meeting.recorder.onstop = () => {
      const blob = new Blob(meeting.chunks, { type: meeting.recorder.mimeType || "audio/webm" });
      if (meetingTimerEl) meetingTimerEl.classList.remove("on");
      if (meetingRecord) {
        meetingRecord.textContent = "Record clip";
        meetingRecord.classList.remove("active");
      }
      meeting.recording = false;
      if (meeting.micStream) { meeting.micStream.getTracks().forEach((t) => t.stop()); meeting.micStream = null; }
      if (blob.size > 0) transcribeAudio(blob, "meeting-" + Date.now() + ".webm");
      else setMeetingStatus("empty recording", "err");
    };
    meeting.recorder.start();
    meeting.timerHandle = setInterval(updateMeetingTimer, 500);
  }

  function stopMeetingRecording() {
    if (meeting.timerHandle) { clearInterval(meeting.timerHandle); meeting.timerHandle = null; }
    if (meeting.recorder && meeting.recorder.state !== "inactive") {
      try { meeting.recorder.stop(); } catch { /* noop */ }
    }
  }

  async function toggleMeetingRecording() {
    if (meeting.recording) { stopMeetingRecording(); return; }
    await startMeetingRecording();
  }

  function bindMeetingControls() {
    if (meetingRecord) meetingRecord.addEventListener("click", toggleMeetingRecording);
    if (meetingUpload && meetingFile) {
      meetingUpload.addEventListener("click", () => meetingFile.click());
      meetingFile.addEventListener("change", () => {
        const file = meetingFile.files && meetingFile.files[0];
        if (!file) return;
        transcribeAudio(file, file.name);
        meetingFile.value = "";
      });
    }
  }

  // =====================================================================
  //  INIT
  // =====================================================================

  function init() {
    startPerceptionCapture();
    startStatusPoll();
    bindVoiceControls();
    bindMeetingControls();
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
      stopVoice();
      if (statusTimer) clearInterval(statusTimer);
    },
  };
})();
