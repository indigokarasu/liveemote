// LiveEmote front-end.
//
// The avatar has its own face (the active character's canonical still or an SVG
// fallback) and its own voice. Webcam input is consumed ONLY as a perception
// signal — the captured frames are downsampled, JPEG-encoded, and POSTed to
// /api/perception/video, where the server-side MediaPipe tracker computes
// focus + energy. The avatar never displays or copies the user.

const els = {
  mode: q('#mode'),
  affect: q('#affect'),
  vad: q('#vad'),
  face: q('#face'),
  gaze: q('#gaze'),
  confidence: q('#confidence'),
  bodyPose: q('#bodyPose'),
  voiceStatus: q('#voiceStatus'),
  rendererStatus: q('#rendererStatus'),
  emote: q('#emote'),
  policy: q('#policy'),
  character: q('#character'),
  style: q('#style'),
  background: q('#background'),
  response: q('#response'),
  raw: q('#raw'),
  avatar: q('#avatarCanvas'),
  avatarPortrait: q('#avatarPortrait'),
  avatarFallback: q('#avatarFallback'),
  avatarEmote: q('#avatarEmote'),
  captureCanvas: q('#captureCanvas'),
  speech: q('#speech'),
  characterSelect: q('#characterSelect'),
  characterPathSelect: q('#characterPathSelect'),
  styleSelect: q('#styleSelect'),
  backgroundSelect: q('#backgroundSelect'),
  syncBackground: q('#syncBackground'),
  workflowSelect: q('#workflowSelect'),
  meetingStatus: q('#meetingStatus'),
  meetingLatency: q('#meetingLatency'),
  meetingDetail: q('#meetingDetail'),
  meetingUrl: q('#meetingUrl'),
  meetingName: q('#meetingName'),
  trackerKind: q('#trackerKind'),
  trackerAvailable: q('#trackerAvailable'),
  attentionMeter: q('#attentionMeter'),
  arousalMeter: q('#arousalMeter'),
  valenceMeter: q('#valenceMeter'),
  tensionMeter: q('#tensionMeter'),
  poseVar: q('#poseVar'),
  blinkRate: q('#blinkRate'),
};

let policy = 'reflect';
let updatingControls = false;
let audioContext, analyser, audioData, captureCtx;
let lastPerceptionAt = 0;

function q(s) { return document.querySelector(s); }

async function post(url, body = {}) {
  const r = await fetch(url, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body)});
  let payload = null;
  try { payload = await r.json(); } catch (_) { payload = null; }
  if (!r.ok) throw new Error((payload && payload.detail) || `Request failed: ${r.status}`);
  return update(payload);
}

function optionLabel(item) {
  return item.name || item.id || item.workflow;
}

function fillSelect(select, items, value, placeholder = null) {
  const next = [];
  if (placeholder) next.push(`<option value="">${placeholder}</option>`);
  for (const item of items) {
    const id = item.id || item.workflow;
    next.push(`<option value="${id}">${optionLabel(item)}</option>`);
  }
  const html = next.join('');
  if (select.innerHTML !== html) select.innerHTML = html;
  select.value = value || '';
}

function applyAvatarTheme(style, background) {
  els.avatar.dataset.style = style?.id || '';
  els.avatar.dataset.background = background?.id || '';
  if (background?.kind === 'color' || background?.kind === 'gradient') {
    els.avatar.style.background = background.value;
  } else if (background?.kind === 'image') {
    els.avatar.style.background = `center / cover no-repeat url(${background.value})`;
  } else {
    els.avatar.style.background = '';
  }
}

function updateControls(s) {
  updatingControls = true;
  fillSelect(els.characterSelect, s.characters || [], s.character_id);
  fillSelect(els.styleSelect, s.styles || [], s.active_style_id);
  fillSelect(els.backgroundSelect, s.backgrounds || [], s.active_background_id);
  fillSelect(els.workflowSelect, s.workflow_style_rules || [], '', 'Apply workflow…');
  els.syncBackground.checked = Boolean(s.sync_background_to_style);
  updatingControls = false;
}

// ----- Avatar visual layer ----------------------------------------------------

function renderAvatar(s) {
  const a = s.avatar || {};
  const visual = (s.capabilities && s.capabilities.renderer && s.capabilities.renderer.avatar_visual) || null;
  const kind = visual?.portrait_kind || 'svg_fallback';
  const canonicalUrl = visual?.canonical_url || null;
  const emoteUrl = visual?.active_emote_url || null;

  // Apply behavior to the avatar container for CSS animations.
  els.avatar.className = `mode-${a.mode || 'idle'} intensity-${bandIntensity(a.intensity)} emote-${a.emote_id || 'none'}`;
  els.avatar.dataset.gaze = a.gaze_target || 'soft_forward';
  els.avatar.dataset.affect = a.affect || 'neutral';

  if (kind === 'canonical' && canonicalUrl) {
    // The avatar has its own face. Inline the character canonical still + emote overlay.
    ensurePortraitImage(canonicalUrl);
    els.avatarFallback.hidden = true;
  } else {
    ensureFallbackFace(a);
    els.avatarFallback.hidden = false;
  }

  if (emoteUrl) {
    els.avatarEmote.hidden = false;
    if (els.avatarEmote.src !== emoteUrl) els.avatarEmote.src = emoteUrl;
  } else {
    els.avatarEmote.hidden = true;
  }
}

let portraitImg;
function ensurePortraitImage(src) {
  if (!portraitImg) {
    portraitImg = document.createElement('img');
    portraitImg.alt = 'Avatar portrait';
    portraitImg.className = 'avatar-portrait';
    els.avatarPortrait.appendChild(portraitImg);
  }
  if (portraitImg.src !== src) {
    portraitImg.src = src;
    portraitImg.hidden = false;
  }
  portraitImg.hidden = false;
}

function ensureFallbackFace(behavior) {
  // Tear down any prior SVG so we always re-render against the latest behavior.
  const existing = els.avatarPortrait.querySelector('svg.avatar-face');
  if (existing) existing.remove();
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 240 240');
  svg.classList.add('avatar-face');
  // Head ellipse.
  const head = document.createElementNS(svg.namespaceURI, 'ellipse');
  head.setAttribute('cx', 120); head.setAttribute('cy', 120); head.setAttribute('rx', 90); head.setAttribute('ry', 100);
  head.setAttribute('class', 'avatar-head');
  svg.appendChild(head);

  const intensity = clamp01(behavior.intensity ?? 0.3);
  const isSpeaking = behavior.mode === 'speaking';
  const isListening = behavior.mode === 'listening';
  const isThinking = behavior.mode === 'thinking';

  // Mouth shape.
  const mouth = document.createElementNS(svg.namespaceURI, 'path');
  const mouthY = 168;
  let d;
  if (isSpeaking) {
    d = `M 96 ${mouthY} Q 120 ${mouthY + 8 + intensity * 10} 144 ${mouthY}`;
  } else if (behavior.affect && behavior.affect.includes('warm')) {
    d = `M 96 ${mouthY} Q 120 ${mouthY + 12} 144 ${mouthY}`;
  } else if (behavior.affect && behavior.affect.includes('concern')) {
    d = `M 96 ${mouthY + 6} Q 120 ${mouthY - 4} 144 ${mouthY + 6}`;
  } else if (behavior.affect && behavior.affect.includes('sad')) {
    d = `M 96 ${mouthY + 4} Q 120 ${mouthY - 4} 144 ${mouthY + 4}`;
  } else {
    d = `M 96 ${mouthY} Q 120 ${mouthY + 4} 144 ${mouthY}`;
  }
  mouth.setAttribute('d', d);
  mouth.setAttribute('class', `avatar-mouth ${isSpeaking ? 'speaking' : ''}`);
  svg.appendChild(mouth);

  // Eyes — blink if not attentively listening; otherwise open.
  const eyeOpenness = isThinking ? 0.4 : (isListening ? 0.95 : 0.75 + intensity * 0.2);
  const eyeOffsetY = behavior.gaze_target === 'away' ? -3 : 0;
  for (const cx of [86, 154]) {
    const eye = document.createElementNS(svg.namespaceURI, 'ellipse');
    eye.setAttribute('cx', cx);
    eye.setAttribute('cy', 116 + eyeOffsetY);
    eye.setAttribute('rx', 8);
    eye.setAttribute('ry', Math.max(1.2, 7.5 * eyeOpenness));
    eye.setAttribute('class', 'avatar-eye');
    svg.appendChild(eye);
  }
  // Brow.
  for (const [cx, dy] of [[86, -22], [154, -22]]) {
    const brow = document.createElementNS(svg.namespaceURI, 'path');
    const curve = behavior.affect && behavior.affect.includes('concern') ? -4 : -8;
    brow.setAttribute('d', `M ${cx - 14} ${108 + dy + curve} Q ${cx} ${108 + dy - 2} ${cx + 14} ${108 + dy + curve}`);
    brow.setAttribute('class', 'avatar-brow');
    svg.appendChild(brow);
  }
  els.avatarPortrait.appendChild(svg);
}

function bandIntensity(v) {
  const i = Math.max(0, Math.min(1, Number(v) || 0));
  if (i < 0.25) return 'low';
  if (i < 0.6) return 'mid';
  return 'high';
}
function clamp01(v) { return Math.max(0, Math.min(1, Number(v) || 0)); }
function fmt(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : '-';
}

// ----- Older status update ---------------------------------------------------

function update(s) {
  const a = s.avatar || {};
  const u = s.user || {};
  const m = s.meeting || {};
  const c = s.capabilities || {};
  const style = s.active_style || null;
  const background = s.active_background || null;

  els.mode.textContent = a.mode || '-';
  els.affect.textContent = a.affect || '-';
  els.vad.textContent = u.speaking ? 'speaking' : 'silent';
  els.face.textContent = String(Boolean(u.face_detected));
  els.gaze.textContent = a.gaze_target || '-';
  els.confidence.textContent = `emotion ${fmt(u.emotion_confidence)} / gaze ${fmt(u.gaze_confidence)}`;
  els.bodyPose.textContent = a.full_body_pose || '-';
  els.voiceStatus.textContent = `${c.voice?.last_engine || c.voice?.backend || '-'} (${c.voice?.last_latency_ms ?? 0} ms)`;
  els.rendererStatus.textContent = c.renderer?.online ? `${c.renderer.backend} (${c.renderer.last_latency_ms ?? 0} ms)` : 'offline';
  els.emote.textContent = a.emote_id || '-';
  els.policy.textContent = s.mode_policy || '-';
  els.character.textContent = s.character_name || s.character_id || '-';
  els.style.textContent = style ? `${style.name} (${style.id})` : '-';
  els.background.textContent = background ? `${background.name} (${background.id})` : '-';
  els.response.textContent = s.agent_response_text || s.hermes_response_text || '';
  els.meetingStatus.textContent = m.status || 'idle';
  els.meetingLatency.textContent = m.estimated_join_latency_ms == null ? '-' : `${m.estimated_join_latency_ms} ms`;
  els.meetingDetail.textContent = m.detail || '';
  els.raw.textContent = JSON.stringify(s, null, 2);
  els.avatar.className = `${els.avatar.className} mode-${a.mode || 'idle'}`;
  updateCharacterPaths(s.characters || [], s.character_id);
  applyAvatarTheme(style, background);
  renderAvatar(s);

  // Live affect meters (focus + energy driven by webcam perception).
  els.trackerKind.textContent = c.perception?.backend || '-';
  els.trackerAvailable.textContent = c.perception?.available ? 'yes' : 'no';
  els.attentionMeter.textContent = fmt(u.attention);
  els.arousalMeter.textContent = fmt(u.arousal);
  els.valenceMeter.textContent = fmt(u.valence);
  els.tensionMeter.textContent = fmt(u.tension);
  els.poseVar.textContent = fmt(u.head_pose_variance ?? (s.user && s.user.head_yaw !== undefined ? Math.abs(u.head_yaw).toFixed(2) : '-'));
  els.blinkRate.textContent = fmt(u.blink_rate ?? 0);

  updateControls(s);

  if (s.speech?.audio_path) els.speech.src = `/api/audio?path=${encodeURIComponent(s.speech.audio_path)}`;
  return s;
}

function updateCharacterPaths(characters, activeId) {
  const selected = els.characterPathSelect.value;
  els.characterPathSelect.innerHTML = '';
  characters.forEach(ch => {
    const o = document.createElement('option');
    o.value = ch.path;
    o.textContent = `${ch.name || ch.id}${ch.id === activeId ? ' (active)' : ''} — ${ch.emote_count} emotes`;
    els.characterPathSelect.appendChild(o);
  });
  if ([...els.characterPathSelect.options].some(o => o.value === selected)) els.characterPathSelect.value = selected;
}

async function poll() {
  const r = await fetch('/api/status');
  update(await r.json());
}

// ----- Wiring --------------------------------------------------------------

q('#speak').onclick = () => post('/api/speak', {text: 'Demo user turn complete.'});
q('#toggle').onclick = () => {
  policy = policy === 'reflect' ? 'mirror' : 'reflect';
  post('/api/mode', {mode: policy});
};
els.characterSelect.onchange = () => {
  if (!updatingControls) post('/api/character', {character_id: els.characterSelect.value});
};
els.styleSelect.onchange = () => {
  if (!updatingControls) post('/api/style', {style_id: els.styleSelect.value, sync_background: els.syncBackground.checked});
};
els.backgroundSelect.onchange = () => {
  if (!updatingControls) post('/api/background', {background_id: els.backgroundSelect.value, sync_background: false});
};
els.syncBackground.onchange = () => {
  if (!updatingControls && els.syncBackground.checked) {
    post('/api/style', {style_id: els.styleSelect.value, sync_background: true});
  }
};
els.workflowSelect.onchange = () => {
  if (!updatingControls && els.workflowSelect.value) post('/api/workflow', {workflow: els.workflowSelect.value});
};
q('#joinMeeting').onclick = async () => {
  try {
    await post('/api/meeting/join', {meeting_url: els.meetingUrl.value, display_name: els.meetingName.value});
  } catch (e) {
    els.meetingStatus.textContent = 'error';
    els.meetingDetail.textContent = e.message;
  }
};
q('#leaveMeeting').onclick = () => post('/api/meeting/leave');
q('#selectCharacter').onclick = () => post('/api/character/select', {character_path: els.characterPathSelect.value});
document.querySelectorAll('[data-trigger]').forEach(b => b.onclick = () => post(`/api/trigger/${b.dataset.trigger}`));

// Audio VAD — keeps the local capture path; perception is streamed too.
function audioVad() {
  if (!analyser) return {speaking: false, energy: 0, speech_rate: 0};
  analyser.getByteTimeDomainData(audioData);
  let sum = 0, crossings = 0;
  for (let i = 0; i < audioData.length; i++) {
    const v = (audioData[i] - 128) / 128;
    sum += v * v;
    if (i && (audioData[i - 1] < 128) !== (audioData[i] < 128)) crossings++;
  }
  const energy = Math.min(1, Math.sqrt(sum / audioData.length) * 5);
  return {speaking: energy > 0.08, energy, speech_rate: Math.min(1, crossings / audioData.length * 8)};
}

// Draw the current video frame into the hidden capture canvas at low resolution
// and POST a base64 JPEG to /api/perception/video. The server-side tracker
// turns that into focus + energy signals.
async function streamPerceptionFrame(video) {
  if (!video || !video.videoWidth) return;
  const w = els.captureCanvas.width;
  const h = Math.round((video.videoHeight / video.videoWidth) * w) || els.captureCanvas.height;
  els.captureCanvas.height = h;
  captureCtx.drawImage(video, 0, 0, w, h);
  let jpeg;
  try {
    jpeg = els.captureCanvas.toDataURL('image/jpeg', 0.6);
  } catch (_) {
    return;
  }
  const now = Date.now();
  if (now - lastPerceptionAt < 280) return;
  lastPerceptionAt = now;
  try {
    await fetch('/api/perception/video', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({image: jpeg, timestamp_ms: now}),
    });
  } catch (err) {
    console.warn('perception stream failed', err);
  }
}

// Backwards-compatible legacy telemetry — used when the server-side tracker
// is unavailable. Sends a coarse perception event with bounding box & VAD.
async function frameTelemetry(video) {
  const vad = audioVad();
  const now = Date.now();
  await post('/api/event', {event: {type: 'audio.vad', timestamp_ms: now, ...vad}});
}

async function webcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
    const video = q('#webcam');
    video.srcObject = stream;
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    audioData = new Uint8Array(analyser.fftSize);
    audioContext.createMediaStreamSource(stream).connect(analyser);
    captureCtx = els.captureCanvas.getContext('2d');

    // Server-side perception (focus + energy).
    setInterval(() => streamPerceptionFrame(video).catch(console.warn), 320);
    // Legacy client-side audio VAD for endpoints without server perception.
    setInterval(() => frameTelemetry(video).catch(console.warn), 1500);
  } catch (e) {
    console.warn('webcam init failed', e);
    els.trackerAvailable.textContent = 'no (no camera permission)';
  }
}

webcam();
poll();
setInterval(poll, 1500);
