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
};
let policy = 'reflect';
let updatingControls = false;
let audioContext, analyser, audioData, faceDetector;

function q(s) { return document.querySelector(s); }

async function post(url, body = {}) {
  const r = await fetch(url, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body)});
  const payload = await r.json();
  if (!r.ok) throw new Error(payload.detail || `Request failed: ${r.status}`);
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

function fmt(v) { return Number(v || 0).toFixed(2); }

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
  els.rendererStatus.textContent = c.renderer?.online ? `online (${c.renderer.last_latency_ms ?? 0} ms)` : 'offline / contract checked';
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
  els.avatar.className = a.mode || '';
  updateCharacterPaths(s.characters || [], s.character_id);
  applyAvatarTheme(style, background);
  updateControls(s);
  if (s.speech?.audio_path) els.speech.src = `/api/audio?path=${encodeURIComponent(s.speech.audio_path)}`;
  return s;
}
function fmt(v) { return Number(v || 0).toFixed(2); }

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

async function frameTelemetry(video) {
  let face = null;
  if (faceDetector) {
    try { face = (await faceDetector.detect(video))[0]; } catch (_) { face = null; }
  }
  const now = Date.now();
  const box = face?.boundingBox;
  const center = box ? [(box.x + box.width / 2) / video.videoWidth, (box.y + box.height / 2) / video.videoHeight] : [0.5, 0.45];
  const expression = {smile: 0.12, frown: 0.03, brow_raise: 0.06, eye_open: 0.72};
  await post('/api/event', {event: {type: 'perception.frame', timestamp_ms: now, face_detected: Boolean(box) || video.readyState >= 2, face_center: center, head_yaw: (0.5 - center[0]) * 24, head_pitch: (center[1] - 0.5) * 16, gaze_confidence: box ? 0.85 : 0.45, emotion_confidence: box ? 0.55 : 0.25, expression}});
  const vad = audioVad();
  await post('/api/event', {event: {type: 'audio.vad', timestamp_ms: now, ...vad}});
}

async function webcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
    const video = q('#webcam');
    video.srcObject = stream;
    if ('FaceDetector' in window) faceDetector = new FaceDetector({fastMode: true, maxDetectedFaces: 1});
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    audioData = new Uint8Array(analyser.fftSize);
    audioContext.createMediaStreamSource(stream).connect(analyser);
    setInterval(() => frameTelemetry(video).catch(console.warn), 350);
  } catch (e) { console.warn(e); }
}

webcam();
poll();
setInterval(poll, 1500);
