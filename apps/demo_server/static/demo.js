const els = {
  mode: q('#mode'),
  affect: q('#affect'),
  vad: q('#vad'),
  face: q('#face'),
  gaze: q('#gaze'),
  emote: q('#emote'),
  policy: q('#policy'),
  character: q('#character'),
  style: q('#style'),
  background: q('#background'),
  response: q('#response'),
  raw: q('#raw'),
  avatar: q('#avatarCanvas'),
  characterSelect: q('#characterSelect'),
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

function q(s) { return document.querySelector(s); }

async function post(url, body = {}) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify(body),
  });
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

function update(s) {
  const a = s.avatar || {};
  const u = s.user || {};
  const m = s.meeting || {};
  const style = s.active_style || null;
  const background = s.active_background || null;
  els.mode.textContent = a.mode;
  els.affect.textContent = a.affect;
  els.vad.textContent = u.speaking ? 'speaking' : 'silent';
  els.face.textContent = String(u.face_detected);
  els.gaze.textContent = a.gaze_target;
  els.emote.textContent = a.emote_id || '-';
  els.policy.textContent = s.mode_policy;
  els.character.textContent = s.character_name || s.character_id || '-';
  els.style.textContent = style ? `${style.name} (${style.id})` : '-';
  els.background.textContent = background ? `${background.name} (${background.id})` : '-';
  els.response.textContent = s.hermes_response_text || '';
  els.meetingStatus.textContent = m.status || 'idle';
  els.meetingLatency.textContent = m.estimated_join_latency_ms == null ? '-' : `${m.estimated_join_latency_ms} ms`;
  els.meetingDetail.textContent = m.detail || '';
  els.raw.textContent = JSON.stringify(s, null, 2);
  els.avatar.className = a.mode || '';
  applyAvatarTheme(style, background);
  updateControls(s);
  return s;
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
document.querySelectorAll('[data-trigger]').forEach(b => b.onclick = () => post(`/api/trigger/${b.dataset.trigger}`));

async function webcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
    q('#webcam').srcObject = stream;
    setInterval(() => post('/api/event', {event: {type: 'perception.frame', timestamp_ms: Date.now(), face_detected: true, face_center: [0.5 + Math.sin(Date.now() / 1800) * 0.08, 0.45], head_yaw: Math.sin(Date.now() / 1200) * 6, head_pitch: 1.5, expression: {smile: 0.18 + Math.max(0, Math.sin(Date.now() / 2200)) * 0.35, frown: 0.05, brow_raise: 0.08, eye_open: 0.72}}}), 250);
    setInterval(() => post('/api/event', {event: {type: 'audio.vad', timestamp_ms: Date.now(), speaking: false, energy: 0.01, speech_rate: 0.0}}), 700);
  } catch (e) {
    console.warn(e);
  }
}

webcam();
poll();
setInterval(poll, 1500);
