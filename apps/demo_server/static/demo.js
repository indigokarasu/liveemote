const els = {
  mode: q('#mode'), affect: q('#affect'), vad: q('#vad'), face: q('#face'), gaze: q('#gaze'),
  confidence: q('#confidence'), bodyPose: q('#bodyPose'), voiceStatus: q('#voiceStatus'), rendererStatus: q('#rendererStatus'),
  emote: q('#emote'), policy: q('#policy'), response: q('#response'), raw: q('#raw'), avatar: q('#avatarCanvas'),
  meetingStatus: q('#meetingStatus'), meetingLatency: q('#meetingLatency'), meetingDetail: q('#meetingDetail'),
  meetingUrl: q('#meetingUrl'), meetingName: q('#meetingName'), characterSelect: q('#characterSelect'), speech: q('#speech'),
};
let policy = 'reflect';
let audioContext, analyser, audioData, faceDetector;

function q(s) { return document.querySelector(s); }

async function post(url, body = {}) {
  const r = await fetch(url, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body)});
  const payload = await r.json();
  if (!r.ok) throw new Error(payload.detail || `Request failed: ${r.status}`);
  return update(payload);
}

function update(s) {
  const a = s.avatar || {}, u = s.user || {}, m = s.meeting || {}, c = s.capabilities || {};
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
  els.response.textContent = s.hermes_response_text || '';
  els.meetingStatus.textContent = m.status || 'idle';
  els.meetingLatency.textContent = m.estimated_join_latency_ms == null ? '-' : `${m.estimated_join_latency_ms} ms`;
  els.meetingDetail.textContent = m.detail || '';
  els.raw.textContent = JSON.stringify(s, null, 2);
  els.avatar.className = a.mode || '';
  updateCharacters(s.characters || [], s.character_id);
  if (s.speech?.audio_path) els.speech.src = `/api/audio?path=${encodeURIComponent(s.speech.audio_path)}`;
  return s;
}

function fmt(v) { return Number(v || 0).toFixed(2); }

function updateCharacters(characters, activeId) {
  const selected = els.characterSelect.value;
  els.characterSelect.innerHTML = '';
  characters.forEach(ch => {
    const o = document.createElement('option');
    o.value = ch.path; o.textContent = `${ch.character_id}${ch.character_id === activeId ? ' (active)' : ''} — ${ch.emote_count} emotes`;
    els.characterSelect.appendChild(o);
  });
  if ([...els.characterSelect.options].some(o => o.value === selected)) els.characterSelect.value = selected;
}

async function poll() {
  const r = await fetch('/api/status');
  update(await r.json());
}

q('#speak').onclick = () => post('/api/speak', {text: 'Demo user turn complete.'});
q('#toggle').onclick = () => { policy = policy === 'reflect' ? 'mirror' : 'reflect'; post('/api/mode', {mode: policy}); };
q('#joinMeeting').onclick = async () => { try { await post('/api/meeting/join', {meeting_url: els.meetingUrl.value, display_name: els.meetingName.value}); } catch (e) { els.meetingStatus.textContent = 'error'; els.meetingDetail.textContent = e.message; } };
q('#leaveMeeting').onclick = () => post('/api/meeting/leave');
q('#selectCharacter').onclick = () => post('/api/character/select', {character_path: els.characterSelect.value});
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
