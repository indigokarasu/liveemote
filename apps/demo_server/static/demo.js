const els = {mode: q('#mode'), affect: q('#affect'), vad: q('#vad'), face: q('#face'), gaze: q('#gaze'), emote: q('#emote'), policy: q('#policy'), response: q('#response'), raw: q('#raw'), avatar: q('#avatarCanvas')};
let policy = 'reflect';
function q(s){return document.querySelector(s)}
async function post(url, body={}){const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}); return update(await r.json())}
function update(s){const a=s.avatar||{}, u=s.user||{}; els.mode.textContent=a.mode; els.affect.textContent=a.affect; els.vad.textContent=u.speaking?'speaking':'silent'; els.face.textContent=String(u.face_detected); els.gaze.textContent=a.gaze_target; els.emote.textContent=a.emote_id||'-'; els.policy.textContent=s.mode_policy; els.response.textContent=s.hermes_response_text||''; els.raw.textContent=JSON.stringify(s,null,2); els.avatar.className=a.mode||''; return s}
async function poll(){const r=await fetch('/api/status'); update(await r.json())}
q('#speak').onclick=()=>post('/api/speak',{text:'Demo user turn complete.'});
q('#toggle').onclick=()=>{policy=policy==='reflect'?'mirror':'reflect'; post('/api/mode',{mode:policy})};
document.querySelectorAll('[data-trigger]').forEach(b=>b.onclick=()=>post(`/api/trigger/${b.dataset.trigger}`));
async function webcam(){try{const stream=await navigator.mediaDevices.getUserMedia({video:true,audio:true}); q('#webcam').srcObject=stream; setInterval(()=>post('/api/event',{event:{type:'perception.frame',timestamp_ms:Date.now(),face_detected:true,face_center:[0.5+Math.sin(Date.now()/1800)*0.08,0.45],head_yaw:Math.sin(Date.now()/1200)*6,head_pitch:1.5,expression:{smile:0.18+Math.max(0,Math.sin(Date.now()/2200))*0.35,frown:0.05,brow_raise:0.08,eye_open:0.72}}}),250); setInterval(()=>post('/api/event',{event:{type:'audio.vad',timestamp_ms:Date.now(),speaking:false,energy:0.01,speech_rate:0.0}}),700)}catch(e){console.warn(e)}}
webcam(); poll(); setInterval(poll,1500);
