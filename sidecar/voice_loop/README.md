# Voice Loop — speech-to-speech sidecar

The voice loop gives the avatar a **real spoken conversation channel**:

```
browser mic ──(16 kHz PCM16 over /ws/voice)──► LiveEmote server
                                                  │  VoiceLoopClient relay
                                                  ▼
                        voice-loop sidecar (this daemon, :8766)
                                                  │  spawns + supervises
                                                  ▼
              speech-to-speech pipeline (:8765 /v1/realtime)
                  Silero VAD → STT → LLM → TTS
                                                  │
browser speakers ◄──(response.audio.delta PCM16)──┘
```

The pipeline is the [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech)
package, run as its OpenAI-Realtime-compatible WebSocket server. It is a
separate process because it needs torch/transformers — it never lives in the
main LiveEmote server.

## 1. Install (one time)

```bash
cd sidecar/voice_loop
python -m venv .venv
source .venv/bin/activate
pip install torch          # platform-appropriate build; CPU/MPS fine on a Mac
pip install -r requirements.txt
```

> The pipeline downloads its models (VAD, STT, TTS) on first run. Budget a few
> GB and a couple of minutes the first time you start it.

## 2. Run

```bash
# Terminal A — the sidecar (spawns the pipeline subprocess automatically)
python -m sidecar.voice_loop.app --host 0.0.0.0 --port 8766

# Terminal B — the demo server, with the voice loop enabled
cd <repo root>
PYTHONPATH=packages python -m apps.demo_server.main \
  --voice-loop --host 0.0.0.0 --port 8080
```

Then open the demo page, click **Enable voice**, and talk. The avatar animates
while it listens and while its own voice replies.

## 3. Configuration (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `VOICE_LOOP__AUTOSTART` | `1` | Spawn the pipeline on sidecar boot |
| `VOICE_LOOP__PIPELINE_WS_HOST` / `PORT` | `127.0.0.1` / `8765` | Pipeline realtime WS |
| `VOICE_LOOP__STT__MODE` | `parakeet-tdt` | `parakeet-tdt`, `whisper`, `faster-whisper`, `mlx-audio-whisper`, `whisper-mlx`, `paraformer` |
| `VOICE_LOOP__LLM__MODE` | `chat-completions` | LLM backend: `chat-completions`, `responses-api`, `mlx-lm`, `transformers` |
| `VOICE_LOOP__LLM__BASE_URL` | `$OPENAI_COMPATIBLE_BASE_URL` | Any OpenAI-compatible endpoint → `--responses_api_base_url` |
| `VOICE_LOOP__LLM__MODEL` | `$OPENAI_COMPATIBLE_MODEL` | Model id → `--model_name` |
| `VOICE_LOOP__LLM__API_KEY` | `$OPENAI_COMPATIBLE_API_KEY` | API key → `--responses_api_api_key` |
| `VOICE_LOOP__TTS__MODE` | `qwen3` | `qwen3`, `kokoro`, `pocket`, `chatTTS`, `facebookMMS` |
| `VOICE_LOOP__NUM_PIPELINES` | `1` | Concurrent conversation pipelines |

The LLM defaults point at the same `OPENAI_COMPATIBLE_*` env vars the avatar's
typed-speak path uses, so your existing LLM key works for the voice loop too.

## 4. Main-server integration

* `packages/hermes_avatar/voice/voice_loop.py` — `VoiceLoopClient` (health
  probe + WebSocket relay with transcript/audio snooping).
* `apps/demo_server/voice_ws.py` — `/ws/voice` endpoint the browser dials.
* `/api/health` reports a `voice_loop` component; `/api/status` returns
  `voice_loop` state; the demo page shows a voice panel + mic meter.

## Troubleshooting

* `/api/health` shows `voice_loop: degraded` → the sidecar isn't running or
  the pipeline failed to boot. `curl http://127.0.0.1:8766/status` shows the
  pipeline log tail and exit code.
* Pipeline exits immediately → `curl http://127.0.0.1:8766/status` log tail
  usually shows a model download or arg error; check the `--stt` / `--llm_backend` /
  `--tts` backend names in `build_pipeline_args()` against the installed
  `speech-to-speech` version (`speech-to-speech serve -h` lists the current
  choices; the CLI moved to a flat `serve` subcommand and the old `--module.*`
  spelling no longer works).
* No audio from the avatar → confirm the browser mic permission, that the page
  is served over https (or localhost), and that `voice_loop` shows `reachable`.
