# liveemote

> A real AI live avatar whose motion is keyed off *you*, without ever copying
> *you*. The avatar has its own face and voice; your webcam is read as a
> focus + energy signal that shapes its affect — never as a face to
> re-enact or swap.

## What this is

LiveEmote runs a FastAPI demo server that pairs:

1. **An autonomous avatar** rendered in the browser from the active
   character's canonical stills + emotes. The avatar does its own motion
   (breathing, gaze shifts, emote transitions) driven by an affect runtime
   that is *deliberately not* a literal mirror.
2. **A webcam perception pipeline** (MediaPipe Face Mesh on the server, plus
   an audio VAD on the client) that extracts focus (eye aspect ratio, gaze
   centrality, blink cadence) and energy (head-pose variance, mouth-open
   amplitude). Frames never reach the avatar.
3. **A real LLM** that handles the avatar's spoken replies. The runtime ships
   with an OpenAI-compatible chat-completions adapter so any provider works
   (OpenAI, SambaNova, llama.cpp, Together, OpenRouter, LM Studio). The
   adapter asks the model for a tiny structured JSON contract:
   ``{text, tags:{affect, voice:{pace,warmth,intensity}}}``.

The affect runtime (**packages/hermes_avatar/affect/**) is the heart of the
rebrand. ``reflect_policy.py`` and ``mirror_policy.py`` already encode a
psychological model of *deliberately damped* mimicry: high valence gets a
small delayed smile, high tension gets validated and grounded, low attention
gets a spacious silence. The avatar never amplifies anger or copies movement;
it tunes itself to the user's focus and energy.

## Quick start

```bash
make setup                # pip install -e ".[test]" + sample character + vendor clones
python -m apps.demo_server.main \
    --character ./character_input \
    --renderer web \
    --agent-mode openai_compatible \
    --voice-backend luxtts
```

Then visit ``http://127.0.0.1:8080``. Grant camera + mic permission; the
avatar will appear to the right of the webcam preview and start mirroring /
reflecting your affect.

### Renderers

| Flag | Backend | Behaviour |
|------|---------|-----------|
| `--renderer web` (default) | Self-driven browser avatar | The character's own canonical image + emotes are animated autonomously from the affect runtime. No face re-enactment. |
| `--renderer livetalking` | LiveTalking | Opt-in face-reenactment over the vendored LiveTalking daemon. |
| `--renderer deeplivecam` | Deep-Live-Cam | Opt-in face-swap over the vendored Deep-Live-Cam. |

### Agent / LLM

| Flag | Source | Notes |
|------|--------|-------|
| `--agent-mode openai_compatible` | OpenAI-compatible chat completions | Real LLM. Reads ``OPENAI_COMPATIBLE_*`` env vars. Default. |
| `--agent-mode openclaw\|hermes\|deerflow\|external` | Existing OCAS agent harness | Websocket / HTTP JSON contract, no LLM SDK required. |
| `--agent-mode fake` | Local parametric | Echo / canned reply, useful for UI development. |
| `--agent-mode offline` | None | Affect mirroring / reflecting continues, no spoken text. |

### Voice

The avatar's voice is the configured voice backend. Default is LuxTTS
(local parametric with optional upstream wiring through ``$LUXTTS_COMMAND``).
ElevenLabs is selectable via ``--voice-backend elevenlabs`` (set
``ELEVENLABS_API_KEY`` + ``ELEVENLABS_VOICE_ID``).

## Dependencies

``make setup`` installs the local package, generates tiny local sample
character media, and clones the optional source repositories into vendor/ so
the demo does not depend on global source checkouts:

- vendor/LiveTalking
- vendor/Deep-Live-Cam
- vendor/LuxTTS
- vendor/MOSS-TTS

For server-side perception (``--perception-tracker mediapipe``, default),
install the optional extras:

```bash
pip install -e ".[perception]"
```

Without ``[perception]`` the runtime still imports; the tracker degrades to
``NullFaceTracker`` and the avatar still mirrors/reflects from whatever
fallback signals it sees.

## Architecture

```
apps/demo_server/
  routes.py         ← HTTP + WS (incl. /api/perception/video, /api/character/asset)
  static/           ← browser avatar (canonical + emotes, SVG fallback, breathing)
packages/hermes_avatar/
  affect/           ← mirror/reflect policy + smoothing + reaction-delay
  perception/       ← MediaPipe Face Mesh → focus + energy signals
  protocol/         ← AgentBridge + OpenAI-compatible LLM adapter
  renderer/         ← WebRenderer (default), LiveTalking, Deep-Live-Cam
  voice/            ← LuxTTS, ElevenLabs, no-op
  demo/             ← DemoOrchestrator + meeting join + character discovery
```

## What's intentionally NOT here

- No face-reenactment in the default path. LiveTalking / Deep-Live-Cam are
  reachable only as opt-in choices because they copy the user's face frame
  by frame.
- No voice cloning. The avatar's voice is the configured TTS backend speaking
  the LLM's words; the user's voice is not recorded, cloned, or played back.
- No canned behaviour as the default. ``--agent-mode fake`` exists for
  development but the demo now boots in ``openai_compatible`` mode.

---

*liveemote is part of the [OCAS Agent Suite](https://github.com/indigokarasu).*

## 📄 License
MIT License — see `LICENSE` for details.
