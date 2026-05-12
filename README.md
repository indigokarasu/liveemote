# Hermes Live Character Avatar Demo

Hermes Live Avatar is a local demo system that turns a canonical virtual character folder into a video-call-style embodied assistant. Hermes remains the cognition layer; this repository provides the body runtime: character ingest, affect policies, perception event wrappers, voice backends, renderer adapters, and a browser debug UI.

## Architecture

```text
User mic + webcam
  -> perception events (VAD, face position, gaze proxy, expression proxy)
  -> hermes-affect-runtime (rolling state, mirror/reflect policy, smoothing)
  -> Hermes bridge (fake or external compact summaries)
  -> voice backend (LuxTTS fallback, ElevenLabs, experimental MOSS-TTS)
  -> LiveTalking adapter (lip-sync/WebRTC/virtualcam handoff)
  -> browser demo / virtual camera
```

## Hardware assumptions

- Python 3.11+
- Webcam and microphone for live perception
- CPU works for the skeleton and fallback WAV generation
- NVIDIA GPU, Apple MPS, or a fast CPU is recommended for real LuxTTS and LiveTalking inference
- Virtual camera output requires platform-specific camera plumbing plus `pyvirtualcam`

## Dependency setup

```bash
make setup
```

`make setup` installs the local package, generates tiny local sample character media, and clones the required source repositories into `vendor/` so the demo does not depend on global source checkouts:

- `vendor/LiveTalking`
- `vendor/Deep-Live-Cam`
- `vendor/LuxTTS`
- `vendor/MOSS-TTS`

## Character folder spec

The default character folder is `./character_input`. Binary demo media is generated locally instead of tracked in git so GitHub can render the PR diff cleanly:

```bash
python scripts/create_sample_character.py --character ./character_input
```

```text
character_input/
  canonical/
    canonical.png              # required
    profile.yaml               # optional
    voice_reference.wav        # optional, required for local voice clone
    elevenlabs_voice_id.txt    # optional
  emotes/
    neutral/
    listening/
    thinking/
    happy/
    concerned/
    apologetic/
    amused/
    sad/
    error_recovery/
```

Supported emote media: `.png`, `.jpg`, `.webp`, `.mp4`, `.mov`, `.webm`. Optional paired `.wav` files are currently ignored unless explicitly tagged later.

The ingest path builds a `CharacterIndex` containing the canonical image, optional voice reference, optional ElevenLabs voice id, discovered emotes with deterministic ids such as `listening_001`, and a `training_references` manifest for renderer/model backends. The canonical image is always included as an `identity_anchor`; every static emote image (`.png`, `.jpg`, `.jpeg`, `.webp`) is also included as an `expression_reference` with its emote state and tags. Videos remain playable emotes but are not added as training references because most identity-consistency training or image-reference pipelines expect still frames.

### Using ~30 emote images for a consistent character

Place the images under state folders in `character_input/emotes/<state>/` and keep `character_input/canonical/canonical.png` as the clean neutral identity anchor. For example, put multiple happy images in `emotes/happy/`, sad images in `emotes/sad/`, and attentive/listening images in `emotes/listening/`. When `build_asset_index()` runs, all supported static images are included in `CharacterIndex.training_references`, so a renderer can train or condition on the same character across expression states while still using deterministic emote ids for playback.

Recommended dataset hygiene for consistency:

- Use the same character design, outfit, camera angle range, crop, and background style across all expression images.
- Keep one high-quality neutral/canonical image; it receives the strongest identity-anchor role.
- Spread the ~30 images across expression states instead of putting all of them in one folder, so the manifest labels the expression coverage.
- Avoid mixing different art styles or revised character designs unless you intend the model to learn that variation.


## Launch the WebRTC/browser demo

```bash
make demo CHARACTER=./character_input
```

Open <http://127.0.0.1:8080>. The page shows local webcam preview, avatar output placeholder, current behavior mode, VAD state, face detected state, gaze target, active emote, Hermes text, and manual controls.

For a dependency-light demo path using fake Hermes:

```bash
make demo-fake-hermes CHARACTER=./character_input
```


## Google Meet WebRTC test mode

The browser demo includes a **Google Meet WebRTC test mode** panel for checking the avatar path against real meeting latency. Paste a Meet link such as `https://meet.google.com/abc-defg-hij`, optionally set the participant name, and click **Join Meet**. The demo validates Meet URLs, asks the LiveTalking adapter to join through `/avatar/join_meeting`, and reports the measured local join handoff latency in the UI.

If the running LiveTalking process does not yet expose meeting-join support, the local demo falls back to opening a Chromium-family browser with WebRTC camera/microphone permissions pre-approved. Google may still require sign-in, lobby confirmation, or host admission before the avatar appears in the meeting.

## Launch virtual camera mode

```bash
make demo-virtualcam CHARACTER=./character_input
```

The LiveTalking adapter exposes a `start_virtualcam()` hook. Real virtual camera output requires LiveTalking and platform virtual camera dependencies to be installed and configured.

## Voice backends

### Local LuxTTS voice clone

Use the default backend:

```bash
python -m apps.demo_server.main --character ./character_input --voice-backend luxtts
```

Place a reference voice at `character_input/canonical/voice_reference.wav`. The adapter caches the reference prompt location and writes generated WAVs to `cache/voice`. In this skeleton, if LuxTTS is not wired at runtime, the adapter creates a deterministic placeholder WAV so the rest of the pipeline can be tested offline.

### ElevenLabs voice

Set environment variables or `.env` values:

```bash
export ELEVENLABS_API_KEY=...
export ELEVENLABS_VOICE_ID=...
python -m apps.demo_server.main --character ./character_input --voice-backend elevenlabs
```

Alternatively, store the voice id in `character_input/canonical/elevenlabs_voice_id.txt` and pass it into a custom config/adapter.

### MOSS-TTS

MOSS-TTS is experimental and only enabled explicitly:

```bash
python -m apps.demo_server.main --character ./character_input --voice-backend moss
```

The adapter currently raises a clear `NotImplementedError` until the optional streaming model dependencies are installed and integrated.

## Connecting real Hermes

Use external mode and configure the URL in `packages/hermes_avatar/config/defaults.yaml` or your own config:

```yaml
hermes:
  mode: external
  url: ws://127.0.0.1:18789/avatar
  send_events: [user.transcript, affect.summary, interruption]
  receive_events: [hermes.response, hermes.behavior_hint]
```

Launch with:

```bash
python -m apps.demo_server.main --character ./character_input --hermes-mode external
```

The bridge sends compact transcripts and affect summaries. It does **not** send raw video frames to Hermes by default.

## Affect runtime behavior

The deterministic baseline runtime runs independently of the LLM and consumes JSON events:

- `perception.frame`
- `audio.vad`
- `hermes.response`

It maintains rolling user/conversation/avatar state, smooths continuous values with exponential moving averages, latches expression changes with dwell time, clamps gaze/head movement, and supports `mirror` and `reflect` policy modes. Assistant speaking enables lip-sync and prevents camera mirroring from overriding the mouth.

## LiveTalking adapter

`LiveTalkingAdapter` is intentionally non-invasive first. It can load a `CharacterIndex`, set behavior/emotes, send pre-generated WAV speech, interrupt speech, and request WebRTC or virtual camera startup through HTTP endpoints if a patched or compatible LiveTalking process is running. If LiveTalking is offline, calls degrade to no-op/offline responses so the browser, affect, and voice paths remain testable.

## Deep-Live-Cam safety note

Deep-Live-Cam is optional and off by default. Do not use a real person's face identity as output unless you have explicit permission. Clearly label shared synthetic/deepfake output; the adapter includes a synthetic-output watermark string for this reason.

## Required commands

```bash
make setup
make demo CHARACTER=./character_input
make demo-fake-hermes CHARACTER=./character_input
make demo-virtualcam CHARACTER=./character_input
make test
```


## PR push safety

The third-party upstream repositories are cloned into `vendor/` for local development only and are intentionally ignored by git. Use this check before pushing a PR branch:

```bash
make push-check
```

The check fails if cloned vendor payloads, nested `.git` metadata, binary files, or large generated files are accidentally tracked. It also warns when the local checkout has no remote configured, because that environment cannot push until a remote is added by the caller.

## Limitations

- This is a v1 local demo skeleton, not a production avatar stack.
- LuxTTS integration has an offline WAV fallback until full vendor API wiring is configured.
- LiveTalking is controlled through a non-invasive HTTP adapter; endpoint support depends on the running LiveTalking app or future fork patch.
- Browser perception telemetry is intentionally lightweight and approximate.
- Emotion recognition, eye tracking, TTS latency, and full-body animation are not production-grade.
- No cloud deployment, mobile support, or multi-character scene support is included.
