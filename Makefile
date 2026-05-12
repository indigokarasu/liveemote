PYTHON ?= python
CHARACTER ?= ./character_input
VOICE_BACKEND ?= luxtts
RENDERER ?= livetalking
TRANSPORT ?= webrtc

setup:
	$(PYTHON) -m pip install -e ".[test]"
	mkdir -p vendor cache/voice
	$(PYTHON) scripts/create_sample_character.py --character $(CHARACTER)
	@test -d vendor/LiveTalking || git clone --depth 1 https://github.com/lipku/LiveTalking.git vendor/LiveTalking
	@test -d vendor/Deep-Live-Cam || git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git vendor/Deep-Live-Cam
	@test -d vendor/LuxTTS || git clone --depth 1 https://github.com/ysharma3501/LuxTTS.git vendor/LuxTTS
	@test -d vendor/MOSS-TTS || git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git vendor/MOSS-TTS
	$(PYTHON) scripts/setup_deeplivecam_models.py

demo:
	$(PYTHON) -m apps.demo_server.main --character $(CHARACTER) --renderer $(RENDERER) --voice-backend $(VOICE_BACKEND) --transport $(TRANSPORT)

demo-fake-hermes:
	$(PYTHON) -m apps.demo_server.main --character $(CHARACTER) --renderer $(RENDERER) --voice-backend $(VOICE_BACKEND) --transport webrtc --hermes-mode fake

demo-virtualcam:
	$(PYTHON) -m apps.demo_server.main --character $(CHARACTER) --renderer $(RENDERER) --voice-backend $(VOICE_BACKEND) --transport virtualcam

deeplivecam-models:
	mkdir -p vendor
	@test -d vendor/Deep-Live-Cam || git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git vendor/Deep-Live-Cam
	$(PYTHON) scripts/setup_deeplivecam_models.py

check-deeplivecam-models:
	$(PYTHON) scripts/setup_deeplivecam_models.py --check-only

test:
	$(PYTHON) scripts/create_sample_character.py --character ./character_input
	$(PYTHON) -m pytest

push-check:
	$(PYTHON) scripts/check_push_ready.py
