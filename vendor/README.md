# Vendor repositories

`make setup` clones the required upstream repositories into this directory:

- LiveTalking -> `vendor/LiveTalking`
- Deep-Live-Cam -> `vendor/Deep-Live-Cam`
- LuxTTS -> `vendor/LuxTTS`
- MOSS-TTS -> `vendor/MOSS-TTS`

The cloned source trees are intentionally git-ignored to avoid vendoring large third-party histories into this repository.


## Deep-Live-Cam model files

Run `python scripts/setup_deeplivecam_models.py` from the repository root to download the recommended Deep-Live-Cam models into `vendor/Deep-Live-Cam/models`. Run `python scripts/setup_deeplivecam_models.py --check-only` to verify placement without downloading.
