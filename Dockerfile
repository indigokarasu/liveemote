# -- Fly.io Dockerfile for Hermes Live Avatar Demo --
# MediaPipe requires Debian-based image (no Alpine/musl)
FROM python:3.11-slim-bookworm

# System deps for MediaPipe, OpenCV, and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
# mediapipe pinned for reproducibility; the rest from pyproject.toml
RUN pip install --no-cache-dir \
    mediapipe>=0.10.14 \
    opencv-python-headless>=4.9 \
    fastapi>=0.110 \
    "uvicorn[standard]>=0.27" \
    pydantic>=2.6 \
    pyyaml>=6.0 \
    httpx>=0.26 \
    websockets>=12.0 \
    numpy>=1.24 \
    soundfile>=0.12 \
    prometheus_client>=0.20 \
    python-multipart>=0.0.9 \
    Pillow>=10.0 \
    typing_extensions>=4.6

WORKDIR /app

# Copy only what the server needs at runtime
COPY packages/ ./packages/
COPY apps/     ./apps/

ENV PYTHONPATH=/app/packages
ENV PYTHONUNBUFFERED=1

# Fly.io injects PORT; default to 8080 for local dev
EXPOSE 8080

CMD ["sh", "-c", "exec python3 -m apps.demo_server.main --host 0.0.0.0 --port ${PORT:-8080}"]
