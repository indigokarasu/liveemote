# Main LiveEmote Dockerfile — Python 3.10 base.
#
# FaceFusion was moved out of this image into sidecar/Dockerfile.facefusion
# because FaceFusion 3.x hard-requires typing.NotRequired (Python 3.11+) plus
# a heavy ONNX / insightface stack that bloats the main image and forces a
# 3.11 base. Main process stays slim on Python 3.10; FaceFusion runs in a
# 3.11 sidecar and is reached over HTTP (see compose file).
#
# Deep-Live-Cam continues to run in this image — its Python surface works on
# 3.10 and doesn't require the typing.NotRequired shim.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgl1 + libglib2.0-0 are required by the OpenCV / insightface stack that
# Deep-Live-Cam still imports in-process.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY apps ./apps
COPY packages ./packages
COPY scripts ./scripts
COPY character_input ./character_input

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e .

EXPOSE 8080

# FACESWAP__SIDECAR__URL is wired so the adapter auto-picks the HTTP sidecar
# daemon over the in-process FaceFusionVendorDaemon (which would fail under
# Python 3.10). Override at runtime via docker-compose or your orchestrator.
ENV FACESWAP__SIDECAR__SIDECAR_URL=http://liveemote-facefusion:8001
CMD ["python", "-m", "apps.demo_server.main", "--host", "0.0.0.0", "--port", "8080", "--character", "./character_input", "--hermes-mode", "fake"]
