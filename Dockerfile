FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY apps ./apps
COPY packages ./packages
COPY scripts ./scripts
COPY character_input ./character_input

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["python", "-m", "apps.demo_server.main", "--host", "0.0.0.0", "--port", "8080", "--character", "./character_input", "--hermes-mode", "fake"]
