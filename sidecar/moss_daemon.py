"""``sidecar/moss_daemon.py`` — sidecar daemon for MOSS-Transcribe-Diarize.

MOSS-Transcribe-Diarize is a single 0.9B model that performs transcription +
speaker diarization + timestamps in one pass, emitting the compact format

    [0.48][S01]Welcome everyone[1.66][12.26][S02]The pipeline is ready[13.81]

It requires Python 3.12 and Transformers 5.x, which the main LiveEmote server
(Python 3.10) cannot host. Like the FaceFusion sidecar, this daemon runs in
its own venv and exposes HTTP:

* ``GET  /health``           — model loaded? device? degraded reason?
* ``POST /warmup``           — force the model into memory ahead of time
* ``POST /api/v1/transcribe`` — multipart audio upload -> JSON segments

Run (from a Python 3.12 venv with ``moss-transcribe-diarize`` installed):

    python sidecar/moss_daemon.py --host 127.0.0.1 --port 8899 \
        --model OpenMOSS-Team/MOSS-Transcribe-Diarize

Env overrides: ``MOSS__SIDECAR__MODEL``, ``MOSS__SIDECAR__DEVICE``
(``auto`` | ``cuda`` | ``cpu``), ``MOSS__SIDECAR__DTYPE`` (``bf16`` |
``fp16`` | ``fp32``), ``MOSS__SIDECAR__AUTH_REQUIRED`` + ``MOSS__SIDECAR__API_KEY``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("sidecar.moss")

MODEL_ID = os.getenv("MOSS__SIDECAR__MODEL", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
DEVICE = os.getenv("MOSS__SIDECAR__DEVICE", "auto")
DTYPE = os.getenv("MOSS__SIDECAR__DTYPE", "bf16")
API_KEY = os.getenv("MOSS__SIDECAR__API_KEY", "")
AUTH_REQUIRED = os.getenv("MOSS__SIDECAR__AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")

# MOSS is only importable inside its own Python 3.12 venv. Import lazily so
# this module can be syntax-checked / imported anywhere; the daemon degrades
# gracefully with a clear reason when the package is missing.
_runner: Any = None
_runner_error: str | None = None


def _load_runner() -> Any:
    """Lazily build the ModelRunner. Imported only on first use."""
    global _runner, _runner_error
    if _runner is not None or _runner_error is not None:
        return _runner
    try:
        from moss_transcribe_diarize.app.model_runner import ModelRunner  # type: ignore
        _runner = ModelRunner(MODEL_ID, device=DEVICE, dtype=DTYPE)
    except Exception as exc:
        _runner_error = f"moss-transcribe-diarize unavailable: {exc}"
        logger.warning("MOSS runner unavailable: %s", exc)
        return None
    return _runner


def _parse_segments(text: str) -> list[dict[str, Any]]:
    """Parse the compact ``[start][Sxx]text[end]`` transcript into segments.

    Uses the repo's streaming parser when available; falls back to a compact
    regex so the daemon never crashes on an odd transcript shape.
    """
    segments: list[dict[str, Any]] = []
    try:
        from moss_transcribe_diarize.transcript_parser import (  # type: ignore
            TranscriptStreamParser,
        )
        parser = TranscriptStreamParser()
        segs = parser.feed(text)
        try:
            segs += parser.close()
        except Exception:
            pass
        for s in segs:
            segments.append(
                {"start": float(s.start), "end": float(s.end) if s.end is not None else None,
                 "speaker": s.speaker, "text": s.text}
            )
        if segments:
            return segments
    except Exception:
        pass

    import re
    header = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\]")
    tail = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*$")
    matches = list(header.finditer(text))
    for i, m in enumerate(matches):
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end_pos]
        end: float | None = None
        t = tail.search(body)
        if t:
            end = float(t.group(1))
            body = body[: t.start()]
        body = body.strip()
        if body:
            segments.append(
                {"start": float(m.group(1)), "end": end, "speaker": m.group(2), "text": body}
            )
    return segments


def _normalize_wav(raw: bytes) -> bytes:
    """Best-effort normalization to 16 kHz mono WAV via ffmpeg (optional).

    The model expects 16 kHz mono; ffmpeg is a soft dependency. If ffmpeg is
    missing we pass the bytes through untouched and let the model's processor
    complain if the input is genuinely incompatible.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return raw
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
            input=raw, capture_output=True, timeout=180,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
    except Exception as exc:
        logger.warning("ffmpeg normalization failed; passing raw bytes: %s", exc)
    return raw


async def _bearer_dep(authorization: str | None = Header(default=None)) -> None:
    if not AUTH_REQUIRED:
        return
    if not API_KEY:
        raise HTTPException(status_code=503, detail="MOSS sidecar auth not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization.split("Bearer ", 1)[1].strip() != API_KEY:
        raise HTTPException(status_code=403, detail="invalid bearer token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="LiveEmote MOSS-Transcribe-Diarize Sidecar", version="0.1.0", lifespan=lifespan)


def _health_detail() -> dict[str, Any]:
    if _runner is None and _runner_error is None:
        _load_runner()
    loaded = _runner is not None and bool(getattr(_runner, "is_loaded", False))
    return {
        "status": "ok" if loaded else "degraded",
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": DTYPE,
        "model_loaded": loaded,
        "error": _runner_error,
    }


@app.get("/health")
async def health(dep: None = Depends(_bearer_dep)) -> JSONResponse:
    detail = _health_detail()
    return JSONResponse(detail, status_code=200 if detail["status"] == "ok" else 503)


@app.post("/warmup")
async def warmup(dep: None = Depends(_bearer_dep)) -> JSONResponse:
    runner = _load_runner()
    if runner is None:
        return JSONResponse({"healthy": False, "error": _runner_error}, status_code=503)
    try:
        # ModelRunner loads on first transcribe; force it here so the first
        # real request doesn't pay the download+load cost.
        if not getattr(runner, "is_loaded", False):
            runner._ensure_loaded()  # type: ignore[attr-defined]
        return JSONResponse({"healthy": True, "model_loaded": True})
    except Exception as exc:
        return JSONResponse({"healthy": False, "error": str(exc)}, status_code=503)


@app.post("/api/v1/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    prompt: str | None = Form(None),
    max_new_tokens: int = Form(2048),
    decoding: str = Form("greedy"),
    dep: None = Depends(_bearer_dep),
) -> JSONResponse:
    runner = _load_runner()
    if runner is None:
        return JSONResponse(
            {"available": False, "segments": [], "reason": _runner_error}, status_code=503
        )

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio upload")

    wav = _normalize_wav(raw)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav)
        tmp_path = tmp.name
    try:
        result = await asyncio.to_thread(
            runner.transcribe,
            tmp_path,
            prompt=prompt or None,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
        )
        segments = _parse_segments(result.text)
        return JSONResponse(
            {
                "available": True,
                "segments": segments,
                "text": result.text,
                "model": getattr(result, "model", MODEL_ID),
                "elapsed_sec": round(float(getattr(result, "elapsed_sec", 0.0)), 3),
                "decoding": decoding,
            }
        )
    except Exception as exc:
        logger.warning("MOSS transcribe failed: %s", exc)
        return JSONResponse(
            {"available": False, "segments": [], "reason": f"transcribe failed: {exc}"},
            status_code=500,
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="MOSS-Transcribe-Diarize sidecar daemon")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--model", default=None)
    args = p.parse_args()
    logging.basicConfig(level=os.getenv("SIDECAR_LOG_LEVEL", "INFO"))
    if args.model:
        global MODEL_ID
        MODEL_ID = args.model
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
