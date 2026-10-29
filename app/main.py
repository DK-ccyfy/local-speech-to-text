"""FastAPI application: local speech-to-text web service.

Endpoints (fixed by the assignment):

- GET  /health          -> {"status": "ok", ...}
- POST /api/transcribe  -> multipart(file, language) -> transcription JSON
- GET  /                -> the web page (static/index.html)
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import config
from app.asr.base import ASRService
from app.asr.whisper import WhisperASRService
from app.audio import decode_audio
from app.errors import AppError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

VALID_LANGUAGES = ("zh", "en", "auto")


def normalize_language(language: str) -> str | None:
    """Validate the language form field; return the Whisper language (None=auto)."""
    lang = (language or "").strip().lower()
    if lang not in VALID_LANGUAGES:
        raise AppError(
            f"不支持的语言参数 language={language!r}，请使用 zh、en 或 auto", 400
        )
    return None if lang == "auto" else lang


async def read_upload(file: UploadFile) -> bytes:
    """Read the uploaded file, enforcing the size limit (413) and emptiness (400)."""
    limit = config.MAX_UPLOAD_BYTES
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise AppError(f"文件大小超过上限 {config.MAX_UPLOAD_MB} MB", 413)
        chunks.append(chunk)
    if total == 0:
        raise AppError("上传的文件为空", 400)
    return b"".join(chunks)


def create_app(asr_service: ASRService | None = None) -> FastAPI:
    """Build the app; `asr_service` lets tests inject a fake model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.asr.warm_up()
        yield

    app = FastAPI(title="Local Speech-to-Text", lifespan=lifespan)
    app.state.asr = asr_service or WhisperASRService()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model": request.app.state.asr.status}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/transcribe")
    async def transcribe(
        request: Request,
        file: UploadFile = File(..., description="音频文件（WAV/WebM/MP3 等）"),
        language: str = Form(..., description="zh / en / auto"),
    ):
        lang = normalize_language(language)
        data = await read_upload(file)

        decode_started = time.monotonic()
        pcm, duration = await run_in_threadpool(decode_audio, data)
        decode_seconds = time.monotonic() - decode_started

        result = await run_in_threadpool(request.app.state.asr.transcribe, pcm, lang)

        return {
            "text": result.text,
            "language": language.strip().lower(),
            "duration_seconds": round(duration, 3),
            "processing_seconds": round(decode_seconds + result.inference_seconds, 3),
        }

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})

    return app


app = create_app()
