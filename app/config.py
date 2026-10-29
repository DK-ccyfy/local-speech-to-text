"""Application configuration.

Every value can be overridden through environment variables so the unified
start command from the assignment works unchanged::

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# --- ASR model -----------------------------------------------------------
# faster-whisper model size: tiny / base / small / medium ...
# "small" gives good zh+en accuracy while staying fast on CPU (int8).
ASR_MODEL = os.environ.get("ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8")

# 0 means "auto": use at most 8 physical threads.
_asr_threads = _env_int("ASR_CPU_THREADS", 0)
ASR_CPU_THREADS = _asr_threads if _asr_threads > 0 else max(1, min(8, os.cpu_count() or 4))

# Seconds a transcription request waits for the first model load
# (which may include the one-time model download) before failing.
ASR_LOAD_TIMEOUT = float(os.environ.get("ASR_LOAD_TIMEOUT", "900"))

# --- Upload limits -------------------------------------------------------
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 10)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Browser recording length cap (seconds).
MAX_RECORD_SECONDS = float(os.environ.get("MAX_RECORD_SECONDS", "60"))
