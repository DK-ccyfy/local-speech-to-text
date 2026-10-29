"""faster-whisper backed ASR service.

- Runs the open-source Whisper model locally via CTranslate2 (CPU, int8).
- Model weights are downloaded once on first use and cached under the
  user's HuggingFace cache; after that the service works fully offline.
- Loading happens in a background thread at startup ("warm up"); a request
  arriving while the model is still loading waits up to ASR_LOAD_TIMEOUT.
"""

import logging
import threading
import time

import numpy as np

from app import config
from app.asr.base import ASRService, Transcription
from app.errors import AppError

log = logging.getLogger(__name__)

# Bias Whisper's Chinese tokenizer toward simplified characters (Mandarin
# speech is expected; without this it often emits 繁体).
ZH_SIMPLIFIED_PROMPT = "以下是普通话的句子，请使用简体中文。"


class WhisperASRService(ASRService):
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or config.ASR_MODEL
        self._lock = threading.Lock()
        self._state = "idle"  # idle -> loading -> ready | failed
        self._error: str | None = None
        self._model = None
        self._ready_event = threading.Event()

    # ------------------------------------------------------------------ state
    @property
    def status(self) -> str:
        with self._lock:
            return self._state

    def warm_up(self) -> None:
        """Kick off background model loading (called at app startup)."""
        with self._lock:
            if self._state != "idle":
                return
            self._state = "loading"
        threading.Thread(target=self._load, name="asr-model-load", daemon=True).start()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        """Download/load the model; must only run once. Lock is NOT held."""
        try:
            from faster_whisper import WhisperModel  # deferred: keeps import cheap

            started = time.monotonic()
            common = dict(
                device=config.ASR_DEVICE,
                compute_type=config.ASR_COMPUTE_TYPE,
                cpu_threads=config.ASR_CPU_THREADS,
            )
            try:
                # Cached weights first: instant load and works fully offline.
                model = WhisperModel(self._model_name, local_files_only=True, **common)
            except Exception:
                # No usable local cache -> one-time online download.
                model = WhisperModel(self._model_name, **common)
        except Exception as exc:  # download failure, bad model name, OOM ...
            log.exception("ASR model failed to load")
            with self._lock:
                self._state = "failed"
                self._error = str(exc)
            self._ready_event.set()
            return

        with self._lock:
            self._model = model
            self._state = "ready"
            self._error = None
        self._ready_event.set()
        log.info(
            "ASR model %r loaded on %s/%s in %.1fs",
            self._model_name, config.ASR_DEVICE, config.ASR_COMPUTE_TYPE,
            time.monotonic() - started,
        )

    def _ensure_ready(self) -> None:
        with self._lock:
            state = self._state
            if state == "ready":
                return
            if state == "failed":
                raise AppError(f"语音识别模型加载失败：{self._error}", 503)
            lazy_load = state == "idle"
            if lazy_load:
                self._state = "loading"

        if lazy_load:
            # Lifespan warm-up did not run (e.g. bare ASGI usage): load here.
            self._load()
        elif not self._ready_event.wait(timeout=config.ASR_LOAD_TIMEOUT):
            raise AppError(
                "语音识别模型仍在加载中（首次运行需要下载模型权重），请稍后重试", 503
            )

        with self._lock:
            if self._state == "ready":
                return
            if self._state == "loading":
                raise AppError(
                    "语音识别模型仍在加载中（首次运行需要下载模型权重），请稍后重试", 503
                )
            raise AppError(f"语音识别模型加载失败：{self._error}", 503)

    # ------------------------------------------------------------- transcribe
    def transcribe(self, audio: np.ndarray, language: str | None) -> Transcription:
        self._ensure_ready()
        assert self._model is not None
        started = time.monotonic()
        kwargs: dict = {"vad_filter": True, "beam_size": 5}
        if language == "zh":
            kwargs["initial_prompt"] = ZH_SIMPLIFIED_PROMPT
        try:
            segments, info = self._model.transcribe(audio, language=language, **kwargs)
            text = "".join(segment.text for segment in segments).strip()
            if language is None and info.language == "zh":
                # auto mode detected Mandarin: redo with the simplified-Chinese
                # prompt so the bonus feature matches explicit-zh output.
                segments, _ = self._model.transcribe(
                    audio, language="zh", initial_prompt=ZH_SIMPLIFIED_PROMPT,
                    vad_filter=True, beam_size=5,
                )
                text = "".join(segment.text for segment in segments).strip()
        except AppError:
            raise
        except Exception as exc:
            log.exception("transcription failed")
            raise AppError(f"语音识别失败：{exc}", 500)
        if language is None:
            log.info("auto-detected language: %s (prob %.2f)", info.language, info.language_probability)
        return Transcription(text=text, inference_seconds=time.monotonic() - started)
