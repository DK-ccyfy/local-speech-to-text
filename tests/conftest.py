"""Shared fixtures & helpers for the test suite.

The default suite (`python -m pytest -q`) uses FakeASRService so it never
downloads the real model and never touches a microphone.
"""

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.asr.base import ASRService, Transcription  # noqa: E402
from app.main import create_app  # noqa: E402


# --------------------------------------------------------------------- audio
def make_wav(seconds: float = 1.0, sample_rate: int = 16000, channels: int = 1,
             frequency: float = 440.0, amplitude: float = 0.5) -> bytes:
    """Generate an in-memory 16-bit PCM WAV (sine wave or silence)."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    mono = (amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype("<i2")
    data = mono if channels == 1 else np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.astype("<i2").tobytes())
    return buf.getvalue()


def make_webm(seconds: float = 1.0, sample_rate: int = 48000) -> bytes:
    """Generate a small WebM/Opus file with PyAV (the browser MediaRecorder format family)."""
    import av

    buf = io.BytesIO()
    container = av.open(buf, "w", format="webm")
    stream = container.add_stream("libopus", rate=sample_rate)
    stream.layout = "mono"

    frame_size = 960  # samples per Opus frame @48k
    n_frames = int(seconds * sample_rate / frame_size) + 1
    for i in range(n_frames):
        t = (np.arange(frame_size) + i * frame_size) / sample_rate
        arr = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="flt", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = i * frame_size
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue()


# ------------------------------------------------------------------- fakes
class FakeASRService(ASRService):
    """Deterministic stand-in for the real Whisper service."""

    def __init__(self, texts: dict | None = None, error: Exception | None = None):
        self.texts = texts or {
            "zh": "你好，欢迎使用语音识别系统。",
            "en": "Hello from the fake model.",
            None: "自动检测的文本。",
        }
        self.error = error
        self.calls: list[tuple[str, tuple | None, str | None]] = []

    def transcribe(self, audio, language):
        self.calls.append((str(getattr(audio, "dtype", type(audio))), getattr(audio, "shape", None), language))
        if self.error is not None:
            raise self.error
        return Transcription(text=self.texts.get(language, ""), inference_seconds=0.001)


def make_client(service: ASRService) -> TestClient:
    return TestClient(create_app(asr_service=service))


# ----------------------------------------------------------------- fixtures
@pytest.fixture()
def fake_asr() -> FakeASRService:
    return FakeASRService()


@pytest.fixture()
def client(fake_asr) -> TestClient:
    with make_client(fake_asr) as c:
        yield c
