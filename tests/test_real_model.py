"""Real-model tests — NOT part of the default suite.

Run with (downloads the model on first use, ~460 MB for `small`):

    python -m pytest -q -m realmodel

Requires samples/chinese.wav and samples/english.wav (see scripts/make_samples.sh).
"""

import time
from pathlib import Path

import pytest

from app.audio import decode_audio
from app.main import create_app
from app.asr.whisper import WhisperASRService
from tests.conftest import make_client

pytestmark = pytest.mark.realmodel

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
CHINESE_WAV = SAMPLES / "chinese.wav"
ENGLISH_WAV = SAMPLES / "english.wav"

requires_samples = pytest.mark.skipif(
    not (CHINESE_WAV.exists() and ENGLISH_WAV.exists()),
    reason="sample audio files not found (run scripts/make_samples.sh)",
)

ZH_KEYWORDS = ["大家好", "天气", "公园", "散步"]
EN_KEYWORDS = ["hello", "speech", "recognition", "computer"]


@pytest.fixture(scope="module")
def service():
    svc = WhisperASRService()
    svc.warm_up()
    if not svc._ready_event.wait(timeout=1200):  # first run includes model download
        pytest.fail("model did not become ready in time")
    if svc.status != "ready":
        pytest.fail("model failed to load")
    return svc


def _hits(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw.lower() in text.lower())


@requires_samples
def test_real_model_chinese(service):
    pcm, duration = decode_audio(CHINESE_WAV.read_bytes())
    assert 4.0 < duration < 20.0, f"unexpected sample duration: {duration}"
    result = service.transcribe(pcm, "zh")
    assert result.text, "Chinese transcription is empty"
    assert _hits(result.text, ZH_KEYWORDS) >= 2, (
        f"transcription does not match reference well enough: {result.text!r}"
    )
    print(f"\n[chinese] duration={duration:.1f}s inference={result.inference_seconds:.2f}s text={result.text}")


@requires_samples
def test_real_model_english(service):
    pcm, duration = decode_audio(ENGLISH_WAV.read_bytes())
    assert 4.0 < duration < 20.0, f"unexpected sample duration: {duration}"
    result = service.transcribe(pcm, "en")
    assert result.text, "English transcription is empty"
    assert _hits(result.text, EN_KEYWORDS) >= 2, (
        f"transcription does not match reference well enough: {result.text!r}"
    )
    print(f"\n[english] duration={duration:.1f}s inference={result.inference_seconds:.2f}s text={result.text}")


@requires_samples
def test_real_model_auto_detect_language(service):
    pcm, _ = decode_audio(CHINESE_WAV.read_bytes())
    result = service.transcribe(pcm, None)  # bonus: auto language detection
    assert result.text


@requires_samples
def test_real_model_via_http_api(service):
    """Full-stack: multipart upload -> decode -> real model -> response JSON."""
    with make_client(service) as client:
        resp = client.post(
            "/api/transcribe",
            files={"file": ("chinese.wav", CHINESE_WAV.read_bytes(), "audio/wav")},
            data={"language": "zh"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] and _hits(body["text"], ZH_KEYWORDS) >= 2
    assert body["language"] == "zh"
    assert body["duration_seconds"] > 0
    assert body["processing_seconds"] > 0


@requires_samples
def test_real_model_processing_time_excludes_model_load(service):
    pcm, _ = decode_audio(CHINESE_WAV.read_bytes())
    time.sleep(0.05)
    result = service.transcribe(pcm, "zh")
    assert 0 < result.inference_seconds < 120
