"""POST /api/transcribe — API contract, validation and error handling (mocked model)."""

import pytest

from tests.conftest import FakeASRService, make_wav, make_webm, make_client


def post(client, content: bytes, filename: str = "test.wav", language: str | None = "zh",
         include_file: bool = True, include_language: bool = True):
    files = {"file": (filename, content, "application/octet-stream")} if include_file else None
    data = {"language": language} if include_language else None
    return client.post("/api/transcribe", files=files, data=data)


# ------------------------------------------------------------------- happy
def test_transcribe_chinese(client):
    resp = post(client, make_wav(1.5), language="zh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "你好，欢迎使用语音识别系统。"
    assert body["language"] == "zh"
    assert isinstance(body["duration_seconds"], float)
    assert abs(body["duration_seconds"] - 1.5) < 0.05
    assert isinstance(body["processing_seconds"], float)
    assert body["processing_seconds"] >= 0


def test_transcribe_english(client):
    resp = post(client, make_wav(1.0), filename="clip.wav", language="en")
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Hello from the fake model."
    assert body["language"] == "en"


def test_transcribe_auto_language(client):
    resp = post(client, make_wav(1.0), language="auto")
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "自动检测的文本。"
    assert body["language"] == "auto"


def test_language_is_case_insensitive(client):
    resp = post(client, make_wav(0.5), language="ZH")
    assert resp.status_code == 200
    assert resp.json()["language"] == "zh"


def test_browser_recording_format_webm(client):
    """The WebM/Opus family used by MediaRecorder must decode (shared backend logic)."""
    resp = post(client, make_webm(1.0), filename="recording.webm", language="zh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["duration_seconds"] > 0.5
    assert body["text"] == "你好，欢迎使用语音识别系统。"


def test_silence_returns_empty_text_not_error():
    service = FakeASRService(texts={"zh": "", "en": "", None: ""})
    with make_client(service) as c:
        resp = post(c, make_wav(2.0, amplitude=0.0), language="zh")
    assert resp.status_code == 200
    assert resp.json()["text"] == ""


def test_model_receives_expected_pcm(client, fake_asr):
    post(client, make_wav(1.0, sample_rate=44100, channels=2), language="en")
    (dtype, shape, language), = fake_asr.calls
    assert dtype == "float32"          # 16 kHz mono float32 for faster-whisper
    assert shape is not None and shape[0]  # ~1 s -> ~16000 samples
    assert 15000 <= shape[0] <= 17000
    assert language == "en"


# ---------------------------------------------------------- invalid params
@pytest.mark.parametrize("language", ["fr", "chinese", "e", "zh-CN"])
def test_invalid_language_rejected(client, language):
    resp = post(client, make_wav(0.5), language=language)
    assert resp.status_code == 400
    assert "language" in resp.json()["detail"]


def test_empty_language_rejected_as_4xx(client):
    resp = post(client, make_wav(0.5), language="")
    assert resp.status_code in (400, 422)


def test_language_with_whitespace_accepted(client):
    resp = post(client, make_wav(0.5), language=" ZH ")
    assert resp.status_code == 200
    assert resp.json()["language"] == "zh"


def test_missing_language_field(client):
    resp = post(client, make_wav(0.5), include_language=False)
    assert resp.status_code == 422  # FastAPI validation error (4xx)


def test_missing_file_field(client):
    resp = post(client, b"", include_file=False)
    assert resp.status_code == 422


# ------------------------------------------------------------ invalid files
def test_empty_file_rejected(client):
    resp = post(client, b"")
    assert resp.status_code == 400
    assert "空" in resp.json()["detail"]


def test_garbage_bytes_rejected(client):
    resp = post(client, b"this is definitely not audio data " * 20)
    assert resp.status_code == 400
    assert "解码" in resp.json()["detail"]


def test_text_file_named_wav_rejected(client):
    resp = post(client, ("hello world\n" * 50).encode(), filename="fake.wav")
    assert resp.status_code == 400


def test_wav_header_without_frames_rejected(client):
    resp = post(client, make_wav(0.0))
    assert resp.status_code == 400


def test_oversize_file_rejected(client):
    resp = post(client, b"\x00" * (10 * 1024 * 1024 + 1))
    assert resp.status_code == 413
    assert "10 MB" in resp.json()["detail"]


# --------------------------------------------------------- model failures
def test_model_runtime_error_maps_to_500():
    from fastapi.testclient import TestClient

    from app.main import create_app

    service = FakeASRService(error=RuntimeError("boom"))
    with TestClient(create_app(asr_service=service), raise_server_exceptions=False) as c:
        resp = post(c, make_wav(0.5))
    assert resp.status_code == 500
    # understandable message from the catch-all handler; frontend offers retry
    assert "服务器内部错误" in resp.json()["detail"]


def test_model_not_ready_maps_to_503():
    from app.errors import AppError

    service = FakeASRService(error=AppError("语音识别模型加载失败：mock", 503))
    with make_client(service) as c:
        resp = post(c, make_wav(0.5))
    assert resp.status_code == 503
    assert "模型" in resp.json()["detail"]
