"""WhisperASRService unit tests with an injected stub model (no download, no network)."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.asr.whisper import WhisperASRService
from app.errors import AppError

AUDIO = np.zeros(16000, dtype=np.float32)


def make_ready_service(segments=None, error=None, detected_language="en"):
    svc = WhisperASRService()
    svc._model = StubModel(segments=segments, error=error, detected_language=detected_language)
    with svc._lock:
        svc._state = "ready"
    return svc


class StubModel:
    def __init__(self, segments, error, detected_language="en"):
        self.segments = segments or []
        self.error = error
        self.detected_language = detected_language
        self.calls = []

    def transcribe(self, audio, language=None, **kwargs):
        self.calls.append({"language": language, "kwargs": kwargs, "samples": audio.size})
        if self.error is not None:
            raise self.error
        info = SimpleNamespace(language=self.detected_language, language_probability=0.99)
        return iter(self.segments), info


def test_transcribe_passes_language_and_vad():
    segs = [SimpleNamespace(text=" 你好，"), SimpleNamespace(text="世界。")]
    svc = make_ready_service(segments=segs)
    result = svc.transcribe(AUDIO, "zh")
    assert result.text == "你好，世界。"
    assert result.inference_seconds >= 0
    call = svc._model.calls[0]
    assert call["language"] == "zh"
    assert call["kwargs"]["vad_filter"] is True
    assert call["kwargs"]["initial_prompt"]  # simplified-Chinese bias prompt
    assert call["samples"] == AUDIO.size


def test_transcribe_auto_passes_none_language():
    svc = make_ready_service(segments=[SimpleNamespace(text="hello")], detected_language="en")
    svc.transcribe(AUDIO, None)
    assert len(svc._model.calls) == 1  # non-Chinese: no re-run
    assert svc._model.calls[0]["language"] is None


def test_transcribe_auto_detected_chinese_reruns_with_simplified_prompt():
    svc = make_ready_service(segments=[SimpleNamespace(text="你好")], detected_language="zh")
    result = svc.transcribe(AUDIO, None)
    assert len(svc._model.calls) == 2
    assert svc._model.calls[0]["language"] is None      # detection pass
    second = svc._model.calls[1]
    assert second["language"] == "zh"                   # simplified-Chinese redo
    assert second["kwargs"]["initial_prompt"]
    assert result.text == "你好"


def test_inference_error_wrapped_as_500():
    svc = make_ready_service(error=RuntimeError("core dumped"))
    with pytest.raises(AppError) as exc:
        svc.transcribe(AUDIO, "en")
    assert exc.value.status_code == 500
    assert "语音识别失败" in exc.value.message


def test_failed_model_state_gives_503():
    svc = WhisperASRService()
    with svc._lock:
        svc._state = "failed"
        svc._error = "mock 下载失败"
    with pytest.raises(AppError) as exc:
        svc.transcribe(AUDIO, "zh")
    assert exc.value.status_code == 503
    assert "模型加载失败" in exc.value.message


def _stub_whisper_module(monkeypatch, fake_cls):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=fake_cls)
    )


def test_load_prefers_cached_weights_and_skips_network(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)

    _stub_whisper_module(monkeypatch, FakeWhisperModel)
    svc = WhisperASRService()
    svc._load()
    assert svc.status == "ready"
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True  # cache hit -> never goes online


def test_load_falls_back_to_online_download_when_no_cache(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("no cache")

    _stub_whisper_module(monkeypatch, FakeWhisperModel)
    svc = WhisperASRService()
    svc._load()
    assert svc.status == "ready"
    assert len(calls) == 2
    assert calls[0]["local_files_only"] is True
    assert "local_files_only" not in calls[1]  # second attempt goes online


def test_load_failure_sets_failed_state(monkeypatch):
    class FakeWhisperModel:
        def __init__(self, name, **kwargs):
            raise RuntimeError("network unreachable")

    _stub_whisper_module(monkeypatch, FakeWhisperModel)
    svc = WhisperASRService()
    svc._load()
    assert svc.status == "failed"
    assert "network unreachable" in svc._error
