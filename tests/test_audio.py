"""Audio decoding unit tests (app.audio)."""

import pytest

from app.audio import decode_audio
from app.errors import AppError
from tests.conftest import make_wav, make_webm


def test_decode_wav_returns_float32_mono_and_duration():
    pcm, duration = decode_audio(make_wav(1.5))
    assert pcm.dtype.name == "float32"
    assert pcm.ndim == 1
    assert abs(duration - 1.5) < 0.05
    assert 24000 * 0.95 <= pcm.size <= 24000 * 1.05  # 1.5 s @ 16 kHz
    assert float(pcm.max()) <= 1.0 and float(pcm.min()) >= -1.0


def test_decode_resamples_44k_stereo_to_16k_mono():
    pcm, duration = decode_audio(make_wav(2.0, sample_rate=44100, channels=2))
    assert abs(duration - 2.0) < 0.1
    assert 31000 <= pcm.size <= 34000  # ~2 s @16 kHz


def test_decode_webm_opus():
    pcm, duration = decode_audio(make_webm(1.0))
    assert duration > 0.5
    assert pcm.size > 8000


def test_decode_empty_bytes_rejected():
    with pytest.raises(AppError) as exc:
        decode_audio(b"")
    assert exc.value.status_code == 400


def test_decode_garbage_rejected():
    with pytest.raises(AppError) as exc:
        decode_audio(b"RIFF----WAVEjunkjunkjunk" * 10)
    assert exc.value.status_code == 400


def test_decode_wav_without_frames_rejected():
    with pytest.raises(AppError) as exc:
        decode_audio(make_wav(0.0))
    assert exc.value.status_code == 400
