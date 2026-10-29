"""Audio validation and decoding.

All uploaded/recorded audio (WAV, WebM/Opus, MP3, ...) is decoded with PyAV
(bundled FFmpeg libraries, no system FFmpeg required) into 16 kHz mono
float32 PCM, which is exactly what faster-whisper expects.
"""

import io
import logging

import av
import numpy as np

from app.errors import AppError

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def decode_audio(data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Decode arbitrary audio bytes to (float32 mono PCM @sample_rate, duration_seconds).

    Raises AppError(400) for empty, corrupt or non-audio input.
    """
    if not data:
        raise AppError("上传的文件为空", 400)

    try:
        container = av.open(io.BytesIO(data))
    except Exception:
        raise AppError("无法解码音频文件：文件可能已损坏、为空或格式不受支持", 400)

    try:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise AppError("文件中不包含音频流，请上传有效的音频文件", 400)

        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks: list[np.ndarray] = []
        total = 0
        try:
            for frame in container.decode(audio_streams[0]):
                total += _resample_into(resampler, frame, chunks)
            # Flush any samples buffered inside the resampler (ignore if the
            # installed PyAV does not support flushing).
            try:
                total += _resample_into(resampler, None, chunks)
            except Exception:
                pass
        except AppError:
            raise
        except Exception as exc:
            log.warning("audio decode failed mid-stream: %s", exc)
            raise AppError(f"音频解码失败：文件可能已损坏（{exc}）", 400)

        if total == 0:
            raise AppError("音频内容为空或无法解码", 400)

        pcm_int16 = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        duration = pcm_int16.size / sample_rate
        return pcm_int16.astype(np.float32) / 32768.0, float(duration)
    finally:
        container.close()


def _resample_into(resampler, frame, chunks: list[np.ndarray]) -> int:
    """Push one frame (or None to flush) through the resampler; append PCM."""
    out = resampler.resample(frame)
    frames = out if isinstance(out, list) else ([] if out is None else [out])
    count = 0
    for resampled in frames:
        arr = np.asarray(resampled.to_ndarray()).reshape(-1)
        if arr.size:
            chunks.append(arr)
            count += arr.size
    return count
