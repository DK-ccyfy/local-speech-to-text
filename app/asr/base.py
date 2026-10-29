"""ASR service abstraction.

The FastAPI app depends only on this interface, so automated tests can inject
a fake service instead of downloading the real model (requirement: `pytest -q`
must not trigger a large model download).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Transcription:
    text: str
    # Wall time of the pure model inference. Model loading/downloading is
    # deliberately excluded so `processing_seconds` matches the assignment.
    inference_seconds: float = 0.0


class ASRService(ABC):
    """A thread-safe speech-to-text service over 16 kHz mono float32 PCM."""

    @property
    def status(self) -> str:
        """One of: idle / loading / ready / failed (default: ready)."""
        return "ready"

    def warm_up(self) -> None:
        """Start loading the model in the background (no-op by default)."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None) -> Transcription:
        """Transcribe PCM audio. `language` is 'zh'/'en' or None for auto-detect."""
        raise NotImplementedError
