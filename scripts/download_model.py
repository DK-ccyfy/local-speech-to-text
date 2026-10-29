"""Pre-download the ASR model so the app can run fully offline afterwards.

Usage:
    python -m scripts.download_model

The weights are cached by HuggingFace under the user's home directory
(~/.cache/huggingface) — nothing lands inside the repository.
"""

import logging
import sys
import time

from app.asr.whisper import WhisperASRService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main() -> int:
    service = WhisperASRService()
    print("Downloading/loading ASR model (first run only)…", flush=True)
    service.warm_up()
    if not service._ready_event.wait(timeout=1800):
        print("Timed out waiting for the model to load.")
        return 1
    if service.status != "ready":
        print(f"Model failed to load: {service._error}")
        return 1
    print(f"Model ready ({service.status}). Offline recognition is now possible.")
    return 0


if __name__ == "__main__":
    started = time.monotonic()
    code = main()
    print(f"took {time.monotonic() - started:.1f}s")
    sys.exit(code)
