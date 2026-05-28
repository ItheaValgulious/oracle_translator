"""Speech-to-Text module using sherpa-onnx + sounddevice."""

from __future__ import annotations

from pathlib import Path
from threading import Thread, Event

import numpy as np

from src.game import config as cfg


class SpeechToText:
    """Speech recognition using sherpa-onnx.

    Usage:
        stt = SpeechToText()
        if stt.available:
            stt.start()
            # ... user speaks ...
            text = stt.stop()
    """

    def __init__(self, model_dir: str | None = None) -> None:
        self._recognizer = None
        self._available = False
        self._chunks: list[np.ndarray] = []
        self._stop_event = Event()
        self._thread: Thread | None = None

        resolved = Path(model_dir or cfg.STT_MODEL_DIR)
        if not resolved.is_absolute():
            resolved = Path(__file__).resolve().parents[2] / resolved

        encoder = resolved / "encoder-epoch-99-avg-1.int8.onnx"
        decoder = resolved / "decoder-epoch-99-avg-1.onnx"
        joiner = resolved / "joiner-epoch-99-avg-1.int8.onnx"
        tokens = resolved / "tokens.txt"

        if not all(f.exists() for f in [encoder, decoder, joiner, tokens]):
            print(f"[STT] Model files not found in {resolved}. STT disabled.")
            return

        try:
            import sherpa_onnx
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens),
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                num_threads=cfg.STT_NUM_THREADS,
            )
            self._available = True
            print("[STT] sherpa-onnx model loaded successfully.")
        except Exception as e:
            print(f"[STT] Failed to load model: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        """Start recording from microphone in a background thread."""
        if not self._available:
            return
        self._chunks.clear()
        self._stop_event.clear()

        def _record():
            import sounddevice as sd
            blocksize = 4000  # ~250ms at 16kHz
            with sd.InputStream(
                samplerate=cfg.STT_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
            ) as stream:
                while not self._stop_event.is_set():
                    data, overflowed = stream.read(blocksize)
                    self._chunks.append(data[:, 0].copy())

        self._thread = Thread(target=_record, daemon=True)
        self._thread.start()

    def stop(self) -> str:
        """Stop recording and return transcribed text."""
        if not self._available or self._thread is None:
            return ""

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

        if not self._chunks:
            return ""

        samples = np.concatenate(self._chunks)
        self._chunks.clear()
        return self._transcribe(samples)

    def _transcribe(self, samples: np.ndarray) -> str:
        """Transcribe a numpy float32 array of audio samples."""
        if not self._available or len(samples) == 0:
            return ""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(cfg.STT_SAMPLE_RATE, samples)
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        return self._recognizer.get_result(stream).strip()
