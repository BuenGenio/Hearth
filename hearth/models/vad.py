"""Voice activity detection with Silero VAD (through sherpa-onnx)."""
from __future__ import annotations

import numpy as np


class SherpaVad:
    """Feed audio, get back finished speech stretches as (start_sample, samples).

    `start_sample` counts from the first sample fed to this instance. Make a new
    instance for each recording session.
    """

    def __init__(self, model_path: str, threshold: float = 0.5, min_silence: float = 0.6,
                 min_speech: float = 0.25, max_speech: float = 30.0) -> None:
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = min_speech
        cfg.silero_vad.max_speech_duration = max_speech
        cfg.sample_rate = 16000
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max_speech * 2 + 10)
        self._window = cfg.silero_vad.window_size
        self._pending = np.zeros(0, dtype=np.float32)

    def _drain(self) -> list[tuple[int, np.ndarray]]:
        out = []
        while not self._vad.empty():
            seg = self._vad.front
            out.append((int(seg.start), np.asarray(seg.samples, dtype=np.float32)))
            self._vad.pop()
        return out

    def feed(self, samples: np.ndarray) -> list[tuple[int, np.ndarray]]:
        buf = np.concatenate([self._pending, samples]) if len(self._pending) else samples
        n = len(buf) - len(buf) % self._window
        for k in range(0, n, self._window):
            self._vad.accept_waveform(buf[k : k + self._window])
        self._pending = buf[n:].copy()
        return self._drain()

    def flush(self) -> list[tuple[int, np.ndarray]]:
        """End of the session: close any speech still open."""
        self._vad.flush()
        return self._drain()
