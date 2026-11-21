"""Signal features shared by detectors. No I/O, no model runtimes."""
from __future__ import annotations

import numpy as np


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if len(x) else 0.0


def rms_dbfs(x: np.ndarray) -> float:
    """Level in dB relative to full scale; -120 for digital silence."""
    if not len(x):
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return max(-120.0, 20 * np.log10(r)) if r > 0 else -120.0


class Agc:
    """Slow automatic gain for a microphone whose level is too low.

    The detectors (wake phrases, VAD, YAMNet) were trained on audio at normal
    speaking level; a quiet input makes them miss. This follows the loudest
    recent block and raises the gain so that level reaches `target_dbfs`,
    never above `max_gain_db` and never below unity. A sudden loud sound gets
    its lower gain in the same block, so the gain change itself cannot clip it.
    """

    def __init__(self, target_dbfs: float = -20.0, max_gain_db: float = 30.0, half_life: float = 20.0,
                 block_seconds: float = 0.1) -> None:
        self.target = 10 ** (target_dbfs / 20)
        self.max_gain = 10 ** (max_gain_db / 20)
        self.decay = 0.5 ** (block_seconds / half_life)
        self.env = 0.0
        self.gain = 1.0

    def process(self, block: np.ndarray) -> np.ndarray:
        if not len(block):
            return block
        level = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        self.env = max(level, self.env * self.decay)
        self.gain = min(self.max_gain, max(1.0, self.target / self.env)) if self.env > 0 else self.max_gain
        if self.gain == 1.0:
            return block
        return np.clip(block * self.gain, -1.0, 1.0).astype(np.float32)
