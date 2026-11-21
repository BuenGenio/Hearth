"""YAMNet (AudioSet sound classes) through LiteRT.

Input: 15600 samples (0.975 s) of 16 kHz mono float32 in [-1, 1].
Output: 521 class scores. The class names ship inside the model file.
"""
from __future__ import annotations

import zipfile

import numpy as np

WINDOW = 15600


def read_labels(model_path: str) -> list[str]:
    """The MediaPipe build of YAMNet embeds its label list as a zip member."""
    with zipfile.ZipFile(model_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        return z.read(name).decode().splitlines()


class Yamnet:
    def __init__(self, model_path: str, threads: int = 1) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.labels = read_labels(model_path)
        self._it = Interpreter(model_path=model_path, num_threads=threads)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        self._out = self._it.get_output_details()[0]["index"]

    def scores(self, window: np.ndarray) -> np.ndarray:
        if len(window) != WINDOW:
            raise ValueError(f"YAMNet needs {WINDOW} samples, got {len(window)}")
        self._it.set_tensor(self._in["index"], window.astype(np.float32).reshape(self._in["shape"]))
        self._it.invoke()
        return self._it.get_tensor(self._out).reshape(-1)
