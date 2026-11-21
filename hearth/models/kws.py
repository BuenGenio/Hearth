"""Open-vocabulary keyword spotting with sherpa-onnx.

Phrases are plain text: they are split into the model's BPE tokens here, so a
new wake phrase needs a config change, not a trained model.
"""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np


def normalise(phrase: str) -> str:
    """Upper-case and strip punctuation, the form the English model was trained on."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z' ]+", " ", phrase)).strip().upper()


class SherpaKws:
    def __init__(self, model_dir: str, phrases: list[str], threshold: float = 0.25,
                 boost: float = 1.5, threads: int = 1) -> None:
        import sentencepiece as spm
        import sherpa_onnx

        def model(prefix: str) -> str:
            hits = sorted(f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".onnx")
                          and "int8" not in f)
            if not hits:
                raise FileNotFoundError(f"no {prefix}*.onnx in {model_dir}")
            return os.path.join(model_dir, hits[0])

        sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "bpe.model"))
        self.phrases = [normalise(p) for p in phrases]
        lines = [" ".join(sp.encode(p, out_type=str)) for p in self.phrases]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines) + "\n")
        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=os.path.join(model_dir, "tokens.txt"),
                encoder=model("encoder"), decoder=model("decoder"), joiner=model("joiner"),
                num_threads=threads, keywords_file=f.name,
                keywords_score=boost, keywords_threshold=threshold, provider="cpu")
        finally:
            os.unlink(f.name)
        self._stream = self._spotter.create_stream()

    def feed(self, samples: np.ndarray) -> list[str]:
        """Feed audio; returns the normalised phrases recognised in it."""
        hits = []
        self._stream.accept_waveform(16000, samples)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                hits.append(normalise(result))
                self._spotter.reset_stream(self._stream)
        return hits
