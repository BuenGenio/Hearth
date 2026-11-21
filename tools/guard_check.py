"""Run the sound guard over WAV files and report what it would have raised.

    python tools/guard_check.py models/yamnet.tflite clip1.wav clip2.wav ...

For each file: the guard events (with the default or configured thresholds) and
the three strongest YAMNet classes, so thresholds can be tuned on real audio.
"""
from __future__ import annotations

import sys

import numpy as np

from hearth import config as config_mod
from hearth.capture.sources import read_wav
from hearth.guard import GuardLogic
from hearth.models.yamnet import WINDOW, Yamnet


def check(model: Yamnet, path: str, categories, hop: int = 3900) -> tuple[list[str], list[tuple[str, float]]]:
    audio = read_wav(path)
    logic = GuardLogic(categories, model.labels)
    fired, peak = [], np.zeros(len(model.labels), dtype=np.float32)
    padded = np.concatenate([audio, np.zeros(WINDOW, np.float32)])
    for i, k in enumerate(range(0, len(audio), hop)):
        scores = model.scores(padded[k : k + WINDOW])
        peak = np.maximum(peak, scores)
        fired += [name for name, _ in logic.update(scores, i * hop / 16000)]
    top = [(model.labels[j], float(peak[j])) for j in np.argsort(peak)[::-1][:3]]
    return fired, top


def main() -> None:
    model = Yamnet(sys.argv[1])
    cfg = config_mod.load()
    categories, hop = cfg.guard.categories, int(cfg.guard.hop_seconds * 16000)
    for path in sys.argv[2:]:
        fired, top = check(model, path, categories, hop)
        tops = ", ".join(f"{n} {s:.2f}" for n, s in top)
        print(f"{path.rsplit('/', 1)[-1]:28s} -> {', '.join(fired) or '-':24s} | {tops}")


if __name__ == "__main__":
    main()
