"""Phase 0 probe: does every native dependency run on this CPU, and how fast?

Each check runs in its own subprocess so that an `Illegal instruction` (SIGILL)
in one library is reported instead of killing the whole probe.

Usage: python probe.py <models_dir> <speech.wav>
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap

MODELS, WAV = sys.argv[1], sys.argv[2]

COMMON = f"""
import time, wave, numpy as np
MODELS = {MODELS!r}
def load(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return sr, x
sr, x = load({WAV!r})
import soxr
x16 = soxr.resample(x, sr, 16000).astype(np.float32)
"""

CHECKS = {
    "numpy+soxr": """
t = time.perf_counter()
rs = soxr.ResampleStream(48000, 16000, 1, dtype="float32")
n = 0
blk = np.random.randn(4800).astype(np.float32) * 0.1
for _ in range(600):   # 60 s of 48 kHz audio in 100 ms blocks
    n += len(rs.resample_chunk(blk))
dt = time.perf_counter() - t
print({"out_samples": n, "cpu_fraction": round(dt / 60, 4)})
""",
    "sqlite-fts5": """
import sqlite3
c = sqlite3.connect(":memory:")
c.execute("create virtual table t using fts5(body)")
c.execute("insert into t values ('call the plumber before friday')")
print({"sqlite": sqlite3.sqlite_version, "hit": c.execute("select count(*) from t where t match 'plumber'").fetchone()[0]})
""",
    "litert-yamnet": """
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=MODELS + "/yamnet.tflite", num_threads=1)
it.allocate_tensors()
i = it.get_input_details()[0]; o = it.get_output_details()
w = np.zeros(15600, np.float32); w[:min(15600, len(x16))] = x16[:15600]
it.set_tensor(i["index"], w.reshape(i["shape"])); it.invoke()
t = time.perf_counter()
for _ in range(40):
    it.set_tensor(i["index"], w.reshape(i["shape"])); it.invoke()
ms = (time.perf_counter() - t) / 40 * 1000
scores = it.get_tensor(o[0]["index"]).reshape(-1)
print({"input_shape": i["shape"].tolist(), "outputs": [d["shape"].tolist() for d in o],
       "ms_per_inference": round(ms, 1), "top_class_index": int(scores.argmax()), "top_score": round(float(scores.max()), 3)})
""",
    "sherpa-vad": """
import sherpa_onnx
cfg = sherpa_onnx.VadModelConfig()
cfg.silero_vad.model = MODELS + "/silero_vad.onnx"
cfg.silero_vad.min_silence_duration = 0.5
cfg.sample_rate = 16000
vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
pad = np.zeros(16000, np.float32)
sig = np.concatenate([pad, x16, pad, pad])
t = time.perf_counter(); segs = []
for k in range(0, len(sig) - 512, 512):
    vad.accept_waveform(sig[k:k + 512])
    while not vad.empty():
        segs.append((vad.front.start, len(vad.front.samples))); vad.pop()
dt = time.perf_counter() - t
print({"version": sherpa_onnx.__version__, "segments": segs, "cpu_fraction": round(dt / (len(sig) / 16000), 4)})
""",
    "sherpa-kws": """
import sherpa_onnx, glob
d = MODELS + "/kws"
def one(pat):
    return sorted(glob.glob(d + "/" + pat))[0]
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=d + "/bpe.model")
kw = [sp.encode(p, out_type=str) for p in ["HEARTH START RECORDING", "HEARTH STOP RECORDING"]]
open("/tmp/kw.txt", "w").write("\\n".join(" ".join(k) for k in kw) + "\\n")
spotter = sherpa_onnx.KeywordSpotter(
    tokens=d + "/tokens.txt",
    encoder=one("encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
    decoder=one("decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
    joiner=one("joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
    num_threads=1, keywords_file="/tmp/kw.txt", provider="cpu")
s = spotter.create_stream()
sig = np.concatenate([np.zeros(8000, np.float32), x16, np.zeros(16000, np.float32)])
hits = []; t = time.perf_counter()
for k in range(0, len(sig), 1600):
    s.accept_waveform(16000, sig[k:k + 1600])
    while spotter.is_ready(s):
        spotter.decode_stream(s)
        r = spotter.get_result(s)
        if r:
            hits.append(r); spotter.reset_stream(s)
dt = time.perf_counter() - t
print({"tokens": [" ".join(k) for k in kw], "hits": hits, "cpu_fraction": round(dt / (len(sig) / 16000), 4)})
""",
    "web-stack": """
import fastapi, pydantic, uvicorn, paho.mqtt.client as m, httpx
print({"fastapi": fastapi.__version__, "pydantic": pydantic.__version__, "paho": m.__version__ if hasattr(m, "__version__") else "ok", "httpx": httpx.__version__})
""",
}

results = {}
for name, body in CHECKS.items():
    p = subprocess.run([sys.executable, "-c", COMMON + textwrap.dedent(body)], capture_output=True, text=True, timeout=600)
    if p.returncode == 0:
        results[name] = {"ok": True, "out": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}
    else:
        sig = signal.Signals(-p.returncode).name if p.returncode < 0 else f"exit {p.returncode}"
        results[name] = {"ok": False, "why": sig, "stderr": p.stderr.strip()[-600:]}
    print(name, "→", json.dumps(results[name]), flush=True)

sys.exit(0 if all(r["ok"] for r in results.values()) else 1)
