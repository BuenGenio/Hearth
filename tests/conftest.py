"""Shared fakes. They stand in for the model runtimes, the STT service and the
MQTT broker so the pipeline's own logic can be tested anywhere."""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from hearth.capture.sources import SourceLost
from hearth.stt import SttError

RATE = 16000
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def tone(seconds: float, level: float) -> np.ndarray:
    return np.full(int(seconds * RATE), level, dtype=np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


class ScriptSource:
    """Plays a fixed array in 100 ms blocks, quickly, then idles until closed."""

    def __init__(self, audio: np.ndarray, pace: float = 0.002, then_lost: bool = False) -> None:
        self.audio, self.pace, self.then_lost = audio, pace, then_lost
        self.closed = threading.Event()

    def blocks(self):
        for k in range(0, len(self.audio), 1600):
            if self.closed.is_set():
                return
            yield self.audio[k : k + 1600]
            time.sleep(self.pace)
        if self.then_lost:
            raise SourceLost("script ended")
        self.closed.wait()

    def close(self) -> None:
        self.closed.set()


class FakeVad:
    """Speech is anything louder than 0.2; 0.3 s of quiet ends a segment."""

    def __init__(self) -> None:
        self.n = 0
        self.start = None
        self.buf: list[np.ndarray] = []
        self.quiet = 0

    def feed(self, samples):
        out = []
        for k in range(0, len(samples), 160):
            frame = samples[k : k + 160]
            loud = bool(np.max(np.abs(frame)) > 0.2)
            if loud:
                if self.start is None:
                    self.start, self.buf = self.n, []
                self.quiet = 0
            elif self.start is not None:
                self.quiet += len(frame)
            if self.start is not None:
                self.buf.append(frame)
                if self.quiet >= 4800:
                    out.append((self.start, np.concatenate(self.buf)))
                    self.start = None
            self.n += len(frame)
        return out

    def flush(self):
        if self.start is None:
            return []
        seg = (self.start, np.concatenate(self.buf))
        self.start = None
        return [seg]


class FakeYamnet:
    labels = ["Speech", "Shatter", "Breaking", "Screaming", "Smoke detector, smoke alarm", "Fire alarm",
              "Crying, sobbing", "Baby cry, infant cry", "Alarm", "Siren", "Civil defense siren", "Car alarm", "Buzzer"]

    def __init__(self) -> None:
        self.calls = 0

    def scores(self, window):
        self.calls += 1
        s = np.zeros(len(self.labels), dtype=np.float32)
        if np.max(np.abs(window)) >= 0.99:      # a full-scale burst stands for breaking glass
            s[self.labels.index("Shatter")] = 0.9
        return s


class FakeSpotter:
    """A stretch held at exactly 0.05 stands for 'hearth start recording', 0.07 for 'stop'."""

    def feed(self, samples):
        hits = []
        if np.isclose(samples, 0.05).sum() >= 400:
            hits.append("HEARTH START RECORDING")
        if np.isclose(samples, 0.07).sum() >= 400:
            hits.append("HEARTH STOP RECORDING")
        return hits


class FakeStt:
    def __init__(self, name="fake", text="hello world", fail=False, available=True) -> None:
        self.name, self.text, self.fail, self.available = name, text, fail, available
        self.calls: list[bytes] = []

    def transcribe(self, wav):
        self.calls.append(wav)
        if self.fail:
            raise SttError(f"{self.name}: down")
        return self.text, "en"


class FakeMqtt:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, bool]] = []
        self.subscribed = []
        self.will = None
        self.on_connect = self.on_message = None

    def will_set(self, topic, payload, qos=0, retain=False):
        self.will = (topic, payload, retain)

    def connect_async(self, host, port, keepalive=60):
        self.host = host

    def loop_start(self):
        self.on_connect(self, None, None, 0, None)

    def loop_stop(self): ...
    def disconnect(self): ...

    def subscribe(self, topics):
        self.subscribed += topics

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, retain))

    def last(self, topic):
        return next((p for t, p, _ in reversed(self.published) if t == topic), None)

    def message(self, topic, payload):
        class M: ...
        m = M()
        m.topic, m.payload = topic, payload.encode()
        self.on_message(self, None, m)


class FakeLlm:
    name = "fake"
    model = "fake-1"

    def __init__(self, reply="- did things", fail=False) -> None:
        self.reply, self.fail, self.calls = reply, fail, []

    def complete(self, system, user):
        from hearth.llm import LlmError

        self.calls.append((system, user))
        if self.fail:
            raise LlmError("fake: down")
        return self.reply


def wait_for(cond, timeout=5.0, step=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def models_dir():
    d = os.environ.get("HEARTH_MODELS") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    if not os.path.exists(os.path.join(d, "yamnet.tflite")):
        pytest.skip("models not downloaded (run tools/fetch_models.sh)")
    return d
