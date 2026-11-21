"""Audio sources. Each yields 16 kHz mono float32 blocks of about 100 ms."""
from __future__ import annotations

import subprocess
import threading
import time
import wave
from typing import Iterator, Protocol

import numpy as np
import soxr

RATE = 16000
BLOCK_SECONDS = 0.1


class SourceLost(Exception):
    """The source stopped delivering audio (device unplugged, process died...)."""


class Source(Protocol):
    def blocks(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class AlsaSource:
    """Capture from an ALSA device through `arecord`.

    The device is opened at its native rate and resampled here with soxr: ALSA's
    default rate converter does not low-pass, and the aliasing hurts Guard.
    Address the device by card id (`hw:CARD=Receiver,DEV=0`), never by index.
    """

    def __init__(self, device: str, rate: int = 48000, channels: int = 1, arecord: str = "arecord") -> None:
        self.device, self.rate, self.channels, self.arecord = device, rate, channels, arecord
        self._proc: subprocess.Popen | None = None

    def blocks(self) -> Iterator[np.ndarray]:
        cmd = [self.arecord, "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
               "-c", str(self.channels), "-t", "raw", "--buffer-time=200000"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise SourceLost(f"cannot start {self.arecord}: {e}") from e
        resampler = None if self.rate == RATE else soxr.ResampleStream(self.rate, RATE, 1, dtype="float32")
        frame_bytes = 2 * self.channels
        want = int(self.rate * BLOCK_SECONDS) * frame_bytes
        out = self._proc.stdout
        assert out is not None
        try:
            while True:
                raw = out.read(want)
                if len(raw) < frame_bytes:
                    err = (self._proc.stderr.read() or b"").decode(errors="replace").strip() if self._proc.stderr else ""
                    raise SourceLost(err.splitlines()[-1] if err else "arecord ended")
                raw = raw[: len(raw) - len(raw) % frame_bytes]
                x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                if self.channels > 1:
                    x = x.reshape(-1, self.channels).mean(axis=1)
                if resampler is not None:
                    x = resampler.resample_chunk(x)
                if len(x):
                    yield x
        finally:
            self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)
            for stream in (proc.stdout, proc.stderr):
                if stream:
                    stream.close()


class FileSource:
    """Replay a WAV file: tests, fixtures, and dry runs without a microphone."""

    def __init__(self, path: str, realtime: bool = True, loop: bool = False, tail_silence: float = 0.0,
                 hold: bool = False) -> None:
        """`hold`: after the file ends, stay open and silent instead of reporting the source lost
        (which would make the capture loop reopen the file and play it again)."""
        self.path, self.realtime, self.loop, self.tail_silence, self.hold = path, realtime, loop, tail_silence, hold
        self._closed = threading.Event()

    def blocks(self) -> Iterator[np.ndarray]:
        x = read_wav(self.path)
        if self.tail_silence:
            x = np.concatenate([x, np.zeros(int(self.tail_silence * RATE), np.float32)])
        n = int(RATE * BLOCK_SECONDS)
        while True:
            t0 = time.monotonic()
            for i, k in enumerate(range(0, len(x), n)):
                if self.realtime:
                    delay = t0 + i * BLOCK_SECONDS - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield x[k : k + n]
            if not self.loop:
                if self.hold:
                    self._closed.wait()
                    return
                raise SourceLost("end of file")

    def close(self) -> None:
        self._closed.set()


def read_wav(path: str) -> np.ndarray:
    """Read a PCM16 WAV as 16 kHz mono float32."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != RATE:
        x = soxr.resample(x, sr, RATE).astype(np.float32)
    return x
