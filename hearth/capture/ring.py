"""RAM-only ring buffer of 16 kHz mono float32 audio.

There is exactly one copy of recent audio in the process, and it is here. Samples
are addressed by an absolute index that only grows. Readers hold a `Cursor`; a
reader that falls further behind than the ring's capacity skips ahead and the
skipped samples are counted, so a slow consumer can never block capture.
"""
from __future__ import annotations

import threading
import time

import numpy as np


class RingBuffer:
    def __init__(self, seconds: float = 30.0, rate: int = 16000) -> None:
        self.rate = rate
        self.capacity = int(seconds * rate)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._total = 0  # absolute index one past the newest sample
        self._last_write = time.time()
        self._cond = threading.Condition()
        self._closed = False

    @property
    def total(self) -> int:
        with self._cond:
            return self._total

    def write(self, block: np.ndarray) -> int:
        """Append a block; returns the absolute index of its first sample."""
        block = np.asarray(block, dtype=np.float32)
        with self._cond:
            if len(block) > self.capacity:          # keep the newest part; the index still advances in full
                self._total += len(block) - self.capacity
                block = block[-self.capacity :]
            start = self._total
            pos = start % self.capacity
            first = min(len(block), self.capacity - pos)
            self._buf[pos : pos + first] = block[:first]
            if first < len(block):
                self._buf[: len(block) - first] = block[first:]
            self._total += len(block)
            self._last_write = time.time()
            self._cond.notify_all()
            return start

    def read(self, start: int, end: int) -> tuple[int, np.ndarray]:
        """Copy out [start, end), clamped to what the ring still holds."""
        with self._cond:
            return self._read_locked(start, end)

    def _read_locked(self, start: int, end: int) -> tuple[int, np.ndarray]:
        end = min(end, self._total)
        start = max(start, self._total - self.capacity, 0)
        if end <= start:
            return start, np.zeros(0, dtype=np.float32)
        a, b = start % self.capacity, end % self.capacity
        if a < b:
            return start, self._buf[a:b].copy()
        return start, np.concatenate([self._buf[a:], self._buf[:b]])

    def time_of(self, index: int) -> float:
        """Wall-clock time (epoch seconds) at which sample `index` was captured."""
        with self._cond:
            return self._last_write - (self._total - index) / self.rate

    def cursor(self) -> "Cursor":
        return Cursor(self)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class Cursor:
    """A reader's position in the ring. Starts at "now": never reads the past."""

    def __init__(self, ring: RingBuffer) -> None:
        self.ring = ring
        self.index = ring.total
        self.skipped = 0

    def seek_to_now(self) -> int:
        self.index = self.ring.total
        return self.index

    def read(self, timeout: float = 0.5, max_samples: int | None = None) -> tuple[int, np.ndarray]:
        """Block until new audio arrives or `timeout` passes. Returns (start, samples)."""
        ring = self.ring
        with ring._cond:
            if ring._total <= self.index and not ring._closed:
                ring._cond.wait(timeout)
            end = ring._total
            if max_samples is not None:
                end = min(end, self.index + max_samples)
            start, data = ring._read_locked(self.index, end)
            if start > self.index:
                self.skipped += start - self.index
            self.index = max(end, self.index)
            return start, data
