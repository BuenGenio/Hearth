"""Cuts speech into segments, only while recording is ON.

Privacy: this is the only component that copies audio out of the ring for
transcription. It reads nothing while recording is OFF, and when recording
turns ON it starts at "now", so nothing said before the switch can end up in a
segment, pre-roll included.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np

from .capture.ring import Cursor
from .events import Segment

log = logging.getLogger(__name__)
RATE = 16000


class Segmenter(threading.Thread):
    def __init__(self, cursor: Cursor, make_vad: Callable[[], object], recording: Callable[[], bool],
                 emit: Callable[[Segment], None], preroll: float = 0.3, source: str = "hub",
                 history_seconds: float = 90.0) -> None:
        super().__init__(name="segmenter", daemon=True)
        self.cursor, self.make_vad, self.recording, self.emit = cursor, make_vad, recording, emit
        self.preroll = int(preroll * RATE)
        self.source = source
        self.history = int(history_seconds * RATE)
        self._halt = threading.Event()
        self._vad = None
        self._base = 0                      # ring index of the first sample fed this session
        self._hist = np.zeros(0, np.float32)  # audio fed this session (bounded), for pre-roll
        self._hist_start = 0                # session-relative index of _hist[0]
        self._floor = 0                     # session-relative index: pre-roll may not reach before this

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.is_set():
            self.step(timeout=0.25)

    def step(self, timeout: float = 0.0) -> None:
        if not self.recording():
            if self._vad is not None:
                self._close_session()
            self._halt.wait(min(timeout, 0.05))   # poll often: the first word after ON matters
            return
        if self._vad is None:
            self._open_session()
        start, data = self.cursor.read(timeout=timeout)
        if len(data):
            self._feed(data)

    def _open_session(self) -> None:
        self._vad = self.make_vad()
        self._base = self.cursor.seek_to_now()
        self._hist = np.zeros(0, np.float32)
        self._hist_start = 0
        self._floor = 0

    def _close_session(self) -> None:
        for seg in self._vad.flush():
            self._emit(*seg)
        self._vad = None
        self._hist = np.zeros(0, np.float32)

    def _feed(self, data: np.ndarray) -> None:
        self._hist = np.concatenate([self._hist, data])
        if len(self._hist) > self.history:
            cut = len(self._hist) - self.history
            self._hist = self._hist[cut:]
            self._hist_start += cut
        for seg in self._vad.feed(data):
            self._emit(*seg)

    def _emit(self, start: int, samples: np.ndarray) -> None:
        pre_from = max(start - self.preroll, self._floor, self._hist_start)
        pre = self._hist[pre_from - self._hist_start : max(start, pre_from) - self._hist_start]
        if len(pre):
            samples = np.concatenate([pre, samples])
            start = pre_from
        end = start + len(samples)
        self._floor = end                   # the next segment's pre-roll must not overlap this one
        ring = self.cursor.ring
        t0 = datetime.fromtimestamp(ring.time_of(self._base + start), timezone.utc)
        self.emit(Segment(id=str(uuid.uuid4()), source=self.source, started_at=t0,
                          ended_at=t0 + timedelta(seconds=len(samples) / RATE), samples=samples))
