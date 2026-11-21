"""The capture loop: source -> ring buffer, with reconnect and mic health."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from .ring import RingBuffer
from .sources import Source, SourceLost

log = logging.getLogger(__name__)


class Capture(threading.Thread):
    """Reads a source into the ring forever, reopening it when it is lost.

    `on_health(connected, signal)` is called whenever either value changes.
    `signal` is False when the input has been flat (below `silence_floor`) for
    `silence_seconds`: an unplugged capsule or a muted transmitter, which also
    leaves the sound guard deaf.
    """

    def __init__(self, make_source: Callable[[], Source], ring: RingBuffer,
                 on_health: Callable[[bool, bool], None] | None = None,
                 silence_floor: float = 1e-4, silence_seconds: float = 10.0,
                 retry_min: float = 1.0, retry_max: float = 30.0,
                 process: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        super().__init__(name="capture", daemon=True)
        self.make_source, self.ring = make_source, ring
        self.on_health = on_health or (lambda connected, signal: None)
        self.silence_floor, self.silence_seconds = silence_floor, silence_seconds
        self.retry_min, self.retry_max = retry_min, retry_max
        self.process = process          # gain stage; mic health is judged on the raw signal
        self.connected = False
        self.signal = False
        self._halt = threading.Event()
        self._source: Source | None = None

    def stop(self) -> None:
        self._halt.set()
        if self._source is not None:
            self._source.close()

    def _set_health(self, connected: bool, signal: bool) -> None:
        if (connected, signal) != (self.connected, self.signal):
            self.connected, self.signal = connected, signal
            self.on_health(connected, signal)

    def run(self) -> None:
        delay = self.retry_min
        last_error = None
        while not self._halt.is_set():
            try:
                self._source = self.make_source()
                last_loud = None
                for block in self._source.blocks():
                    if self._halt.is_set():
                        return
                    if last_error is not None:
                        log.info("audio source is back")
                        last_error = None
                    delay = self.retry_min
                    self.ring.write(block if self.process is None else self.process(block))
                    now = time.monotonic()
                    if last_loud is None or float(np.max(np.abs(block))) > self.silence_floor:
                        last_loud = now
                    self._set_health(True, now - last_loud < self.silence_seconds)
                raise SourceLost("source ended")
            except SourceLost as e:
                if not self._halt.is_set() and str(e) != last_error:
                    last_error = str(e)             # say it once, not every retry
                    log.warning("audio source unavailable: %s (retrying quietly)", e)
            except Exception:  # noqa: BLE001
                log.exception("capture failed; retrying in %.0f s", delay)
            self._set_health(False, False)
            if self._halt.wait(delay):
                return
            delay = min(delay * 2, self.retry_max)
