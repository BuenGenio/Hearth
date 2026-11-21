"""Sound guard: YAMNet scores -> smoothed category events."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from .bus import EventBus
from .capture.ring import Cursor
from .config import GuardCategory
from .events import SoundEvent

log = logging.getLogger(__name__)
WINDOW = 15600


class GuardLogic:
    """Pure decision logic: per category, N hits in the last M windows, then a cooldown."""

    def __init__(self, categories: dict[str, GuardCategory], labels: list[str]) -> None:
        index = {name: i for i, name in enumerate(labels)}
        self.cats: dict[str, tuple[GuardCategory, list[int]]] = {}
        for name, cat in categories.items():
            if not cat.classes:                 # `classes = []` in the config switches a category off
                continue
            missing = [c for c in cat.classes if c not in index]
            if missing:
                raise ValueError(f"guard category {name!r}: model has no class named {missing}")
            self.cats[name] = (cat, [index[c] for c in cat.classes])
        self._hits = {name: deque(maxlen=cat.window) for name, (cat, _) in self.cats.items()}
        self._last_fire: dict[str, float] = {}

    def update(self, scores: np.ndarray, now: float) -> list[tuple[str, float]]:
        fired = []
        for name, (cat, idx) in self.cats.items():
            score = float(scores[idx].max())
            hits = self._hits[name]
            hits.append(score if score >= cat.threshold else 0.0)
            if sum(1 for h in hits if h) >= cat.min_hits:
                if now - self._last_fire.get(name, -1e18) >= cat.cooldown_seconds:
                    self._last_fire[name] = now
                    fired.append((name, max(hits)))
                    hits.clear()
        return fired


class GuardWorker(threading.Thread):
    """Runs the classifier on the ring whenever `enabled()` is true."""

    def __init__(self, cursor: Cursor, model, logic: GuardLogic, bus: EventBus, enabled, hop_seconds: float = 0.24375,
                 clock=time.monotonic) -> None:
        super().__init__(name="guard", daemon=True)
        self.cursor, self.model, self.logic, self.bus, self.enabled = cursor, model, logic, bus, enabled
        self.hop = max(1, int(hop_seconds * 16000))
        self.clock = clock
        self._halt = threading.Event()
        self._buf = np.zeros(0, dtype=np.float32)

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.is_set():
            _, data = self.cursor.read(timeout=0.5)
            if not self.enabled():
                self._buf = self._buf[:0]
                continue
            if len(data):
                self.process(data)

    def process(self, data: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, data])
        while len(self._buf) >= WINDOW:
            scores = self.model.scores(self._buf[:WINDOW])
            for category, score in self.logic.update(scores, self.clock()):
                log.warning("guard: %s (score %.2f)", category, score)
                self.bus.publish(SoundEvent(category, round(score, 3), datetime.now(timezone.utc)))
            self._buf = self._buf[self.hop :]
