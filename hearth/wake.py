"""Wake phrases: the keyword spotter's hits become state actions."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from .bus import EventBus
from .capture.ring import Cursor
from .events import WakeEvent

log = logging.getLogger(__name__)


class WakeWorker(threading.Thread):
    def __init__(self, cursor: Cursor, spotter, actions: dict[str, str], bus: EventBus) -> None:
        """`actions` maps a normalised phrase (as the spotter reports it) to an action."""
        super().__init__(name="wake", daemon=True)
        self.cursor, self.spotter, self.actions, self.bus = cursor, spotter, actions, bus
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.is_set():
            _, data = self.cursor.read(timeout=0.5)
            if len(data):
                self.process(data)

    def process(self, data) -> None:
        for phrase in self.spotter.feed(data):
            action = self.actions.get(phrase)
            if action:
                log.info("wake phrase %r -> %s", phrase, action)
                self.bus.publish(WakeEvent(phrase, action, datetime.now(timezone.utc)))
