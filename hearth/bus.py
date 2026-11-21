"""A tiny synchronous publish/subscribe bus.

Detectors publish; MQTT, storage and anything added later (an OSC or MIDI path,
say) subscribe. A failing subscriber is logged and does not affect the others.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[type, list[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: type, handler: Callable[[Any], None]) -> None:
        with self._lock:
            self._subs[event_type].append(handler)

    def publish(self, event: Any) -> None:
        with self._lock:
            handlers = list(self._subs.get(type(event), ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - one bad subscriber must not stop the rest
                log.exception("subscriber %r failed on %s", handler, type(event).__name__)
