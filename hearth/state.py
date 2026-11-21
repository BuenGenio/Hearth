"""The two switches (recording, guard) and the live indicator.

State changes come from MQTT, wake phrases and the HTTP API; all of them go
through here. The state is persisted so a restart restores it.
"""
from __future__ import annotations

import json
import logging
import os
import threading

from .bus import EventBus
from .events import MicHealth, StateChanged

log = logging.getLogger(__name__)


class State:
    def __init__(self, bus: EventBus, path: str, guard_default: bool = True, restore_recording: bool = True) -> None:
        self.bus, self.path = bus, path
        self._lock = threading.RLock()
        self.recording = False
        self.guard = guard_default
        self.mic_connected = False
        self.mic_signal = False
        try:
            with open(path) as f:
                saved = json.load(f)
            self.guard = bool(saved.get("guard", guard_default))
            if restore_recording:
                self.recording = bool(saved.get("recording", False))
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            log.warning("could not read %s; starting with recording off", path)
        bus.subscribe(MicHealth, self._on_mic)

    @property
    def live(self) -> bool:
        """True exactly when captured audio can leave the ring buffer."""
        return self.recording and self.mic_connected and self.mic_signal

    def is_live(self) -> bool:
        """`live`, read under the lock: a caller never sees recording ON before
        the change has been published (and the indicator switched on)."""
        with self._lock:
            return self.live

    def snapshot(self) -> StateChanged:
        with self._lock:
            return StateChanged(self.recording, self.guard, self.live)

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w") as f:
                json.dump({"recording": self.recording, "guard": self.guard}, f)
            os.replace(tmp, self.path)
        except OSError:
            log.exception("could not persist state")

    def _on_mic(self, e: MicHealth) -> None:
        with self._lock:
            before = self.live
            self.mic_connected, self.mic_signal = e.connected, e.signal
            if self.live != before:
                self.bus.publish(self.snapshot())

    def set_recording(self, on: bool, by: str = "") -> None:
        with self._lock:
            if on == self.recording:
                return
            self.recording = on
            self._save()
            log.info("recording %s (%s)", "ON" if on else "OFF", by or "unknown")
            # Published while holding the lock: subscribers see ON (and the live
            # indicator goes on) before the segmenter can queue anything.
            self.bus.publish(self.snapshot())

    def set_guard(self, on: bool, by: str = "") -> None:
        with self._lock:
            if on == self.guard:
                return
            self.guard = on
            self._save()
            log.info("guard %s (%s)", "ON" if on else "OFF", by or "unknown")
            self.bus.publish(self.snapshot())

    def apply_action(self, action: str, by: str = "") -> None:
        if action == "recording_on":
            self.set_recording(True, by)
        elif action == "recording_off":
            self.set_recording(False, by)
        elif action == "recording_toggle":
            self.set_recording(not self.recording, by)
        else:
            log.warning("unknown wake action %r", action)
