"""Home Assistant link: MQTT discovery, state and commands.

Everything Home Assistant knows about Hearth arrives through this module. Other
modules never import it; it subscribes to the bus.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from . import __version__
from .bus import EventBus
from .config import Mqtt as MqttConfig, secret
from .events import MicHealth, SoundEvent, StateChanged, SttStatus, SummaryReady
from .state import State
from .store import iso

log = logging.getLogger(__name__)
MAX_ATTR_CHARS = 12000      # Home Assistant drops attributes above 16 kB from its database


def discovery(cfg: MqttConfig, categories: list[str]) -> dict[str, dict[str, Any]]:
    """Discovery payloads keyed by config topic."""
    base, prefix = cfg.base_topic, cfg.discovery_prefix
    device = {"identifiers": ["hearth"], "name": "Hearth", "manufacturer": "Hearth", "model": "Hub",
              "sw_version": __version__}
    common = {"availability_topic": f"{base}/status", "device": device}

    def entity(component: str, key: str, name: str, **extra: Any) -> tuple[str, dict]:
        return (f"{prefix}/{component}/hearth/{key}/config",
                {**common, "name": name, "unique_id": f"hearth_{key}", **extra})

    diag = {"entity_category": "diagnostic"}
    return dict([
        entity("switch", "recording", "Recording", icon="mdi:record-rec",
               state_topic=f"{base}/recording/state", command_topic=f"{base}/recording/set"),
        entity("switch", "guard", "Guard", icon="mdi:shield-home",
               state_topic=f"{base}/guard/state", command_topic=f"{base}/guard/set"),
        entity("binary_sensor", "live", "Live", icon="mdi:microphone", state_topic=f"{base}/live/state"),
        entity("event", "sound", "Sound", icon="mdi:alarm-light", state_topic=f"{base}/sound/event",
               event_types=categories),
        entity("sensor", "last_summary", "Last summary", icon="mdi:text-box-outline", device_class="timestamp",
               state_topic=f"{base}/summary/state", json_attributes_topic=f"{base}/summary/attributes"),
        entity("binary_sensor", "mic_connected", "Microphone", device_class="connectivity",
               state_topic=f"{base}/mic/connected", **diag),
        entity("binary_sensor", "mic_signal", "Microphone signal", device_class="sound",
               state_topic=f"{base}/mic/signal", **diag),
        entity("sensor", "stt_backend", "Transcription backend", icon="mdi:cloud-outline",
               state_topic=f"{base}/stt/backend", **diag),
        entity("sensor", "stt_queue", "Transcription queue", icon="mdi:tray-full", state_class="measurement",
               state_topic=f"{base}/stt/queue", **diag),
        entity("sensor", "stt_dropped", "Dropped segments", icon="mdi:tray-remove", state_class="total_increasing",
               state_topic=f"{base}/stt/dropped", **diag),
    ])


def onoff(value: bool) -> str:
    return "ON" if value else "OFF"


class MqttLink:
    def __init__(self, cfg: MqttConfig, bus: EventBus, state: State, categories: list[str], client=None) -> None:
        self.cfg, self.bus, self.state, self.categories = cfg, bus, state, categories
        self.base = cfg.base_topic
        self._mic = MicHealth(False, False)
        self._stt = SttStatus(None, 0, 0)
        self._summary: SummaryReady | None = None
        if client is None:
            import paho.mqtt.client as mqtt

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="hearth")
            if cfg.username:
                client.username_pw_set(cfg.username, secret(cfg.password_env))
            client.reconnect_delay_set(1, 60)
        self.client = client
        client.will_set(f"{self.base}/status", "offline", qos=1, retain=True)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        bus.subscribe(StateChanged, self._on_state)
        bus.subscribe(SoundEvent, self._on_sound)
        bus.subscribe(MicHealth, self._on_mic)
        bus.subscribe(SttStatus, self._on_stt)
        bus.subscribe(SummaryReady, self._on_summary)

    def start(self) -> None:
        self.client.connect_async(self.cfg.host, self.cfg.port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        self._pub(f"{self.base}/status", "offline")
        self.client.loop_stop()
        self.client.disconnect()

    def _pub(self, topic: str, payload: str, retain: bool = True) -> None:
        self.client.publish(topic, payload, qos=1, retain=retain)

    # -- connection ------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if getattr(reason_code, "is_failure", False):
            log.error("MQTT connection refused: %s", reason_code)
            return
        log.info("MQTT connected to %s:%s", self.cfg.host, self.cfg.port)
        for topic, payload in discovery(self.cfg, self.categories).items():
            self._pub(topic, json.dumps(payload))
        client.subscribe([(f"{self.base}/recording/set", 1), (f"{self.base}/guard/set", 1)])
        self._pub(f"{self.base}/status", "online")
        self._on_state(self.state.snapshot())
        self._on_mic(self._mic)
        self._on_stt(self._stt)
        if self._summary:
            self._on_summary(self._summary)

    def _on_message(self, client, userdata, msg) -> None:
        value = msg.payload.decode(errors="replace").strip().upper()
        if value not in ("ON", "OFF"):
            return
        if msg.topic == f"{self.base}/recording/set":
            self.state.set_recording(value == "ON", by="mqtt")
        elif msg.topic == f"{self.base}/guard/set":
            self.state.set_guard(value == "ON", by="mqtt")
        # Home Assistant shows the switch optimistically; confirm the real state
        # even when the command changed nothing.
        self._on_state(self.state.snapshot())

    # -- bus -> MQTT --------------------------------------------------------------
    def _on_state(self, e: StateChanged) -> None:
        # Live first: the indicator must never trail the recording switch.
        self._pub(f"{self.base}/live/state", onoff(e.live))
        self._pub(f"{self.base}/recording/state", onoff(e.recording))
        self._pub(f"{self.base}/guard/state", onoff(e.guard))

    def _on_sound(self, e: SoundEvent) -> None:
        self._pub(f"{self.base}/sound/event",
                  json.dumps({"event_type": e.category, "score": e.score, "at": iso(e.at)}), retain=False)

    def _on_mic(self, e: MicHealth) -> None:
        self._mic = e
        self._pub(f"{self.base}/mic/connected", onoff(e.connected))
        self._pub(f"{self.base}/mic/signal", onoff(e.signal))

    def _on_stt(self, e: SttStatus) -> None:
        self._stt = e
        self._pub(f"{self.base}/stt/backend", e.backend or "none")
        self._pub(f"{self.base}/stt/queue", str(e.queued))
        self._pub(f"{self.base}/stt/dropped", str(e.dropped))

    def _on_summary(self, e: SummaryReady) -> None:
        self._summary = e
        text = e.text if len(e.text) <= MAX_ATTR_CHARS else e.text[:MAX_ATTR_CHARS] + "\n[truncated]"
        # The state is the timestamp (a state is limited to 255 characters); the text is an attribute.
        self._pub(f"{self.base}/summary/attributes", json.dumps({
            "text": text, "kind": e.kind, "window_start": iso(e.window_start), "window_end": iso(e.window_end),
            "model": e.model, "segments": e.segment_count}))
        self._pub(f"{self.base}/summary/state", e.window_end.isoformat(timespec="seconds"))
