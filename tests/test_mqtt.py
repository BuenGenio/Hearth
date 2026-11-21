import json
from datetime import datetime, timezone

from hearth.bus import EventBus
from hearth.config import Mqtt
from hearth.events import MicHealth, SoundEvent, SummaryReady
from hearth.mqtt import MqttLink, discovery
from hearth.state import State
from conftest import FakeMqtt

CATS = ["smoke_alarm", "glass_break", "scream", "crying", "alarm"]


def link(tmp_path, monkeypatch=None):
    bus, client = EventBus(), FakeMqtt()
    state = State(bus, str(tmp_path / "state.json"))
    m = MqttLink(Mqtt(), bus, state, CATS, client=client)
    m.start()
    return bus, client, state


def test_discovery_announces_the_five_entities_on_one_device():
    d = discovery(Mqtt(), CATS)
    kinds = {t.split("/")[1] + "." + t.split("/")[3] for t in d}
    assert {"switch.recording", "switch.guard", "binary_sensor.live", "event.sound", "sensor.last_summary"} <= kinds
    ids = [p["unique_id"] for p in d.values()]
    assert len(ids) == len(set(ids))
    for p in d.values():
        assert p["device"]["identifiers"] == ["hearth"] and p["availability_topic"] == "hearth/status"
    event = d["homeassistant/event/hearth/sound/config"]
    assert event["event_types"] == CATS
    switch = d["homeassistant/switch/hearth/recording/config"]
    assert switch["command_topic"] == "hearth/recording/set" and switch["state_topic"] == "hearth/recording/state"


def test_connect_publishes_discovery_state_and_availability(tmp_path):
    _, client, _ = link(tmp_path)
    assert client.will == ("hearth/status", "offline", True)
    assert client.last("hearth/status") == "online"
    assert client.last("hearth/recording/state") == "OFF" and client.last("hearth/guard/state") == "ON"
    assert ("hearth/recording/set", 1) in client.subscribed
    assert json.loads(client.last("homeassistant/switch/hearth/guard/config"))["name"] == "Guard"


def test_switch_command_changes_state_and_confirms_it(tmp_path):
    bus, client, state = link(tmp_path)
    bus.publish(MicHealth(True, True))
    client.published.clear()
    client.message("hearth/recording/set", "ON")
    assert state.recording is True
    topics = [t for t, _, _ in client.published]
    # the live indicator is published before the recording state: it never trails the switch
    assert topics.index("hearth/live/state") < topics.index("hearth/recording/state")
    assert client.last("hearth/live/state") == "ON" and client.last("hearth/recording/state") == "ON"
    client.message("hearth/guard/set", "OFF")
    assert state.guard is False and client.last("hearth/guard/state") == "OFF"
    client.message("hearth/recording/set", "garbage")
    assert state.recording is True


def test_live_needs_recording_and_a_working_mic(tmp_path):
    bus, client, state = link(tmp_path)
    state.set_recording(True)
    assert client.last("hearth/live/state") == "OFF"       # recording on, but no microphone yet
    bus.publish(MicHealth(True, True))
    assert client.last("hearth/live/state") == "ON" and client.last("hearth/mic/connected") == "ON"
    bus.publish(MicHealth(True, False))                    # transmitter muted
    assert client.last("hearth/live/state") == "OFF" and client.last("hearth/mic/signal") == "OFF"


def test_sound_event_payload_is_not_retained(tmp_path):
    bus, client, _ = link(tmp_path)
    bus.publish(SoundEvent("glass_break", 0.91, datetime(2026, 11, 21, 12, tzinfo=timezone.utc)))
    topic, payload, retain = client.published[-1]
    assert topic == "hearth/sound/event" and retain is False
    assert json.loads(payload)["event_type"] == "glass_break"


def test_summary_state_is_a_short_timestamp_and_text_is_an_attribute(tmp_path):
    bus, client, _ = link(tmp_path)
    t = datetime(2026, 11, 21, 13, tzinfo=timezone.utc)
    bus.publish(SummaryReady("hourly", t, t, "x" * 50000, "m", 3))
    state = client.last("hearth/summary/state")
    assert len(state) <= 255 and datetime.fromisoformat(state) == t
    attrs = client.last("hearth/summary/attributes")
    assert len(attrs) < 16000 and json.loads(attrs)["kind"] == "hourly"


def test_recording_state_survives_a_restart(tmp_path):
    bus = EventBus()
    s = State(bus, str(tmp_path / "s.json"))
    s.set_recording(True); s.set_guard(False)
    again = State(EventBus(), str(tmp_path / "s.json"))
    assert again.recording is True and again.guard is False
    fresh = State(EventBus(), str(tmp_path / "s.json"), restore_recording=False)
    assert fresh.recording is False and fresh.guard is False


def test_no_secret_reaches_mqtt(tmp_path, monkeypatch):
    for name in ("HEARTH_MQTT_PASSWORD", "LOCALAI_API_KEY", "OPENAI_API_KEY", "HEARTH_API_TOKENS"):
        monkeypatch.setenv(name, "s3cr3t-" + name)
    bus, client, state = link(tmp_path)
    state.set_recording(True)
    assert not any("s3cr3t" in payload for _, payload, _ in client.published)
