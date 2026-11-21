"""The whole hub wired together with fake models: the privacy rules, end to end."""
import os

import numpy as np
import pytest

from hearth import config as config_mod
from hearth.app import App
from hearth.config import Config
from conftest import FakeLlm, FakeMqtt, FakeSpotter, FakeStt, FakeVad, FakeYamnet, ScriptSource, silence, tone, wait_for


def build(tmp_path, audio, pace=0.002, **overrides):
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.api.enabled = False
    cfg.summary.enabled = False
    for key, value in overrides.items():
        section, name = key.split("__")
        setattr(getattr(cfg, section), name, value)
    stt, mqtt = FakeStt(text="we said this while recording"), FakeMqtt()
    app = App(cfg, source_factory=lambda: ScriptSource(audio, pace=pace), yamnet=FakeYamnet(), spotter=FakeSpotter(),
              make_vad=FakeVad, stt_backends=[stt], mqtt_client=mqtt, llms=[FakeLlm()])
    return app, stt, mqtt


def files(tmp_path):
    return sorted(os.path.relpath(os.path.join(d, f), tmp_path) for d, _, fs in os.walk(tmp_path) for f in fs)


def test_recording_off_nothing_is_transcribed_or_written_but_guard_still_works(tmp_path):
    audio = np.concatenate([tone(1.5, 0.5), silence(1.0), tone(2.0, 1.0), silence(1.0)])   # speech, then glass
    app, stt, mqtt = build(tmp_path, audio)
    app.start()
    try:
        assert wait_for(lambda: mqtt.last("hearth/sound/event") is not None)      # guard heard the glass
        assert wait_for(lambda: app.ring.total >= len(audio))
    finally:
        app.stop()
    assert stt.calls == []                                                        # no audio left the box
    assert mqtt.last("hearth/live/state") == "OFF"
    assert all(f.startswith("hearth.sqlite") for f in files(tmp_path))            # no audio, not even state yet
    assert '"glass_break"' in mqtt.last("hearth/sound/event")


def test_wake_phrase_starts_recording_and_only_later_speech_is_transcribed(tmp_path):
    before = tone(1.0, 0.3)                       # speech before the wake phrase: must never be sent
    audio = np.concatenate([before, silence(0.5), tone(0.1, 0.05), silence(1.0),   # "hearth start recording"
                            tone(1.0, 0.5), silence(1.0),                           # speech while recording
                            tone(0.1, 0.07), silence(1.0),                          # "hearth stop recording"
                            tone(1.0, 0.6), silence(1.0)])                          # speech after stop
    app, stt, mqtt = build(tmp_path, audio, pace=0.02)                              # five times real time
    app.start()
    try:
        assert wait_for(lambda: app.store.count_segments() == 1)
        assert wait_for(lambda: mqtt.last("hearth/recording/state") == "OFF" and app.ring.total >= len(audio))
        assert wait_for(lambda: len(app.stt_queue) == 0)
    finally:
        app.stop()
    assert len(stt.calls) == 1
    sent = np.frombuffer(stt.calls[0][44:], dtype="<i2").astype(np.float32) / 32767
    assert np.isclose(sent, 0.5, atol=1e-3).sum() > 15000          # the speech said while ON
    assert not np.isclose(sent, 0.3, atol=1e-3).any()              # nothing from before the switch
    assert not np.isclose(sent, 0.6, atol=1e-3).any()              # nothing from after it
    # the indicator went ON before anything was handed to transcription, and is OFF again now
    assert "ON" in [p for t, p, _ in mqtt.published if t == "hearth/live/state"]
    assert mqtt.last("hearth/live/state") == "OFF"
    assert not any(f.endswith((".wav", ".opus")) for f in files(tmp_path))   # raw audio is off by default


def test_guard_switch_off_silences_events_without_touching_recording(tmp_path):
    audio = np.concatenate([silence(0.5), tone(2.0, 1.0), silence(1.0)])
    app, stt, mqtt = build(tmp_path, audio, guard__enabled_default=False)
    app.start()
    try:
        assert wait_for(lambda: app.ring.total >= len(audio))
    finally:
        app.stop()
    assert mqtt.last("hearth/sound/event") is None and mqtt.last("hearth/guard/state") == "OFF"


def test_recording_restored_after_restart_and_raw_audio_kept_when_enabled(tmp_path):
    audio = np.concatenate([silence(0.3), tone(1.0, 0.5), silence(1.0)])
    app, _, _ = build(tmp_path, silence(0.2))
    app.state.set_recording(True)
    app.stop()
    app, stt, mqtt = build(tmp_path, audio, pace=0.02, storage__store_audio=True)
    assert app.state.recording is True
    app.start()
    try:
        assert wait_for(lambda: app.store.count_segments() == 1)
        assert wait_for(lambda: any(f.startswith("audio/") for f in files(tmp_path)))
    finally:
        app.stop()


def test_config_file_overrides_single_settings_and_rejects_typos(tmp_path):
    p = tmp_path / "hearth.toml"
    p.write_text('[audio]\nsource = "none"\n[stt]\norder = ["openai"]\n'
                 '[[wake.phrases]]\ntext = "computer listen"\naction = "recording_on"\n'
                 '[guard.categories.scream]\nclasses = ["Screaming"]\nthreshold = 0.6\n')
    cfg = config_mod.load(str(p))
    assert cfg.audio.source == "none" and cfg.audio.rate == 48000          # sibling default kept
    assert cfg.stt.order == ["openai"] and "localai" in cfg.stt.backends
    assert [w.text for w in cfg.wake.phrases] == ["computer listen"]
    assert cfg.guard.categories["scream"].threshold == 0.6 and cfg.guard.categories["scream"].min_hits == 3
    assert "glass_break" in cfg.guard.categories                               # other categories keep their defaults
    # overriding one field of one backend keeps the rest of it, and the other backends
    p.write_text('[stt.backends.localai]\nbase_url = "http://gpu:8080/v1"\n[guard.categories.crying]\nclasses = []\n'
                 '[llm.big]\nkind = "openai"\nbase_url = "http://gpu/v1"\nmodel = "m"\n')
    cfg = config_mod.load(str(p))
    assert cfg.stt.backends["localai"].base_url == "http://gpu:8080/v1"
    assert cfg.stt.backends["localai"].model == "whisper-large-turbo-q5_0" and "openai" in cfg.stt.backends
    assert cfg.llm["big"].model == "m" and "anthropic" in cfg.llm
    from hearth.guard import GuardLogic
    assert "crying" not in GuardLogic(cfg.guard.categories, FakeYamnet.labels).cats   # switched off
    for bad in ("[audio]\nsauce = 1\n", "[sauce]\nx = 1\n", "[audio]\nrate = \"fast\"\n",
                "[stt.backends.localai]\nmodle = \"x\"\n"):
        p.write_text(bad)
        with pytest.raises(ValueError):
            config_mod.load(str(p))
