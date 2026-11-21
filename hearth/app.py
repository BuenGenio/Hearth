"""Wiring: builds the pipeline from the config and runs it."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import __version__, config as config_mod
from .bus import EventBus
from .capture import AlsaSource, Capture, FileSource, RingBuffer
from .config import Config, secret
from .events import MicHealth, Segment, SoundEvent, SttStatus, WakeEvent
from .features import Agc
from .guard import GuardLogic, GuardWorker
from .segmenter import Segmenter
from .state import State
from .store import Store, retention_sweep
from .stt import AudioStore, OpenAICompatStt, SttQueue, SttWorker, plain
from .wake import WakeWorker

log = logging.getLogger("hearth")


class App:
    """The running hub. Model runners and backends can be injected for tests."""

    def __init__(self, cfg: Config, *, source_factory=None, yamnet=None, spotter=None, make_vad=None,
                 stt_backends=None, mqtt_client=None, llms=None) -> None:
        self.cfg = cfg
        self.bus = EventBus()
        self.threads: list = []
        self._started = False
        os.makedirs(cfg.storage.data_dir, exist_ok=True)
        self.store = Store(os.path.join(cfg.storage.data_dir, "hearth.sqlite"))
        self.state = State(self.bus, os.path.join(cfg.storage.data_dir, "state.json"),
                           guard_default=cfg.guard.enabled_default, restore_recording=cfg.storage.restore_recording)
        self.ring = RingBuffer(cfg.audio.ring_seconds)
        self.stt_status = SttStatus(None, 0, 0)
        self.bus.subscribe(SttStatus, lambda e: setattr(self, "stt_status", e))
        self.bus.subscribe(SoundEvent, lambda e: self.store.add_sound_event(e.at, e.category, e.score))
        self.bus.subscribe(WakeEvent, lambda e: self.state.apply_action(e.action, by=f"wake phrase {e.phrase!r}"))

        # MQTT first, so it sees every later event.
        self.mqtt = None
        if cfg.mqtt.enabled:
            from .mqtt import MqttLink

            self.mqtt = MqttLink(cfg.mqtt, self.bus, self.state, list(cfg.guard.categories), client=mqtt_client)

        # Capture
        self.capture = None
        factory = source_factory or self._source_factory()
        if factory is not None:
            self.capture = Capture(factory, self.ring,
                                   on_health=lambda c, s: self.bus.publish(MicHealth(c, s)),
                                   silence_floor=cfg.audio.silence_floor, silence_seconds=cfg.audio.silence_seconds,
                                   process=self._gain_stage())
            self.threads.append(self.capture)

            # Guard
            if yamnet is None and cfg.guard.categories:
                from .models.yamnet import Yamnet

                yamnet = Yamnet(cfg.guard.model)
            if yamnet is not None:
                logic = GuardLogic(cfg.guard.categories, yamnet.labels)
                self.threads.append(GuardWorker(self.ring.cursor(), yamnet, logic, self.bus,
                                                enabled=lambda: self.state.guard, hop_seconds=cfg.guard.hop_seconds))

            # Wake phrases
            if cfg.wake.enabled and cfg.wake.phrases:
                from .models.kws import normalise

                if spotter is None:
                    from .models.kws import SherpaKws

                    spotter = SherpaKws(cfg.wake.model_dir, [p.text for p in cfg.wake.phrases],
                                        threshold=cfg.wake.threshold, boost=cfg.wake.boost)
                actions = {normalise(p.text): p.action for p in cfg.wake.phrases}
                self.threads.append(WakeWorker(self.ring.cursor(), spotter, actions, self.bus))

            # Transcription
            self.stt_queue = SttQueue(cfg.stt.queue_seconds)
            if make_vad is None:
                def make_vad():
                    from .models.vad import SherpaVad

                    v = cfg.vad
                    return SherpaVad(v.model, v.threshold, v.min_silence, v.min_speech, v.max_speech)
            self.threads.append(Segmenter(self.ring.cursor(), make_vad, recording=self.state.is_live,
                                          emit=self.stt_queue.put, preroll=cfg.vad.preroll))
            if stt_backends is None:
                stt_backends = [OpenAICompatStt(n, cfg.stt.backends[n]) for n in cfg.stt.order]
            audio_store = AudioStore(cfg.audio_dir) if cfg.storage.store_audio else None
            # A wake phrase said while recording is a command, not something to keep.
            drop = cfg.stt.drop_phrases + ([p.text for p in cfg.wake.phrases] if cfg.wake.enabled else [])
            worker = SttWorker(self.stt_queue, stt_backends, self.store.add_segment, self.bus, audio_store,
                               drop, cfg.stt.retry_min, cfg.stt.retry_max)
            worker.audio_stored = lambda seg: self.store.set_audio_path(seg.id, seg.audio_path)
            if cfg.wake.enabled:
                # Second route for phrases said while recording: the transcript. Whisper
                # hears "hearth stop recording" in conditions where the small spotter misses it.
                by_text = {plain(p.text): p for p in cfg.wake.phrases}
                worker.on_phrase = lambda text: self.bus.publish(
                    WakeEvent(by_text[text].text, by_text[text].action, datetime.now(timezone.utc)))
                worker.phrases = set(by_text)
            self.threads.append(worker)

        # Summaries
        self.summariser = None
        if cfg.summary.enabled:
            from .llm import make_llm
            from .summariser import EmailSink, JournalSink, Summariser, SummaryScheduler, WebhookSink

            tz = ZoneInfo(cfg.summary.timezone)
            sinks = []
            if cfg.summary.journal:
                sinks.append(JournalSink(cfg.journal_dir, tz))
            if cfg.summary.email.enabled:
                sinks.append(EmailSink(cfg.summary.email, tz))
            if cfg.summary.webhook.enabled:
                sinks.append(WebhookSink(cfg.summary.webhook))
            if llms is None:
                llms = [make_llm(n, cfg.llm[n]) for n in cfg.summary.llm]
            self.summariser = Summariser(cfg.summary, self.store, llms, self.bus, sinks)
            self.threads.append(SummaryScheduler(self.summariser, on_tick=self._sweep))

        # HTTP API
        self.api_servers: list = []
        self._stopping = threading.Event()
        if cfg.api.enabled:
            import uvicorn

            from .api import create_app, parse_tokens

            tokens = parse_tokens(secret(cfg.api.tokens_env))
            if not tokens:
                log.warning("no API tokens in $%s: every API request will be refused", cfg.api.tokens_env)
            api = create_app(self.store, tokens, self.state, self._status, self.summariser,
                             cfg.audio_dir if cfg.storage.store_audio else None)
            self._api_app = api
            for host in cfg.api.hosts:
                self.threads.append(threading.Thread(target=self._serve, args=(host,), name=f"api-{host}", daemon=True))

    def _serve(self, host: str) -> None:
        """Serve the API on one address. An address that is not up yet (the
        tailnet at boot, say) is retried instead of taking the hub down."""
        import uvicorn

        while not self._stopping.is_set():
            server = uvicorn.Server(uvicorn.Config(self._api_app, host=host, port=self.cfg.api.port,
                                                   log_level="warning", access_log=False))
            self.api_servers.append(server)
            try:
                server.run()
            except SystemExit:
                pass                                    # uvicorn exits this way when it cannot bind
            except Exception:  # noqa: BLE001
                log.exception("API server on %s failed", host)
            self.api_servers.remove(server)
            if self._stopping.wait(15):
                return
            log.warning("API on %s:%s is not listening; retrying", host, self.cfg.api.port)

    def _gain_stage(self):
        a = self.cfg.audio
        fixed = 10 ** (a.gain_db / 20)
        agc = Agc(a.agc_target_dbfs, a.agc_max_gain_db) if a.agc else None
        if agc is None and fixed == 1.0:
            return None

        def stage(block):
            if fixed != 1.0:
                block = np.clip(block * fixed, -1.0, 1.0).astype(np.float32)
            return agc.process(block) if agc else block

        self.agc = agc
        return stage

    def _source_factory(self):
        a = self.cfg.audio
        if a.source == "alsa":
            return lambda: AlsaSource(a.device, a.rate, a.channels)
        if a.source == "file":
            return lambda: FileSource(a.file, realtime=True, loop=a.file_loop, hold=True)
        if a.source == "none":
            return None
        raise ValueError(f"audio.source: unknown source {a.source!r}")

    def _status(self) -> dict:
        s = self.stt_status
        out = {"stt_backend": s.backend, "stt_queued": s.queued, "stt_dropped": s.dropped}
        if getattr(self, "agc", None) is not None:
            out["input_gain_db"] = round(20 * float(np.log10(self.agc.gain)), 1)
        return out

    def _sweep(self) -> None:
        s = self.cfg.storage
        retention_sweep(self.store, s.transcript_retention_days, s.audio_retention_days)

    def start(self) -> None:
        self._started = True
        if self.mqtt:
            self.mqtt.start()
        for t in self.threads:
            t.start()
        log.info("hearth %s running: recording %s, guard %s", __version__,
                 "ON" if self.state.recording else "OFF", "ON" if self.state.guard else "OFF")

    def stop(self) -> None:
        self._stopping.set()
        for server in list(self.api_servers):
            server.should_exit = True
        for t in self.threads:
            if hasattr(t, "stop"):
                t.stop()
        self.ring.close()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=5)
        if self.mqtt and self._started:
            self.mqtt.stop()
        self.store.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="hearth")
    parser.add_argument("--config", default=None, help="TOML file (default: $HEARTH_CONFIG)")
    parser.add_argument("--check", action="store_true", help="load the config and models, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("HEARTH_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # its INFO lines log every request URL
    app = App(config_mod.load(args.config))
    if args.check:
        print("ok")
        return
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    signal.signal(signal.SIGINT, lambda *_: done.set())
    app.start()
    done.wait()
    log.info("stopping")
    app.stop()


if __name__ == "__main__":
    main()
