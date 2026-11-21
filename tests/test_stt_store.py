import os
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest

from hearth.bus import EventBus
from hearth.config import SttBackend
from hearth.events import Segment, SttStatus, TranscriptCommitted
from hearth.store import Store, retention_sweep
from hearth.stt import AudioStore, OpenAICompatStt, SttError, SttQueue, SttWorker
from conftest import FakeStt, tone

T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def seg(i=0, seconds=1.0, text=""):
    return Segment(id=f"00000000-0000-0000-0000-{i:012d}", source="hub", started_at=T0 + timedelta(minutes=i),
                   ended_at=T0 + timedelta(minutes=i, seconds=seconds), samples=tone(seconds, 0.5), text=text)


def worker(backends, store, **kw):
    bus, committed = EventBus(), []
    bus.subscribe(TranscriptCommitted, committed.append)
    w = SttWorker(SttQueue(60), backends, store.add_segment, bus, **kw)
    return w, committed


def test_falls_through_to_the_next_backend():
    store, a, b = Store(":memory:"), FakeStt("a", fail=True), FakeStt("b", text="from b")
    w, committed = worker([a, b], store)
    assert w.process(seg()) is True
    assert len(a.calls) == 1 and committed[0].segment.backend == "b"
    assert store.segments_between(T0, T0 + timedelta(hours=1))[0]["text"] == "from b"
    assert committed[0].segment.samples is None           # audio released from RAM after commit


def test_backend_without_a_key_is_skipped_not_called():
    store, a, b = Store(":memory:"), FakeStt("a", available=False), FakeStt("b")
    w, _ = worker([a, b], store)
    assert w.process(seg()) and a.calls == []


def test_all_backends_down_keeps_the_segment_for_later():
    store, a = Store(":memory:"), FakeStt("a", fail=True)
    w, committed = worker([a], store)
    s = seg()
    assert w.process(s) is False
    assert committed == [] and store.count_segments() == 0 and s.samples is not None
    a.fail = False
    assert w.process(s) is True and store.count_segments() == 1


def test_whisper_filler_on_silence_is_discarded():
    store = Store(":memory:")
    w, committed = worker([FakeStt(text="Thank you.")], store, drop_phrases=["thank you", "hearth stop recording"])
    assert w.process(seg()) is True
    assert committed == [] and store.count_segments() == 0
    w.backends[0].text = "Hearth, stop recording."         # a wake phrase heard while recording: a command, not a note
    assert w.process(seg(1)) is True and store.count_segments() == 0
    w.backends[0].text = "Thank you for stopping the recording."
    assert w.process(seg(2)) is True and store.count_segments() == 1


def test_queue_drops_the_oldest_and_counts():
    q = SttQueue(max_seconds=2.5)
    for i in range(4):
        q.put(seg(i, seconds=1.0))
    assert len(q) == 2 and q.dropped == 2
    assert q.get(0).id.endswith("2")                      # the two oldest went


def test_status_reports_queue_and_drops():
    bus, seen = EventBus(), []
    bus.subscribe(SttStatus, seen.append)
    q = SttQueue(1.5)
    q.put(seg(0)); q.put(seg(1))
    w = SttWorker(q, [FakeStt()], lambda s: True, bus)
    w._status()
    assert seen[-1] == SttStatus(None, 1, 1)


def test_audio_is_written_only_after_commit_and_only_when_enabled(tmp_path):
    store = Store(":memory:")
    order = []

    def commit(s):
        order.append(("commit", os.listdir(tmp_path)))
        return store.add_segment(s)

    audio = AudioStore(str(tmp_path))
    audio._opusenc = None                                 # force the WAV path so the test needs no opus-tools
    bus = EventBus()
    w = SttWorker(SttQueue(60), [FakeStt()], commit, bus, audio_store=audio)
    w.audio_stored = lambda s: store.set_audio_path(s.id, s.audio_path)
    w.process(seg())
    assert order == [("commit", [])]                      # nothing on disk when the transcript was committed
    row = store.segments_between(T0, T0 + timedelta(hours=1))[0]
    assert row["audio_path"] and os.path.exists(row["audio_path"])
    # all backends failing: no transcript, so no audio either
    failing = SttWorker(SttQueue(60), [FakeStt(fail=True)], commit, bus, audio_store=AudioStore(str(tmp_path / "x")))
    failing.process(seg(1))
    assert not os.path.exists(tmp_path / "x")
    # storage off: no AudioStore at all, nothing written
    off = SttWorker(SttQueue(60), [FakeStt()], store.add_segment, bus)
    off.process(seg(2))
    assert store.segments_between(T0 + timedelta(minutes=2), T0 + timedelta(minutes=3))[0]["audio_path"] is None


def test_http_client_sends_openai_shaped_request_and_hides_the_key(monkeypatch):
    monkeypatch.setenv("K", "sk-secret-value")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], request.read()
        return httpx.Response(200, json={"text": " Hearth, start recording. "})

    b = OpenAICompatStt("localai", SttBackend("http://x/v1", "whisper-1", "K"), httpx.Client(transport=httpx.MockTransport(handler)))
    text, _ = b.transcribe(b"RIFFdata")
    assert text == "Hearth, start recording."
    assert seen["url"] == "http://x/v1/audio/transcriptions" and seen["auth"] == "Bearer sk-secret-value"
    assert b'name="model"' in seen["body"] and b"whisper-1" in seen["body"] and b"RIFFdata" in seen["body"]
    down = OpenAICompatStt("localai", SttBackend("http://x/v1", "m", "K"),
                           httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad sk-secret-value"))))
    with pytest.raises(SttError) as e:
        down.transcribe(b"x")
    assert "sk-secret-value" not in str(e.value)


def test_store_is_idempotent_and_search_takes_any_text():
    store = Store(":memory:")
    s = seg(0, text="call the plumber before friday")
    assert store.add_segment(s) is True and store.add_segment(s) is False
    assert store.count_segments() == 1
    assert [r["id"] for r in store.search("plumber")] == [s.id]
    for hostile in ['"', 'a" OR "b', "NEAR(", "plumber*", "col:x", "-"]:
        store.search(hostile)                             # must not raise


def test_retention_removes_old_transcripts_and_audio(tmp_path):
    store = Store(":memory:")
    old, new = seg(0, text="old"), seg(1, text="new")
    old.started_at = datetime.now(timezone.utc) - timedelta(days=40)
    new.started_at = datetime.now(timezone.utc)
    f = tmp_path / "old.wav"; f.write_bytes(b"x")
    old.audio_path = str(f)
    store.add_segment(old); store.add_segment(new)
    retention_sweep(store, transcript_days=0, audio_days=30)
    assert not f.exists() and store.count_segments() == 2  # audio gone, transcript kept
    retention_sweep(store, transcript_days=30, audio_days=0)
    assert store.count_segments() == 1 and store.search("old") == []


def test_wake_phrase_ending_a_transcript_acts_as_a_command_and_is_not_stored():
    store, heard = Store(":memory:"), []
    w, committed = worker([FakeStt(text="Hearth, stop recording.")], store)
    w.phrases, w.on_phrase = {"hearth stop recording"}, heard.append
    assert w.process(seg()) is True
    assert heard == ["hearth stop recording"] and committed == [] and store.count_segments() == 0
    # run together with the sentence before it: the sentence is kept, the command is not
    w.backends[0].text = "Call the plumber before Friday. Hearth, stop recording."
    w.process(seg(1))
    assert len(heard) == 2 and committed[-1].segment.text == "Call the plumber before Friday."
    w.backends[0].text = "I told it hearth stop recording and it did"     # mid-sentence is talk, not a command
    w.process(seg(2))
    assert len(heard) == 2 and store.count_segments() == 2
