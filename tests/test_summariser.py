import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

from hearth.bus import EventBus
from hearth.config import Summary, Webhook
from hearth.events import Segment, SummaryReady
from hearth.store import Store
from hearth.summariser import JournalSink, Summariser, WebhookSink, chunks
from conftest import FakeLlm

TZ = ZoneInfo("Europe/London")
T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def add(store, minute, text):
    t = T0 + timedelta(minutes=minute)
    store.add_segment(Segment(id=f"00000000-0000-0000-0000-{minute:012d}", source="hub", started_at=t,
                              ended_at=t + timedelta(seconds=3), text=text))


def summariser(store, llms, sinks=None, **cfg):
    bus, events = EventBus(), []
    bus.subscribe(SummaryReady, events.append)
    return Summariser(Summary(**cfg), store, llms, bus, sinks), events


def test_empty_window_calls_no_model():
    llm = FakeLlm()
    s, events = summariser(Store(":memory:"), [llm])
    assert s.summarise("hourly", T0, T0 + timedelta(hours=1)) is None
    assert llm.calls == [] and events == []


def test_summary_is_stored_published_and_sent_with_local_times():
    store, llm = Store(":memory:"), FakeLlm("- plumber on Friday")
    add(store, 5, "call the plumber before friday")
    s, events = summariser(store, [llm])
    e = s.summarise("hourly", T0, T0 + timedelta(hours=1))
    assert e.text == "- plumber on Friday" and e.segment_count == 1 and events == [e]
    assert "[12:05] call the plumber before friday" in llm.calls[0][1]
    assert store.latest_summary("hourly")["model"] == "fake/fake-1"


def test_long_transcripts_are_summarised_in_parts_then_merged():
    store, llm = Store(":memory:"), FakeLlm()
    for i in range(30):
        add(store, i, "word " * 40)
    s, _ = summariser(store, [llm], chunk_chars=1500)
    s.summarise("nightly", T0, T0 + timedelta(hours=1))
    assert len(llm.calls) > 2 and "Merge these partial digests" in llm.calls[-1][0]
    assert all(len(p) <= 1500 for p in chunks("\n".join(["x" * 700] * 9) + "\n" + "y" * 4000, 1500))


def test_falls_back_to_the_next_model():
    store = Store(":memory:"); add(store, 1, "hello")
    bad, good = FakeLlm(fail=True), FakeLlm("ok")
    s, _ = summariser(store, [bad, good])
    assert s.summarise("hourly", T0, T0 + timedelta(hours=1)).text == "ok"


def test_one_failing_sink_does_not_block_the_others(tmp_path):
    store = Store(":memory:"); add(store, 1, "hello")

    def broken(event):
        raise RuntimeError("smtp down")

    s, _ = summariser(store, [FakeLlm("- note")], sinks=[broken, JournalSink(str(tmp_path), TZ)])
    s.summarise("hourly", T0, T0 + timedelta(hours=1))
    text = (tmp_path / "2026-11-21.md").read_text()
    assert text.startswith("# Saturday 21 November 2026") and "- note" in text


def test_webhook_is_signed(monkeypatch):
    monkeypatch.setenv("HEARTH_WEBHOOK_SECRET", "shh")
    got = {}

    def handler(request):
        got["sig"], got["body"] = request.headers["x-hearth-signature"], request.read()
        return httpx.Response(204)

    sink = WebhookSink(Webhook(True, "http://journal/hook"), httpx.Client(transport=httpx.MockTransport(handler)))
    sink(SummaryReady("nightly", T0, T0, "text", "m", 2))
    assert got["sig"] == "sha256=" + hmac.new(b"shh", got["body"], hashlib.sha256).hexdigest()
    assert json.loads(got["body"])["text"] == "text"


def test_schedule_runs_each_hour_once_and_retries_after_a_model_failure():
    store, llm = Store(":memory:"), FakeLlm()
    add(store, 5, "something said at five past twelve")
    s, _ = summariser(store, [llm], nightly_at="")
    assert len(s.run_due(T0 + timedelta(minutes=30))) == 0        # 12:00-13:00 is not over yet
    llm.fail = True
    assert s.run_due(T0 + timedelta(hours=1, minutes=1)) == []    # model down: nothing recorded as done
    llm.fail = False
    done = s.run_due(T0 + timedelta(hours=1, minutes=2))
    assert len(done) == 1 and done[0].window_end == T0 + timedelta(hours=1)
    assert s.run_due(T0 + timedelta(hours=1, minutes=3)) == []    # not twice
    calls = len(llm.calls)
    assert s.run_due(T0 + timedelta(hours=2, minutes=1)) == [] and len(llm.calls) == calls   # empty hour: no call


def test_nightly_covers_the_day_up_to_the_configured_local_time():
    store, llm = Store(":memory:"), FakeLlm()
    add(store, 5, "midday talk")
    s, _ = summariser(store, [llm], hourly=False, nightly_at="23:30")
    assert s.run_due(datetime(2026, 11, 21, 23, 29, tzinfo=timezone.utc)) == []
    done = s.run_due(datetime(2026, 11, 21, 23, 31, tzinfo=timezone.utc))
    assert len(done) == 1 and done[0].kind == "nightly"
    assert done[0].window_end == datetime(2026, 11, 21, 23, 30, tzinfo=timezone.utc)
