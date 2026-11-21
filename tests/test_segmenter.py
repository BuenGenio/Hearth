import numpy as np

from hearth.capture import RingBuffer
from hearth.segmenter import Segmenter
from conftest import FakeVad, silence, tone


def make(ring, recording, **kw):
    out, made = [], []

    def make_vad():
        made.append(1)
        return FakeVad()

    seg = Segmenter(ring.cursor(), make_vad, recording=lambda: recording[0], emit=out.append, **kw)
    return seg, out, made


def test_reads_nothing_while_recording_is_off():
    ring, recording = RingBuffer(30), [False]
    seg, out, made = make(ring, recording)
    ring.write(np.concatenate([tone(1.0, 0.5), silence(1.0)]))
    for _ in range(5):
        seg.step()
    assert out == [] and made == []            # no VAD was even created
    assert seg.cursor.index == 0               # and the cursor never moved: no audio was copied out


def test_speech_before_switch_on_never_reaches_a_segment():
    ring, recording = RingBuffer(30), [False]
    seg, out, _ = make(ring, recording, preroll=1.0)
    ring.write(tone(1.0, 0.3))                 # said while OFF; 0.3 is the tell-tale value
    seg.step()
    recording[0] = True
    seg.step()                                 # session opens at "now"
    ring.write(np.concatenate([tone(1.0, 0.5), silence(1.0)]))
    seg.step()
    assert len(out) == 1
    assert not np.any(np.isclose(out[0].samples, 0.3))     # even with a 1 s pre-roll asked for
    assert np.isclose(out[0].samples[0], 0.5)


def test_preroll_is_taken_from_this_session_only_and_segments_do_not_overlap():
    ring, recording = RingBuffer(30), [True]
    seg, out, _ = make(ring, recording, preroll=0.2)
    seg.step()
    ring.write(np.concatenate([tone(0.5, 0.01), tone(1.0, 0.5), silence(0.35), tone(1.0, 0.6), silence(1.0)]))
    seg.step()
    assert len(out) == 2
    first, second = out
    assert np.isclose(first.samples[0], 0.01)              # 0.2 s of pre-roll before the speech
    assert np.isclose(first.samples[int(0.2 * 16000)], 0.5)
    assert second.started_at >= first.ended_at             # pre-roll did not reach back into the first
    assert abs(first.seconds - (0.2 + 1.0 + 0.3)) < 0.05


def test_switching_off_mid_speech_keeps_what_was_said_while_on():
    ring, recording = RingBuffer(30), [True]
    seg, out, _ = make(ring, recording)
    seg.step()
    ring.write(tone(1.0, 0.5))                 # still talking
    seg.step()
    assert out == []
    recording[0] = False
    ring.write(tone(1.0, 0.4))                 # said after OFF
    seg.step()
    assert len(out) == 1
    assert not np.any(np.isclose(out[0].samples, 0.4))


def test_segment_times_follow_the_wall_clock():
    import time
    from datetime import datetime, timezone

    ring, recording = RingBuffer(30), [True]
    seg, out, _ = make(ring, recording, preroll=0.0)
    seg.step()
    ring.write(np.concatenate([silence(1.0), tone(1.0, 0.5), silence(1.0)]))
    now = time.time()
    seg.step()
    started = out[0].started_at.timestamp()
    assert abs(started - (now - 2.0)) < 0.2    # the speech began two seconds before the last write
    assert out[0].started_at.tzinfo == timezone.utc and isinstance(out[0].ended_at, datetime)
