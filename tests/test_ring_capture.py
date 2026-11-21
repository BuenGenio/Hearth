import numpy as np

from hearth.capture import Capture, RingBuffer
from hearth.capture.sources import FileSource, SourceLost
from conftest import ScriptSource, silence, tone, wait_for, FIXTURES


def test_ring_wraps_and_clamps():
    ring = RingBuffer(seconds=1.0)                  # 16000 samples
    ring.write(np.arange(10000, dtype=np.float32))
    ring.write(np.arange(10000, 24000, dtype=np.float32))
    start, data = ring.read(0, 24000)               # the first 8000 have been overwritten
    assert start == 8000 and len(data) == 16000
    assert data[0] == 8000 and data[-1] == 23999


def test_cursor_starts_at_now_and_counts_skips():
    ring = RingBuffer(seconds=1.0)
    ring.write(tone(0.5, 0.3))
    cur = ring.cursor()
    assert cur.read(timeout=0)[1].size == 0         # nothing from before the cursor existed
    ring.write(tone(2.5, 0.1))                      # 40000 samples, far more than the ring holds
    start, data = cur.read(timeout=0)
    assert len(data) == 16000 and cur.skipped == 24000


def test_capture_reconnects_and_reports_health():
    ring, health, made = RingBuffer(5), [], []

    def make():
        made.append(1)
        return ScriptSource(tone(0.3, 0.2), then_lost=len(made) == 1)

    cap = Capture(make, ring, on_health=lambda c, s: health.append((c, s)), retry_min=0.01)
    cap.start()
    assert wait_for(lambda: len(made) >= 2 and health[-1] == (True, True))
    cap.stop()
    assert (False, False) in health                 # the loss was reported
    assert health[0] == (True, True)


def test_flat_line_is_reported_as_no_signal():
    ring, health = RingBuffer(5), []
    cap = Capture(lambda: ScriptSource(np.concatenate([tone(0.2, 0.2), silence(1.0)]), pace=0.03), ring,
                  on_health=lambda c, s: health.append((c, s)), silence_seconds=0.3)
    cap.start()
    assert wait_for(lambda: (True, False) in health)
    cap.stop()


def test_file_source_resamples_to_16k():
    src = FileSource(f"{FIXTURES}/tts_1.wav", realtime=False)     # 24 kHz fixture
    total = 0
    try:
        for block in src.blocks():
            assert block.dtype == np.float32
            total += len(block)
    except SourceLost:
        pass
    import wave
    with wave.open(f"{FIXTURES}/tts_1.wav") as w:
        expected = w.getnframes() * 16000 / w.getframerate()
    assert abs(total - expected) < 50


def test_agc_lifts_a_quiet_input_and_never_clips_a_sudden_loud_one():
    from hearth.features import Agc

    agc = Agc(target_dbfs=-20, max_gain_db=30)
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(1600) * 0.002).astype(np.float32)        # about -54 dBFS
    for _ in range(20):
        out = agc.process(quiet)
    assert 25 < 20 * np.log10(agc.gain) <= 30                              # raised, but capped
    assert np.sqrt((out ** 2).mean()) > 10 * np.sqrt((quiet ** 2).mean())
    loud = (rng.standard_normal(1600) * 0.3).astype(np.float32)           # a shout straight after
    out = agc.process(loud)
    assert agc.gain == 1.0 and np.array_equal(out, loud)                   # passed through untouched
    hot = Agc()
    assert np.array_equal(hot.process(loud), loud)                         # a healthy level is never attenuated


def test_mic_health_is_judged_before_the_gain_stage():
    ring, health = RingBuffer(5), []
    cap = Capture(lambda: ScriptSource(np.concatenate([tone(0.2, 0.2), silence(1.0)]), pace=0.03), ring,
                  on_health=lambda c, s: health.append((c, s)), silence_seconds=0.3,
                  process=lambda b: b + 0.5)                               # a gain stage that would hide the flat line
    cap.start()
    assert wait_for(lambda: (True, False) in health)
    cap.stop()
