import time

import numpy as np
import pytest

from hearth.bus import EventBus
from hearth.capture import RingBuffer
from hearth.config import GuardCategory, default_categories
from hearth.events import SoundEvent
from hearth.guard import GuardLogic, GuardWorker
from conftest import FakeYamnet, tone, wait_for


def scores(labels, **named):
    s = np.zeros(len(labels), dtype=np.float32)
    for name, v in named.items():
        s[labels.index(name)] = v
    return s


def test_needs_n_of_m_then_cools_down():
    labels = ["Screaming"]
    logic = GuardLogic({"scream": GuardCategory(["Screaming"], threshold=0.5, min_hits=2, window=3,
                                                cooldown_seconds=10)}, labels)
    hi, lo = scores(labels, Screaming=0.8), scores(labels)
    assert logic.update(hi, 0.0) == []                    # one hit is not enough
    assert logic.update(lo, 0.5) == []
    assert logic.update(hi, 1.0) == [("scream", pytest.approx(0.8))]
    assert logic.update(hi, 1.5) == [] and logic.update(hi, 2.0) == []     # cooling down
    assert logic.update(hi, 12.0) != []                   # still sounding once the cooldown ends: fires again


def test_default_categories_use_real_class_names():
    GuardLogic(default_categories(), FakeYamnet.labels)   # raises if a name is unknown


def test_unknown_class_name_is_a_config_error():
    with pytest.raises(ValueError, match="no class named"):
        GuardLogic({"x": GuardCategory(["Carbon monoxide detector"])}, FakeYamnet.labels)


def test_worker_publishes_event_and_skips_model_when_disabled():
    bus, events, model = EventBus(), [], FakeYamnet()
    bus.subscribe(SoundEvent, events.append)
    enabled = [False]
    ring = RingBuffer(10)
    worker = GuardWorker(ring.cursor(), model, GuardLogic(default_categories(), model.labels), bus,
                         enabled=lambda: enabled[0])
    worker.start()
    ring.write(tone(2.0, 1.0))                            # "breaking glass" while the guard is off
    time.sleep(0.3)
    assert model.calls == 0 and events == []              # the classifier is not even run
    enabled[0] = True
    ring.write(tone(2.0, 1.0))
    assert wait_for(lambda: [e.category for e in events] == ["glass_break"])
    worker.stop()
