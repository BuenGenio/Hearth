"""Audio capture: sources, the RAM ring buffer and the capture loop.

This package has no Home Assistant, MQTT or SQLite imports, so it can be reused
by other projects (and by an OSC/MIDI fast path) as it is.
"""
from .ring import RingBuffer, Cursor  # noqa: F401
from .sources import AlsaSource, FileSource, Source, SourceLost  # noqa: F401
from .loop import Capture  # noqa: F401

RATE = 16000
