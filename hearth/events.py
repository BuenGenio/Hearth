"""Events passed over the in-process bus. Plain data, no I/O imports."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class SoundEvent:
    category: str
    score: float
    at: datetime


@dataclass(frozen=True)
class WakeEvent:
    phrase: str
    action: str
    at: datetime


@dataclass(frozen=True)
class MicHealth:
    connected: bool
    signal: bool


@dataclass(frozen=True)
class StateChanged:
    recording: bool
    guard: bool
    live: bool


@dataclass
class Segment:
    """A stretch of speech. `samples` is 16 kHz mono float32 and lives in RAM only."""

    id: str
    source: str
    started_at: datetime
    ended_at: datetime
    samples: np.ndarray | None = None
    text: str = ""
    language: str | None = None
    backend: str | None = None
    audio_path: str | None = None

    @property
    def seconds(self) -> float:
        return 0.0 if self.samples is None else len(self.samples) / 16000


@dataclass(frozen=True)
class TranscriptCommitted:
    segment: Segment


@dataclass(frozen=True)
class SttStatus:
    backend: str | None
    queued: int
    dropped: int


@dataclass(frozen=True)
class SummaryReady:
    kind: str
    window_start: datetime
    window_end: datetime
    text: str
    model: str
    segment_count: int
