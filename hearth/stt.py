"""Speech-to-text: a bounded RAM queue and a worker that tries backends in order.

Both backends (a LocalAI box on the network, OpenAI) speak the OpenAI-compatible
`/audio/transcriptions` API, so there is one client with two configurations.
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import threading
import wave
from collections import deque
from typing import Callable

import httpx
import numpy as np

from .bus import EventBus
from .config import SttBackend, secret
from .events import Segment, SttStatus, TranscriptCommitted

log = logging.getLogger(__name__)
RATE = 16000


def wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def plain(text: str) -> str:
    """Lower-case letters and digits only, for comparing short phrases."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class SttError(Exception):
    pass


class OpenAICompatStt:
    def __init__(self, name: str, cfg: SttBackend, client: httpx.Client | None = None) -> None:
        self.name, self.cfg = name, cfg
        self._client = client or httpx.Client(timeout=cfg.timeout)

    @property
    def available(self) -> bool:
        """A backend whose key is not configured is skipped, not failed."""
        return bool(self.cfg.base_url) and (not self.cfg.api_key_env or bool(secret(self.cfg.api_key_env)))

    def transcribe(self, wav: bytes) -> tuple[str, str | None]:
        data = {"model": self.cfg.model, "response_format": "json"}
        if self.cfg.language:
            data["language"] = self.cfg.language
        headers = {}
        key = secret(self.cfg.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            r = self._client.post(self.cfg.base_url.rstrip("/") + "/audio/transcriptions", data=data,
                                  files={"file": ("segment.wav", wav, "audio/wav")}, headers=headers)
            r.raise_for_status()
            body = r.json()
        except (httpx.HTTPError, ValueError) as e:
            # Never include request headers here: they carry the key.
            raise SttError(f"{self.name}: {type(e).__name__}: {str(e)[:200]}") from None
        return str(body.get("text", "")).strip(), body.get("language")


class SttQueue:
    """Segments waiting for transcription. RAM only, bounded by audio duration:
    when full, the oldest speech is dropped and counted."""

    def __init__(self, max_seconds: float) -> None:
        self.max_seconds = max_seconds
        self._items: deque[Segment] = deque()
        self._seconds = 0.0
        self.dropped = 0
        self._cond = threading.Condition()

    def put(self, seg: Segment, front: bool = False) -> None:
        with self._cond:
            (self._items.appendleft if front else self._items.append)(seg)
            self._seconds += seg.seconds
            while self._seconds > self.max_seconds and len(self._items) > 1:
                old = self._items.popleft()
                self._seconds -= old.seconds
                self.dropped += 1
                log.warning("stt queue full: dropped %.1f s of speech", old.seconds)
            self._cond.notify()

    def get(self, timeout: float) -> Segment | None:
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            if not self._items:
                return None
            seg = self._items.popleft()
            self._seconds -= seg.seconds
            return seg

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)


class AudioStore:
    """Writes a segment's audio to disk. Only ever called after its transcript
    has been committed, and only when raw-audio storage is switched on."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._opusenc = shutil.which("opusenc")

    def write(self, seg: Segment) -> str:
        day = seg.started_at.strftime("%Y/%m/%d")
        folder = os.path.join(self.directory, day)
        os.makedirs(folder, exist_ok=True)
        wav = wav_bytes(seg.samples)
        if self._opusenc:
            path = os.path.join(folder, seg.id + ".opus")
            p = subprocess.run([self._opusenc, "--quiet", "--bitrate", "24", "-", path], input=wav,
                               capture_output=True)
            if p.returncode == 0:
                return path
            log.warning("opusenc failed (%s); storing WAV", p.stderr.decode(errors="replace")[:120])
        path = os.path.join(folder, seg.id + ".wav")
        with open(path, "wb") as f:
            f.write(wav)
        return path


class SttWorker(threading.Thread):
    def __init__(self, queue: SttQueue, backends: list, commit: Callable[[Segment], bool], bus: EventBus,
                 audio_store: AudioStore | None = None, drop_phrases: list[str] | None = None,
                 retry_min: float = 5.0, retry_max: float = 300.0) -> None:
        super().__init__(name="stt", daemon=True)
        self.queue, self.backends, self.commit, self.bus = queue, backends, commit, bus
        self.audio_store = audio_store
        self.drop_phrases = {plain(p) for p in (drop_phrases or [])}
        self.retry_min, self.retry_max = retry_min, retry_max
        self.last_backend: str | None = None
        self.phrases: set[str] = set()              # wake phrases, in `plain` form
        self.on_phrase: Callable[[str], None] = lambda text: None
        self._halt = threading.Event()
        self._last_status: SttStatus | None = None

    def stop(self) -> None:
        self._halt.set()

    def _status(self) -> None:
        status = SttStatus(self.last_backend, len(self.queue), self.queue.dropped)
        if status != self._last_status:
            self._last_status = status
            self.bus.publish(status)

    def run(self) -> None:
        delay = self.retry_min
        while not self._halt.is_set():
            self._status()
            seg = self.queue.get(timeout=1.0)
            if seg is None:
                continue
            if self.process(seg):
                delay = self.retry_min
            else:
                self.queue.put(seg, front=True)     # keep it; the queue bound still applies
                self._status()
                log.warning("no STT backend reachable; retrying in %.0f s", delay)
                if self._halt.wait(delay):
                    return
                delay = min(delay * 2, self.retry_max)

    def process(self, seg: Segment) -> bool:
        """Transcribe and commit one segment. False means: nothing worked, try again later."""
        wav = wav_bytes(seg.samples)
        for backend in self.backends:
            if not backend.available:
                continue
            try:
                text, language = backend.transcribe(wav)
            except SttError as e:
                log.warning("stt backend failed: %s", e)
                continue
            self.last_backend = backend.name
            # A wake phrase at the end of what was said is a command: act on it and
            # keep only what came before it.
            words = plain(text).split()
            for phrase in self.phrases:
                n = len(phrase.split())
                if words[-n:] == phrase.split():
                    self.on_phrase(phrase)
                    text = " ".join(text.split()[:-n]).rstrip(" ,;:-")
                    break
            if not text or plain(text) in self.drop_phrases:
                seg.samples = None
                return True
            seg.text, seg.language, seg.backend = text, language, backend.name
            if self.commit(seg) and self.audio_store is not None:
                try:
                    seg.audio_path = self.audio_store.write(seg)
                    self.audio_stored(seg)
                except OSError:
                    log.exception("could not store audio for %s", seg.id)
            seg.samples = None                      # the audio leaves RAM here
            self.bus.publish(TranscriptCommitted(seg))
            return True
        return False

    def audio_stored(self, seg: Segment) -> None:  # replaced by the app with Store.set_audio_path
        pass
