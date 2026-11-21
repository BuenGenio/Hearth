"""HTTP API: phone ingest (Hearth Mobile), search, state and status.

Bound to one address (the tailnet address in production) and guarded by a
bearer token per device. `python -m hearth.api` prints the OpenAPI document,
which is the contract the mobile app is written against.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from . import __version__
from .events import Segment
from .store import Store

MAX_BATCH = 500
AUDIO_TYPES = {"opus": ".opus", "ogg": ".ogg", "wav": ".wav", "m4a": ".m4a", "flac": ".flac"}


class SegmentIn(BaseModel):
    id: str = Field(pattern=r"^[0-9a-fA-F-]{36}$", description="UUID chosen by the device; makes retries safe.")
    started_at: datetime
    ended_at: datetime
    text: str = Field(min_length=1, max_length=20000)
    language: str | None = Field(default=None, max_length=16)
    stt: str | None = Field(default=None, max_length=64, description="What transcribed it, e.g. 'android-on-device'.")
    audio_b64: str | None = Field(default=None, description="Optional audio. Stored only if the hub keeps raw audio.")
    audio_format: str | None = Field(default=None, description="opus, ogg, wav, m4a or flac.")


class SegmentsIn(BaseModel):
    segments: list[SegmentIn] = Field(max_length=MAX_BATCH)


class SegmentsOut(BaseModel):
    accepted: int
    duplicates: int


class StateIn(BaseModel):
    recording: bool | None = None
    guard: bool | None = None


class SummaryRunIn(BaseModel):
    kind: str = Field(default="manual", max_length=32)
    hours: float = Field(default=1.0, gt=0, le=72)


def parse_tokens(raw: str) -> dict[str, str]:
    """"phone:abc,ha:def" -> {"abc": "phone", "def": "ha"}"""
    out = {}
    for item in raw.split(","):
        name, _, token = item.strip().partition(":")
        if name and token:
            out[token] = name
    return out


def create_app(store: Store, tokens: dict[str, str], state=None, status: Callable[[], dict] | None = None,
               summariser=None, audio_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Hearth", version=__version__,
                  description="Hub API for Hearth: segment ingest from devices, search, state and summaries.")

    def device(authorization: str = Header(default="")) -> str:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            for known, name in tokens.items():
                if hmac.compare_digest(known.encode(), token.encode()):
                    return name
        raise HTTPException(401, "invalid or missing bearer token", {"WWW-Authenticate": "Bearer"})

    @app.get("/healthz", summary="Liveness; no authentication")
    def healthz() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/api/v1/status")
    def get_status(_: str = Depends(device)) -> dict:
        out = {"version": __version__, "segments": store.count_segments()}
        if state is not None:
            snap = state.snapshot()
            out.update(recording=snap.recording, guard=snap.guard, live=snap.live,
                       mic_connected=state.mic_connected, mic_signal=state.mic_signal)
        if status:
            out.update(status())
        return out

    @app.post("/api/v1/state", summary="Switch recording and/or guard")
    def set_state(body: StateIn, who: str = Depends(device)) -> dict:
        if state is None:
            raise HTTPException(503, "state is not available")
        if body.recording is not None:
            state.set_recording(body.recording, by=f"api:{who}")
        if body.guard is not None:
            state.set_guard(body.guard, by=f"api:{who}")
        snap = state.snapshot()
        return {"recording": snap.recording, "guard": snap.guard, "live": snap.live}

    @app.post("/api/v1/segments", response_model=SegmentsOut,
              summary="Upload finished segments; idempotent on segment id")
    def post_segments(body: SegmentsIn, who: str = Depends(device)) -> SegmentsOut:
        accepted = duplicates = 0
        for s in body.segments:
            if s.ended_at < s.started_at:
                raise HTTPException(422, f"segment {s.id}: ended_at is before started_at")
            seg = Segment(id=s.id.lower(), source=who, started_at=_utc(s.started_at), ended_at=_utc(s.ended_at),
                          text=s.text.strip(), language=s.language, backend=s.stt or "device")
            if not store.add_segment(seg):
                duplicates += 1
                continue
            accepted += 1
            if s.audio_b64 and audio_dir:
                ext = AUDIO_TYPES.get((s.audio_format or "").lower())
                if ext is None:
                    raise HTTPException(422, f"segment {s.id}: unsupported audio_format")
                try:
                    data = base64.b64decode(s.audio_b64, validate=True)
                except (binascii.Error, ValueError):
                    raise HTTPException(422, f"segment {s.id}: audio_b64 is not valid base64") from None
                folder = os.path.join(audio_dir, seg.started_at.strftime("%Y/%m/%d"))
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, seg.id + ext)
                with open(path, "wb") as f:
                    f.write(data)
                store.set_audio_path(seg.id, path)
        return SegmentsOut(accepted=accepted, duplicates=duplicates)

    @app.get("/api/v1/segments", summary="Transcripts in a time range")
    def get_segments(start: datetime, end: datetime, limit: int = Query(1000, ge=1, le=10000),
                     _: str = Depends(device)) -> dict:
        return {"segments": [_public(r) for r in store.segments_between(_utc(start), _utc(end), limit)]}

    @app.get("/api/v1/search", summary="Full-text search over transcripts")
    def search(q: str = Query(min_length=1, max_length=200), limit: int = Query(50, ge=1, le=500),
               _: str = Depends(device)) -> dict:
        return {"segments": [_public(r) for r in store.search(q, limit)]}

    @app.get("/api/v1/summaries/latest")
    def latest_summary(kind: str | None = None, _: str = Depends(device)) -> dict:
        return {"summary": store.latest_summary(kind)}

    @app.post("/api/v1/summaries/run", summary="Summarise the last N hours now")
    def run_summary(body: SummaryRunIn, _: str = Depends(device)) -> dict:
        if summariser is None:
            raise HTTPException(503, "summaries are not enabled")
        from .llm import LlmError

        end = datetime.now(timezone.utc)
        try:
            event = summariser.summarise(body.kind, end - timedelta(hours=body.hours), end)
        except LlmError as e:
            raise HTTPException(502, f"summary model failed: {e}") from None
        return {"summary": None if event is None else {"text": event.text, "model": event.model,
                                                       "segments": event.segment_count}}

    @app.get("/api/v1/sounds", summary="Recent guard events")
    def sounds(limit: int = Query(100, ge=1, le=1000), _: str = Depends(device)) -> dict:
        return {"events": store.sound_events(limit)}

    return app


def _utc(t: datetime) -> datetime:
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _public(row: dict) -> dict:
    row = dict(row)
    row["has_audio"] = bool(row.pop("audio_path", None))    # never expose host paths
    return row


if __name__ == "__main__":
    print(json.dumps(create_app(Store(":memory:"), {}).openapi(), indent=2))
