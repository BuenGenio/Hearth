import base64
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from hearth.api import create_app, parse_tokens
from hearth.bus import EventBus
from hearth.state import State
from hearth.store import Store

AUTH = {"Authorization": "Bearer tok-phone"}
SEG = {"id": "3f2b8c1e-0000-4000-8000-000000000001", "started_at": "2026-11-21T12:00:00Z",
       "ended_at": "2026-11-21T12:00:04Z", "text": "pick up the parcel", "language": "en"}


def client(tmp_path, audio=False):
    store = State and Store(":memory:")
    state = State(EventBus(), str(tmp_path / "state.json"))
    app = create_app(store, parse_tokens("phone:tok-phone, ha:tok-ha"), state,
                     audio_dir=str(tmp_path / "audio") if audio else None)
    return TestClient(app), store, state


def test_requires_a_known_bearer_token(tmp_path):
    c, _, _ = client(tmp_path)
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/v1/status").status_code == 401
    assert c.get("/api/v1/status", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/api/v1/segments", json={"segments": [SEG]}).status_code == 401
    assert c.get("/api/v1/status", headers=AUTH).json()["recording"] is False


def test_ingest_is_idempotent_and_tags_the_device_from_the_token(tmp_path):
    c, store, _ = client(tmp_path)
    r = c.post("/api/v1/segments", json={"segments": [SEG]}, headers=AUTH)
    assert r.json() == {"accepted": 1, "duplicates": 0}
    r = c.post("/api/v1/segments", json={"segments": [SEG]}, headers=AUTH)       # the phone retries
    assert r.json() == {"accepted": 0, "duplicates": 1}
    rows = c.get("/api/v1/search", params={"q": "parcel"}, headers=AUTH).json()["segments"]
    assert len(rows) == 1 and rows[0]["source"] == "phone" and "audio_path" not in rows[0]
    got = c.get("/api/v1/segments", params={"start": "2026-11-21T00:00:00Z", "end": "2026-11-22T00:00:00Z"},
                headers=AUTH).json()["segments"]
    assert got[0]["text"] == "pick up the parcel"


def test_rejects_malformed_segments(tmp_path):
    c, store, _ = client(tmp_path)
    for bad in ({**SEG, "id": "not-a-uuid"}, {**SEG, "text": ""}, {**SEG, "ended_at": "2026-11-21T11:00:00Z"}):
        assert c.post("/api/v1/segments", json={"segments": [bad]}, headers=AUTH).status_code == 422
    assert store.count_segments() == 0


def test_phone_audio_is_kept_only_when_the_hub_stores_audio(tmp_path):
    body = {"segments": [{**SEG, "audio_b64": base64.b64encode(b"OggS....").decode(), "audio_format": "opus"}]}
    c, store, _ = client(tmp_path, audio=False)
    c.post("/api/v1/segments", json=body, headers=AUTH)
    assert not (tmp_path / "audio").exists()
    c, store, _ = client(tmp_path, audio=True)
    c.post("/api/v1/segments", json=body, headers=AUTH)
    files = list((tmp_path / "audio").rglob("*.opus"))
    assert len(files) == 1 and files[0].read_bytes() == b"OggS...."
    assert c.get("/api/v1/search", params={"q": "parcel"}, headers=AUTH).json()["segments"][0]["has_audio"] is True


def test_state_can_be_switched_over_the_api(tmp_path):
    c, _, state = client(tmp_path)
    r = c.post("/api/v1/state", json={"recording": True}, headers=AUTH)
    assert r.json()["recording"] is True and state.recording is True and state.guard is True
