"""SQLite storage: transcripts, sound events, summaries."""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from .events import Segment

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    text TEXT NOT NULL,
    language TEXT,
    backend TEXT,
    audio_path TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS segments_started ON segments(started_at);
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(text, content='segments', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TABLE IF NOT EXISTS sound_events (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    category TEXT NOT NULL,
    score REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    text TEXT NOT NULL,
    model TEXT NOT NULL,
    segment_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def iso(t: datetime) -> str:
    """UTC, fixed width, so string comparison orders correctly."""
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class Store:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- segments -----------------------------------------------------------
    def add_segment(self, seg: Segment) -> bool:
        """Insert a transcript. Returns False if this id is already stored."""
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO segments VALUES (?,?,?,?,?,?,?,?,?)",
                (seg.id, seg.source, iso(seg.started_at), iso(seg.ended_at), seg.text, seg.language,
                 seg.backend, seg.audio_path, iso(datetime.now(timezone.utc))))
            self._db.commit()
            return cur.rowcount == 1

    def set_audio_path(self, segment_id: str, path: str) -> None:
        with self._lock:
            self._db.execute("UPDATE segments SET audio_path=? WHERE id=?", (path, segment_id))
            self._db.commit()

    def segments_between(self, start: datetime, end: datetime, limit: int = 10000) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM segments WHERE started_at >= ? AND started_at < ? ORDER BY started_at LIMIT ?",
                (iso(start), iso(end), limit)).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 50) -> list[dict]:
        # Quote each term so user input cannot be read as FTS5 query syntax.
        terms = " ".join('"' + t.replace('"', '""') + '"' for t in query.split())
        if not terms:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT s.* FROM segments_fts f JOIN segments s ON s.rowid = f.rowid "
                "WHERE segments_fts MATCH ? ORDER BY s.started_at DESC LIMIT ?", (terms, limit)).fetchall()
        return [dict(r) for r in rows]

    def count_segments(self) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM segments").fetchone()[0]

    # -- sound events ---------------------------------------------------------
    def add_sound_event(self, at: datetime, category: str, score: float) -> None:
        with self._lock:
            self._db.execute("INSERT INTO sound_events(at, category, score) VALUES (?,?,?)", (iso(at), category, score))
            self._db.commit()

    def sound_events(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM sound_events ORDER BY at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- summaries --------------------------------------------------------------
    def add_summary(self, kind: str, start: datetime, end: datetime, text: str, model: str, count: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO summaries(kind, window_start, window_end, text, model, segment_count, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (kind, iso(start), iso(end), text, model, count, iso(datetime.now(timezone.utc))))
            self._db.commit()

    def latest_summary(self, kind: str | None = None) -> dict | None:
        q, args = "SELECT * FROM summaries", ()
        if kind:
            q, args = q + " WHERE kind=?", (kind,)
        with self._lock:
            row = self._db.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone()
        return dict(row) if row else None

    # -- meta / retention ---------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                             (key, value))
            self._db.commit()

    def purge_before(self, cutoff: datetime) -> list[str]:
        """Delete transcripts older than `cutoff`. Returns their audio paths."""
        with self._lock:
            paths = [r[0] for r in self._db.execute(
                "SELECT audio_path FROM segments WHERE started_at < ? AND audio_path IS NOT NULL", (iso(cutoff),))]
            self._db.execute("DELETE FROM segments WHERE started_at < ?", (iso(cutoff),))
            self._db.commit()
        return paths

    def audio_older_than(self, cutoff: datetime) -> list[tuple[str, str]]:
        with self._lock:
            return [(r[0], r[1]) for r in self._db.execute(
                "SELECT id, audio_path FROM segments WHERE started_at < ? AND audio_path IS NOT NULL", (iso(cutoff),))]

    def clear_audio_path(self, segment_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE segments SET audio_path=NULL WHERE id=?", (segment_id,))
            self._db.commit()


def retention_sweep(store: Store, transcript_days: int, audio_days: int) -> None:
    now = datetime.now(timezone.utc)
    if audio_days > 0:
        for seg_id, path in store.audio_older_than(now - timedelta(days=audio_days)):
            _unlink(path)
            store.clear_audio_path(seg_id)
    if transcript_days > 0:
        for path in store.purge_before(now - timedelta(days=transcript_days)):
            _unlink(path)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
