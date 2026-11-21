"""Scheduled summaries of the transcripts, and where they are sent."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import smtplib
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import httpx

from .bus import EventBus
from .config import Email, Summary as SummaryConfig, Webhook, secret
from .events import SummaryReady
from .llm import LlmError
from .store import Store, iso, parse

log = logging.getLogger(__name__)

SYSTEM = """You write digests of what was said in a home, from an automatic transcript of a room microphone.
The transcript has errors: words are misheard, speakers are not labelled, and fragments are missing.
Write a short digest in Markdown with these sections, leaving out any that would be empty:
**Topics** (what was talked about), **Decisions**, **To do** (tasks, with who and when if stated),
**Reminders and dates**.
Only report what the transcript supports. Do not guess names or details, and do not add advice.
Keep times and dates in the words that were used ("tomorrow", "before Friday"); never turn them into calendar dates.
If the transcript holds nothing worth reporting, reply with exactly: Nothing notable."""

PART_SYSTEM = SYSTEM + "\nThis is one part of a longer transcript; another pass will merge the parts."
MERGE_SYSTEM = """Merge these partial digests of one period into a single digest with the same sections
(**Topics**, **Decisions**, **To do**, **Reminders and dates**). Remove repeats. Add nothing new."""


def render(segments: list[dict], tz: ZoneInfo) -> str:
    return "\n".join(f"[{parse(s['started_at']).astimezone(tz):%H:%M}] {s['text']}" for s in segments)


def chunks(text: str, size: int) -> list[str]:
    """Split on line boundaries into pieces of at most `size` characters."""
    out, cur = [], ""
    for line in text.splitlines():
        while len(line) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:size])
            line = line[size:]
        if cur and len(cur) + len(line) + 1 > size:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


class Summariser:
    def __init__(self, cfg: SummaryConfig, store: Store, llms: list, bus: EventBus, sinks: list | None = None) -> None:
        self.cfg, self.store, self.llms, self.bus = cfg, store, llms, bus
        self.sinks = sinks or []
        self.tz = ZoneInfo(cfg.timezone)

    def _complete(self, system: str, user: str) -> tuple[str, str]:
        errors = []
        for llm in self.llms:
            try:
                return llm.complete(system, user), f"{llm.name}/{llm.model}"
            except LlmError as e:
                log.warning("summary model failed: %s", e)
                errors.append(str(e))
        raise LlmError("; ".join(errors) or "no summary model configured")

    def summarise(self, kind: str, start: datetime, end: datetime) -> SummaryReady | None:
        """Summarise [start, end). Returns None when nothing was said."""
        segments = self.store.segments_between(start, end)
        if not segments:
            return None
        header = f"Transcript from {start.astimezone(self.tz):%A %d %B %Y, %H:%M} to {end.astimezone(self.tz):%H:%M}.\n\n"
        parts = chunks(render(segments, self.tz), self.cfg.chunk_chars)
        if len(parts) == 1:
            text, model = self._complete(SYSTEM, header + parts[0])
        else:
            partial = [self._complete(PART_SYSTEM, f"{header}Part {i} of {len(parts)}.\n\n{p}")[0]
                       for i, p in enumerate(parts, 1)]
            text, model = self._complete(MERGE_SYSTEM, "\n\n---\n\n".join(partial))
        self.store.add_summary(kind, start, end, text, model, len(segments))
        event = SummaryReady(kind, start, end, text, model, len(segments))
        self.bus.publish(event)
        for sink in self.sinks:
            try:
                sink(event)
            except Exception:  # noqa: BLE001 - one failing sink must not block the others
                log.exception("summary sink %s failed", getattr(sink, "__name__", type(sink).__name__))
        return event

    # -- schedule -----------------------------------------------------------------
    def _last_end(self, kind: str, default: datetime) -> datetime:
        saved = self.store.get_meta(f"summary_end_{kind}")
        return parse(saved) if saved else default

    def run_due(self, now: datetime | None = None) -> list[SummaryReady]:
        """Run whatever is due at `now`. Safe to call often."""
        now = now or datetime.now(timezone.utc)
        local = now.astimezone(self.tz)
        done = []
        if self.cfg.hourly:
            boundary = local.replace(minute=0, second=0, microsecond=0)
            done += self._run("hourly", boundary, boundary - timedelta(hours=1), max_window=timedelta(hours=6))
        if self.cfg.nightly_at:
            hh, mm = (int(x) for x in self.cfg.nightly_at.split(":"))
            boundary = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if boundary > local:
                boundary -= timedelta(days=1)
            done += self._run("nightly", boundary, boundary - timedelta(days=1), max_window=timedelta(days=2))
        return done

    def _run(self, kind: str, boundary: datetime, default_start: datetime, max_window: timedelta) -> list[SummaryReady]:
        boundary = boundary.astimezone(timezone.utc)
        start = max(self._last_end(kind, default_start.astimezone(timezone.utc)), boundary - max_window)
        if start >= boundary:
            return []
        try:
            event = self.summarise(kind, start, boundary)
        except LlmError as e:
            log.error("%s summary failed, will retry: %s", kind, e)
            return []
        self.store.set_meta(f"summary_end_{kind}", iso(boundary))
        return [event] if event else []


class SummaryScheduler(threading.Thread):
    def __init__(self, summariser: Summariser, on_tick=None, interval: float = 60.0) -> None:
        super().__init__(name="summariser", daemon=True)
        self.summariser, self.on_tick, self.interval = summariser, on_tick, interval
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                self.summariser.run_due()
                if self.on_tick:
                    self.on_tick()
            except Exception:  # noqa: BLE001
                log.exception("summary run failed")


# -- sinks ----------------------------------------------------------------------------
def title(e: SummaryReady, tz: ZoneInfo) -> str:
    return f"{e.kind.capitalize()} summary, {e.window_start.astimezone(tz):%a %d %b %H:%M}–{e.window_end.astimezone(tz):%H:%M}"


class JournalSink:
    """One Markdown file per day."""

    def __init__(self, directory: str, tz: ZoneInfo) -> None:
        self.directory, self.tz = directory, tz

    def __call__(self, e: SummaryReady) -> None:
        day = e.window_start.astimezone(self.tz)
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"{day:%Y-%m-%d}.md")
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(f"# {day:%A %d %B %Y}\n")
            f.write(f"\n## {title(e, self.tz)}\n\n{e.text.strip()}\n")


class EmailSink:
    def __init__(self, cfg: Email, tz: ZoneInfo) -> None:
        self.cfg, self.tz = cfg, tz

    def __call__(self, e: SummaryReady) -> None:
        if e.kind not in self.cfg.kinds:
            return
        msg = EmailMessage()
        msg["Subject"] = f"Hearth: {title(e, self.tz)}"
        msg["From"] = self.cfg.sender
        msg["To"] = ", ".join(self.cfg.to)
        msg.set_content(e.text)
        with smtplib.SMTP(self.cfg.host, self.cfg.port, timeout=30) as smtp:
            if self.cfg.starttls:
                smtp.starttls()
            if self.cfg.username:
                smtp.login(self.cfg.username, secret(self.cfg.password_env))
            smtp.send_message(msg)


class WebhookSink:
    """POSTs the summary as JSON, signed: `X-Hearth-Signature: sha256=<hex HMAC of the body>`."""

    def __init__(self, cfg: Webhook, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._client = client or httpx.Client(timeout=30)

    def __call__(self, e: SummaryReady) -> None:
        body = json.dumps({"kind": e.kind, "window_start": iso(e.window_start), "window_end": iso(e.window_end),
                           "text": e.text, "model": e.model, "segments": e.segment_count}).encode()
        headers = {"Content-Type": "application/json"}
        key = secret(self.cfg.secret_env)
        if key:
            headers["X-Hearth-Signature"] = "sha256=" + hmac.new(key.encode(), body, hashlib.sha256).hexdigest()
        self._client.post(self.cfg.url, content=body, headers=headers).raise_for_status()
