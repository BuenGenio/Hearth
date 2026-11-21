# Hearth

An always-on listening hub for one household, built to sit next to Home Assistant.
From a room microphone it does three things:

- **Guard** recognises danger sounds (smoke alarm, glass break, scream, crying, alarm)
  and raises an event in Home Assistant. It runs whether or not recording is on.
- **Wake phrases** are recognised locally and switch recording on and off.
- **Transcription** runs only while the recording switch is ON: speech is cut into
  segments, sent to a speech-to-text service you choose, stored in SQLite, and
  summarised by an LLM every hour and every night.

```
 microphone ──► capture ──► ring buffer (RAM only)
                              ├─► Guard: YAMNet ───────────────┐
                              ├─► wake phrases ────────────────┤
                              └─► VAD ─► STT queue (RAM) ──► SQLite
                                         only while recording   │
 phone app ──► ingest API ──────────────────────────────────────┤
                         summaries (hourly, nightly) ◄──────────┘
                              └─► MQTT discovery ─► Home Assistant
```

The repository holds two things:

- `hearth/` — the hub, a Python service run in Docker on a machine with the microphone.
- `custom_components/hearth/` — a small Home Assistant integration (installable from the
  Marketplace, or HACS before 2026.11) that adds actions for searching transcripts and
  making summaries.

## What it keeps private

These are tested properties, not intentions (`tests/test_app.py`, `tests/test_segmenter.py`).

- Audio lives in a RAM ring buffer. While recording is OFF nothing is written to disk
  and nothing leaves the machine. Guard and wake-phrase detection keep only decisions.
- When recording turns ON, capture for transcription starts at that moment. Nothing
  said before the switch can end up in a transcript.
- Raw audio is stored only if `storage.store_audio` is on, and only after its transcript
  has been committed. It is off by default.
- Speech waiting for a transcription service waits in RAM, bounded by
  `stt.queue_seconds`. Beyond that the oldest is dropped and counted.
- The **Live** indicator is ON exactly while captured audio can be sent for
  transcription, and is published before the recording state.
- Secrets come from environment variables and never appear in MQTT, logs or the API.

Transcription sends audio to the service you configure. With a LocalAI box on your own
network nothing leaves the house; with OpenAI it does.

## Run the hub

Needs Docker and a machine with the microphone attached. It was checked on a two-core
Celeron without AVX: listening costs about 17 % of one core and 110 MB of RAM.

```sh
cd deploy
cp .env.example .env && chmod 600 .env      # fill in keys and tokens
cp hearth.example.toml hearth.toml          # set the microphone and the endpoints
docker compose up -d --build
```

Find the microphone's ALSA name with `arecord -L` and use the card id form
(`hw:CARD=<id>,DEV=0`), never an index. `source = "file"` replays a WAV instead, and
`source = "none"` runs the hub with phone ingest only.

Every setting and its default is in [`hearth/config.py`](hearth/config.py). The config
file only needs the settings you change.

### Home Assistant entities

The hub announces itself over MQTT discovery, so Home Assistant needs only the MQTT
integration and a broker.

| Entity | What it is |
|---|---|
| `switch.hearth_recording` | Transcription on or off |
| `switch.hearth_guard` | Sound guard on or off (on by default) |
| `binary_sensor.hearth_live` | Audio can currently be sent for transcription |
| `event.hearth_sound` | Guard events: `smoke_alarm`, `glass_break`, `scream`, `crying`, `alarm` |
| `sensor.hearth_last_summary` | Time of the last summary; the text is in the `text` attribute |
| diagnostics | Microphone connected, microphone signal, transcription backend, queue, dropped segments |

To turn recording on or off: say a wake phrase, use the switch, expose the switch to
Assist, or drive it from any automation (a button, a Zigbee remote). Muting the
transmitter also stops everything, including Guard; the "Microphone signal" sensor
goes off when that happens.

An automation for Guard:

```yaml
automation:
  - alias: Hearth alarm sound
    triggers:
      - trigger: state
        entity_id: event.hearth_sound
    conditions:
      - condition: template
        value_template: "{{ trigger.to_state.attributes.event_type in ['smoke_alarm', 'glass_break'] }}"
    actions:
      - action: notify.notify
        data:
          title: Hearth
          message: "Heard {{ trigger.to_state.attributes.event_type }} at home"
```

Guard supplements certified smoke, CO and intruder alarms. It does not replace them.

### Wake phrases

Phrases are plain text in the config; no model training is needed. The bundled model
is English only.

```toml
[[wake.phrases]]
text = "hearth start recording"
action = "recording_on"          # recording_on | recording_off | recording_toggle
```

A phrase said while recording is also caught from the transcript, which is more
reliable than the small on-device model, and is then left out of the stored text.

### Summaries

`summary.llm` lists the models to try in order. `kind = "openai"` works with any
OpenAI-compatible endpoint (LocalAI, OpenAI), `"anthropic"` uses the Claude API and
needs `pip install anthropic` in the image, and `"opusos"` uses an OpusOS bridge.
Summaries go to the MQTT sensor, a Markdown journal (one file per day under
`<data>/journal`), email, and a signed webhook (`X-Hearth-Signature: sha256=<HMAC>`).

### API

`docs/openapi.json` is the contract; regenerate it with `python -m hearth.api`.
Every call except `/healthz` needs `Authorization: Bearer <token>`, with one token per
device in `HEARTH_API_TOKENS`. `POST /api/v1/segments` takes finished transcript
segments from a phone and is idempotent on the segment id.

## Home Assistant integration

The integration adds four actions that call the hub's API. It creates no entities:
those come from MQTT discovery.

1. Home Assistant 2026.11 and later: Settings → Marketplace → **Add from link**, enter
   `BuenGenio/Hearth`, then open Hearth and select **Install**. Adding a repository by
   link needs the Marketplace connected to a GitHub account. On older versions, add
   this repository to HACS as a custom repository of type *Integration* and install it.
   Restart Home Assistant.
2. Settings → Devices & services → Add integration → Hearth. Enter the hub's URL
   (an address Home Assistant can reach, port 8377) and one of the API tokens.

| Action | Returns |
|---|---|
| `hearth.search` | Transcript segments matching `query` |
| `hearth.get_transcript` | Segments between `start` and `end` |
| `hearth.summarise` | A fresh summary of the last `hours` |
| `hearth.latest_summary` | The most recent stored summary |

It needs Home Assistant 2026.9 or newer.

## Limits worth knowing

- YAMNet has no carbon-monoxide class. A CO alarm is caught, if at all, as a smoke alarm
  or a generic alarm.
- Guard thresholds were set on a small set of recordings (see below). Tune them on your
  own room with `tools/guard_check.py`, which prints what Guard would raise for any WAV.
- The detectors need speech at a normal level. A quiet input is raised automatically by
  up to 20 dB (`audio.agc`); the current gain is in `/api/v1/status`.
- Deleting a transcript does not remove it from filesystem snapshots or backups.
- Small local models make weak summaries. A 4B model dropped details and, before the
  prompt forbade it, invented calendar dates.

## Development

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[models,dev]"
sh tools/fetch_models.sh models       # checksummed; tests that need models skip without them
.venv/bin/python -m pytest            # the hub
.venv-ha/bin/python -m pytest tests_ha  # the integration (see tests_ha/conftest.py)
```

- `hearth.capture`, `hearth.features` and `hearth.models` import nothing from MQTT,
  Home Assistant or SQLite, so they can be reused on their own. Detectors publish to an
  in-process bus (`hearth/bus.py`); MQTT is one subscriber.
- `tools/probe/probe.py` checks that every native dependency runs on a given CPU.
- `tools/e2e.sh` runs the built image against a recording, a real STT endpoint and a
  throwaway MQTT broker, and prints what reached MQTT and SQLite.

### What has been checked

- 57 hub tests, including the privacy properties above; each of seven deliberate
  breakages of those properties made a test fail.
- 58 integration tests on Home Assistant 2026.10, plus a live run on 2026.11.
  `hassfest` passes in GitHub Actions. The HACS validation action passes every check
  except the licence, which the repository does not have yet.
- Guard on public recordings: glass break 3 of 3, sirens 3 of 3, alarm clocks 2 of 3,
  baby crying 1 of 3, and no event on 11 recordings of knocking, vacuuming and speech.
  No real smoke-alarm recording was tested.
- End to end on the target machine with a recording, and over the air on a laptop
  microphone (speech and wake phrases; glass break was detected in one of two runs).
