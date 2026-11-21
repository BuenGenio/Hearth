"""Configuration: one TOML file, secrets from the environment.

Secrets never live in the TOML file. A setting named `*_env` holds the *name*
of an environment variable; the value is looked up when it is needed.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints


def secret(env_name: str) -> str:
    return os.environ.get(env_name, "") if env_name else ""


@dataclass
class Audio:
    source: str = "alsa"            # alsa | file | none
    device: str = "default"         # ALSA device, by card id: hw:CARD=<id>,DEV=0
    rate: int = 48000               # the device's native rate
    channels: int = 1
    file: str = ""                  # for source = "file"
    file_loop: bool = False
    ring_seconds: float = 30.0
    gain_db: float = 0.0            # fixed gain applied first
    agc: bool = True                # then raise a quiet input towards agc_target_dbfs
    agc_target_dbfs: float = -26.0
    agc_max_gain_db: float = 20.0
    silence_floor: float = 1e-4     # raw peak below this counts as "no signal"
    silence_seconds: float = 10.0


@dataclass
class GuardCategory:
    classes: list[str] = field(default_factory=list)
    threshold: float = 0.3
    min_hits: int = 3               # N of the last `window` inferences (four inferences a second)
    window: int = 8
    cooldown_seconds: float = 60.0


def default_categories() -> dict[str, GuardCategory]:
    # Names are YAMNet's own (read from the model file; checked at start-up).
    # YAMNet has no carbon-monoxide class: a CO alarm is caught, if at all, as
    # a smoke alarm or a generic alarm.
    return {
        "smoke_alarm": GuardCategory(["Smoke detector, smoke alarm", "Fire alarm"], threshold=0.25, min_hits=3, window=8),
        "glass_break": GuardCategory(["Shatter", "Breaking"], threshold=0.3, min_hits=1, window=2),
        "scream": GuardCategory(["Screaming"], threshold=0.35, min_hits=3, window=8),
        "crying": GuardCategory(["Crying, sobbing", "Baby cry, infant cry"], threshold=0.3, min_hits=3, window=12,
                                cooldown_seconds=300),
        "alarm": GuardCategory(["Alarm", "Siren", "Civil defense siren", "Car alarm", "Buzzer"], threshold=0.3,
                               min_hits=3, window=12),
    }


@dataclass
class Guard:
    enabled_default: bool = True
    model: str = "/models/yamnet.tflite"
    hop_seconds: float = 0.24375    # a quarter of YAMNet's window: a short sound cannot fall between windows
    categories: dict[str, GuardCategory] = field(default_factory=default_categories)


@dataclass
class WakePhrase:
    text: str = ""
    action: str = ""                # recording_on | recording_off | recording_toggle


def default_phrases() -> list[WakePhrase]:
    return [WakePhrase("hearth start recording", "recording_on"),
            WakePhrase("hearth stop recording", "recording_off")]


@dataclass
class Wake:
    enabled: bool = True
    model_dir: str = "/models/kws"
    threshold: float = 0.25
    boost: float = 1.5
    phrases: list[WakePhrase] = field(default_factory=default_phrases)


@dataclass
class Vad:
    model: str = "/models/silero_vad.onnx"
    threshold: float = 0.5
    min_silence: float = 0.7
    min_speech: float = 0.3
    max_speech: float = 30.0
    preroll: float = 0.3


@dataclass
class SttBackend:
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    language: str = ""              # empty: let the model detect
    timeout: float = 120.0


def default_stt_backends() -> dict[str, SttBackend]:
    return {
        "localai": SttBackend("http://localhost:8080/v1", "whisper-large-turbo-q5_0", "LOCALAI_API_KEY"),
        "openai": SttBackend("https://api.openai.com/v1", "gpt-4o-mini-transcribe", "OPENAI_API_KEY"),
    }


@dataclass
class Stt:
    order: list[str] = field(default_factory=lambda: ["localai", "openai"])
    backends: dict[str, SttBackend] = field(default_factory=default_stt_backends)
    queue_seconds: float = 1800.0   # RAM queue bound; the oldest speech is dropped beyond it
    retry_min: float = 5.0
    retry_max: float = 300.0
    # Whisper invents these on near-silence; segments that are only this are discarded.
    drop_phrases: list[str] = field(default_factory=lambda: ["thank you", "thanks for watching", "you"])


@dataclass
class Storage:
    data_dir: str = "/data"
    audio_dir: str = ""             # default: <data_dir>/audio
    store_audio: bool = False
    restore_recording: bool = True  # false: always start with recording OFF
    transcript_retention_days: int = 0   # 0 = keep
    audio_retention_days: int = 0


@dataclass
class Mqtt:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 1883
    username: str = ""
    password_env: str = "HEARTH_MQTT_PASSWORD"
    discovery_prefix: str = "homeassistant"
    base_topic: str = "hearth"


@dataclass
class Llm:
    kind: str = "openai"            # openai (any compatible endpoint) | anthropic | opusos
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    max_tokens: int = 2000
    timeout: float = 300.0
    # Appended to the transcript. "/no_think" stops Qwen3 spending its whole
    # token budget on hidden reasoning (which came back as an empty summary).
    prompt_suffix: str = ""


def default_llms() -> dict[str, Llm]:
    return {
        "localai": Llm("openai", "http://localhost:8080/v1", "qwen3-4b", "LOCALAI_API_KEY",
                       prompt_suffix="\n/no_think"),
        "anthropic": Llm("anthropic", "https://api.anthropic.com", "claude-opus-5-5", "ANTHROPIC_API_KEY"),
        "opusos": Llm("opusos", "http://127.0.0.1:8001", "", "OPUSOS_TOKEN"),
    }


@dataclass
class Email:
    enabled: bool = False
    host: str = ""
    port: int = 587
    starttls: bool = True
    username: str = ""
    password_env: str = "HEARTH_SMTP_PASSWORD"
    sender: str = ""
    to: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=lambda: ["nightly"])


@dataclass
class Webhook:
    enabled: bool = False
    url: str = ""
    secret_env: str = "HEARTH_WEBHOOK_SECRET"


@dataclass
class Summary:
    enabled: bool = True
    hourly: bool = True
    nightly_at: str = "23:30"       # local time, HH:MM; empty to disable
    timezone: str = "UTC"
    llm: list[str] = field(default_factory=lambda: ["localai"])   # tried in order
    chunk_chars: int = 12000        # transcripts longer than this are summarised in parts
    journal: bool = True
    journal_dir: str = ""           # default: <data_dir>/journal
    email: Email = field(default_factory=Email)
    webhook: Webhook = field(default_factory=Webhook)


@dataclass
class Api:
    enabled: bool = True
    # Addresses to listen on. Add the tailnet address for phones and the
    # address Home Assistant reaches the host on; never 0.0.0.0 unless firewalled.
    hosts: list[str] = field(default_factory=lambda: ["127.0.0.1"])
    port: int = 8377
    tokens_env: str = "HEARTH_API_TOKENS"   # "phone:token1,ha:token2"


@dataclass
class Config:
    audio: Audio = field(default_factory=Audio)
    guard: Guard = field(default_factory=Guard)
    wake: Wake = field(default_factory=Wake)
    vad: Vad = field(default_factory=Vad)
    stt: Stt = field(default_factory=Stt)
    storage: Storage = field(default_factory=Storage)
    mqtt: Mqtt = field(default_factory=Mqtt)
    llm: dict[str, Llm] = field(default_factory=default_llms)
    summary: Summary = field(default_factory=Summary)
    api: Api = field(default_factory=Api)

    @property
    def audio_dir(self) -> str:
        return self.storage.audio_dir or os.path.join(self.storage.data_dir, "audio")

    @property
    def journal_dir(self) -> str:
        return self.summary.journal_dir or os.path.join(self.storage.data_dir, "journal")


def _scalar(tp: Any, value: Any, path: str) -> Any:
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(tp, type) and not isinstance(value, tp):
        raise ValueError(f"{path}: expected {tp.__name__}, got {type(value).__name__}")
    return value


def _build(tp: Any, value: Any, path: str) -> Any:
    """Make a value of type `tp` from parsed TOML."""
    origin = get_origin(tp)
    if is_dataclass(tp):
        return _merge(tp(), value, path)
    if origin is dict:
        return {k: _build(get_args(tp)[1], v, f"{path}.{k}") for k, v in value.items()}
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a list")
        return [_build(get_args(tp)[0], v, f"{path}[{i}]") for i, v in enumerate(value)]
    return _scalar(tp, value, path)


def _merge(obj: Any, value: Any, path: str) -> Any:
    """Lay parsed TOML over an existing dataclass, in place.

    Tables merge key by key, so overriding one setting (or one backend, or one
    guard category) keeps every default around it. Lists are replaced whole.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a table")
    hints = get_type_hints(type(obj))
    unknown = set(value) - {f.name for f in fields(obj)}
    if unknown:
        raise ValueError(f"{path or 'config'}: unknown setting(s) {sorted(unknown)}")
    for key, v in value.items():
        tp, here = hints[key], f"{path}.{key}" if path else key
        current = getattr(obj, key)
        if is_dataclass(tp):
            _merge(current, v, here)
        elif get_origin(tp) is dict and is_dataclass(get_args(tp)[1]):
            if not isinstance(v, dict):
                raise ValueError(f"{here}: expected a table")
            for name, entry in v.items():
                if name in current:
                    _merge(current[name], entry, f"{here}.{name}")
                else:
                    current[name] = _build(get_args(tp)[1], entry, f"{here}.{name}")
        else:
            setattr(obj, key, _build(tp, v, here))
    return obj


def load(path: str | None = None) -> Config:
    """Load the TOML at `path` (or $HEARTH_CONFIG) over the defaults."""
    path = path or os.environ.get("HEARTH_CONFIG", "")
    cfg = Config()
    if path:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f), "")
    return cfg
