"""LLM providers for summaries. One small interface, three implementations."""
from __future__ import annotations

import re

import httpx

from .config import Llm as LlmConfig, secret


class LlmError(Exception):
    pass


def strip_reasoning(text: str) -> str:
    """Local reasoning models (Qwen3) may emit their scratchpad in <think> tags."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class OpenAICompatLlm:
    """Any OpenAI-compatible chat endpoint: a LocalAI box on the network, OpenAI itself."""

    def __init__(self, name: str, cfg: LlmConfig, client: httpx.Client | None = None) -> None:
        self.name, self.cfg = name, cfg
        self._client = client or httpx.Client(timeout=cfg.timeout)

    @property
    def model(self) -> str:
        return self.cfg.model

    def complete(self, system: str, user: str) -> str:
        key = secret(self.cfg.api_key_env)
        try:
            r = self._client.post(
                self.cfg.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"} if key else {},
                json={"model": self.cfg.model, "max_tokens": self.cfg.max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user + self.cfg.prompt_suffix}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LlmError(f"{self.name}: {type(e).__name__}: {str(e)[:200]}") from None
        text = strip_reasoning(text)
        if not text:
            raise LlmError(f"{self.name}: empty reply")
        return text


class AnthropicLlm:
    """Claude through the official SDK (`pip install anthropic`)."""

    def __init__(self, name: str, cfg: LlmConfig) -> None:
        self.name, self.cfg = name, cfg

    @property
    def model(self) -> str:
        return self.cfg.model

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise LlmError(f"{self.name}: the 'anthropic' package is not installed") from None
        key = secret(self.cfg.api_key_env)
        if not key:
            raise LlmError(f"{self.name}: {self.cfg.api_key_env} is not set")
        client = anthropic.Anthropic(api_key=key, timeout=self.cfg.timeout)
        try:
            # Server-side fallbacks: if the model declines, the API reruns the
            # request on its default fallback model inside the same call.
            response = client.beta.messages.create(
                model=self.cfg.model, max_tokens=max(self.cfg.max_tokens, 16000), system=system,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": user}])
        except anthropic.RateLimitError as e:
            raise LlmError(f"{self.name}: rate limited: {e.message}") from None
        except anthropic.APIStatusError as e:
            raise LlmError(f"{self.name}: HTTP {e.status_code}: {e.message}") from None
        except anthropic.APIConnectionError as e:
            raise LlmError(f"{self.name}: connection error: {e}") from None
        if response.stop_reason == "refusal":
            raise LlmError(f"{self.name}: the model declined this request")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise LlmError(f"{self.name}: empty reply")
        return text


class OpusOsLlm:
    """The OpusOS bridge (`POST /api/ai/ask`), which holds its own provider keys."""

    def __init__(self, name: str, cfg: LlmConfig, client: httpx.Client | None = None) -> None:
        self.name, self.cfg = name, cfg
        self._client = client or httpx.Client(timeout=cfg.timeout)

    @property
    def model(self) -> str:
        return self.cfg.model or "opusos-default"

    def complete(self, system: str, user: str) -> str:
        body = {"prompt": user[:32000], "system": system[:8000], "max_tokens": min(self.cfg.max_tokens, 8000)}
        if self.cfg.model:
            body["model"] = self.cfg.model
        try:
            r = self._client.post(self.cfg.base_url.rstrip("/") + "/api/ai/ask", json=body, headers={
                "Authorization": f"Bearer {secret(self.cfg.api_key_env)}", "Accept": "application/json"})
            r.raise_for_status()
            text = str(r.json().get("reply", "")).strip()
        except (httpx.HTTPError, ValueError) as e:
            raise LlmError(f"{self.name}: {type(e).__name__}: {str(e)[:200]}") from None
        if not text:
            raise LlmError(f"{self.name}: empty reply")
        return text


def make_llm(name: str, cfg: LlmConfig):
    kinds = {"openai": OpenAICompatLlm, "anthropic": AnthropicLlm, "opusos": OpusOsLlm}
    if cfg.kind not in kinds:
        raise ValueError(f"llm.{name}: unknown kind {cfg.kind!r}")
    return kinds[cfg.kind](name, cfg)
