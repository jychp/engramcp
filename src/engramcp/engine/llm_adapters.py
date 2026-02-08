"""Concrete LLM adapters and factory helpers."""

from __future__ import annotations

import asyncio
import json
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from engramcp.config import LLMConfig
from engramcp.engine.extraction import LLMAdapter
from engramcp.engine.extraction import LLMError


class NoopLLMAdapter(LLMAdapter):
    """Deterministic adapter that returns an empty extraction payload."""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        del prompt, temperature, max_tokens, timeout_seconds
        return (
            '{"entities":[],"relations":[],"claims":[],'
            '"fragment_ids_processed":[],"errors":[]}'
        )


class OpenAICompatibleLLMAdapter(LLMAdapter):
    """OpenAI-compatible chat-completions adapter."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        return await asyncio.to_thread(
            self._complete_sync,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def _complete_sync(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"provider HTTP {exc.code}: {detail[:200]}") from exc
        except URLError as exc:
            raise LLMError(f"provider network error: {exc.reason}") from exc
        except OSError as exc:
            raise LLMError(f"provider IO error: {exc}") from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(
                "provider response missing choices[0].message.content"
            ) from exc

        if isinstance(content, str):
            return content
        raise LLMError("provider response content must be a string")


def build_llm_adapter(config: LLMConfig) -> LLMAdapter:
    """Create a concrete adapter from ``LLMConfig``."""

    provider = config.provider.strip().lower()
    if provider == "openai":
        if not config.api_key:
            raise ValueError("llm_config.api_key is required when provider='openai'")
        return OpenAICompatibleLLMAdapter(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    if provider == "noop":
        return NoopLLMAdapter()
    raise ValueError(
        f"Unsupported llm_config.provider '{config.provider}'. "
        "Supported providers: openai, noop."
    )
