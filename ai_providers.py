"""
Shared provider utilities for interacting with OpenRouter and Cohere.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

import httpx
import cohere
from langchain_core.messages import BaseMessage


def _message_to_dict(message: BaseMessage) -> Mapping[str, str]:
    """Convert a LangChain message into the OpenAI/OpenRouter schema."""
    role_map = {
        "human": "user",
        "user": "user",
        "system": "system",
        "ai": "assistant",
        "assistant": "assistant",
        "tool": "tool",
    }
    role = role_map.get(getattr(message, "type", "user"), "user")
    content = getattr(message, "content", "")
    if isinstance(content, list):
        # Flatten multimodal content to string segments
        content = " ".join(str(part) for part in content)
    return {"role": role, "content": content}


def _default_openrouter_headers() -> MutableMapping[str, str]:
    headers: MutableMapping[str, str] = {
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_APP_TITLE", "Digital Me Orchestrator")
    headers["X-Title"] = title
    return headers


class OpenRouterChat:
    """
    Minimal chat client compatible with LangChain prompt messages, backed by OpenRouter.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 60.0,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        if not model:
            raise ValueError("OpenRouter model is required.")

        self.api_key = api_key
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        headers = _default_openrouter_headers()
        headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self.headers = headers

    def invoke(self, messages: Sequence[BaseMessage]) -> SimpleNamespace:
        """Invoke the chat completion API and return an object with a .content attribute."""
        payload = {
            "model": self.model,
            "messages": [_message_to_dict(message) for message in messages],
            "temperature": self.temperature,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = httpx.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError("OpenRouter response did not include choices.")
                message = choices[0].get("message", {})
                content = message.get("content")
                if content is None:
                    raise ValueError("OpenRouter response missing message content.")
                return SimpleNamespace(content=content)
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                if attempt + 1 >= self.max_retries:
                    break
                backoff = 2 ** attempt
                time.sleep(backoff)

        raise RuntimeError(f"OpenRouter request failed: {last_exc}") from last_exc


class CohereEmbeddingsClient:
    """Simple embedding client that exposes embed_query and embed_documents."""

    def __init__(
        self,
        api_key: str,
        model: str,
        truncate: Optional[str] = "END",
    ) -> None:
        if not api_key:
            raise ValueError("Cohere API key is required.")
        if not model:
            raise ValueError("Cohere embedding model is required.")

        self.client = cohere.Client(api_key=api_key)
        self.model = model
        self.truncate = truncate

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
                truncate=self.truncate,
            )
            return list(response.embeddings[0])
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Cohere embed_query failed: {exc}") from exc

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        batch = list(texts)
        if not batch:
            return []
        try:
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
                truncate=self.truncate,
            )
            return [list(embedding) for embedding in response.embeddings]
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Cohere embed_documents failed: {exc}") from exc


