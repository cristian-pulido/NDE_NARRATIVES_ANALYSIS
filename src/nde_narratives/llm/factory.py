from __future__ import annotations

from ..config import LLMRuntimeConfig
from .base import LLMProvider
from .ollama import OllamaProvider


def build_llm_provider(runtime: LLMRuntimeConfig) -> LLMProvider:
    provider = runtime.provider.lower().strip()
    if provider == "ollama":
        return OllamaProvider(
            base_url=runtime.base_url,
            timeout_seconds=runtime.timeout_seconds,
            temperature=runtime.temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {runtime.provider}")
