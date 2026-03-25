from __future__ import annotations

from ..config import LLMRuntimeConfig
from .base import LLMProvider
from .bedrock import BedrockProvider
from .ollama import OllamaProvider


def build_llm_provider(runtime: LLMRuntimeConfig) -> LLMProvider:
    provider = runtime.provider.lower().strip()
    if provider == "ollama":
        return OllamaProvider(
            base_url=runtime.base_url,
            timeout_seconds=runtime.timeout_seconds,
            temperature=runtime.temperature,
        )
    if provider == "bedrock":
        return BedrockProvider(
            region_name=runtime.aws_region,
            timeout_seconds=runtime.timeout_seconds,
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            top_p=runtime.top_p,
            top_k=runtime.top_k,
            stop_sequences=runtime.stop_sequences,
            aws_profile=runtime.aws_profile,
        )
    raise ValueError(f"Unsupported LLM provider: {runtime.provider}")
