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
        # PreprocessingConfig also calls this factory but does not expose Bedrock runtime fields.
        # Preserve the previous actionable behavior (unsupported provider) instead of AttributeError.
        if not hasattr(runtime, "aws_region"):
            raise ValueError(f"Unsupported LLM provider: {runtime.provider}")
        return BedrockProvider(
            region_name=getattr(runtime, "aws_region", "us-east-1"),
            timeout_seconds=runtime.timeout_seconds,
            temperature=runtime.temperature,
            max_tokens=getattr(runtime, "max_tokens", 512),
            top_p=getattr(runtime, "top_p", None),
            top_k=getattr(runtime, "top_k", None),
            stop_sequences=getattr(runtime, "stop_sequences", None),
            aws_profile=getattr(runtime, "aws_profile", None),
        )
    raise ValueError(f"Unsupported LLM provider: {runtime.provider}")
