from __future__ import annotations

import json
from typing import Any

from .base import LLMProvider
from .types import LLMProviderResponse, LLMRequest


class BedrockProvider(LLMProvider):
    def __init__(
        self,
        *,
        region_name: str,
        timeout_seconds: int,
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        aws_profile: str | None = None,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Bedrock provider requires boto3/botocore. Install project dependencies including boto3."
            ) from exc

        session_kwargs: dict[str, Any] = {}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        session = boto3.Session(**session_kwargs)

        self.client = session.client(
            "bedrock-runtime",
            region_name=region_name,
            config=Config(connect_timeout=timeout_seconds, read_timeout=timeout_seconds),
        )
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = list(stop_sequences or [])

    def _inference_config(self, request: LLMRequest) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": self.temperature if request.temperature is None else float(request.temperature),
            "maxTokens": self.max_tokens,
        }
        if self.top_p is not None:
            config["topP"] = float(self.top_p)
        if self.stop_sequences:
            config["stopSequences"] = list(self.stop_sequences)
        return config

    def _additional_model_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if self.top_k is not None:
            fields["top_k"] = int(self.top_k)
        return fields

    @staticmethod
    def _extract_text_from_converse(payload: dict[str, Any]) -> str:
        output = payload.get("output")
        if not isinstance(output, dict):
            raise RuntimeError("Bedrock converse response did not include output payload.")

        message = output.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Bedrock converse response did not include output.message.")

        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise RuntimeError("Bedrock converse response did not include output.message.content.")

        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)

        if not text_parts:
            raise RuntimeError("Bedrock converse response did not include text content.")
        return "\n".join(text_parts)

    def generate_structured(self, request: LLMRequest) -> LLMProviderResponse:
        response = self.client.converse(
            modelId=request.model,
            messages=[{"role": "user", "content": [{"text": request.prompt}]}],
            inferenceConfig=self._inference_config(request),
            additionalModelRequestFields=self._additional_model_fields(),
        )

        raw_text = self._extract_text_from_converse(response)
        metadata: dict[str, Any] = {
            "stop_reason": response.get("stopReason"),
            "usage": response.get("usage", {}),
            "metrics": response.get("metrics", {}),
        }
        return LLMProviderResponse(
            provider="bedrock",
            model=request.model,
            raw_text=raw_text,
            metadata=metadata,
        )

