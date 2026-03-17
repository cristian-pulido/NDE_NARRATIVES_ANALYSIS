from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMRequest:
    participant_code: str
    section: str
    prompt: str
    response_schema: dict[str, Any]
    model: str
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMProviderResponse:
    provider: str
    model: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMExecutionResult:
    provider: str
    model: str
    raw_text: str
    parsed_prediction: dict[str, str]
    provider_metadata: dict[str, Any] = field(default_factory=dict)
