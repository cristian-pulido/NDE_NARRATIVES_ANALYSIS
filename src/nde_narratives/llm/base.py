from __future__ import annotations

from abc import ABC, abstractmethod

from .types import LLMProviderResponse, LLMRequest


class LLMProvider(ABC):
    @abstractmethod
    def generate_structured(self, request: LLMRequest) -> LLMProviderResponse:
        raise NotImplementedError
