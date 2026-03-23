from __future__ import annotations

import json
from urllib import error, request as urllib_request

from .base import LLMProvider
from .types import LLMProviderResponse, LLMRequest


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, timeout_seconds: int, temperature: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.temperature = float(temperature)
        self.generate_url = self._resolve_generate_url(self.base_url)

    @staticmethod
    def _resolve_generate_url(base_url: str) -> str:
        if base_url.endswith("/api/generate"):
            return base_url
        return f"{base_url}/api/generate"

    @staticmethod
    def _extract_raw_text(payload: dict[str, object]) -> tuple[str, str]:
        response_text = payload.get("response")
        if isinstance(response_text, str) and response_text.strip():
            return response_text, "response"

        # Some reasoning-capable Ollama models return the structured payload in
        # `thinking` while leaving `response` blank when `format` is supplied.
        thinking_text = payload.get("thinking")
        if isinstance(thinking_text, str) and thinking_text.strip():
            return thinking_text, "thinking"

        raise RuntimeError("Ollama response did not include a non-empty 'response' field.")

    @staticmethod
    def _optional_positive_int(value: object) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def generate_structured(self, request: LLMRequest) -> LLMProviderResponse:
        options: dict[str, float] = {}
        effective_temperature = self.temperature if request.temperature is None else float(request.temperature)
        options["temperature"] = effective_temperature
        num_ctx = self._optional_positive_int(request.metadata.get("num_ctx"))
        if num_ctx is not None:
            options["num_ctx"] = float(num_ctx)

        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "format": request.response_schema,
            "options": options,
        }
        encoded = json.dumps(payload).encode("utf-8")
        http_request = urllib_request.Request(
            self.generate_url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.generate_url}: {exc.reason}") from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response body.") from exc

        raw_text, response_field = self._extract_raw_text(payload)
        metadata = {key: value for key, value in payload.items() if key not in {"response", "thinking"}}
        metadata["response_field"] = response_field
        return LLMProviderResponse(
            provider="ollama",
            model=request.model,
            raw_text=raw_text,
            metadata=metadata,
        )
