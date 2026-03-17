from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Model response was blank.")

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Could not extract a JSON object from the model response.")


def _normalize_enum_value(value: object, allowed: list[str], field_name: str) -> str:
    if isinstance(value, bool) and set(allowed) == {"yes", "no"}:
        return "yes" if value else "no"

    normalized = str(value).strip().lower()
    if normalized in allowed:
        return normalized
    raise ValueError(f"Invalid value for {field_name}: {value!r}. Allowed values: {allowed}")


def validate_and_normalize_payload(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, str]:
    properties = dict(schema.get("properties", {}))
    required = [str(field) for field in schema.get("required", [])]

    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Model response is missing required fields: {missing}")

    if schema.get("additionalProperties") is False:
        extras = sorted(set(payload) - set(properties))
        if extras:
            raise ValueError(f"Model response contains unexpected fields: {extras}")

    normalized: dict[str, str] = {}
    for field_name, rules in properties.items():
        if field_name not in payload:
            continue
        if rules.get("type") != "string":
            raise ValueError(f"Unsupported schema type for {field_name}: {rules.get('type')}")
        allowed = [str(value).lower() for value in rules.get("enum", [])]
        normalized[field_name] = _normalize_enum_value(payload[field_name], allowed, field_name)

    return normalized


def parse_structured_response(raw_text: str, schema: dict[str, Any]) -> dict[str, str]:
    payload = extract_json_object(raw_text)
    return validate_and_normalize_payload(payload, schema)
