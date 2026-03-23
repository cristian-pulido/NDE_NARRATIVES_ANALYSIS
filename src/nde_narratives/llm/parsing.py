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


def _normalize_by_schema(value: Any, schema: dict[str, Any], field_name: str) -> Any:
    schema_type = schema.get("type")

    if schema_type == "string":
        allowed = [str(item).lower() for item in schema.get("enum", [])]
        if allowed:
            return _normalize_enum_value(value, allowed, field_name)
        if value is None:
            return ""
        return str(value).strip()

    if schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"Invalid value for {field_name}: expected an array.")

        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < int(min_items):
            raise ValueError(f"Invalid value for {field_name}: expected at least {min_items} item(s).")
        if max_items is not None and len(value) > int(max_items):
            raise ValueError(f"Invalid value for {field_name}: expected at most {max_items} item(s).")

        items_schema = dict(schema.get("items", {}))
        return [_normalize_by_schema(item, items_schema, f"{field_name}[{index}]") for index, item in enumerate(value)]

    if schema_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"Invalid value for {field_name}: expected an object.")

        properties = dict(schema.get("properties", {}))
        required = [str(field) for field in schema.get("required", [])]
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"Model response is missing required fields in {field_name}: {missing}")

        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ValueError(f"Model response contains unexpected fields in {field_name}: {extras}")

        normalized: dict[str, Any] = {}
        for child_name, child_rules in properties.items():
            if child_name not in value:
                continue
            qualified = f"{field_name}.{child_name}" if field_name else child_name
            normalized[child_name] = _normalize_by_schema(value[child_name], dict(child_rules), qualified)
        return normalized

    raise ValueError(f"Unsupported schema type for {field_name or 'root'}: {schema_type}")


def validate_and_normalize_payload(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_by_schema(payload, schema, "")
    if not isinstance(normalized, dict):
        raise ValueError("Model response must normalize to a JSON object.")
    return normalized


def parse_structured_response(raw_text: str, schema: dict[str, Any]) -> dict[str, Any]:
    payload = extract_json_object(raw_text)
    return validate_and_normalize_payload(payload, schema)
