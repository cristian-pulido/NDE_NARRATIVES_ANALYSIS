from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib import error, request as urllib_request

from .config import LLMConfig, PathsConfig, PreprocessingConfig, StudyConfig
from .constants import PROJECT_ROOT
from .llm import LLMProvider, LLMRequest, build_llm_provider
from .llm.parsing import parse_structured_response
from .prompting import load_response_schema, render_prompt
from .sampling import is_meaningful_text


def normalize_ollama_base_url(base_url: str) -> str:
    normalized = str(base_url).strip().rstrip("/")
    if not normalized:
        raise ValueError("Ollama base URL cannot be blank.")
    for suffix in ("/api/generate", "/api/tags"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def parse_ollama_tags_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return []

    discovered: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_name = item.get("name") or item.get("model")
        if model_name is None:
            continue
        normalized = str(model_name).strip()
        if normalized:
            discovered.add(normalized)
    return sorted(discovered)


def list_ollama_models(base_url: str, *, timeout_seconds: int = 10) -> list[str]:
    normalized = normalize_ollama_base_url(base_url)
    tags_url = f"{normalized}/api/tags"
    request = urllib_request.Request(tags_url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=int(timeout_seconds)) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {tags_url}: {exc.reason}"
        ) from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ollama /api/tags returned a non-JSON response body."
        ) from exc
    return parse_ollama_tags_payload(payload)


def configured_model_fallbacks(
    llm_config: LLMConfig, preprocessing: PreprocessingConfig
) -> list[str]:
    model_names: set[str] = set()
    if preprocessing.model:
        model_names.add(str(preprocessing.model).strip())
    for experiment in llm_config.experiments:
        if experiment.model:
            model_names.add(str(experiment.model).strip())
    return sorted(value for value in model_names if value)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _require_meaningful_text(value: str, *, field_name: str) -> str:
    if not is_meaningful_text(value):
        raise ValueError(f"{field_name} must include meaningful text.")
    return str(value)


def _build_runtime_config(
    llm_config: LLMConfig, *, base_url: str, temperature: float
) -> Any:
    runtime = llm_config.runtime
    return replace(
        runtime,
        base_url=normalize_ollama_base_url(base_url),
        temperature=float(temperature),
    )


def _run_section_prediction(
    *,
    provider: LLMProvider,
    section_name: str,
    input_text: str,
    model: str,
    temperature: float,
    prompt_variant: str | None,
    paths: PathsConfig,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    prompt = render_prompt(
        section_name,
        input_text,
        prompt_variant=prompt_variant,
        paths=paths,
    )
    schema = load_response_schema(section_name)
    request = LLMRequest(
        participant_code="interactive",
        section=section_name,
        prompt=prompt,
        response_schema=schema,
        model=str(model),
        temperature=float(temperature),
        metadata={"mode": "interactive"},
    )
    response = provider.generate_structured(request)
    parsed = parse_structured_response(response.raw_text, schema)
    return parsed, response.raw_text, dict(response.metadata)


def analyze_three_sections(
    *,
    study: StudyConfig,
    paths: PathsConfig,
    llm_config: LLMConfig,
    model: str,
    context_text: str,
    experience_text: str,
    aftereffects_text: str,
    prompt_variant: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    provider_factory: Callable[[Any], LLMProvider] | None = None,
) -> dict[str, Any]:
    context_text = _require_meaningful_text(context_text, field_name="Context")
    experience_text = _require_meaningful_text(experience_text, field_name="Experience")
    aftereffects_text = _require_meaningful_text(
        aftereffects_text, field_name="Aftereffects"
    )

    effective_temperature = (
        llm_config.runtime.temperature if temperature is None else float(temperature)
    )
    effective_base_url = (
        llm_config.runtime.base_url if base_url is None else str(base_url)
    )
    runtime = _build_runtime_config(
        llm_config, base_url=effective_base_url, temperature=effective_temperature
    )
    provider = (provider_factory or build_llm_provider)(runtime)

    section_inputs = {
        "context": context_text,
        "experience": experience_text,
        "aftereffects": aftereffects_text,
    }
    predictions: dict[str, dict[str, Any]] = {}
    raw_outputs: dict[str, str] = {}
    provider_metadata: dict[str, dict[str, Any]] = {}
    for section_name in study.section_order:
        parsed, raw_text, metadata = _run_section_prediction(
            provider=provider,
            section_name=section_name,
            input_text=section_inputs[section_name],
            model=model,
            temperature=effective_temperature,
            prompt_variant=prompt_variant,
            paths=paths,
        )
        predictions[section_name] = parsed
        raw_outputs[section_name] = raw_text
        provider_metadata[section_name] = metadata

    return {
        "mode": "three_sections",
        "provider": llm_config.runtime.provider,
        "base_url": normalize_ollama_base_url(effective_base_url),
        "model": str(model),
        "prompt_variant": prompt_variant,
        "temperature": effective_temperature,
        "segmentation": {
            "context": context_text,
            "experience": experience_text,
            "aftereffects": aftereffects_text,
        },
        "predictions": predictions,
        "raw_outputs": raw_outputs,
        "provider_metadata": provider_metadata,
        "generated_at": _utc_now(),
    }


def _render_resegmentation_prompt(single_narrative_text: str) -> str:
    prompt_path = (
        Path(PROJECT_ROOT)
        / "prompts"
        / "preprocessing"
        / "resegment_narrative_prompt.md"
    )
    template = prompt_path.read_text(encoding="utf-8")
    return template.replace("{{merged_text}}", single_narrative_text)


def _run_single_text_resegmentation(
    *,
    provider: LLMProvider,
    model: str,
    temperature: float,
    single_narrative_text: str,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    schema_path = (
        Path(PROJECT_ROOT) / "schemas" / "preprocess_resegmentation_output.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    prompt = _render_resegmentation_prompt(single_narrative_text)
    request = LLMRequest(
        participant_code="interactive",
        section="interactive_preprocess_resegment",
        prompt=prompt,
        response_schema=schema,
        model=str(model),
        temperature=float(temperature),
        metadata={"mode": "interactive", "step": "resegment"},
    )
    response = provider.generate_structured(request)
    parsed = parse_structured_response(response.raw_text, schema)
    return (
        {
            "context": str(parsed.get("context", "")),
            "experience": str(parsed.get("experience", "")),
            "aftereffects": str(parsed.get("aftereffects", "")),
        },
        response.raw_text,
        dict(response.metadata),
    )


def analyze_single_narrative(
    *,
    study: StudyConfig,
    paths: PathsConfig,
    llm_config: LLMConfig,
    model: str,
    single_narrative_text: str,
    prompt_variant: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    provider_factory: Callable[[Any], LLMProvider] | None = None,
) -> dict[str, Any]:
    single_narrative_text = _require_meaningful_text(
        single_narrative_text, field_name="Narrative"
    )
    effective_temperature = (
        llm_config.runtime.temperature if temperature is None else float(temperature)
    )
    effective_base_url = (
        llm_config.runtime.base_url if base_url is None else str(base_url)
    )
    runtime = _build_runtime_config(
        llm_config, base_url=effective_base_url, temperature=effective_temperature
    )
    provider = (provider_factory or build_llm_provider)(runtime)

    segmented, resegment_raw_text, resegment_metadata = _run_single_text_resegmentation(
        provider=provider,
        model=model,
        temperature=effective_temperature,
        single_narrative_text=single_narrative_text,
    )
    analysis = analyze_three_sections(
        study=study,
        paths=paths,
        llm_config=llm_config,
        model=model,
        context_text=segmented["context"],
        experience_text=segmented["experience"],
        aftereffects_text=segmented["aftereffects"],
        prompt_variant=prompt_variant,
        base_url=effective_base_url,
        temperature=effective_temperature,
        provider_factory=lambda _runtime: provider,
    )

    analysis["mode"] = "single_narrative"
    analysis["input_narrative"] = single_narrative_text
    analysis["resegmentation"] = {
        "parsed": segmented,
        "raw_output": resegment_raw_text,
        "provider_metadata": resegment_metadata,
    }
    return analysis


def build_evidence_summary_markdown(predictions: dict[str, dict[str, Any]]) -> str:
    lines = ["# Section Evidence Summary"]
    for section_name in ("context", "experience", "aftereffects"):
        payload = predictions.get(section_name, {})
        section_payload = (
            payload.get(section_name, {}) if isinstance(payload, dict) else {}
        )
        tone = "unknown"
        evidence: list[str] = []
        if isinstance(section_payload, dict):
            tone = str(section_payload.get("tone", "unknown"))
            raw_evidence = section_payload.get("evidence_segments", [])
            if isinstance(raw_evidence, list):
                evidence = [str(item) for item in raw_evidence]
        lines.append(f"## {section_name.title()}")
        lines.append(f"- Tone: {tone}")
        if evidence:
            for segment in evidence:
                lines.append(f"- Evidence: {segment}")
        else:
            lines.append("- Evidence: none returned")
    return "\n".join(lines)


def build_predictions_table_rows(
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for section_name in ("context", "experience", "aftereffects"):
        payload = predictions.get(section_name, {})
        section_payload = (
            payload.get(section_name, {}) if isinstance(payload, dict) else {}
        )
        if not isinstance(section_payload, dict):
            section_payload = {}

        tone = str(section_payload.get("tone", ""))
        evidence_items = section_payload.get("evidence_segments", [])
        evidence = ""
        if isinstance(evidence_items, list):
            evidence = " | ".join(str(item) for item in evidence_items)

        labels: list[str] = []
        for key, value in section_payload.items():
            if key in {"tone", "evidence_segments", "death_context_nature"}:
                continue
            labels.append(f"{key}={value}")

        rows.append(
            {
                "section": section_name,
                "tone": tone,
                "death_context_nature": str(
                    section_payload.get("death_context_nature", "")
                ),
                "evidence_segments": evidence,
                "binary_labels": " | ".join(labels),
            }
        )
    return rows
