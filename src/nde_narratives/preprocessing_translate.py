from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import PathsConfig, StudyConfig, TranslateConfig
from .io_utils import read_tabular_file
from .llm import LLMProvider, LLMRequest, build_llm_provider
from .llm.parsing import parse_structured_response
from .sampling import apply_dataset_row_filters, assign_participant_codes, is_meaningful_text


PARTICIPANT_RESULTS_FILENAME = "participant_results_translate.jsonl"
RAW_RESPONSES_FILENAME = "raw_responses_translate.jsonl"
ERRORS_FILENAME = "errors_translate.jsonl"
RUN_SUMMARY_FILENAME = "run_summary_translate.json"
MANIFEST_FILENAME = "manifest_translate.json"
TRANSLATED_DATASET_FILENAME = "translated_dataset.csv"
TRANSLATED_DATASET_XLSX_FILENAME = "translated_dataset.xlsx"

SUCCESS_STATUS = "success"
FAILED_STATUS = "failed"
PENDING_STATUS = "pending"
SKIPPED_NO_TEXT_STATUS = "skipped_no_text"
EXHAUSTED_STATUS = "exhausted"


def _coerce_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    _atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default))
    return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    content = "".join(f"{json.dumps(record, ensure_ascii=False, default=_json_default)}\n" for record in records)
    _atomic_write_text(path, content)
    return str(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _reset_translate_state(output_dir: Path) -> None:
    for filename in (
        PARTICIPANT_RESULTS_FILENAME,
        RAW_RESPONSES_FILENAME,
        ERRORS_FILENAME,
        RUN_SUMMARY_FILENAME,
        MANIFEST_FILENAME,
        TRANSLATED_DATASET_FILENAME,
        TRANSLATED_DATASET_XLSX_FILENAME,
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _escape_literal_newlines_inside_json_strings(raw_text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for char in raw_text:
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
                continue
            if char == "\\":
                out.append(char)
                escaped = True
                continue
            if char == '"':
                out.append(char)
                in_string = False
                continue
            if char == "\n":
                out.append("\\n")
                continue
            if char == "\r":
                continue
            out.append(char)
            continue

        out.append(char)
        if char == '"':
            in_string = True
            escaped = False
    return "".join(out)


def _parse_translate_response(raw_text: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_structured_response(raw_text, schema)
    except ValueError:
        normalized = _escape_literal_newlines_inside_json_strings(raw_text)
        if normalized == raw_text:
            raise
        return parse_structured_response(normalized, schema)


def _render_translate_prompt(section_name: str, section_text: str, prompt_root: Path) -> str:
    template = _load_prompt(prompt_root / "translate_prompt.md")
    return template.replace("{{section_name}}", section_name).replace("{{section_text}}", _coerce_text(section_text))


def _estimate_prompt_tokens(prompt: str, chars_per_token: float) -> int:
    safe_chars_per_token = max(float(chars_per_token), 0.1)
    return max(1, int(math.ceil(len(prompt) / safe_chars_per_token)))


def _resolve_dynamic_num_ctx(prompt: str, translate: TranslateConfig) -> int | None:
    dynamic_enabled = bool(getattr(translate, "dynamic_context_enabled", True))
    if not dynamic_enabled:
        return None
    chars_per_token = float(getattr(translate, "chars_per_token", 4.0))
    estimated_tokens = _estimate_prompt_tokens(prompt, chars_per_token)
    target_tokens = estimated_tokens + 1024
    minimum = int(getattr(translate, "num_ctx_min", 4096))
    maximum = int(getattr(translate, "num_ctx_max", 16384))
    if target_tokens <= minimum:
        return minimum
    bucket = minimum
    while bucket < target_tokens and bucket < maximum:
        bucket *= 2
    return min(bucket, maximum)


def _build_request_metadata(prompt: str, translate: TranslateConfig) -> dict[str, Any]:
    chars_per_token = float(getattr(translate, "chars_per_token", 4.0))
    metadata = {
        "prompt_chars": len(prompt),
        "prompt_tokens_estimate": _estimate_prompt_tokens(prompt, chars_per_token),
    }
    dynamic_num_ctx = _resolve_dynamic_num_ctx(prompt, translate)
    if dynamic_num_ctx is not None:
        metadata["num_ctx"] = dynamic_num_ctx
    return metadata


def _source_frame(study: StudyConfig, paths: PathsConfig, input_path: Path | None, *, all_records: bool, limit: int | None) -> pd.DataFrame:
    survey_path = Path(input_path or paths.survey_csv)
    if not survey_path.exists():
        raise FileNotFoundError(f"Survey source not found: {survey_path}")
    raw = read_tabular_file(survey_path)
    if all_records:
        prepared = raw.copy()
    else:
        prepared = apply_dataset_row_filters(raw, study)
        text_columns = [study.sections[name].source_column for name in study.section_order]
        valid_mask = prepared[text_columns].apply(lambda column: column.apply(is_meaningful_text))
        prepared = prepared[valid_mask.any(axis=1)].copy()
    prepared = prepared.sort_values(study.id_column).reset_index(drop=True)
    prepared = assign_participant_codes(prepared, study)
    if limit is not None:
        prepared = prepared.head(limit).copy()
    return prepared


def _record_has_meaningful_text(row: pd.Series, study: StudyConfig) -> bool:
    return any(is_meaningful_text(row.get(study.sections[name].source_column)) for name in study.section_order)


def _base_record(row: pd.Series, study: StudyConfig) -> dict[str, Any]:
    participant_code = row.get("participant_code")
    if participant_code is None and getattr(row, "name", None) is not None:
        participant_code = row.name
    return {
        study.id_column: row[study.id_column],
        "participant_code": participant_code,
        "status": PENDING_STATUS,
        "attempts_translate": 0,
        "translate_status": "pending",
        "context_translation": "",
        "experience_translation": "",
        "aftereffects_translation": "",
        "detected_language_context": "",
        "detected_language_experience": "",
        "detected_language_aftereffects": "",
        "detected_language_row": "",
        "last_error_type": None,
        "last_error_message": None,
        "updated_at": _utc_now(),
    }


def _dominant_language(values: list[str]) -> str:
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        return "unknown"
    counts: dict[str, int] = {}
    for lang in cleaned:
        counts[lang] = counts.get(lang, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _process_row(
    row: pd.Series,
    record: dict[str, Any],
    *,
    study: StudyConfig,
    translate: TranslateConfig,
    provider: LLMProvider,
    prompt_root: Path,
    translate_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_responses: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if not _record_has_meaningful_text(row, study):
        original_context = _coerce_text(row.get(study.sections["context"].source_column))
        original_experience = _coerce_text(row.get(study.sections["experience"].source_column))
        original_aftereffects = _coerce_text(row.get(study.sections["aftereffects"].source_column))
        return {
            **record,
            "status": SKIPPED_NO_TEXT_STATUS,
            "translate_status": "skipped_no_text",
            "context_translation": original_context,
            "experience_translation": original_experience,
            "aftereffects_translation": original_aftereffects,
            "detected_language_context": "unknown",
            "detected_language_experience": "unknown",
            "detected_language_aftereffects": "unknown",
            "detected_language_row": "unknown",
            "updated_at": _utc_now(),
        }, raw_responses, errors

    translated_values = {
        "context_translation": _coerce_text(record.get("context_translation")) or _coerce_text(row.get(study.sections["context"].source_column)),
        "experience_translation": _coerce_text(record.get("experience_translation"))
        or _coerce_text(row.get(study.sections["experience"].source_column)),
        "aftereffects_translation": _coerce_text(record.get("aftereffects_translation"))
        or _coerce_text(row.get(study.sections["aftereffects"].source_column)),
    }
    detected_languages = {
        "context": _coerce_text(record.get("detected_language_context")).lower() or "unknown",
        "experience": _coerce_text(record.get("detected_language_experience")).lower() or "unknown",
        "aftereffects": _coerce_text(record.get("detected_language_aftereffects")).lower() or "unknown",
    }

    section_errors = 0
    meaningful_sections = 0
    for section_name in study.section_order:
        source_column = study.sections[section_name].source_column
        section_text = _coerce_text(row.get(source_column))
        if not is_meaningful_text(section_text):
            translated_values[f"{section_name}_translation"] = ""
            detected_languages[section_name] = "unknown"
            continue

        meaningful_sections += 1
        translate_prompt = _render_translate_prompt(section_name, section_text, prompt_root)
        translate_request = LLMRequest(
            participant_code=str(row["participant_code"]),
            section=f"preprocess_translate_{section_name}",
            prompt=translate_prompt,
            response_schema=translate_schema,
            model=str(translate.model),
            temperature=translate.temperature,
            metadata=_build_request_metadata(translate_prompt, translate),
        )

        try:
            translate_response = provider.generate_structured(translate_request)
            raw_responses.append(
                {
                    study.id_column: row[study.id_column],
                    "participant_code": row["participant_code"],
                    "stage": f"translate_{section_name}",
                    "raw_text": translate_response.raw_text,
                    "provider_metadata": translate_response.metadata,
                    "request_metadata": translate_request.metadata,
                }
            )
            parsed = _parse_translate_response(translate_response.raw_text, translate_schema)
            translated_values[f"{section_name}_translation"] = _coerce_text(parsed["translation"]) or section_text
            detected_languages[section_name] = _coerce_text(parsed["source_language"]).lower() or "unknown"
        except Exception as exc:  # noqa: BLE001
            section_errors += 1
            errors.append(
                {
                    study.id_column: row[study.id_column],
                    "participant_code": row["participant_code"],
                    "stage": f"translate_{section_name}",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "request_metadata": translate_request.metadata,
                    "updated_at": _utc_now(),
                }
            )

    updated_attempts = int(record.get("attempts_translate", 0)) + 1
    detected_row = _dominant_language([detected_languages["context"], detected_languages["experience"], detected_languages["aftereffects"]])
    if section_errors > 0:
        status = EXHAUSTED_STATUS if updated_attempts >= translate.max_attempts else FAILED_STATUS
        return {
            **record,
            "attempts_translate": updated_attempts,
            "status": status,
            "translate_status": "partially_translated" if meaningful_sections > section_errors else "failed",
            **translated_values,
            "detected_language_context": detected_languages["context"],
            "detected_language_experience": detected_languages["experience"],
            "detected_language_aftereffects": detected_languages["aftereffects"],
            "detected_language_row": detected_row,
            "last_error_type": "SectionTranslationError",
            "last_error_message": f"{section_errors} section(s) failed translation",
            "updated_at": _utc_now(),
        }, raw_responses, errors

    return {
        **record,
        "attempts_translate": updated_attempts,
        "translate_status": "translated",
        "status": SUCCESS_STATUS,
        **translated_values,
        "detected_language_context": detected_languages["context"],
        "detected_language_experience": detected_languages["experience"],
        "detected_language_aftereffects": detected_languages["aftereffects"],
        "detected_language_row": detected_row,
        "last_error_type": None,
        "last_error_message": None,
        "updated_at": _utc_now(),
    }, raw_responses, errors


def _build_translated_dataset(source_df: pd.DataFrame, records: dict[str, dict[str, Any]], study: StudyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_code = source_df.set_index("participant_code")
    passthrough_columns = [
        study.stratify_column,
        str(study.dataset.get("quality_label_column") or ""),
        str(study.dataset.get("to_drop_column") or ""),
    ]
    passthrough_columns = [column for column in passthrough_columns if column and column in source_df.columns]

    def _resolved_translation(record_value: object, original_value: object) -> str:
        translated_value = _coerce_text(record_value)
        if translated_value:
            return translated_value
        return _coerce_text(original_value)

    for participant_code in by_code.index:
        row = by_code.loc[participant_code]
        record = records.get(str(participant_code), _base_record(row, study))
        translated = {
            study.id_column: row[study.id_column],
            "participant_code": participant_code,
            study.sections["context"].source_column: _resolved_translation(
                record.get("context_translation"), row.get(study.sections["context"].source_column)
            ),
            study.sections["experience"].source_column: _resolved_translation(
                record.get("experience_translation"), row.get(study.sections["experience"].source_column)
            ),
            study.sections["aftereffects"].source_column: _resolved_translation(
                record.get("aftereffects_translation"), row.get(study.sections["aftereffects"].source_column)
            ),
            "translate_status": record.get("translate_status"),
            "translate_attempts": record.get("attempts_translate"),
            "translate_run_status": record.get("status"),
            "detected_language_context": record.get("detected_language_context", "unknown"),
            "detected_language_experience": record.get("detected_language_experience", "unknown"),
            "detected_language_aftereffects": record.get("detected_language_aftereffects", "unknown"),
            "detected_language_row": record.get("detected_language_row", "unknown"),
            "original_context": _coerce_text(row.get(study.sections["context"].source_column)),
            "original_experience": _coerce_text(row.get(study.sections["experience"].source_column)),
            "original_aftereffects": _coerce_text(row.get(study.sections["aftereffects"].source_column)),
        }
        for column in passthrough_columns:
            translated[column] = row.get(column)
        rows.append(translated)
    return pd.DataFrame(rows)


def _write_translated_outputs(output_dir: Path, translated_df: pd.DataFrame) -> dict[str, str]:
    csv_path = output_dir / TRANSLATED_DATASET_FILENAME
    xlsx_path = output_dir / TRANSLATED_DATASET_XLSX_FILENAME
    translated_df.to_csv(csv_path, index=False)
    translated_df.to_excel(xlsx_path, index=False)
    return {"translated_dataset_csv": str(csv_path), "translated_dataset_xlsx": str(xlsx_path)}


def _status_counts(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {SUCCESS_STATUS: 0, FAILED_STATUS: 0, PENDING_STATUS: 0, SKIPPED_NO_TEXT_STATUS: 0, EXHAUSTED_STATUS: 0}
    for record in records.values():
        status = str(record.get("status"))
        counts.setdefault(status, 0)
        counts[status] += 1
    return counts


def run_translate_pipeline(
    *,
    study: StudyConfig,
    paths: PathsConfig,
    translate: TranslateConfig,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    all_records: bool = False,
    retry_exhausted: bool = False,
    from_scratch: bool = False,
    provider_factory: Callable[[TranslateConfig], LLMProvider] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not translate.model:
        raise ValueError("Missing translate.model in paths config. Configure a dedicated translation model before running this command.")

    effective_output_dir = Path(output_dir or paths.preprocessing_output_dir)
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    if from_scratch:
        _reset_translate_state(effective_output_dir)

    project_root = Path(__file__).resolve().parents[2]
    prompt_root = project_root / "prompts" / "preprocessing"
    translate_schema = _load_schema(project_root / "schemas" / "preprocess_translate_output.schema.json")
    source_df = _source_frame(study, paths, input_path, all_records=all_records, limit=limit)

    participant_results_path = effective_output_dir / PARTICIPANT_RESULTS_FILENAME
    raw_responses_path = effective_output_dir / RAW_RESPONSES_FILENAME
    errors_path = effective_output_dir / ERRORS_FILENAME
    run_summary_path = effective_output_dir / RUN_SUMMARY_FILENAME
    manifest_path = effective_output_dir / MANIFEST_FILENAME

    existing_records = {str(record["participant_code"]): record for record in _load_jsonl(participant_results_path)}
    raw_responses = _load_jsonl(raw_responses_path)
    errors = _load_jsonl(errors_path)

    provider = (provider_factory or build_llm_provider)(translate)
    no_op = True

    total_rows = int(len(source_df))
    processed_rows = 0
    for _, row in source_df.iterrows():
        participant_code = str(row["participant_code"])
        record = existing_records.get(participant_code, _base_record(row, study))
        status = str(record.get("status", PENDING_STATUS))
        if status == SUCCESS_STATUS or status == SKIPPED_NO_TEXT_STATUS:
            processed_rows += 1
            if progress_callback is not None:
                progress_callback({"stage": "translate", "current": processed_rows, "total": total_rows, "participant_code": participant_code, "status": "skipped_completed"})
            continue
        if status == EXHAUSTED_STATUS and not retry_exhausted:
            processed_rows += 1
            if progress_callback is not None:
                progress_callback({"stage": "translate", "current": processed_rows, "total": total_rows, "participant_code": participant_code, "status": "skipped_exhausted"})
            continue

        updated, new_raw, new_errors = _process_row(
            row,
            record,
            study=study,
            translate=translate,
            provider=provider,
            prompt_root=prompt_root,
            translate_schema=translate_schema,
        )
        existing_records[participant_code] = updated
        raw_responses.extend(new_raw)
        errors.extend(new_errors)
        no_op = False
        _write_jsonl(participant_results_path, [existing_records[key] for key in sorted(existing_records)])
        _write_jsonl(raw_responses_path, raw_responses)
        _write_jsonl(errors_path, errors)
        processed_rows += 1
        if progress_callback is not None:
            progress_callback({"stage": "translate", "current": processed_rows, "total": total_rows, "participant_code": participant_code, "status": str(updated.get("status", "unknown"))})

    translated_df = _build_translated_dataset(source_df, existing_records, study)
    written = {
        "participant_results_file": _write_jsonl(participant_results_path, [existing_records[key] for key in sorted(existing_records)]),
        "raw_responses_file": _write_jsonl(raw_responses_path, raw_responses),
        "errors_file": _write_jsonl(errors_path, errors),
    }
    written.update(_write_translated_outputs(effective_output_dir, translated_df))

    summary = {
        "input_path": str(Path(input_path or paths.survey_csv)),
        "output_dir": str(effective_output_dir),
        "prompt_root": str(prompt_root),
        "records": int(len(source_df)),
        "status_counts": _status_counts(existing_records),
        "n_translated_rows": int(len(translated_df)),
    }
    manifest = {
        "pipeline": "translate",
        "input_path": str(Path(input_path or paths.survey_csv)),
        "study_config": str(study.path),
        "prompt_root": str(prompt_root),
        "config": translate.to_dict(),
        "records": int(len(source_df)),
    }
    written["manifest_file"] = _write_json(manifest_path, manifest)
    written["run_summary_file"] = _write_json(run_summary_path, summary)
    return {
        "no_op": no_op,
        "message": "No pending translate calls." if no_op else "Translate run completed.",
        "summary": summary,
        **written,
    }
