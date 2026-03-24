from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import PathsConfig, PreprocessingConfig, StudyConfig
from .io_utils import read_tabular_file
from .llm import LLMProvider, LLMRequest, build_llm_provider
from .llm.parsing import parse_structured_response
from .prompting import resolve_prompt_root
from .sampling import apply_dataset_row_filters, assign_participant_codes, is_meaningful_text


PARTICIPANT_RESULTS_FILENAME = "participant_results.jsonl"
RAW_RESPONSES_FILENAME = "raw_responses.jsonl"
ERRORS_FILENAME = "errors.jsonl"
RUN_SUMMARY_FILENAME = "run_summary.json"
MANIFEST_FILENAME = "manifest.json"
CLEANED_DATASET_FILENAME = "cleaned_dataset.csv"
CLEANED_DATASET_XLSX_FILENAME = "cleaned_dataset.xlsx"
VALIDATION_SAMPLE_FILENAME = "preprocessing_validation_sample.xlsx"
VALIDATION_MAPPING_FILENAME = "preprocessing_validation_mapping_private.xlsx"
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


def _reset_preprocessing_state(output_dir: Path) -> None:
    for filename in (
        PARTICIPANT_RESULTS_FILENAME,
        RAW_RESPONSES_FILENAME,
        ERRORS_FILENAME,
        RUN_SUMMARY_FILENAME,
        MANIFEST_FILENAME,
        CLEANED_DATASET_FILENAME,
        CLEANED_DATASET_XLSX_FILENAME,
        VALIDATION_SAMPLE_FILENAME,
        VALIDATION_MAPPING_FILENAME,
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _render_validation_prompt(row: pd.Series, study: StudyConfig, prompt_root: Path) -> str:
    template = _load_prompt(prompt_root / "validate_sections_prompt.md")
    replacements = {
        "{{context_text}}": _coerce_text(row.get(study.sections["context"].source_column)),
        "{{experience_text}}": _coerce_text(row.get(study.sections["experience"].source_column)),
        "{{aftereffects_text}}": _coerce_text(row.get(study.sections["aftereffects"].source_column)),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _render_resegment_prompt(row: pd.Series, study: StudyConfig, prompt_root: Path) -> str:
    template = _load_prompt(prompt_root / "resegment_narrative_prompt.md")
    merged = "\n\n".join(
        [
            f"Context:\n{_coerce_text(row.get(study.sections['context'].source_column))}",
            f"Experience:\n{_coerce_text(row.get(study.sections['experience'].source_column))}",
            f"Aftereffects:\n{_coerce_text(row.get(study.sections['aftereffects'].source_column))}",
        ]
    )
    return template.replace("{{merged_text}}", merged)


def _estimate_prompt_tokens(prompt: str, chars_per_token: float) -> int:
    safe_chars_per_token = max(float(chars_per_token), 0.1)
    return max(1, int(math.ceil(len(prompt) / safe_chars_per_token)))


def _resolve_dynamic_num_ctx(prompt: str, preprocessing: PreprocessingConfig) -> int | None:
    dynamic_enabled = bool(getattr(preprocessing, "dynamic_context_enabled", True))
    if not dynamic_enabled:
        return None
    chars_per_token = float(getattr(preprocessing, "chars_per_token", 4.0))
    estimated_tokens = _estimate_prompt_tokens(prompt, chars_per_token)
    target_tokens = estimated_tokens + 1024
    minimum = int(getattr(preprocessing, "num_ctx_min", 4096))
    maximum = int(getattr(preprocessing, "num_ctx_max", 16384))
    if target_tokens <= minimum:
        return minimum
    bucket = minimum
    while bucket < target_tokens and bucket < maximum:
        bucket *= 2
    return min(bucket, maximum)


def _build_request_metadata(prompt: str, preprocessing: PreprocessingConfig) -> dict[str, Any]:
    chars_per_token = float(getattr(preprocessing, "chars_per_token", 4.0))
    metadata = {
        "prompt_chars": len(prompt),
        "prompt_tokens_estimate": _estimate_prompt_tokens(prompt, chars_per_token),
    }
    dynamic_num_ctx = _resolve_dynamic_num_ctx(prompt, preprocessing)
    if dynamic_num_ctx is not None:
        metadata["num_ctx"] = dynamic_num_ctx
    return metadata


def _base_record(row: pd.Series, study: StudyConfig) -> dict[str, Any]:
    participant_code = row.get("participant_code")
    if participant_code is None and getattr(row, "name", None) is not None:
        participant_code = row.name
    return {
        study.id_column: row[study.id_column],
        "participant_code": participant_code,
        "status": PENDING_STATUS,
        "attempts_validation": 0,
        "attempts_resegmentation": 0,
        "n_valid_sections": 0,
        "preprocessing_status": "pending",
        "context_clean": "",
        "experience_clean": "",
        "aftereffects_clean": "",
        "changed_sections": [],
        "last_error_type": None,
        "last_error_message": None,
        "updated_at": _utc_now(),
    }


def _coerce_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"valid", "invalid", "empty"}:
        return normalized
    raise ValueError(f"Unsupported section assessment: {value!r}")


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


def _updated_cleaned_texts_from_original(row: pd.Series, study: StudyConfig) -> dict[str, str]:
    return {
        "context_clean": _coerce_text(row.get(study.sections["context"].source_column)),
        "experience_clean": _coerce_text(row.get(study.sections["experience"].source_column)),
        "aftereffects_clean": _coerce_text(row.get(study.sections["aftereffects"].source_column)),
    }


def _count_meaningful_sections_from_texts(section_texts: dict[str, str]) -> int:
    return sum(1 for value in section_texts.values() if is_meaningful_text(value))


def _process_row(
    row: pd.Series,
    record: dict[str, Any],
    *,
    study: StudyConfig,
    preprocessing: PreprocessingConfig,
    provider: LLMProvider,
    prompt_root: Path,
    validation_schema: dict[str, Any],
    resegmentation_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_responses: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if not _record_has_meaningful_text(row, study):
        updated = {
            **record,
            **_updated_cleaned_texts_from_original(row, study),
            "status": SKIPPED_NO_TEXT_STATUS,
            "preprocessing_status": "unchanged",
            "updated_at": _utc_now(),
        }
        return updated, raw_responses, errors

    validation_prompt = _render_validation_prompt(row, study, prompt_root)
    validation_request = LLMRequest(
        participant_code=str(row["participant_code"]),
        section="preprocess_validate",
        prompt=validation_prompt,
        response_schema=validation_schema,
        model=str(preprocessing.model),
        temperature=preprocessing.temperature,
        metadata=_build_request_metadata(validation_prompt, preprocessing),
    )

    try:
        validation_response = provider.generate_structured(validation_request)
        raw_responses.append(
            {
                study.id_column: row[study.id_column],
                "participant_code": row["participant_code"],
                "stage": "validation",
                "raw_text": validation_response.raw_text,
                "provider_metadata": validation_response.metadata,
                "request_metadata": validation_request.metadata,
            }
        )
        parsed_validation = parse_structured_response(validation_response.raw_text, validation_schema)
        section_states = {
            "context": _coerce_status(parsed_validation["context_assessment"]),
            "experience": _coerce_status(parsed_validation["experience_assessment"]),
            "aftereffects": _coerce_status(parsed_validation["aftereffects_assessment"]),
        }
        n_valid_sections_original = sum(1 for value in section_states.values() if value == "valid")
    except Exception as exc:  # noqa: BLE001
        updated_attempts = int(record.get("attempts_validation", 0)) + 1
        status = EXHAUSTED_STATUS if updated_attempts >= preprocessing.max_attempts else FAILED_STATUS
        errors.append(
            {
                study.id_column: row[study.id_column],
                "participant_code": row["participant_code"],
                "stage": "validation",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "request_metadata": validation_request.metadata,
                "updated_at": _utc_now(),
            }
        )
        return {
            **record,
            "attempts_validation": updated_attempts,
            "status": status,
            "last_error_type": type(exc).__name__,
            "last_error_message": str(exc),
            "updated_at": _utc_now(),
        }, raw_responses, errors

    unchanged = n_valid_sections_original == len(study.section_order)
    if unchanged:
        cleaned_texts = _updated_cleaned_texts_from_original(row, study)
        n_valid_sections_cleaned = _count_meaningful_sections_from_texts(cleaned_texts)
        return {
            **record,
            **cleaned_texts,
            "attempts_validation": int(record.get("attempts_validation", 0)) + 1,
            "n_valid_sections": n_valid_sections_cleaned,
            "n_valid_sections_original": n_valid_sections_original,
            "n_valid_sections_cleaned": n_valid_sections_cleaned,
            "preprocessing_status": "unchanged",
            "status": SUCCESS_STATUS,
            "changed_sections": [],
            "last_error_type": None,
            "last_error_message": None,
            "updated_at": _utc_now(),
        }, raw_responses, errors

    resegment_prompt = _render_resegment_prompt(row, study, prompt_root)
    resegment_request = LLMRequest(
        participant_code=str(row["participant_code"]),
        section="preprocess_resegment",
        prompt=resegment_prompt,
        response_schema=resegmentation_schema,
        model=str(preprocessing.model),
        temperature=preprocessing.temperature,
        metadata=_build_request_metadata(resegment_prompt, preprocessing),
    )
    try:
        resegment_response = provider.generate_structured(resegment_request)
        raw_responses.append(
            {
                study.id_column: row[study.id_column],
                "participant_code": row["participant_code"],
                "stage": "resegmentation",
                "raw_text": resegment_response.raw_text,
                "provider_metadata": resegment_response.metadata,
                "request_metadata": resegment_request.metadata,
            }
        )
        parsed_resegmentation = parse_structured_response(resegment_response.raw_text, resegmentation_schema)
        original = _updated_cleaned_texts_from_original(row, study)
        cleaned = {
            "context_clean": parsed_resegmentation["context"],
            "experience_clean": parsed_resegmentation["experience"],
            "aftereffects_clean": parsed_resegmentation["aftereffects"],
        }
        n_valid_sections_cleaned = _count_meaningful_sections_from_texts(cleaned)
        changed_sections = [
            section_name
            for section_name in study.section_order
            if cleaned[f"{section_name}_clean"] != original[f"{section_name}_clean"]
        ]
        preprocessing_status = "fully_resegmented" if len(changed_sections) == len(study.section_order) else "partially_corrected"
        return {
            **record,
            **cleaned,
            "attempts_validation": int(record.get("attempts_validation", 0)) + 1,
            "attempts_resegmentation": int(record.get("attempts_resegmentation", 0)) + 1,
            "n_valid_sections": n_valid_sections_cleaned,
            "n_valid_sections_original": n_valid_sections_original,
            "n_valid_sections_cleaned": n_valid_sections_cleaned,
            "preprocessing_status": preprocessing_status,
            "status": SUCCESS_STATUS,
            "changed_sections": changed_sections,
            "last_error_type": None,
            "last_error_message": None,
            "updated_at": _utc_now(),
        }, raw_responses, errors
    except Exception as exc:  # noqa: BLE001
        updated_attempts = int(record.get("attempts_resegmentation", 0)) + 1
        status = EXHAUSTED_STATUS if updated_attempts >= preprocessing.max_attempts else FAILED_STATUS
        errors.append(
            {
                study.id_column: row[study.id_column],
                "participant_code": row["participant_code"],
                "stage": "resegmentation",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "request_metadata": resegment_request.metadata,
                "updated_at": _utc_now(),
            }
        )
        return {
            **record,
            "attempts_validation": int(record.get("attempts_validation", 0)) + 1,
            "attempts_resegmentation": updated_attempts,
            "n_valid_sections": int(record.get("n_valid_sections", 0)),
            "n_valid_sections_original": n_valid_sections_original,
            "n_valid_sections_cleaned": int(record.get("n_valid_sections_cleaned", 0)),
            "status": status,
            "last_error_type": type(exc).__name__,
            "last_error_message": str(exc),
            "updated_at": _utc_now(),
        }, raw_responses, errors


def _status_counts(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {SUCCESS_STATUS: 0, FAILED_STATUS: 0, PENDING_STATUS: 0, SKIPPED_NO_TEXT_STATUS: 0, EXHAUSTED_STATUS: 0}
    for record in records.values():
        status = str(record.get("status"))
        counts.setdefault(status, 0)
        counts[status] += 1
    return counts


def _count_rows_with_exact_valid_sections(cleaned_df: pd.DataFrame, column: str, expected: int) -> int:
    if column not in cleaned_df.columns or cleaned_df.empty:
        return 0
    return int((pd.to_numeric(cleaned_df[column], errors="coerce").fillna(-1).astype(int) == int(expected)).sum())


def _build_cleaned_dataset(source_df: pd.DataFrame, records: dict[str, dict[str, Any]], study: StudyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_code = source_df.set_index("participant_code")
    passthrough_columns = [
        study.stratify_column,
        str(study.dataset.get("quality_label_column") or ""),
        str(study.dataset.get("to_drop_column") or ""),
    ]
    passthrough_columns = [column for column in passthrough_columns if column and column in source_df.columns]
    for participant_code in by_code.index:
        row = by_code.loc[participant_code]
        record = records.get(str(participant_code), _base_record(row, study))
        cleaned = {
            study.id_column: row[study.id_column],
            "participant_code": participant_code,
            study.sections["context"].source_column: record.get("context_clean", ""),
            study.sections["experience"].source_column: record.get("experience_clean", ""),
            study.sections["aftereffects"].source_column: record.get("aftereffects_clean", ""),
            "preprocessing_status": record.get("preprocessing_status"),
            "n_valid_sections": record.get("n_valid_sections"),
            "n_valid_sections_original": record.get("n_valid_sections_original"),
            "n_valid_sections_cleaned": record.get("n_valid_sections_cleaned"),
            "preprocessing_run_status": record.get("status"),
            "changed_sections": "|".join(record.get("changed_sections", [])),
            "original_context": _coerce_text(row.get(study.sections["context"].source_column)),
            "original_experience": _coerce_text(row.get(study.sections["experience"].source_column)),
            "original_aftereffects": _coerce_text(row.get(study.sections["aftereffects"].source_column)),
        }
        for column in passthrough_columns:
            cleaned[column] = row.get(column)
        rows.append(cleaned)
    return pd.DataFrame(rows)


def _write_cleaned_outputs(output_dir: Path, cleaned_df: pd.DataFrame) -> dict[str, str]:
    csv_path = output_dir / CLEANED_DATASET_FILENAME
    xlsx_path = output_dir / CLEANED_DATASET_XLSX_FILENAME
    cleaned_df.to_csv(csv_path, index=False)
    cleaned_df.to_excel(xlsx_path, index=False)
    return {"cleaned_dataset_csv": str(csv_path), "cleaned_dataset_xlsx": str(xlsx_path)}


def _build_validation_sample(cleaned_df: pd.DataFrame, output_dir: Path, *, n_total: int | None, random_state: int | None, force: bool) -> dict[str, str]:
    sample_path = output_dir / VALIDATION_SAMPLE_FILENAME
    mapping_path = output_dir / VALIDATION_MAPPING_FILENAME
    if not force and (sample_path.exists() or mapping_path.exists()):
        raise FileExistsError("Refusing to overwrite existing preprocessing validation artifacts without --force-validation-sample")

    if cleaned_df.empty:
        sample_df = cleaned_df.copy()
    else:
        n = min(int(n_total or len(cleaned_df)), len(cleaned_df))
        sampled_parts: list[pd.DataFrame] = []
        grouped = cleaned_df.groupby("preprocessing_status", dropna=False)
        for _, frame in grouped:
            per_group = min(len(frame), max(1, round(n * len(frame) / len(cleaned_df))))
            sampled_parts.append(frame.sample(n=per_group, random_state=random_state))
        sample_df = pd.concat(sampled_parts, axis=0)
        sample_df = sample_df.drop_duplicates(subset=["participant_code"]).head(n).reset_index(drop=True)

    mapping_df = sample_df[["participant_code"]].copy()
    sample_df.to_excel(sample_path, index=False)
    with pd.ExcelWriter(mapping_path) as writer:
        mapping_df.to_excel(writer, index=False, sheet_name="mapping")
    return {"validation_sample_workbook": str(sample_path), "validation_mapping_workbook": str(mapping_path)}


def run_preprocessing_pipeline(
    *,
    study: StudyConfig,
    paths: PathsConfig,
    preprocessing: PreprocessingConfig,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    all_records: bool = False,
    retry_exhausted: bool = False,
    from_scratch: bool = False,
    generate_validation_sample: bool = False,
    validation_n_total: int | None = None,
    validation_random_state: int | None = None,
    force_validation_sample: bool = False,
    provider_factory: Callable[[PreprocessingConfig], LLMProvider] | None = None,
) -> dict[str, Any]:
    if not preprocessing.model:
        raise ValueError("Missing preprocessing.model in paths config. Configure a dedicated preprocessing model before running this command.")

    effective_output_dir = Path(output_dir or paths.preprocessing_output_dir)
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    if from_scratch:
        _reset_preprocessing_state(effective_output_dir)
    project_root = Path(__file__).resolve().parents[2]
    prompt_root = project_root / "prompts" / "preprocessing"
    validation_schema = _load_schema(project_root / "schemas" / "preprocess_validation_output.schema.json")
    resegmentation_schema = _load_schema(project_root / "schemas" / "preprocess_resegmentation_output.schema.json")
    source_df = _source_frame(study, paths, input_path, all_records=all_records, limit=limit)

    participant_results_path = effective_output_dir / PARTICIPANT_RESULTS_FILENAME
    raw_responses_path = effective_output_dir / RAW_RESPONSES_FILENAME
    errors_path = effective_output_dir / ERRORS_FILENAME
    run_summary_path = effective_output_dir / RUN_SUMMARY_FILENAME
    manifest_path = effective_output_dir / MANIFEST_FILENAME

    existing_records = {str(record["participant_code"]): record for record in _load_jsonl(participant_results_path)}
    raw_responses = _load_jsonl(raw_responses_path)
    errors = _load_jsonl(errors_path)

    provider = (provider_factory or build_llm_provider)(preprocessing)
    no_op = True

    for _, row in source_df.iterrows():
        participant_code = str(row["participant_code"])
        record = existing_records.get(participant_code, _base_record(row, study))
        status = str(record.get("status", PENDING_STATUS))
        if status == SUCCESS_STATUS or status == SKIPPED_NO_TEXT_STATUS:
            continue
        if status == EXHAUSTED_STATUS and not retry_exhausted:
            continue
        updated, new_raw, new_errors = _process_row(
            row,
            record,
            study=study,
            preprocessing=preprocessing,
            provider=provider,
            prompt_root=prompt_root,
            validation_schema=validation_schema,
            resegmentation_schema=resegmentation_schema,
        )
        existing_records[participant_code] = updated
        raw_responses.extend(new_raw)
        errors.extend(new_errors)
        no_op = False
        _write_jsonl(participant_results_path, [existing_records[key] for key in sorted(existing_records)])
        _write_jsonl(raw_responses_path, raw_responses)
        _write_jsonl(errors_path, errors)

    cleaned_df = _build_cleaned_dataset(source_df, existing_records, study)
    written = {
        "participant_results_file": _write_jsonl(participant_results_path, [existing_records[key] for key in sorted(existing_records)]),
        "raw_responses_file": _write_jsonl(raw_responses_path, raw_responses),
        "errors_file": _write_jsonl(errors_path, errors),
    }
    written.update(_write_cleaned_outputs(effective_output_dir, cleaned_df))
    if generate_validation_sample:
        written.update(
            _build_validation_sample(
                cleaned_df,
                effective_output_dir,
                n_total=validation_n_total,
                random_state=validation_random_state,
                force=force_validation_sample,
            )
        )

    summary = {
        "input_path": str(Path(input_path or paths.survey_csv)),
        "output_dir": str(effective_output_dir),
        "prompt_root": str(prompt_root),
        "records": int(len(source_df)),
        "status_counts": _status_counts(existing_records),
        "n_cleaned_rows": int(len(cleaned_df)),
        "n_rows_with_3_valid_sections_original": _count_rows_with_exact_valid_sections(cleaned_df, "n_valid_sections_original", 3),
        "n_rows_with_3_valid_sections_cleaned": _count_rows_with_exact_valid_sections(cleaned_df, "n_valid_sections_cleaned", 3),
    }
    manifest = {
        "pipeline": "preprocessing",
        "input_path": str(Path(input_path or paths.survey_csv)),
        "study_config": str(study.path),
        "prompt_root": str(prompt_root),
        "config": preprocessing.to_dict(),
        "records": int(len(source_df)),
    }
    written["manifest_file"] = _write_json(manifest_path, manifest)
    written["run_summary_file"] = _write_json(run_summary_path, summary)
    return {
        "no_op": no_op,
        "message": "No pending preprocessing calls." if no_op else "Preprocessing run completed.",
        "summary": summary,
        **written,
    }
