from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import (
    ExperimentMetadata,
    LLMConfig,
    LLMExperimentConfig,
    LLMRuntimeConfig,
    PathsConfig,
    StudyConfig,
)
from .llm import LLMProvider, LLMRequest, build_llm_provider
from .llm.parsing import parse_structured_response
from .prompting import build_llm_batch_records, load_batch_source, resolve_prompt_root
from .sampling import is_meaningful_text


SECTION_RESULTS_FILENAME = "section_results.jsonl"
PREDICTIONS_FILENAME = "predictions.jsonl"
RAW_RESPONSES_FILENAME = "raw_responses.jsonl"
ERRORS_FILENAME = "errors.jsonl"
RUN_SUMMARY_FILENAME = "run_summary.json"
MANIFEST_FILENAME = "manifest.json"
INTERNAL_ARTIFACT_FILENAMES = {
    SECTION_RESULTS_FILENAME,
    RAW_RESPONSES_FILENAME,
    ERRORS_FILENAME,
}
SUCCESS_STATUS = "success"
FAILED_STATUS = "failed"
PENDING_STATUS = "pending"
SKIPPED_NO_TEXT_STATUS = "skipped_no_text"
EXHAUSTED_STATUS = "exhausted"


@dataclass(frozen=True)
class ResolvedExperimentRun:
    metadata: ExperimentMetadata
    model: str
    temperature: float
    runtime: LLMRuntimeConfig
    prompt_root: Path
    source: str
    all_records: bool


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment_id: str
    artifact_id: str
    no_op: bool
    message: str
    paths: dict[str, str]
    summary: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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
    content = "".join(
        f"{json.dumps(record, ensure_ascii=False, default=_json_default)}\n"
        for record in records
    )
    _atomic_write_text(path, content)
    return str(path)


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    if not records:
        path.touch(exist_ok=True)
        return str(path)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            handle.write("\n")
    return str(path)


def _ledger_key(participant_code: str, section: str) -> tuple[str, str]:
    return participant_code, section


def _sorted_ledger_records(ledger: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    return [ledger[key] for key in sorted(ledger.keys(), key=lambda item: (str(ledger[item].get("response_id")), item[0], item[1]))]


def _load_ledger(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    ledger: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _load_jsonl(path):
        key = _ledger_key(str(record["participant_code"]), str(record["section"]))
        ledger[key] = record
    return ledger


def _base_ledger_record(response_id: object, participant_code: str, section: str) -> dict[str, Any]:
    return {
        "response_id": response_id,
        "participant_code": participant_code,
        "section": section,
        "status": PENDING_STATUS,
        "attempts": 0,
        "prediction": None,
        "last_error_type": None,
        "last_error_message": None,
        "updated_at": _utc_now(),
    }


def _resolve_experiment_run(
    experiment: LLMExperimentConfig,
    runtime: LLMRuntimeConfig,
    paths: PathsConfig,
    *,
    all_records_override: bool | None,
) -> ResolvedExperimentRun:
    metadata = ExperimentMetadata(
        experiment_id=experiment.experiment_id,
        prompt_variant=experiment.prompt_variant,
        run_id=experiment.run_id,
        model_variant=experiment.model_variant or experiment.model,
    )
    prompt_root = resolve_prompt_root(paths=paths, prompt_variant=experiment.prompt_variant)
    all_records = runtime.all_records if all_records_override is None else bool(all_records_override)
    return ResolvedExperimentRun(
        metadata=metadata,
        model=str(experiment.model),
        temperature=float(runtime.temperature if experiment.temperature is None else experiment.temperature),
        runtime=runtime,
        prompt_root=prompt_root,
        source=runtime.source,
        all_records=all_records,
    )


def _select_experiments(
    llm_config: LLMConfig,
    experiment_ids: list[str] | None,
    all_experiments: bool,
) -> list[LLMExperimentConfig]:
    configured = list(llm_config.experiments)
    if not configured:
        raise ValueError("No LLM experiments are configured. Add [[llm.experiments]] entries to paths.local.toml.")

    if all_experiments:
        selected = [experiment for experiment in configured if experiment.enabled]
        if not selected:
            raise ValueError("No enabled LLM experiments were found in the configuration.")
        return selected

    if not experiment_ids:
        raise ValueError("Pass at least one --experiment-id or use --all-experiments.")

    index = {experiment.experiment_id: experiment for experiment in configured}
    selected: list[LLMExperimentConfig] = []
    for experiment_id in experiment_ids:
        if experiment_id not in index:
            raise ValueError(f"Unknown LLM experiment_id: {experiment_id}")
        selected.append(index[experiment_id])
    return selected


def _prediction_records(
    ledger: dict[tuple[str, str], dict[str, Any]],
    source_df: pd.DataFrame,
    study: StudyConfig,
) -> list[dict[str, Any]]:
    by_participant = source_df.set_index("participant_code")
    records: list[dict[str, Any]] = []

    for participant_code in sorted(by_participant.index):
        aggregate: dict[str, Any] = {
            "participant_code": participant_code,
            study.id_column: by_participant.loc[participant_code, study.id_column],
        }
        is_complete = True
        for section_name in study.section_order:
            key = _ledger_key(participant_code, section_name)
            entry = ledger.get(key)
            if not entry or entry.get("status") != SUCCESS_STATUS or not isinstance(entry.get("prediction"), dict):
                is_complete = False
                break
            aggregate.update(entry["prediction"])
        if is_complete:
            records.append(aggregate)
    return records


def _status_counts(ledger: dict[tuple[str, str], dict[str, Any]]) -> dict[str, int]:
    counts = {
        SUCCESS_STATUS: 0,
        FAILED_STATUS: 0,
        PENDING_STATUS: 0,
        SKIPPED_NO_TEXT_STATUS: 0,
        EXHAUSTED_STATUS: 0,
    }
    for record in ledger.values():
        status = str(record.get("status"))
        counts.setdefault(status, 0)
        counts[status] += 1
    return counts


def _manifest_payload(
    resolved: ResolvedExperimentRun,
    source_path: Path,
    total_rows: int,
) -> dict[str, Any]:
    return {
        **resolved.metadata.to_dict(),
        "provider": resolved.runtime.provider,
        "model": resolved.model,
        "base_url": resolved.runtime.base_url,
        "timeout_seconds": resolved.runtime.timeout_seconds,
        "max_attempts": resolved.runtime.max_attempts,
        "temperature": resolved.temperature,
        "source": resolved.source,
        "all_records": resolved.all_records,
        "input_path": str(source_path),
        "prompt_root": str(resolved.prompt_root),
        "records": total_rows,
    }


def _artifact_paths(artifact_dir: Path) -> dict[str, str]:
    return {
        "manifest_file": str(artifact_dir / MANIFEST_FILENAME),
        "section_results_file": str(artifact_dir / SECTION_RESULTS_FILENAME),
        "predictions_file": str(artifact_dir / PREDICTIONS_FILENAME),
        "summary_file": str(artifact_dir / RUN_SUMMARY_FILENAME),
        "raw_responses_file": str(artifact_dir / RAW_RESPONSES_FILENAME),
        "errors_file": str(artifact_dir / ERRORS_FILENAME),
    }


def _assert_manifest_compatible(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    for key, expected_value in expected.items():
        if existing.get(key) != expected_value:
            raise ValueError(
                f"Existing artifact at {path.parent} was created with a different {key}. "
                "Use a new run_id or experiment_id for a different configuration."
            )


def _source_input_path(paths: PathsConfig, source: str, input_path: Path | None) -> Path:
    if input_path is not None:
        return Path(input_path)
    if source == "survey":
        return paths.survey_csv
    if source == "sampled-private":
        return paths.sampled_private_workbook
    raise ValueError(f"Unsupported LLM source: {source}")


def _write_artifacts(
    artifact_dir: Path,
    ledger: dict[tuple[str, str], dict[str, Any]],
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(artifact_dir)
    _write_json(Path(paths["manifest_file"]), manifest)
    _write_jsonl(Path(paths["section_results_file"]), _sorted_ledger_records(ledger))
    _write_jsonl(Path(paths["predictions_file"]), predictions)
    _write_json(Path(paths["summary_file"]), summary)
    _append_jsonl(Path(paths["raw_responses_file"]), [])
    _append_jsonl(Path(paths["errors_file"]), [])
    return paths


def _build_run_summary(
    *,
    manifest: dict[str, Any],
    source_df: pd.DataFrame,
    study: StudyConfig,
    ledger: dict[tuple[str, str], dict[str, Any]],
    predictions: list[dict[str, Any]],
    work_items: int,
    skipped_existing_success: int,
    skipped_exhausted: int,
    skipped_no_text: int,
    final: bool,
) -> dict[str, Any]:
    no_op = final and work_items == 0
    if final:
        message = (
            f"No pending LLM calls for artifact {manifest['artifact_id']} with the current configuration."
            if no_op
            else f"Executed {work_items} LLM calls for artifact {manifest['artifact_id']}."
        )
    else:
        message = f"Processed {work_items} LLM calls so far for artifact {manifest['artifact_id']}."

    return {
        "experiment": manifest,
        "coverage": {
            "n_source_rows": int(len(source_df)),
            "n_section_tasks": int(len(source_df) * len(study.section_order)),
            "n_complete_predictions": int(len(predictions)),
        },
        "execution": {
            "no_op": no_op,
            "message": message,
            "n_calls_attempted": int(work_items),
            "n_skipped_existing_success": int(skipped_existing_success),
            "n_skipped_exhausted": int(skipped_exhausted),
            "n_skipped_no_text": int(skipped_no_text),
        },
        "status_counts": _status_counts(ledger),
        "updated_at": _utc_now(),
    }


def _persist_run_state(
    artifact_dir: Path,
    *,
    manifest: dict[str, Any],
    ledger: dict[tuple[str, str], dict[str, Any]],
    source_df: pd.DataFrame,
    study: StudyConfig,
    work_items: int,
    skipped_existing_success: int,
    skipped_exhausted: int,
    skipped_no_text: int,
    raw_record: dict[str, Any] | None = None,
    error_record: dict[str, Any] | None = None,
    final: bool,
) -> dict[str, Any]:
    predictions = _prediction_records(ledger, source_df, study)
    summary = _build_run_summary(
        manifest=manifest,
        source_df=source_df,
        study=study,
        ledger=ledger,
        predictions=predictions,
        work_items=work_items,
        skipped_existing_success=skipped_existing_success,
        skipped_exhausted=skipped_exhausted,
        skipped_no_text=skipped_no_text,
        final=final,
    )
    _write_artifacts(
        artifact_dir,
        ledger,
        predictions,
        summary,
        manifest,
    )
    if raw_record is not None:
        _append_jsonl(artifact_dir / RAW_RESPONSES_FILENAME, [raw_record])
    if error_record is not None:
        _append_jsonl(artifact_dir / ERRORS_FILENAME, [error_record])
    return summary


def _run_single_experiment(
    study: StudyConfig,
    paths: PathsConfig,
    source_df: pd.DataFrame,
    resolved: ResolvedExperimentRun,
    *,
    input_path: Path | None,
    output_root: Path,
    retry_exhausted: bool,
    provider_factory: Callable[[LLMRuntimeConfig], LLMProvider] | None,
) -> ExperimentRunResult:
    artifact_dir = output_root / resolved.metadata.artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / MANIFEST_FILENAME
    ledger_path = artifact_dir / SECTION_RESULTS_FILENAME

    source_path = _source_input_path(paths, resolved.source, input_path)
    manifest = _manifest_payload(resolved, source_path=source_path, total_rows=int(len(source_df)))
    _assert_manifest_compatible(manifest_path, manifest)
    _write_json(manifest_path, manifest)
    _append_jsonl(artifact_dir / RAW_RESPONSES_FILENAME, [])
    _append_jsonl(artifact_dir / ERRORS_FILENAME, [])

    provider = (provider_factory or build_llm_provider)(resolved.runtime)
    ledger = _load_ledger(ledger_path)
    work_items = 0
    skipped_existing_success = 0
    skipped_exhausted = 0
    skipped_no_text = 0

    batches = build_llm_batch_records(source_df, study, experiment=resolved.metadata, prompt_root=resolved.prompt_root, paths=paths)
    source_rows = source_df.set_index("participant_code")

    for section_name in study.section_order:
        for batch_record in batches[section_name]:
            participant_code = str(batch_record["participant_code"])
            response_id = source_rows.loc[participant_code, study.id_column]
            key = _ledger_key(participant_code, section_name)
            current = dict(ledger.get(key, _base_ledger_record(response_id, participant_code, section_name)))
            current["response_id"] = response_id

            if not is_meaningful_text(batch_record["input_text"]):
                current.update(
                    {
                        "status": SKIPPED_NO_TEXT_STATUS,
                        "prediction": None,
                        "last_error_type": None,
                        "last_error_message": None,
                        "updated_at": _utc_now(),
                    }
                )
                ledger[key] = current
                skipped_no_text += 1
                _persist_run_state(
                    artifact_dir,
                    manifest=manifest,
                    ledger=ledger,
                    source_df=source_df,
                    study=study,
                    work_items=work_items,
                    skipped_existing_success=skipped_existing_success,
                    skipped_exhausted=skipped_exhausted,
                    skipped_no_text=skipped_no_text,
                    final=False,
                )
                continue

            attempts = int(current.get("attempts", 0))
            status = str(current.get("status", PENDING_STATUS))
            if status == SUCCESS_STATUS:
                ledger[key] = current
                skipped_existing_success += 1
                continue
            if status == EXHAUSTED_STATUS and not retry_exhausted:
                ledger[key] = current
                skipped_exhausted += 1
                continue
            if status in {FAILED_STATUS, PENDING_STATUS} and attempts >= resolved.runtime.max_attempts and not retry_exhausted:
                current["status"] = EXHAUSTED_STATUS
                current["updated_at"] = _utc_now()
                ledger[key] = current
                skipped_exhausted += 1
                _persist_run_state(
                    artifact_dir,
                    manifest=manifest,
                    ledger=ledger,
                    source_df=source_df,
                    study=study,
                    work_items=work_items,
                    skipped_existing_success=skipped_existing_success,
                    skipped_exhausted=skipped_exhausted,
                    skipped_no_text=skipped_no_text,
                    final=False,
                )
                continue

            work_items += 1
            attempt_number = attempts + 1
            request = LLMRequest(
                participant_code=participant_code,
                section=section_name,
                prompt=str(batch_record["prompt"]),
                response_schema=dict(batch_record["response_schema"]),
                model=resolved.model,
                temperature=resolved.temperature,
                metadata={"artifact_id": resolved.metadata.artifact_id},
            )
            provider_response = None
            raw_record = None
            error_record = None
            try:
                provider_response = provider.generate_structured(request)
                prediction = parse_structured_response(provider_response.raw_text, request.response_schema)
                current.update(
                    {
                        "status": SUCCESS_STATUS,
                        "attempts": attempt_number,
                        "prediction": prediction,
                        "last_error_type": None,
                        "last_error_message": None,
                        "updated_at": _utc_now(),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                next_status = FAILED_STATUS
                if attempt_number >= resolved.runtime.max_attempts or status == EXHAUSTED_STATUS:
                    next_status = EXHAUSTED_STATUS
                current.update(
                    {
                        "status": next_status,
                        "attempts": attempt_number,
                        "prediction": None,
                        "last_error_type": type(exc).__name__,
                        "last_error_message": str(exc),
                        "updated_at": _utc_now(),
                    }
                )
                error_record = {
                    "timestamp": current["updated_at"],
                    "artifact_id": resolved.metadata.artifact_id,
                    "response_id": response_id,
                    "participant_code": participant_code,
                    "section": section_name,
                    "attempt": attempt_number,
                    "status": current["status"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            if provider_response is not None:
                raw_record = {
                    "timestamp": _utc_now(),
                    "artifact_id": resolved.metadata.artifact_id,
                    "response_id": response_id,
                    "participant_code": participant_code,
                    "section": section_name,
                    "attempt": attempt_number,
                    "provider": provider_response.provider,
                    "model": provider_response.model,
                    "raw_text": provider_response.raw_text,
                    "metadata": provider_response.metadata,
                }
            ledger[key] = current
            _persist_run_state(
                artifact_dir,
                manifest=manifest,
                ledger=ledger,
                source_df=source_df,
                study=study,
                work_items=work_items,
                skipped_existing_success=skipped_existing_success,
                skipped_exhausted=skipped_exhausted,
                skipped_no_text=skipped_no_text,
                raw_record=raw_record,
                error_record=error_record,
                final=False,
            )

    summary = _persist_run_state(
        artifact_dir,
        manifest=manifest,
        ledger=ledger,
        source_df=source_df,
        study=study,
        work_items=work_items,
        skipped_existing_success=skipped_existing_success,
        skipped_exhausted=skipped_exhausted,
        skipped_no_text=skipped_no_text,
        final=True,
    )
    written = _artifact_paths(artifact_dir)
    no_op = work_items == 0
    message = str(summary["execution"]["message"])
    return ExperimentRunResult(
        experiment_id=resolved.metadata.experiment_id,
        artifact_id=resolved.metadata.artifact_id,
        no_op=no_op,
        message=message,
        paths=written,
        summary=summary,
    )


def run_llm_experiments(
    study: StudyConfig,
    paths: PathsConfig,
    llm_config: LLMConfig,
    *,
    experiment_ids: list[str] | None = None,
    all_experiments: bool = False,
    input_path: Path | None = None,
    limit: int | None = None,
    all_records: bool | None = None,
    retry_exhausted: bool = False,
    output_dir: Path | None = None,
    provider_factory: Callable[[LLMRuntimeConfig], LLMProvider] | None = None,
) -> dict[str, Any]:
    selected = _select_experiments(llm_config, experiment_ids, all_experiments)
    output_root = Path(output_dir or paths.llm_results_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime = llm_config.runtime
    source_df = load_batch_source(
        study=study,
        paths=paths,
        source=runtime.source,
        input_path=Path(input_path) if input_path is not None else None,
        limit=limit,
        all_records=(runtime.all_records if all_records is None else bool(all_records)),
    )

    experiment_results: list[ExperimentRunResult] = []
    for experiment in selected:
        resolved = _resolve_experiment_run(
            experiment,
            runtime,
            paths,
            all_records_override=all_records,
        )
        experiment_results.append(
            _run_single_experiment(
                study,
                paths,
                source_df,
                resolved,
                input_path=Path(input_path) if input_path is not None else None,
                output_root=output_root,
                retry_exhausted=retry_exhausted,
                provider_factory=provider_factory,
            )
        )

    return {
        "source": {
            "type": runtime.source,
            "rows": int(len(source_df)),
            "all_records": bool(runtime.all_records if all_records is None else all_records),
            "input_path": str(_source_input_path(paths, runtime.source, Path(input_path) if input_path is not None else None)),
            "limit": limit,
        },
        "experiments": [
            {
                "experiment_id": result.experiment_id,
                "artifact_id": result.artifact_id,
                "no_op": result.no_op,
                "message": result.message,
                **result.paths,
                "summary": result.summary,
            }
            for result in experiment_results
        ],
    }

