from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any
import tomllib

import pandas as pd

from .config import ExperimentMetadata, PathsConfig, StudyConfig
from .constants import ANNOTATION_SHEET, SAMPLED_PRIVATE_SHEET
from .evaluation_report import write_alignment_outputs
from .io_utils import read_tabular_file
from .llm_runner import INTERNAL_ARTIFACT_FILENAMES
from .sampling import assign_participant_codes, filter_source_data
from .vader_analysis import (
    default_vader_scores_path,
    load_vader_scores,
    run_vader_sensitivity,
    vader_scores_to_tone_predictions,
)


PREDICTION_SUFFIXES = {".jsonl", ".csv", ".xlsx", ".xls"}
HUMAN_SUFFIXES = {".xlsx", ".xls"}


def _is_blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning NaN if denominator is zero."""
    return numerator / denominator if denominator else float("nan")


def _normalize_identifier(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "artifact"


def _validate_labels(df: pd.DataFrame, columns: list[str], study: StudyConfig, source_name: str) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"{source_name} is missing required column: {column}")
        blank_mask = df[column].apply(_is_blank)
        if blank_mask.any():
            raise ValueError(f"{source_name} contains blank values in required column: {column}")
        allowed = set(study.allowed_labels_for_column(column))
        invalid = sorted({str(value) for value in df.loc[~blank_mask, column] if str(value) not in allowed})
        if invalid:
            raise ValueError(f"{source_name} contains invalid labels in {column}: {invalid}")


def _normalize_human_annotations(annotation_workbook: Path, study: StudyConfig) -> pd.DataFrame:
    if not annotation_workbook.exists():
        raise FileNotFoundError(f"Human annotation workbook not found: {annotation_workbook}")

    df = pd.read_excel(annotation_workbook, sheet_name=ANNOTATION_SHEET)
    rename_map = study.visible_to_internal_annotation_columns()
    required_visible = list(rename_map.keys())
    missing_visible = [column for column in required_visible if column not in df.columns]
    if missing_visible:
        raise ValueError(f"Human annotation workbook is missing columns: {missing_visible}")

    normalized = df[required_visible].rename(columns=rename_map).copy()
    normalized["participant_code"] = normalized["participant_code"].apply(lambda value: "" if pd.isna(value) else str(value).strip())
    blank_codes = normalized["participant_code"] == ""
    if blank_codes.any():
        raise ValueError("Human annotation workbook contains blank participant_code values.")
    return normalized


def load_human_annotations(annotation_workbook: Path, study: StudyConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    normalized = _normalize_human_annotations(annotation_workbook, study)
    label_columns = study.annotation_internal_columns()
    blank_matrix = normalized[label_columns].map(_is_blank)
    fully_blank_mask = blank_matrix.all(axis=1)
    partially_complete_mask = blank_matrix.any(axis=1) & ~fully_blank_mask

    if partially_complete_mask.any():
        partial_codes = normalized.loc[partially_complete_mask, "participant_code"].tolist()
        raise ValueError(
            "Human annotations contain partially completed rows. "
            f"Complete or clear all required labels for participant_code: {partial_codes}."
        )

    evaluable = normalized.loc[~fully_blank_mask].copy()
    _validate_labels(evaluable, label_columns, study, "Human annotations")

    coverage = {
        "n_rows_total": int(len(normalized)),
        "n_rows_evaluable": int(len(evaluable)),
        "n_rows_skipped_blank": int(fully_blank_mask.sum()),
    }
    if coverage["n_rows_evaluable"] == 0:
        raise ValueError("Human annotation workbook does not contain any fully annotated rows to evaluate.")
    return evaluable, coverage


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _flatten_llm_payload(payload: dict[str, Any], study: StudyConfig) -> dict[str, Any]:
    row: dict[str, Any] = {}

    # New nested section contract: {"context": {...}, "experience": {...}, "aftereffects": {...}}
    for section_name in study.section_order:
        section_config = study.sections[section_name]
        section_block = payload.get(section_name)
        if not isinstance(section_block, dict):
            continue

        tone_value = section_block.get("tone")
        if tone_value is not None:
            row[section_config.tone_internal_column] = tone_value

        for column in section_config.binary_labels:
            if column in section_block:
                row[column] = section_block[column]

    # Legacy flat contract fallback
    for column in study.annotation_internal_columns():
        if column in payload and column not in row:
            row[column] = payload[column]

    return row


def load_llm_predictions(prediction_path: Path, study: StudyConfig) -> pd.DataFrame:
    if not prediction_path.exists():
        raise FileNotFoundError(f"LLM predictions file not found: {prediction_path}")

    suffix = prediction_path.suffix.lower()
    if suffix == ".jsonl":
        records = _jsonl_records(prediction_path)
        rows: list[dict[str, Any]] = []
        for record in records:
            participant_code = record.get("participant_code")
            if not participant_code:
                raise ValueError("Every prediction record must include participant_code.")
            payload = record.get("prediction", record)
            row = {"participant_code": participant_code}
            row.update(_flatten_llm_payload(payload, study))
            rows.append(row)
        df = pd.DataFrame(rows)
        predictions = df.groupby("participant_code", as_index=False).first()
    elif suffix == ".csv":
        predictions = pd.read_csv(prediction_path)
    elif suffix in {".xlsx", ".xls"}:
        predictions = pd.read_excel(prediction_path)
    else:
        raise ValueError(f"Unsupported predictions format: {prediction_path}")

    if "participant_code" not in predictions.columns:
        raise ValueError("LLM predictions must include participant_code.")

    required = ["participant_code"] + study.annotation_internal_columns()
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"LLM predictions are missing columns: {missing}")

    normalized = predictions[required].copy()
    _validate_labels(normalized, study.annotation_internal_columns(), study, "LLM predictions")
    return normalized


def _load_metadata_from_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        with path.open("rb") as handle:
            return tomllib.load(handle)
    raise ValueError(f"Unsupported metadata file: {path}")


def _artifact_metadata(artifact_path: Path) -> tuple[dict[str, Any], Path | None]:
    candidates = [
        artifact_path.with_name(f"{artifact_path.stem}.manifest.json"),
        artifact_path.with_name(f"{artifact_path.stem}.manifest.toml"),
        artifact_path.parent / "manifest.json",
        artifact_path.parent / "manifest.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_metadata_from_file(candidate), candidate
    return {}, None


def _discover_files(root_dir: Path, suffixes: set[str]) -> list[Path]:
    if not root_dir.exists():
        return []
    return sorted(path for path in root_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _annotator_id_for(path: Path, metadata: dict[str, Any]) -> str:
    raw = str(metadata.get("annotator_id") or metadata.get("id") or path.stem)
    return _normalize_identifier(raw)


def _experiment_metadata_for(path: Path, metadata: dict[str, Any]) -> ExperimentMetadata:
    experiment_id = str(metadata.get("experiment_id") or metadata.get("id") or path.parent.name or path.stem)
    return ExperimentMetadata(
        experiment_id=_normalize_identifier(experiment_id),
        prompt_variant=(str(metadata["prompt_variant"]) if metadata.get("prompt_variant") is not None else None),
        run_id=(str(metadata["run_id"]) if metadata.get("run_id") is not None else None),
        model_variant=(str(metadata["model_variant"]) if metadata.get("model_variant") is not None else None),
    )


def discover_human_annotations(
    study: StudyConfig,
    paths: PathsConfig,
    human_annotation_workbook: Path | None = None,
    human_annotations_dir: Path | None = None,
    annotator_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    allowed_ids = {_normalize_identifier(value) for value in annotator_ids} if annotator_ids else None

    if human_annotation_workbook is not None:
        candidates = [Path(human_annotation_workbook)]
    else:
        root = Path(human_annotations_dir or paths.human_annotations_dir)
        candidates = _discover_files(root, HUMAN_SUFFIXES)

    for candidate in candidates:
        metadata, metadata_path = _artifact_metadata(candidate)
        annotator_id = _annotator_id_for(candidate, metadata)
        artifact = {
            "annotator_id": annotator_id,
            "artifact_path": str(candidate),
            "metadata_path": str(metadata_path) if metadata_path else None,
        }
        if allowed_ids and annotator_id not in allowed_ids:
            continue
        try:
            df, coverage = load_human_annotations(candidate, study)
        except Exception as exc:  # noqa: BLE001
            rejected.append({**artifact, "reason": str(exc)})
            continue
        df["annotator_id"] = annotator_id
        accepted.append({**artifact, "coverage": coverage, "rows": int(len(df)), "data": df})

    if not accepted:
        raise ValueError("No valid human annotation artifacts were found.")
    return accepted, rejected


def discover_llm_prediction_artifacts(
    study: StudyConfig,
    paths: PathsConfig,
    llm_predictions_path: Path | None = None,
    llm_results_dir: Path | None = None,
    experiment_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    allowed_ids = {_normalize_identifier(value) for value in experiment_ids} if experiment_ids else None

    if llm_predictions_path is not None:
        candidates = [Path(llm_predictions_path)]
    else:
        root = Path(llm_results_dir or paths.llm_results_dir)
        candidates = [candidate for candidate in _discover_files(root, PREDICTION_SUFFIXES) if candidate.name not in INTERNAL_ARTIFACT_FILENAMES]

    seen_artifact_ids: set[str] = set()
    for candidate in candidates:
        metadata, metadata_path = _artifact_metadata(candidate)
        experiment = _experiment_metadata_for(candidate, metadata)
        if allowed_ids and experiment.experiment_id not in allowed_ids and experiment.artifact_id not in allowed_ids:
            continue
        artifact = {
            **experiment.to_dict(),
            "artifact_path": str(candidate),
            "metadata_path": str(metadata_path) if metadata_path else None,
        }
        if experiment.artifact_id in seen_artifact_ids:
            rejected.append({**artifact, "reason": f"Duplicate artifact_id discovered: {experiment.artifact_id}"})
            continue
        try:
            df = load_llm_predictions(candidate, study)
        except Exception as exc:  # noqa: BLE001
            rejected.append({**artifact, "reason": str(exc)})
            continue
        seen_artifact_ids.add(experiment.artifact_id)
        accepted.append({**artifact, "rows": int(len(df)), "data": df})
    return accepted, rejected


def _map_questionnaire_value(value: object, yes_values: list[str], no_values: list[str], source_column: str) -> str:
    if _is_blank(value):
        raise ValueError(f"Questionnaire column {source_column} contains blank values.")
    normalized = str(value).strip()
    if normalized in yes_values:
        return "yes"
    if normalized in no_values:
        return "no"
    raise ValueError(f"Unexpected questionnaire value in {source_column}: {normalized}")


def load_sampled_private_data(sampled_private_workbook: Path) -> pd.DataFrame:
    if not sampled_private_workbook.exists():
        raise FileNotFoundError(f"Sampled private workbook not found: {sampled_private_workbook}")
    return pd.read_excel(sampled_private_workbook, sheet_name=SAMPLED_PRIVATE_SHEET)


def _filter_participant_subset(df: pd.DataFrame, participant_codes: set[str] | None) -> pd.DataFrame:
    if participant_codes is None:
        return df.copy()
    return df[df["participant_code"].isin(participant_codes)].copy()


def load_questionnaire_labels(
    source_path: Path,
    study: StudyConfig,
    participant_codes: set[str] | None = None,
) -> pd.DataFrame:
    placeholder_columns = study.placeholder_questionnaire_columns()
    if placeholder_columns:
        raise ValueError(f"Replace questionnaire placeholders before evaluation: {placeholder_columns}")

    df = read_tabular_file(source_path)
    df = filter_source_data(df, study)
    if "participant_code" not in df.columns:
        df = assign_participant_codes(df, study)
    df = _filter_participant_subset(df, participant_codes)

    out = pd.DataFrame({"participant_code": df["participant_code"]})
    stratify_column = study.stratify_column
    if stratify_column not in df.columns:
        raise ValueError(f"Sampled private workbook is missing questionnaire tone column: {stratify_column}")

    tone_map = {
        "positive": "positive",
        "negative": "negative",
        "mixed": "mixed",
    }
    experience_tone_column = study.sections["experience"].tone_internal_column
    out[experience_tone_column] = df[stratify_column].apply(
        lambda value: tone_map.get(str(value).strip().lower(), str(value).strip().lower()) if pd.notna(value) else pd.NA
    )

    for block_name in ("m8", "m9"):
        block = study.questionnaire[block_name]
        for internal_column, source_column in block["columns"].items():
            if source_column not in df.columns:
                raise ValueError(f"Sampled private workbook is missing questionnaire column: {source_column}")
            out[internal_column] = df[source_column].apply(
                lambda value: _map_questionnaire_value(value, block["yes_values"], block["no_values"], source_column)
            )

    _validate_labels(out, [experience_tone_column], study, "Questionnaire tone labels")
    _validate_labels(out, study.binary_columns(), study, "Questionnaire labels")
    return out


def load_vader_predictions(
    study: StudyConfig,
    paths: PathsConfig,
    sampled_private_workbook: Path,
    participant_codes: set[str] | None = None,
    vader_scores_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    survey_df = read_tabular_file(paths.survey_csv)
    survey_df = filter_source_data(survey_df, study)
    survey_df = assign_participant_codes(survey_df, study)
    sampled_private_df = _filter_participant_subset(load_sampled_private_data(sampled_private_workbook), participant_codes)
    resolved_vader_scores = Path(vader_scores_path or default_vader_scores_path(paths))

    if vader_scores_path is not None and not resolved_vader_scores.exists():
        raise FileNotFoundError(f"VADER scores file not found: {resolved_vader_scores}")

    vader_summary: dict[str, Any] | None = None
    if resolved_vader_scores.exists():
        scores_df = load_vader_scores(resolved_vader_scores, study)
        summary_path = resolved_vader_scores.with_name("vader_summary.json")
        if summary_path.exists():
            vader_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        sample_output_dir = Path(output_dir or paths.evaluation_output_dir) / "vader_sentiment_sample"
        scores_df, vader_summary, _ = run_vader_sensitivity(
            study,
            paths,
            output_dir=sample_output_dir,
            all_records=False,
            source_df=survey_df,
        )

    if "participant_code" in scores_df.columns:
        blank_codes = scores_df["participant_code"].fillna("").astype(str).str.strip() == ""
        if blank_codes.any() and study.id_column in scores_df.columns and study.id_column in survey_df.columns:
            participant_lookup = survey_df[[study.id_column, "participant_code"]].drop_duplicates()
            scores_df = scores_df.drop(columns=["participant_code"]).merge(
                participant_lookup,
                on=study.id_column,
                how="left",
            )

    predictions = vader_scores_to_tone_predictions(scores_df, survey_df, study)
    _validate_labels(predictions, study.tone_columns(), study, "VADER predictions")
    return predictions, vader_summary


def _outer_merge(reference_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    return reference_df.merge(candidate_df, on="participant_code", how="outer", suffixes=("_reference", "_candidate"))


def accuracy_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true == y_pred).mean())


def cohen_kappa_score(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    matrix = pd.crosstab(
        pd.Categorical(y_true, categories=labels),
        pd.Categorical(y_pred, categories=labels),
        dropna=False,
    )
    total = float(matrix.to_numpy().sum())
    if total == 0:
        return float("nan")
    observed = float(matrix.to_numpy().trace()) / total
    row_totals = matrix.sum(axis=1).to_numpy(dtype=float)
    column_totals = matrix.sum(axis=0).to_numpy(dtype=float)
    expected = float((row_totals * column_totals).sum()) / (total**2)
    denominator = 1.0 - expected
    if denominator == 0:
        return 1.0 if observed == 1.0 else 0.0
    return float((observed - expected) / denominator)


def macro_f1_score(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return float(sum(scores) / len(scores))


def binary_positive_metrics(y_true: pd.Series, y_pred: pd.Series, positive_label: str = "yes") -> dict[str, float]:
    tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    fp = int(((y_true != positive_label) & (y_pred == positive_label)).sum())
    fn = int(((y_true == positive_label) & (y_pred != positive_label)).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    prevalence_reference = float((y_true == positive_label).mean()) if len(y_true) else float("nan")
    prevalence_candidate = float((y_pred == positive_label).mean()) if len(y_pred) else float("nan")
    return {
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "prevalence_reference_yes": prevalence_reference,
        "prevalence_candidate_yes": prevalence_candidate,
        "prevalence_gap_yes": abs(prevalence_candidate - prevalence_reference) if len(y_true) else float("nan"),
    }


def compute_comparison_metrics(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    columns: list[str],
    study: StudyConfig,
    comparison_name: str,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    merged = _outer_merge(reference_df, candidate_df)
    rows: list[dict[str, Any]] = []
    for column in columns:
        labels = study.allowed_labels_for_column(column)
        reference_values = merged[f"{column}_reference"]
        candidate_values = merged[f"{column}_candidate"]
        valid_mask = ~reference_values.apply(_is_blank) & ~candidate_values.apply(_is_blank)
        y_true = reference_values.loc[valid_mask]
        y_pred = candidate_values.loc[valid_mask]
        if len(y_true) == 0:
            accuracy = float("nan")
            kappa = float("nan")
            macro_f1 = float("nan")
            extra_metrics = {
                "precision_yes": float("nan"),
                "recall_yes": float("nan"),
                "f1_yes": float("nan"),
                "prevalence_reference_yes": float("nan"),
                "prevalence_candidate_yes": float("nan"),
                "prevalence_gap_yes": float("nan"),
            }
        else:
            accuracy = accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred, labels)
            macro_f1 = macro_f1_score(y_true, y_pred, labels)
            extra_metrics = binary_positive_metrics(y_true, y_pred) if labels == ["yes", "no"] or labels == ["no", "yes"] else {
                "precision_yes": float("nan"),
                "recall_yes": float("nan"),
                "f1_yes": float("nan"),
                "prevalence_reference_yes": float("nan"),
                "prevalence_candidate_yes": float("nan"),
                "prevalence_gap_yes": float("nan"),
            }
        row: dict[str, Any] = {
            "comparison": comparison_name,
            "field": column,
            "n": int(len(y_true)),
            "accuracy": accuracy,
            "cohen_kappa": kappa,
            "macro_f1": macro_f1,
            **extra_metrics,
        }
        if metadata:
            row.update(metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def consolidate_majority_human_reference(human_artifacts: list[dict[str, Any]], study: StudyConfig) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    frames = [artifact["data"][["participant_code", *study.annotation_internal_columns(), "annotator_id"]].copy() for artifact in human_artifacts]
    combined = pd.concat(frames, ignore_index=True)
    reference = pd.DataFrame({"participant_code": sorted(combined["participant_code"].drop_duplicates())})
    adjudication_rows: list[dict[str, Any]] = []
    unresolved_total = 0

    for column in study.annotation_internal_columns():
        value_counts = (
            combined[["participant_code", column]]
            .groupby(["participant_code", column], as_index=False)
            .size()
            .rename(columns={"size": "votes"})
            .sort_values(["participant_code", "votes", column], ascending=[True, False, True])
        )
        resolved_map: dict[str, str | None] = {}
        for participant_code, subset in value_counts.groupby("participant_code"):
            top_votes = int(subset.iloc[0]["votes"])
            leaders = subset[subset["votes"] == top_votes]
            if len(leaders) == 1:
                resolved_map[str(participant_code)] = str(leaders.iloc[0][column])
            else:
                resolved_map[str(participant_code)] = None
                unresolved_total += 1
        reference[column] = reference["participant_code"].map(resolved_map)
        adjudicated_n = int(reference[column].notna().sum())
        unresolved_n = int(reference[column].isna().sum())
        adjudication_rows.append(
            {
                "field": column,
                "n_participants": int(len(reference)),
                "n_adjudicated": adjudicated_n,
                "n_unresolved": unresolved_n,
                "n_annotators": int(combined["annotator_id"].nunique()),
            }
        )

    reference = reference.copy()
    summary = {
        "n_valid_annotators": int(combined["annotator_id"].nunique()),
        "n_reference_participants": int(len(reference)),
        "n_fields": int(len(study.annotation_internal_columns())),
        "n_unresolved_field_participant_pairs": int(unresolved_total),
    }
    return reference, summary, combined


def compute_human_agreement_metrics(human_long_df: pd.DataFrame, study: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_frames: list[pd.DataFrame] = []
    annotator_ids = sorted(human_long_df["annotator_id"].drop_duplicates())
    for annotator_a, annotator_b in combinations(annotator_ids, 2):
        left = human_long_df[human_long_df["annotator_id"] == annotator_a].drop(columns=["annotator_id"])
        right = human_long_df[human_long_df["annotator_id"] == annotator_b].drop(columns=["annotator_id"])
        metrics = compute_comparison_metrics(
            left,
            right,
            study.annotation_internal_columns(),
            study,
            comparison_name=f"annotator_pair:{annotator_a}_vs_{annotator_b}",
            metadata={"annotator_a": annotator_a, "annotator_b": annotator_b},
        )
        pairwise_frames.append(metrics)

    if pairwise_frames:
        pairwise_df = pd.concat(pairwise_frames, ignore_index=True)
        summary_df = (
            pairwise_df.groupby("field", as_index=False)[["n", "accuracy", "cohen_kappa", "macro_f1"]]
            .mean(numeric_only=True)
            .rename(
                columns={
                    "n": "n_mean",
                    "accuracy": "accuracy_mean",
                    "cohen_kappa": "cohen_kappa_mean",
                    "macro_f1": "macro_f1_mean",
                }
            )
        )
        summary_df["n_pairs"] = int(len(pairwise_frames))
        return pairwise_df, summary_df

    columns = ["comparison", "field", "n", "accuracy", "cohen_kappa", "macro_f1", "annotator_a", "annotator_b"]
    summary_columns = ["field", "n_mean", "accuracy_mean", "cohen_kappa_mean", "macro_f1_mean", "n_pairs"]
    return pd.DataFrame(columns=columns), pd.DataFrame(columns=summary_columns)


def _comparison_summary(metrics_df: pd.DataFrame) -> dict[str, Any]:
    if metrics_df.empty:
        return {}
    return {
        comparison: {
            "fields": int(len(group)),
            "accuracy_mean": float(group["accuracy"].mean()),
            "cohen_kappa_mean": float(group["cohen_kappa"].mean()),
            "macro_f1_mean": float(group["macro_f1"].mean()),
        }
        for comparison, group in metrics_df.groupby("comparison")
    }


def _comparison_source(
    kind: str,
    df: pd.DataFrame,
    study: StudyConfig,
    *,
    artifact_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "df": df,
        "columns": [column for column in study.annotation_internal_columns() if column in df.columns],
        "artifact_id": artifact_id,
        "metadata": metadata or {},
    }


def _pairwise_comparison_name(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_kind = str(left["kind"])
    right_kind = str(right["kind"])
    if left_kind == "llm" and right_kind == "llm":
        return f"llm_vs_llm:{left['artifact_id']}__vs__{right['artifact_id']}"
    if right_kind == "llm":
        return f"{left_kind}_vs_llm:{right['artifact_id']}"
    return f"{left_kind}_vs_{right_kind}"


def _pairwise_comparison_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_kind = str(left["kind"])
    right_kind = str(right["kind"])
    metadata: dict[str, Any] = {}
    if left_kind == "llm":
        metadata.update({f"left_{key}": value for key, value in dict(left.get("metadata", {})).items()})
    if right_kind == "llm":
        right_metadata = dict(right.get("metadata", {}))
        if left_kind == "llm":
            metadata.update({f"right_{key}": value for key, value in right_metadata.items()})
        else:
            metadata.update(right_metadata)
    return metadata


def _compute_source_pair_metrics(left: dict[str, Any], right: dict[str, Any], study: StudyConfig) -> pd.DataFrame | None:
    shared_columns = [column for column in left["columns"] if column in right["columns"]]
    if not shared_columns:
        return None
    return compute_comparison_metrics(
        left["df"],
        right["df"],
        shared_columns,
        study,
        comparison_name=_pairwise_comparison_name(left, right),
        metadata=_pairwise_comparison_metadata(left, right),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def evaluate_outputs(
    study: StudyConfig,
    paths: PathsConfig,
    human_annotation_workbook: Path | None = None,
    human_annotations_dir: Path | None = None,
    llm_predictions_path: Path | None = None,
    llm_results_dir: Path | None = None,
    annotator_ids: list[str] | None = None,
    experiment_ids: list[str] | None = None,
    sampled_private_workbook: Path | None = None,
    output_dir: Path | None = None,
    vader_scores_path: Path | None = None,
    figure_dpi: int = 300,
    export_figures_pdf: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    resolved_sampled_private = Path(sampled_private_workbook or paths.sampled_private_workbook)
    evaluation_dir = Path(output_dir or paths.evaluation_output_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    human_artifacts, rejected_humans = discover_human_annotations(
        study,
        paths,
        human_annotation_workbook=human_annotation_workbook,
        human_annotations_dir=human_annotations_dir,
        annotator_ids=annotator_ids,
    )
    reference_df, adjudication_summary, human_long_df = consolidate_majority_human_reference(human_artifacts, study)
    participant_codes = set(reference_df["participant_code"])

    questionnaire_df = load_questionnaire_labels(paths.survey_csv, study)
    vader_df, vader_summary = load_vader_predictions(
        study,
        paths,
        resolved_sampled_private,
        vader_scores_path=Path(vader_scores_path) if vader_scores_path else None,
        output_dir=evaluation_dir,
    )

    questionnaire_source = _comparison_source("questionnaire", questionnaire_df, study)
    vader_source = _comparison_source("vader", vader_df, study)

    questionnaire_tone_columns = [study.sections["experience"].tone_internal_column]
    metrics_frames = [
        compute_comparison_metrics(reference_df, questionnaire_df, questionnaire_tone_columns, study, "human_reference_vs_questionnaire"),
        compute_comparison_metrics(reference_df, questionnaire_df, study.binary_columns(), study, "human_reference_vs_questionnaire"),
        compute_comparison_metrics(reference_df, vader_df, study.tone_columns(), study, "human_reference_vs_vader"),
    ]

    llm_candidates, rejected_llm = discover_llm_prediction_artifacts(
        study,
        paths,
        llm_predictions_path=llm_predictions_path,
        llm_results_dir=llm_results_dir,
        experiment_ids=experiment_ids,
    )
    accepted_llm: list[dict[str, Any]] = []
    accepted_llm_sources: list[dict[str, Any]] = []
    experiment_output_dir = evaluation_dir / "experiments"
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    for artifact in llm_candidates:
        llm_df = artifact["data"]
        comparison_name = f"human_reference_vs_llm:{artifact['artifact_id']}"
        metadata = {
            "experiment_id": artifact["experiment_id"],
            "artifact_id": artifact["artifact_id"],
            "prompt_variant": artifact.get("prompt_variant"),
            "run_id": artifact.get("run_id"),
            "model_variant": artifact.get("model_variant"),
        }
        experiment_metrics = compute_comparison_metrics(
            reference_df,
            llm_df,
            study.annotation_internal_columns(),
            study,
            comparison_name=comparison_name,
            metadata=metadata,
        )
        if int(experiment_metrics["n"].fillna(0).sum()) == 0:
            rejected_llm.append({**{k: v for k, v in artifact.items() if k != "data"}, "reason": "No overlap with adjudicated human reference."})
            continue
        accepted_llm.append({k: v for k, v in artifact.items() if k != "data"})
        accepted_llm_sources.append(
            _comparison_source(
                "llm",
                llm_df,
                study,
                artifact_id=str(artifact["artifact_id"]),
                metadata=metadata,
            )
        )
        metrics_frames.append(experiment_metrics)

        artifact_dir = experiment_output_dir / artifact["artifact_id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_metrics_path = artifact_dir / "evaluation_metrics.csv"
        artifact_summary_path = artifact_dir / "evaluation_summary.json"
        artifact_metrics = experiment_metrics.copy()
        artifact_metrics.to_csv(artifact_metrics_path, index=False)
        artifact_summary = {
            "experiment": {key: artifact.get(key) for key in ("experiment_id", "artifact_id", "prompt_variant", "run_id", "model_variant")},
            "comparisons": _comparison_summary(artifact_metrics),
        }
        artifact_summary_path.write_text(json.dumps(artifact_summary, indent=2), encoding="utf-8")

    additional_sources = [questionnaire_source, vader_source, *accepted_llm_sources]
    for left_source, right_source in combinations(additional_sources, 2):
        pair_metrics = _compute_source_pair_metrics(left_source, right_source, study)
        if pair_metrics is not None:
            metrics_frames.append(pair_metrics)

    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    human_pairwise_df, human_summary_df = compute_human_agreement_metrics(human_long_df, study)

    sampled_private_df = load_sampled_private_data(resolved_sampled_private)
    coverage = {
        "n_sampled_total": int(len(sampled_private_df)),
        "n_valid_human_artifacts": int(len(human_artifacts)),
        "n_rejected_human_artifacts": int(len(rejected_humans)),
        "n_reference_participants": adjudication_summary["n_reference_participants"],
        "n_valid_llm_artifacts": int(len(accepted_llm)),
        "n_rejected_llm_artifacts": int(len(rejected_llm)),
    }

    summary = {
        "coverage": coverage,
        "adjudication": {
            **adjudication_summary,
            "fields": human_summary_df.to_dict(orient="records") if not human_summary_df.empty else [],
        },
        "comparisons": _comparison_summary(metrics_df),
        "comparison_scopes": {
            comparison: {
                "fields": int(len(group)),
                "max_overlap_n": int(group["n"].max()) if not group.empty else 0,
                "min_overlap_n": int(group["n"].min()) if not group.empty else 0,
            }
            for comparison, group in metrics_df.groupby("comparison")
        },
        "human_artifacts": {
            "accepted": [{k: v for k, v in artifact.items() if k != "data"} for artifact in human_artifacts],
            "rejected": rejected_humans,
        },
        "llm_artifacts": {
            "accepted": accepted_llm,
            "rejected": rejected_llm,
        },
    }

    metrics_path = evaluation_dir / "evaluation_metrics.csv"
    summary_path = evaluation_dir / "evaluation_summary.json"
    reference_path = evaluation_dir / "human_reference_majority.csv"
    adjudication_path = evaluation_dir / "adjudication_summary.csv"
    human_pairwise_path = evaluation_dir / "human_agreement_pairwise.csv"
    human_summary_path = evaluation_dir / "human_agreement_summary.csv"
    human_manifest_path = evaluation_dir / "human_artifacts_manifest.json"
    llm_manifest_path = evaluation_dir / "llm_artifacts_manifest.json"

    metrics_df.to_csv(metrics_path, index=False)
    reference_df.to_csv(reference_path, index=False)
    pd.DataFrame(summary["adjudication"]["fields"]).to_csv(adjudication_path, index=False)
    human_pairwise_df.to_csv(human_pairwise_path, index=False)
    human_summary_df.to_csv(human_summary_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_json(human_manifest_path, summary["human_artifacts"])
    _write_json(llm_manifest_path, summary["llm_artifacts"])

    reporting_outputs = write_alignment_outputs(
        study=study,
        metrics_df=metrics_df,
        summary=summary,
        output_dir=evaluation_dir,
        vader_summary=vader_summary,
        figure_dpi=figure_dpi,
        export_figures_pdf=export_figures_pdf,
    )

    return metrics_df, summary, {
        "metrics_file": str(metrics_path),
        "summary_file": str(summary_path),
        "human_reference_file": str(reference_path),
        "adjudication_summary_file": str(adjudication_path),
        "human_agreement_pairwise_file": str(human_pairwise_path),
        "human_agreement_summary_file": str(human_summary_path),
        "human_artifacts_manifest_file": str(human_manifest_path),
        "llm_artifacts_manifest_file": str(llm_manifest_path),
        **reporting_outputs,
    }
