from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PathsConfig, StudyConfig
from .constants import ANNOTATION_SHEET, SAMPLED_PRIVATE_SHEET


def _is_blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


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


def load_human_annotations(annotation_workbook: Path, study: StudyConfig) -> pd.DataFrame:
    if not annotation_workbook.exists():
        raise FileNotFoundError(f"Human annotation workbook not found: {annotation_workbook}")

    df = pd.read_excel(annotation_workbook, sheet_name=ANNOTATION_SHEET)
    rename_map = study.visible_to_internal_annotation_columns()
    required_visible = list(rename_map.keys())
    missing_visible = [column for column in required_visible if column not in df.columns]
    if missing_visible:
        raise ValueError(f"Human annotation workbook is missing columns: {missing_visible}")

    normalized = df[required_visible].rename(columns=rename_map)
    _validate_labels(normalized, study.annotation_internal_columns(), study, "Human annotations")
    return normalized


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
            for column in study.annotation_internal_columns():
                if column in payload:
                    row[column] = payload[column]
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


def _map_questionnaire_value(value: object, yes_values: list[str], no_values: list[str], source_column: str) -> str:
    if _is_blank(value):
        raise ValueError(f"Questionnaire column {source_column} contains blank values.")
    normalized = str(value).strip()
    if normalized in yes_values:
        return "yes"
    if normalized in no_values:
        return "no"
    raise ValueError(f"Unexpected questionnaire value in {source_column}: {normalized}")


def load_questionnaire_labels(sampled_private_workbook: Path, study: StudyConfig) -> pd.DataFrame:
    if not sampled_private_workbook.exists():
        raise FileNotFoundError(f"Sampled private workbook not found: {sampled_private_workbook}")

    placeholder_columns = study.placeholder_questionnaire_columns()
    if placeholder_columns:
        raise ValueError(f"Replace questionnaire placeholders before evaluation: {placeholder_columns}")

    df = pd.read_excel(sampled_private_workbook, sheet_name=SAMPLED_PRIVATE_SHEET)
    if "participant_code" not in df.columns:
        raise ValueError("Sampled private workbook must include participant_code.")

    out = pd.DataFrame({"participant_code": df["participant_code"]})
    for block_name in ("m8", "m9"):
        block = study.questionnaire[block_name]
        for internal_column, source_column in block["columns"].items():
            if source_column not in df.columns:
                raise ValueError(f"Sampled private workbook is missing questionnaire column: {source_column}")
            out[internal_column] = df[source_column].apply(
                lambda value: _map_questionnaire_value(value, block["yes_values"], block["no_values"], source_column)
            )

    _validate_labels(out, study.binary_columns(), study, "Questionnaire labels")
    return out


def _ensure_same_participants(reference: pd.DataFrame, other: pd.DataFrame, reference_name: str, other_name: str) -> None:
    reference_codes = set(reference["participant_code"])
    other_codes = set(other["participant_code"])
    if reference_codes != other_codes:
        missing_from_other = sorted(reference_codes - other_codes)
        missing_from_reference = sorted(other_codes - reference_codes)
        raise ValueError(
            f"Participant mismatch between {reference_name} and {other_name}. "
            f"Missing from {other_name}: {missing_from_other}. Missing from {reference_name}: {missing_from_reference}."
        )


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
        return 0.0
    observed = float(matrix.to_numpy().trace()) / total
    row_totals = matrix.sum(axis=1).to_numpy(dtype=float)
    column_totals = matrix.sum(axis=0).to_numpy(dtype=float)
    expected = float((row_totals * column_totals).sum()) / (total ** 2)
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


def compute_comparison_metrics(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    columns: list[str],
    study: StudyConfig,
    comparison_name: str,
) -> pd.DataFrame:
    merged = reference_df.merge(candidate_df, on="participant_code", suffixes=("_reference", "_candidate"))
    rows: list[dict[str, Any]] = []
    for column in columns:
        labels = study.allowed_labels_for_column(column)
        y_true = merged[f"{column}_reference"]
        y_pred = merged[f"{column}_candidate"]
        rows.append(
            {
                "comparison": comparison_name,
                "field": column,
                "n": int(len(merged)),
                "accuracy": accuracy_score(y_true, y_pred),
                "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels),
                "macro_f1": macro_f1_score(y_true, y_pred, labels),
            }
        )
    return pd.DataFrame(rows)


def evaluate_outputs(
    study: StudyConfig,
    paths: PathsConfig,
    human_annotation_workbook: Path | None = None,
    llm_predictions_path: Path | None = None,
    sampled_private_workbook: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    human_df = load_human_annotations(Path(human_annotation_workbook or paths.human_annotation_workbook), study)
    llm_df = load_llm_predictions(Path(llm_predictions_path or paths.llm_predictions_path), study)
    questionnaire_df = load_questionnaire_labels(Path(sampled_private_workbook or paths.sampled_private_workbook), study)

    _ensure_same_participants(human_df, llm_df, "human annotations", "LLM predictions")
    _ensure_same_participants(human_df, questionnaire_df, "human annotations", "questionnaire labels")

    metrics_frames = [
        compute_comparison_metrics(human_df, llm_df, study.annotation_internal_columns(), study, "human_vs_llm"),
        compute_comparison_metrics(human_df, questionnaire_df, study.binary_columns(), study, "human_vs_questionnaire"),
        compute_comparison_metrics(llm_df, questionnaire_df, study.binary_columns(), study, "llm_vs_questionnaire"),
    ]
    metrics_df = pd.concat(metrics_frames, ignore_index=True)

    summary = {
        comparison: {
            "fields": int(len(group)),
            "accuracy_mean": float(group["accuracy"].mean()),
            "cohen_kappa_mean": float(group["cohen_kappa"].mean()),
            "macro_f1_mean": float(group["macro_f1"].mean()),
        }
        for comparison, group in metrics_df.groupby("comparison")
    }

    evaluation_dir = Path(output_dir or paths.evaluation_output_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = evaluation_dir / "evaluation_metrics.csv"
    summary_path = evaluation_dir / "evaluation_summary.json"
    metrics_df.to_csv(metrics_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return metrics_df, summary, {"metrics_file": str(metrics_path), "summary_file": str(summary_path)}
