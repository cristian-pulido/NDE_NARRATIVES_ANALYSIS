from __future__ import annotations

import json
import re
from collections import Counter
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
from .prompting import load_batch_source
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


def _select_top_by_pareto(
    df: pd.DataFrame,
    *,
    top_n: int,
    metric_columns: tuple[str, str] = ("macro_f1", "cohen_kappa"),
) -> pd.DataFrame:
    """Select top rows using Pareto-front ranking over two metrics.

    Primary rule: non-dominated sorting (maximize both metrics).
    Tie-break inside each front: max-min balance over normalized metrics.
    """
    if df.empty:
        return df

    m1, m2 = metric_columns
    work = df.copy()
    work = work[work[m1].notna() & work[m2].notna()].copy()
    if work.empty:
        return df.head(0).copy()

    remaining = work.reset_index(drop=True)
    selected_fronts: list[pd.DataFrame] = []

    while not remaining.empty and sum(len(front) for front in selected_fronts) < top_n:
        idx = remaining.index.to_list()
        frontier: list[int] = []
        for i in idx:
            i_m1 = float(remaining.at[i, m1])
            i_m2 = float(remaining.at[i, m2])
            dominated = False
            for j in idx:
                if i == j:
                    continue
                j_m1 = float(remaining.at[j, m1])
                j_m2 = float(remaining.at[j, m2])
                j_dominates_i = (j_m1 >= i_m1 and j_m2 >= i_m2) and (j_m1 > i_m1 or j_m2 > i_m2)
                if j_dominates_i:
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)

        front_df = remaining.loc[frontier].copy()
        # Balance tie-break: maximize the weaker metric after min-max normalization.
        span_m1 = float(front_df[m1].max() - front_df[m1].min())
        span_m2 = float(front_df[m2].max() - front_df[m2].min())
        front_df["_m1_norm"] = 0.5 if span_m1 == 0 else (front_df[m1] - float(front_df[m1].min())) / span_m1
        front_df["_m2_norm"] = 0.5 if span_m2 == 0 else (front_df[m2] - float(front_df[m2].min())) / span_m2
        front_df["_balance"] = front_df[["_m1_norm", "_m2_norm"]].min(axis=1)
        front_df = front_df.sort_values(["_balance", m1, m2], ascending=[False, False, False], na_position="last")
        selected_fronts.append(front_df.drop(columns=["_m1_norm", "_m2_norm", "_balance"]))
        remaining = remaining.drop(index=frontier)

    out = pd.concat(selected_fronts, ignore_index=True) if selected_fronts else work.head(0).copy()
    return out.head(top_n)


def _normalize_identifier(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "artifact"


def _infer_prompt_variant_from_metadata(metadata: dict[str, Any]) -> str | None:
    explicit_prompt_variant = metadata.get("prompt_variant")
    if explicit_prompt_variant is not None:
        resolved = str(explicit_prompt_variant).strip()
        if resolved:
            return resolved

    prompt_root_raw = metadata.get("prompt_root")
    if prompt_root_raw is None:
        return None
    prompt_root_text = str(prompt_root_raw).strip()
    if not prompt_root_text:
        return None

    prompt_root = Path(prompt_root_text)
    if prompt_root.parent.name == "prompt_variants":
        candidate = prompt_root.name.strip()
        return candidate or None

    if prompt_root.name == "analysis":
        return "default"

    candidate = prompt_root.name.strip()
    return candidate or None


def _validate_labels(
    df: pd.DataFrame,
    columns: list[str],
    study: StudyConfig,
    source_name: str,
    allow_blank_columns: set[str] | None = None,
) -> None:
    allow_blank = allow_blank_columns or set()
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"{source_name} is missing required column: {column}")
        blank_mask = df[column].apply(_is_blank)
        if blank_mask.any() and column not in allow_blank:
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

    return row


def _normalize_evidence_segments(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    segments: list[str] = []
    for segment in value:
        text = str(segment).strip()
        if text:
            segments.append(text)
    return segments


def _extract_section_evidence(payload: dict[str, Any], section_name: str) -> list[str]:
    section_block = payload.get(section_name)
    if isinstance(section_block, dict):
        return _normalize_evidence_segments(section_block.get("evidence_segments"))
    return []


def load_llm_predictions_with_evidence(prediction_path: Path, study: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not prediction_path.exists():
        raise FileNotFoundError(f"LLM predictions file not found: {prediction_path}")

    suffix = prediction_path.suffix.lower()
    evidence_rows: list[dict[str, Any]] = []
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

            for section_name in study.section_order:
                segments = _extract_section_evidence(payload, section_name)
                if segments:
                    evidence_rows.append(
                        {
                            "participant_code": str(participant_code),
                            "section": section_name,
                            "evidence_segments": segments,
                        }
                    )

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

    if not evidence_rows:
        return normalized, pd.DataFrame(columns=["participant_code", "section", "evidence_segments"])

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df = (
        evidence_df.groupby(["participant_code", "section"], as_index=False)["evidence_segments"]
        .agg(
            lambda values: [
                segment
                for segments in values
                for segment in segments
            ]
        )
    )
    return normalized, evidence_df


def load_llm_predictions(prediction_path: Path, study: StudyConfig) -> pd.DataFrame:
    predictions, _ = load_llm_predictions_with_evidence(prediction_path, study)
    return predictions


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
        prompt_variant=_infer_prompt_variant_from_metadata(metadata),
        run_id=(str(metadata["run_id"]) if metadata.get("run_id") is not None else None),
        model_variant=(str(metadata["model_variant"]) if metadata.get("model_variant") is not None else None),
    )


def discover_human_annotations(
    study: StudyConfig,
    paths: PathsConfig,
    human_annotation_workbook: Path | None = None,
    human_annotations_dir: Path | None = None,
    annotator_ids: list[str] | None = None,
    allow_empty: bool = False,
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
        scanned_root = Path(human_annotations_dir or paths.human_annotations_dir)
        if allow_empty and not candidates:
            return [], []
        if not candidates:
            raise ValueError(
                "No valid human annotation artifacts were found. "
                f"No .xlsx/.xls files were discovered under: {scanned_root}"
            )
        reasons_preview = "; ".join(
            f"{Path(str(item.get('artifact_path', 'unknown'))).name}: {item.get('reason', 'unknown reason')}"
            for item in rejected[:5]
        )
        raise ValueError(
            "No valid human annotation artifacts were found. "
            f"Scanned: {scanned_root}. "
            f"Found {len(candidates)} candidate file(s), rejected {len(rejected)}. "
            f"First rejections: {reasons_preview or 'n/a'}"
        )
    return accepted, rejected


def discover_llm_prediction_artifacts(
    study: StudyConfig,
    paths: PathsConfig,
    llm_predictions_path: Path | None = None,
    llm_results_dir: Path | None = None,
    experiment_ids: list[str] | None = None,
    prompt_variants: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    allowed_ids = {_normalize_identifier(value) for value in experiment_ids} if experiment_ids else None
    allowed_prompt_variants = {_normalize_identifier(value) for value in prompt_variants} if prompt_variants else None

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
        normalized_prompt_variant = _normalize_identifier(str(experiment.prompt_variant or "")) if experiment.prompt_variant else None
        if allowed_prompt_variants and (normalized_prompt_variant is None or normalized_prompt_variant not in allowed_prompt_variants):
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
            df, evidence_df = load_llm_predictions_with_evidence(candidate, study)
        except Exception as exc:  # noqa: BLE001
            rejected.append({**artifact, "reason": str(exc)})
            continue
        seen_artifact_ids.add(experiment.artifact_id)
        accepted.append({**artifact, "rows": int(len(df)), "data": df, "evidence_data": evidence_df})
    return accepted, rejected


def _strict_polarity_contradiction(questionnaire_label: object, candidate_label: object) -> bool:
    questionnaire = str(questionnaire_label).strip().lower()
    candidate = str(candidate_label).strip().lower()
    return (questionnaire == "positive" and candidate == "negative") or (questionnaire == "negative" and candidate == "positive")


def _tokenize_terms(text: str) -> list[str]:
    stopwords = {
        "the", "and", "for", "that", "with", "this", "from", "have", "were", "was", "they", "them", "then", "when", "into", "onto", "very", "just", "about", "because", "there", "their", "would", "could", "should", "after", "before", "while", "where", "which", "what", "been", "being", "also", "than", "some", "more", "less", "over", "under", "such", "only", "felt", "feel", "made", "make", "like", "into", "still", "much", "many", "your", "ours", "ourselves", "myself", "itself", "herself", "himself", "through", "during", "about", "between", "within", "without", "across", "around", "against", "again", "ever", "never", "always", "often", "sometimes",
        "you", "youre", "youve", "youll", "you'd", "you", "yours", "our", "ours", "we", "us", "i", "me", "my", "mine", "he", "she", "his", "her", "hers", "its", "it", "an", "a", "to", "of", "in", "on", "at", "by", "as", "or", "if", "but", "so", "do", "did", "does", "done", "doing", "can", "cant", "cannot", "couldnt", "shouldnt", "wouldnt", "dont", "didnt", "doesnt", "isnt", "arent", "wasnt", "werent", "hadnt", "hasnt", "havent", "wont", "im", "ive", "ill", "id", "thats", "there's", "theres", "can't", "couldn't", "didn't", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "won't", "i'm", "i've", "i'll", "i'd",
    }
    raw_tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower().replace("’", "'"))
    tokens: list[str] = []
    for token in raw_tokens:
        normalized = token.strip("'")
        compact = normalized.replace("'", "")
        if len(compact) < 3:
            continue
        if normalized in stopwords or compact in stopwords:
            continue
        tokens.append(compact)
    return tokens


def _top_ngrams(texts: list[str], n: int, top_k: int = 15) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = _tokenize_terms(text)
        if len(tokens) < n:
            continue
        for index in range(0, len(tokens) - n + 1):
            gram = " ".join(tokens[index : index + n])
            counter[gram] += 1
    return [{"ngram": gram, "count": int(count)} for gram, count in counter.most_common(top_k)]


def _serialize_contradiction_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if df.empty:
        return out
    for _, row in df.iterrows():
        out.append(
            {
                "comparison": str(row["comparison"]),
                "source_kind": str(row["source_kind"]),
                "artifact_id": (str(row["artifact_id"]) if pd.notna(row["artifact_id"]) else None),
                "participant_code": str(row["participant_code"]),
                "questionnaire_label": str(row["questionnaire_label"]),
                "candidate_label": str(row["candidate_label"]),
                "direction": str(row.get("direction", "")) if pd.notna(row.get("direction", "")) else "",
                "evidence_text": str(row["evidence_text"]) if pd.notna(row["evidence_text"]) else "",
            }
        )
    return out


def _contradiction_direction(questionnaire_label: object, candidate_label: object) -> str:
    questionnaire = str(questionnaire_label).strip().lower()
    candidate = str(candidate_label).strip().lower()
    if questionnaire == "positive" and candidate == "negative":
        return "q_positive_llm_negative"
    if questionnaire == "negative" and candidate == "positive":
        return "q_negative_llm_positive"
    return ""


def _build_questionnaire_contradiction_analysis(
    study: StudyConfig,
    metrics_df: pd.DataFrame,
    questionnaire_df: pd.DataFrame,
    vader_df: pd.DataFrame,
    accepted_llm_artifacts: list[dict[str, Any]],
    *,
    top_n: int = 3,
) -> dict[str, Any]:
    tone_column = study.sections["experience"].tone_internal_column
    questionnaire_tones = questionnaire_df[["participant_code", tone_column]].rename(columns={tone_column: "questionnaire_label"})

    llm_metric_rows = metrics_df[
        metrics_df["comparison"].astype(str).str.startswith("questionnaire_vs_llm:") & (metrics_df["field"] == tone_column)
    ].copy()
    llm_metric_rows = _select_top_by_pareto(llm_metric_rows, top_n=top_n, metric_columns=("macro_f1", "cohen_kappa"))
    selected_comparisons = llm_metric_rows["comparison"].tolist()
    selected_artifact_ids = [comparison.split(":", 1)[1] for comparison in selected_comparisons if ":" in comparison]

    selected_map = {
        str(artifact["artifact_id"]): artifact
        for artifact in accepted_llm_artifacts
        if str(artifact.get("artifact_id")) in set(selected_artifact_ids)
    }

    contradiction_rows: list[dict[str, Any]] = []
    overview_rows: list[dict[str, Any]] = []

    # VADER baseline (quantitative only)
    vader_candidate = vader_df[["participant_code", tone_column]].rename(columns={tone_column: "candidate_label"})
    vader_merged = questionnaire_tones.merge(vader_candidate, on="participant_code", how="inner")
    vader_merged = vader_merged[vader_merged["questionnaire_label"].isin(["positive", "negative"])]
    vader_contradictions = vader_merged[
        vader_merged.apply(lambda row: _strict_polarity_contradiction(row["questionnaire_label"], row["candidate_label"]), axis=1)
    ]
    overview_rows.append(
        {
            "comparison": "questionnaire_vs_vader",
            "source_kind": "vader",
            "artifact_id": None,
            "n_total": int(len(vader_merged)),
            "n_contradictions": int(len(vader_contradictions)),
            "contradiction_rate": float(len(vader_contradictions) / len(vader_merged)) if len(vader_merged) else 0.0,
            "evidence_available": False,
        }
    )

    # Selected top LLMs (quantitative + qualitative)
    for artifact_id in selected_artifact_ids:
        artifact = selected_map.get(artifact_id)
        if artifact is None:
            continue
        comparison_name = f"questionnaire_vs_llm:{artifact_id}"
        candidate_df = artifact["data"][["participant_code", tone_column]].rename(columns={tone_column: "candidate_label"})
        merged = questionnaire_tones.merge(candidate_df, on="participant_code", how="inner")
        merged = merged[merged["questionnaire_label"].isin(["positive", "negative"])]
        contradictions = merged[
            merged.apply(lambda row: _strict_polarity_contradiction(row["questionnaire_label"], row["candidate_label"]), axis=1)
        ].copy()
        contradictions["direction"] = contradictions.apply(
            lambda row: _contradiction_direction(row["questionnaire_label"], row["candidate_label"]), axis=1
        )

        evidence_df = artifact.get("evidence_data", pd.DataFrame())
        if isinstance(evidence_df, pd.DataFrame) and not evidence_df.empty:
            experience_evidence = evidence_df[evidence_df["section"] == "experience"][["participant_code", "evidence_segments"]].copy()
            experience_evidence["evidence_text"] = experience_evidence["evidence_segments"].apply(
                lambda values: " | ".join(str(value).strip() for value in values if str(value).strip())
            )
            contradictions = contradictions.merge(
                experience_evidence[["participant_code", "evidence_text"]],
                on="participant_code",
                how="left",
            )
        else:
            contradictions["evidence_text"] = ""

        overview_rows.append(
            {
                "comparison": comparison_name,
                "source_kind": "llm",
                "artifact_id": artifact_id,
                "n_total": int(len(merged)),
                "n_contradictions": int(len(contradictions)),
                "contradiction_rate": float(len(contradictions) / len(merged)) if len(merged) else 0.0,
                "evidence_available": True,
            }
        )

        for _, row in contradictions.iterrows():
            contradiction_rows.append(
                {
                    "comparison": comparison_name,
                    "source_kind": "llm",
                    "artifact_id": artifact_id,
                    "participant_code": str(row["participant_code"]),
                    "questionnaire_label": str(row["questionnaire_label"]),
                    "candidate_label": str(row["candidate_label"]),
                    "direction": str(row.get("direction") or ""),
                    "evidence_text": str(row.get("evidence_text") or ""),
                }
            )

    contradiction_df = pd.DataFrame(contradiction_rows)
    overview_df = pd.DataFrame(overview_rows)

    per_comparison_ngrams: list[dict[str, Any]] = []
    for comparison in selected_comparisons:
        subset = contradiction_df[contradiction_df["comparison"] == comparison] if not contradiction_df.empty else pd.DataFrame()
        texts = [
            str(value)
            for value in subset.get("evidence_text", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        ]
        subset_qpos_lneg = subset[subset.get("direction", "") == "q_positive_llm_negative"] if not subset.empty else pd.DataFrame()
        subset_qneg_lpos = subset[subset.get("direction", "") == "q_negative_llm_positive"] if not subset.empty else pd.DataFrame()
        texts_qpos_lneg = [
            str(value)
            for value in subset_qpos_lneg.get("evidence_text", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        ]
        texts_qneg_lpos = [
            str(value)
            for value in subset_qneg_lpos.get("evidence_text", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        ]
        per_comparison_ngrams.append(
            {
                "comparison": comparison,
                "artifact_id": comparison.split(":", 1)[1] if ":" in comparison else None,
                "unigrams": _top_ngrams(texts, 1),
                "bigrams": _top_ngrams(texts, 2),
                "trigrams": _top_ngrams(texts, 3),
                "by_direction": {
                    "q_positive_llm_negative": {
                        "unigrams": _top_ngrams(texts_qpos_lneg, 1),
                        "bigrams": _top_ngrams(texts_qpos_lneg, 2),
                        "trigrams": _top_ngrams(texts_qpos_lneg, 3),
                    },
                    "q_negative_llm_positive": {
                        "unigrams": _top_ngrams(texts_qneg_lpos, 1),
                        "bigrams": _top_ngrams(texts_qneg_lpos, 2),
                        "trigrams": _top_ngrams(texts_qneg_lpos, 3),
                    },
                },
            }
        )

    all_texts = [
        str(value)
        for value in contradiction_df.get("evidence_text", pd.Series(dtype=str)).tolist()
        if str(value).strip()
    ]
    wordcloud_terms = _top_ngrams(all_texts, 1, top_k=80)
    qpos_lneg_all = contradiction_df[contradiction_df.get("direction", "") == "q_positive_llm_negative"] if not contradiction_df.empty else pd.DataFrame()
    qneg_lpos_all = contradiction_df[contradiction_df.get("direction", "") == "q_negative_llm_positive"] if not contradiction_df.empty else pd.DataFrame()
    wordcloud_qpos_lneg = _top_ngrams(
        [str(value) for value in qpos_lneg_all.get("evidence_text", pd.Series(dtype=str)).tolist() if str(value).strip()],
        1,
        top_k=60,
    )
    wordcloud_qneg_lpos = _top_ngrams(
        [str(value) for value in qneg_lpos_all.get("evidence_text", pd.Series(dtype=str)).tolist() if str(value).strip()],
        1,
        top_k=60,
    )

    examples_df = contradiction_df.copy()
    if not examples_df.empty:
        examples_df["evidence_len"] = examples_df["evidence_text"].astype(str).str.len()
        examples_df = examples_df.sort_values(["comparison", "direction", "evidence_len"], ascending=[True, True, False])
        examples_df = examples_df.groupby(["comparison", "direction"], as_index=False).head(3)
        examples_df = examples_df.drop(columns=["evidence_len"])

    return {
        "enabled": bool(selected_artifact_ids),
        "definition": "Strict contradiction = questionnaire positive vs automated negative, or questionnaire negative vs automated positive, on experience_tone.",
        "experience_tone_field": tone_column,
        "top_n_selected": int(top_n),
        "selected_llm_comparisons": selected_comparisons,
        "selected_llm_artifact_ids": selected_artifact_ids,
        "overview": overview_df.to_dict(orient="records"),
        "rows": _serialize_contradiction_rows(contradiction_df),
        "examples": _serialize_contradiction_rows(examples_df),
        "ngrams": {
            "wordcloud_terms": [{"term": row["ngram"], "count": int(row["count"])} for row in wordcloud_terms],
            "wordcloud_terms_by_direction": {
                "q_positive_llm_negative": [{"term": row["ngram"], "count": int(row["count"])} for row in wordcloud_qpos_lneg],
                "q_negative_llm_positive": [{"term": row["ngram"], "count": int(row["count"])} for row in wordcloud_qneg_lpos],
            },
            "per_comparison": [
                {
                    "comparison": row["comparison"],
                    "artifact_id": row["artifact_id"],
                    "unigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["unigrams"]],
                    "bigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["bigrams"]],
                    "trigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["trigrams"]],
                    "by_direction": {
                        "q_positive_llm_negative": {
                            "unigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["by_direction"]["q_positive_llm_negative"]["unigrams"]],
                            "bigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["by_direction"]["q_positive_llm_negative"]["bigrams"]],
                            "trigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["by_direction"]["q_positive_llm_negative"]["trigrams"]],
                        },
                        "q_negative_llm_positive": {
                            "unigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["by_direction"]["q_negative_llm_positive"]["unigrams"]],
                            "bigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["by_direction"]["q_negative_llm_positive"]["bigrams"]],
                            "trigrams": [{"term": item["ngram"], "count": int(item["count"])} for item in row["by_direction"]["q_negative_llm_positive"]["trigrams"]],
                        },
                    },
                }
                for row in per_comparison_ngrams
            ],
        },
        "notes": [
            "VADER is included only in quantitative contradiction rates because it does not provide extractive evidence spans.",
            "Qualitative evidence analysis is limited to the top questionnaire-vs-LLM comparisons on experience_tone, ranked by macro F1 and Cohen kappa.",
        ],
    }


def _tone_label_diagnostics(
    questionnaire_tones: pd.DataFrame,
    candidate_tones: pd.DataFrame,
    comparison: str,
    labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged = questionnaire_tones.merge(candidate_tones, on="participant_code", how="inner")
    valid_mask = ~merged["questionnaire_label"].apply(_is_blank) & ~merged["candidate_label"].apply(_is_blank)
    merged = merged.loc[valid_mask].copy()
    if merged.empty:
        return [], []

    y_true = merged["questionnaire_label"].astype(str)
    y_pred = merged["candidate_label"].astype(str)
    matrix = pd.crosstab(
        pd.Categorical(y_true, categories=labels),
        pd.Categorical(y_pred, categories=labels),
        dropna=False,
    )

    per_label_rows: list[dict[str, Any]] = []
    for label in labels:
        tp = int(matrix.loc[label, label]) if label in matrix.index and label in matrix.columns else 0
        fp = int(matrix[label].sum() - tp) if label in matrix.columns else 0
        fn = int(matrix.loc[label].sum() - tp) if label in matrix.index else 0
        support = int(matrix.loc[label].sum()) if label in matrix.index else 0
        predicted_n = int(matrix[label].sum()) if label in matrix.columns else 0
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall) if pd.notna(precision) and pd.notna(recall) else float("nan")
        per_label_rows.append(
            {
                "comparison": comparison,
                "label": label,
                "support_n": support,
                "predicted_n": predicted_n,
                "precision": float(precision) if pd.notna(precision) else float("nan"),
                "recall": float(recall) if pd.notna(recall) else float("nan"),
                "f1": float(f1) if pd.notna(f1) else float("nan"),
                "reference_prevalence": _safe_divide(support, len(merged)),
                "candidate_prevalence": _safe_divide(predicted_n, len(merged)),
            }
        )

    confusion_rows: list[dict[str, Any]] = []
    for true_label in labels:
        row_total = int(matrix.loc[true_label].sum()) if true_label in matrix.index else 0
        for predicted_label in labels:
            count = int(matrix.loc[true_label, predicted_label]) if true_label in matrix.index and predicted_label in matrix.columns else 0
            confusion_rows.append(
                {
                    "comparison": comparison,
                    "questionnaire_label": true_label,
                    "candidate_label": predicted_label,
                    "count": count,
                    "row_rate": _safe_divide(count, row_total),
                }
            )
    return per_label_rows, confusion_rows


def _build_questionnaire_tone_label_analysis(
    study: StudyConfig,
    metrics_df: pd.DataFrame,
    questionnaire_df: pd.DataFrame,
    vader_df: pd.DataFrame,
    accepted_llm_artifacts: list[dict[str, Any]],
    *,
    top_n: int = 3,
) -> dict[str, Any]:
    tone_column = study.sections["experience"].tone_internal_column
    labels = list(study.allowed_labels_for_column(tone_column))
    questionnaire_tones = questionnaire_df[["participant_code", tone_column]].rename(columns={tone_column: "questionnaire_label"})

    llm_metric_rows = metrics_df[
        metrics_df["comparison"].astype(str).str.startswith("questionnaire_vs_llm:") & (metrics_df["field"] == tone_column)
    ].copy()
    llm_metric_rows = _select_top_by_pareto(llm_metric_rows, top_n=top_n, metric_columns=("macro_f1", "cohen_kappa"))
    selected_llm_comparisons = llm_metric_rows["comparison"].tolist()
    selected_artifact_ids = [comparison.split(":", 1)[1] for comparison in selected_llm_comparisons if ":" in comparison]
    selected_comparisons = ["questionnaire_vs_vader", *selected_llm_comparisons]

    selected_map = {
        str(artifact["artifact_id"]): artifact
        for artifact in accepted_llm_artifacts
        if str(artifact.get("artifact_id")) in set(selected_artifact_ids)
    }

    per_label_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    vader_candidate = vader_df[["participant_code", tone_column]].rename(columns={tone_column: "candidate_label"})
    per_label, confusion = _tone_label_diagnostics(questionnaire_tones, vader_candidate, "questionnaire_vs_vader", labels)
    per_label_rows.extend(per_label)
    confusion_rows.extend(confusion)

    for artifact_id in selected_artifact_ids:
        artifact = selected_map.get(artifact_id)
        if artifact is None:
            continue
        comparison_name = f"questionnaire_vs_llm:{artifact_id}"
        candidate_df = artifact["data"][["participant_code", tone_column]].rename(columns={tone_column: "candidate_label"})
        per_label, confusion = _tone_label_diagnostics(questionnaire_tones, candidate_df, comparison_name, labels)
        per_label_rows.extend(per_label)
        confusion_rows.extend(confusion)

    return {
        "enabled": bool(per_label_rows),
        "experience_tone_field": tone_column,
        "labels": labels,
        "top_n_selected": int(top_n),
        "selected_comparisons": selected_comparisons,
        "selected_llm_comparisons": selected_llm_comparisons,
        "selected_llm_artifact_ids": selected_artifact_ids,
        "per_label": per_label_rows,
        "confusion": confusion_rows,
    }


def _map_questionnaire_value(
    value: object,
    yes_values: list[str],
    no_values: list[str],
    na_values: list[str],
    source_column: str,
) -> str | Any:
    def _normalize_token(raw: object) -> str:
        return " ".join(str(raw).strip().lower().split())

    if _is_blank(value):
        return pd.NA

    normalized = _normalize_token(value)
    if normalized == "missing":
        return pd.NA

    yes_tokens = {_normalize_token(item) for item in yes_values}
    no_tokens = {_normalize_token(item) for item in no_values}
    na_tokens = {_normalize_token(item) for item in na_values}

    if normalized in na_tokens:
        return pd.NA
    if normalized in yes_tokens:
        return "yes"
    if normalized in no_tokens:
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

    for block in study.questionnaire.values():
        for internal_column, source_column in block["columns"].items():
            if source_column not in df.columns:
                raise ValueError(f"Sampled private workbook is missing questionnaire column: {source_column}")
            out[internal_column] = df[source_column].apply(
                lambda value: _map_questionnaire_value(
                    value,
                    block["yes_values"],
                    block["no_values"],
                    list(block.get("na_values", [])),
                    source_column,
                )
            )

    _validate_labels(out, [experience_tone_column], study, "Questionnaire tone labels")
    _validate_labels(
        out,
        study.binary_columns(),
        study,
        "Questionnaire labels",
        allow_blank_columns=set(study.binary_columns()),
    )
    return out


def _complete_case_sources(
    left: dict[str, Any],
    right: dict[str, Any],
    shared_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = left["df"].merge(
        right["df"],
        on="participant_code",
        how="inner",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return left["df"].iloc[0:0].copy(), right["df"].iloc[0:0].copy()

    complete_mask = pd.Series(True, index=merged.index)
    for column in shared_columns:
        complete_mask = complete_mask & ~merged[f"{column}_left"].apply(_is_blank)
        complete_mask = complete_mask & ~merged[f"{column}_right"].apply(_is_blank)

    complete_codes = set(merged.loc[complete_mask, "participant_code"].astype(str))
    if not complete_codes:
        return left["df"].iloc[0:0].copy(), right["df"].iloc[0:0].copy()

    left_cc = left["df"][left["df"]["participant_code"].astype(str).isin(complete_codes)].copy()
    right_cc = right["df"][right["df"]["participant_code"].astype(str).isin(complete_codes)].copy()
    return left_cc, right_cc


def load_vader_predictions(
    study: StudyConfig,
    paths: PathsConfig,
    sampled_private_workbook: Path,
    participant_codes: set[str] | None = None,
    vader_scores_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    survey_df = load_batch_source(
        study=study,
        paths=paths,
        source="survey",
        all_records=False,
        min_valid_sections=3,
    )
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


def f1_by_label(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        metrics[f"f1_label_{_normalize_identifier(str(label))}"] = f1
    return metrics


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

        tone_label_f1_metrics = {
            f"f1_label_{_normalize_identifier(str(label))}": float("nan")
            for label in labels
        } if column == "experience_tone" else {}

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
            if column == "experience_tone":
                tone_label_f1_metrics = f1_by_label(y_true, y_pred, labels)

        row: dict[str, Any] = {
            "comparison": comparison_name,
            "field": column,
            "n": int(len(y_true)),
            "accuracy": accuracy,
            "cohen_kappa": kappa,
            "macro_f1": macro_f1,
            **extra_metrics,
            **tone_label_f1_metrics,
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
            "accuracy_std": float(group["accuracy"].std(ddof=1)),
            "cohen_kappa_mean": float(group["cohen_kappa"].mean()),
            "cohen_kappa_std": float(group["cohen_kappa"].std(ddof=1)),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std(ddof=1)),
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


def _compute_source_pair_metrics(
    left: dict[str, Any],
    right: dict[str, Any],
    study: StudyConfig,
    *,
    complete_case: bool = False,
) -> pd.DataFrame | None:
    shared_columns = [column for column in left["columns"] if column in right["columns"]]
    if not shared_columns:
        return None
    left_df = left["df"]
    right_df = right["df"]
    if complete_case:
        left_df, right_df = _complete_case_sources(left, right, shared_columns)
    return compute_comparison_metrics(
        left_df,
        right_df,
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
    prompt_variants: list[str] | None = None,
    sampled_private_workbook: Path | None = None,
    output_dir: Path | None = None,
    vader_scores_path: Path | None = None,
    skip_vader: bool = False,
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
        allow_empty=human_annotation_workbook is None,
    )
    human_enabled = bool(human_artifacts)
    if human_enabled:
        reference_df, adjudication_summary, human_long_df = consolidate_majority_human_reference(
            human_artifacts, study
        )
    else:
        reference_df = pd.DataFrame(
            columns=["participant_code", *study.annotation_internal_columns()]
        )
        adjudication_summary = {
            "n_valid_annotators": 0,
            "n_reference_participants": 0,
            "n_fields": int(len(study.annotation_internal_columns())),
            "n_unresolved_field_participant_pairs": 0,
        }
        human_long_df = pd.DataFrame(
            columns=["participant_code", "annotator_id", *study.annotation_internal_columns()]
        )
    questionnaire_df = load_questionnaire_labels(
        paths.survey_csv,
        study,
        participant_codes=None,
    )
    if skip_vader:
        vader_df = pd.DataFrame(columns=["participant_code", *study.tone_columns()])
        vader_summary = None
    else:
        vader_df, vader_summary = load_vader_predictions(
            study,
            paths,
            resolved_sampled_private,
            participant_codes=None,
            vader_scores_path=Path(vader_scores_path) if vader_scores_path else None,
            output_dir=evaluation_dir,
        )

    questionnaire_source = _comparison_source("questionnaire", questionnaire_df, study)
    vader_source = _comparison_source("vader", vader_df, study) if not skip_vader else None

    questionnaire_tone_columns = [study.sections["experience"].tone_internal_column]
    metrics_frames: list[pd.DataFrame] = []
    complete_case_frames: list[pd.DataFrame | None] = []
    reference_source = _comparison_source("human_reference", reference_df, study)
    if human_enabled:
        metrics_frames.extend(
            [
                compute_comparison_metrics(
                    reference_df,
                    questionnaire_df,
                    questionnaire_tone_columns,
                    study,
                    "human_reference_vs_questionnaire",
                ),
                compute_comparison_metrics(
                    reference_df,
                    questionnaire_df,
                    study.binary_columns(),
                    study,
                    "human_reference_vs_questionnaire",
                ),
            ]
        )
        complete_case_frames.append(
            _compute_source_pair_metrics(
                reference_source,
                questionnaire_source,
                study,
                complete_case=True,
            )
        )
        if not skip_vader:
            metrics_frames.append(
                compute_comparison_metrics(
                    reference_df,
                    vader_df,
                    study.tone_columns(),
                    study,
                    "human_reference_vs_vader",
                )
            )
            if vader_source is not None:
                complete_case_frames.append(
                    _compute_source_pair_metrics(
                        reference_source,
                        vader_source,
                        study,
                        complete_case=True,
                    )
                )

    llm_candidates, rejected_llm = discover_llm_prediction_artifacts(
        study,
        paths,
        llm_predictions_path=llm_predictions_path,
        llm_results_dir=llm_results_dir,
        experiment_ids=experiment_ids,
        prompt_variants=prompt_variants,
    )
    preprocessed_three_section_codes: set[str] = set()
    for artifact in llm_candidates:
        llm_df = artifact.get("data")
        if isinstance(llm_df, pd.DataFrame) and "participant_code" in llm_df.columns:
            codes = llm_df["participant_code"].dropna().astype(str).str.strip()
            preprocessed_three_section_codes.update(code for code in codes.tolist() if code)
    accepted_llm: list[dict[str, Any]] = []
    accepted_llm_analysis: list[dict[str, Any]] = []
    accepted_llm_sources: list[dict[str, Any]] = []
    experiment_output_dir = evaluation_dir / "experiments"
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    for artifact in llm_candidates:
        llm_df = artifact["data"]
        metadata = {
            "experiment_id": artifact["experiment_id"],
            "artifact_id": artifact["artifact_id"],
            "prompt_variant": artifact.get("prompt_variant"),
            "run_id": artifact.get("run_id"),
            "model_variant": artifact.get("model_variant"),
        }
        if human_enabled:
            experiment_metrics = compute_comparison_metrics(
                reference_df,
                llm_df,
                study.annotation_internal_columns(),
                study,
                comparison_name=f"human_reference_vs_llm:{artifact['artifact_id']}",
                metadata=metadata,
            )
            rejection_reason = "No overlap with adjudicated human reference."
        else:
            llm_source_for_precheck = _comparison_source(
                "llm",
                llm_df,
                study,
                artifact_id=str(artifact["artifact_id"]),
                metadata=metadata,
            )
            experiment_metrics = _compute_source_pair_metrics(
                questionnaire_source,
                llm_source_for_precheck,
                study,
            )
            if experiment_metrics is None:
                rejected_llm.append(
                    {
                        **{k: v for k, v in artifact.items() if k != "data"},
                        "reason": "No shared evaluation columns with questionnaire labels.",
                    }
                )
                continue
            rejection_reason = "No overlap with questionnaire labels."
        if int(experiment_metrics["n"].fillna(0).sum()) == 0:
            rejected_llm.append(
                {
                    **{k: v for k, v in artifact.items() if k != "data"},
                    "reason": rejection_reason,
                }
            )
            continue
        accepted_llm.append({k: v for k, v in artifact.items() if k not in {"data", "evidence_data"}})
        accepted_llm_analysis.append(artifact)
        accepted_llm_sources.append(
            _comparison_source(
                "llm",
                llm_df,
                study,
                artifact_id=str(artifact["artifact_id"]),
                metadata=metadata,
            )
        )
        if human_enabled:
            metrics_frames.append(experiment_metrics)
            complete_case_frames.append(
                _compute_source_pair_metrics(
                    reference_source,
                    accepted_llm_sources[-1],
                    study,
                    complete_case=True,
                )
            )

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

    additional_sources = [questionnaire_source, *accepted_llm_sources]
    if vader_source is not None:
        additional_sources.append(vader_source)
    for left_source, right_source in combinations(additional_sources, 2):
        pair_metrics = _compute_source_pair_metrics(left_source, right_source, study)
        if pair_metrics is not None:
            metrics_frames.append(pair_metrics)
        pair_complete_case = _compute_source_pair_metrics(
            left_source,
            right_source,
            study,
            complete_case=True,
        )
        if pair_complete_case is not None:
            complete_case_frames.append(pair_complete_case)

    if not metrics_frames:
        raise ValueError(
            "No overlap-based comparisons were available to evaluate. "
            "Provide at least one compatible source pair (for example questionnaire + LLM, "
            "or questionnaire + VADER)."
        )

    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    complete_case_frames = [frame for frame in complete_case_frames if frame is not None]
    metrics_complete_case_df = (
        pd.concat(complete_case_frames, ignore_index=True)
        if complete_case_frames
        else pd.DataFrame(columns=metrics_df.columns)
    )
    if human_enabled:
        human_pairwise_df, human_summary_df = compute_human_agreement_metrics(
            human_long_df, study
        )
    else:
        human_pairwise_df = pd.DataFrame(
            columns=[
                "comparison",
                "field",
                "n",
                "accuracy",
                "cohen_kappa",
                "macro_f1",
                "annotator_a",
                "annotator_b",
            ]
        )
        human_summary_df = pd.DataFrame(
            columns=[
                "field",
                "n_mean",
                "accuracy_mean",
                "cohen_kappa_mean",
                "macro_f1_mean",
                "n_pairs",
            ]
        )

    sampled_private_missing = False
    try:
        sampled_private_df = load_sampled_private_data(resolved_sampled_private)
    except FileNotFoundError:
        sampled_private_missing = True
        sampled_private_df = pd.DataFrame(columns=["participant_code"])
        if not skip_vader:
            raise

    sampled_total = int(len(sampled_private_df))
    if sampled_total == 0:
        sampled_total = int(questionnaire_df["participant_code"].nunique())

    coverage = {
        "n_sampled_total": sampled_total,
        "n_preprocessed_three_sections_total": int(len(preprocessed_three_section_codes)),
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
        "comparisons_complete_case": _comparison_summary(metrics_complete_case_df),
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
        "vader": {
            "enabled": not skip_vader,
        },
        "inputs": {
            "sampled_private_workbook": {
                "path": str(resolved_sampled_private),
                "exists": not sampled_private_missing,
            }
        },
        "human_evaluation": {
            "enabled": human_enabled,
            "reason": (
                None
                if human_enabled
                else "No human annotation artifacts were discovered; generated questionnaire-vs-automated outputs only."
            ),
        },
        "sensitivity": {
            "complete_case": {
                "enabled": True,
                "n_metric_rows": int(len(metrics_complete_case_df)),
            }
        },
    }

    tone_label_columns = sorted(column for column in metrics_df.columns if column.startswith("f1_label_"))
    tone_label_rows = metrics_df.loc[metrics_df["field"] == "experience_tone", ["comparison", "field", "n", *tone_label_columns]].copy() if tone_label_columns else pd.DataFrame()
    summary["experience_tone_label_f1"] = {
        "labels": tone_label_columns,
        "rows": tone_label_rows.to_dict(orient="records") if not tone_label_rows.empty else [],
    }

    contradiction_analysis = _build_questionnaire_contradiction_analysis(
        study,
        metrics_df,
        questionnaire_df,
        vader_df,
        accepted_llm_analysis,
    )
    summary["questionnaire_contradictions"] = contradiction_analysis
    summary["questionnaire_tone_label_analysis"] = _build_questionnaire_tone_label_analysis(
        study,
        metrics_df,
        questionnaire_df,
        vader_df,
        accepted_llm_analysis,
    )

    metrics_path = evaluation_dir / "evaluation_metrics.csv"
    metrics_complete_case_path = evaluation_dir / "evaluation_metrics_complete_case.csv"
    summary_path = evaluation_dir / "evaluation_summary.json"
    reference_path = evaluation_dir / "human_reference_majority.csv"
    adjudication_path = evaluation_dir / "adjudication_summary.csv"
    human_pairwise_path = evaluation_dir / "human_agreement_pairwise.csv"
    human_summary_path = evaluation_dir / "human_agreement_summary.csv"
    human_manifest_path = evaluation_dir / "human_artifacts_manifest.json"
    llm_manifest_path = evaluation_dir / "llm_artifacts_manifest.json"
    contradiction_rows_path = evaluation_dir / "questionnaire_contradictions.csv"
    contradiction_examples_path = evaluation_dir / "questionnaire_contradiction_examples.csv"

    metrics_df.to_csv(metrics_path, index=False)
    metrics_complete_case_df.to_csv(metrics_complete_case_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if human_enabled:
        reference_df.to_csv(reference_path, index=False)
        pd.DataFrame(summary["adjudication"]["fields"]).to_csv(adjudication_path, index=False)
        human_pairwise_df.to_csv(human_pairwise_path, index=False)
        human_summary_df.to_csv(human_summary_path, index=False)
        _write_json(human_manifest_path, summary["human_artifacts"])
    _write_json(llm_manifest_path, summary["llm_artifacts"])
    pd.DataFrame(summary["questionnaire_contradictions"].get("rows", [])).to_csv(contradiction_rows_path, index=False)
    pd.DataFrame(summary["questionnaire_contradictions"].get("examples", [])).to_csv(contradiction_examples_path, index=False)

    reporting_outputs = write_alignment_outputs(
        study=study,
        metrics_df=metrics_df,
        summary=summary,
        output_dir=evaluation_dir,
        vader_summary=vader_summary,
        figure_dpi=figure_dpi,
        export_figures_pdf=export_figures_pdf,
    )

    written_files: dict[str, str] = {
        "metrics_file": str(metrics_path),
        "metrics_complete_case_file": str(metrics_complete_case_path),
        "summary_file": str(summary_path),
        "llm_artifacts_manifest_file": str(llm_manifest_path),
        "questionnaire_contradictions_file": str(contradiction_rows_path),
        "questionnaire_contradiction_examples_file": str(contradiction_examples_path),
        **reporting_outputs,
    }
    if human_enabled:
        written_files.update(
            {
                "human_reference_file": str(reference_path),
                "adjudication_summary_file": str(adjudication_path),
                "human_agreement_pairwise_file": str(human_pairwise_path),
                "human_agreement_summary_file": str(human_summary_path),
                "human_artifacts_manifest_file": str(human_manifest_path),
            }
        )

    return metrics_df, summary, written_files
