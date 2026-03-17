from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .config import StudyConfig


def is_meaningful_text(value: object) -> bool:
    if pd.isna(value):
        return False
    normalized = str(value).strip()
    if not normalized:
        return False
    return normalized.lower() not in {"n/a", "na", "none", "nothing", "see above", "-", "--"}


def apply_dataset_row_filters(
    df: pd.DataFrame,
    study: StudyConfig,
    *,
    apply_quality_filter: bool = True,
    quality_values: list[str] | None = None,
    apply_to_drop_filter: bool = True,
    drop_missing_strata: bool | None = None,
) -> pd.DataFrame:
    out = df.copy()

    quality_column = study.dataset.get("quality_label_column")
    effective_quality_values = quality_values if quality_values is not None else study.dataset.get("quality_values_to_use")
    if apply_quality_filter and quality_column and effective_quality_values:
        out = out[out[str(quality_column)].isin(effective_quality_values)].copy()

    to_drop_column = study.dataset.get("to_drop_column")
    to_drop_value = study.dataset.get("to_drop_exclude_value")
    if apply_to_drop_filter and to_drop_column and to_drop_column in out.columns:
        out = out[out[str(to_drop_column)] != to_drop_value].copy()

    effective_drop_missing_strata = (
        bool(study.dataset.get("drop_missing_strata", True))
        if drop_missing_strata is None
        else drop_missing_strata
    )
    if effective_drop_missing_strata:
        stratify_column = study.stratify_column
        missing_label = study.dataset.get("missing_label")
        out = out[out[stratify_column].notna()].copy()
        if missing_label is not None:
            out = out[out[stratify_column] != missing_label].copy()

    return out


def filter_source_data(df: pd.DataFrame, study: StudyConfig) -> pd.DataFrame:
    out = apply_dataset_row_filters(df, study)

    for source_column in study.text_columns().values():
        out[f"__valid_{source_column}"] = out[source_column].apply(is_meaningful_text)

    valid_columns = [f"__valid_{source_column}" for source_column in study.text_columns().values()]
    require_all = bool(study.dataset.get("require_all_texts", True))
    if require_all:
        out = out[out[valid_columns].all(axis=1)].copy()
    else:
        out = out[out[valid_columns].any(axis=1)].copy()

    helper_columns = [column for column in out.columns if column.startswith("__valid_")]
    if helper_columns:
        out = out.drop(columns=helper_columns)
    return out


def proportional_stratified_sample(
    df: pd.DataFrame,
    stratify_column: str,
    n_total: int,
    random_state: int,
) -> pd.DataFrame:
    counts = df[stratify_column].value_counts(dropna=False)
    if counts.empty:
        raise ValueError("No rows available after filtering.")

    n_total = min(n_total, len(df))
    proportions = counts / counts.sum()
    raw_allocation = proportions * n_total
    floor_allocation = np.floor(raw_allocation).astype(int)

    remainder = raw_allocation - floor_allocation
    allocated = int(floor_allocation.sum())
    remaining = n_total - allocated

    if remaining > 0:
        for label in remainder.sort_values(ascending=False).index.tolist()[:remaining]:
            floor_allocation.loc[label] += 1

    rng = np.random.default_rng(random_state)
    parts: list[pd.DataFrame] = []
    for label, per_label in floor_allocation.items():
        label_df = df[df[stratify_column] == label]
        per_label = min(int(per_label), len(label_df))
        chosen = rng.choice(label_df.index.to_numpy(), size=per_label, replace=False)
        parts.append(df.loc[chosen])

    sampled = pd.concat(parts, axis=0).copy()
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def make_participant_code(real_id: object, prefix: str, digits: int, salt: str) -> str:
    token = f"{salt}|{real_id}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest().upper()
    stable_width = max(int(digits), 6)
    return f"{prefix}_{digest[:stable_width]}"


def assign_participant_codes(df: pd.DataFrame, study: StudyConfig) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    prefix = str(study.sampling["aux_prefix"])
    digits = int(study.sampling["aux_id_digits"])
    salt = str(study.sampling["hash_salt"])
    out.insert(
        0,
        "participant_code",
        [
            make_participant_code(real_id=row[study.id_column], prefix=prefix, digits=digits, salt=salt)
            for _, row in out.iterrows()
        ],
    )
    return out


def build_visible_column_map(study: StudyConfig) -> pd.DataFrame:
    rows = [{"section": "", "internal_column": "participant_code", "visible_column": "Participant Code"}]

    for section_name in study.section_order:
        section = study.sections[section_name]
        rows.append(
            {
                "section": section_name,
                "internal_column": f"{section_name}_text",
                "visible_column": section.text_label,
            }
        )
        rows.append(
            {
                "section": section_name,
                "internal_column": section.tone_internal_column,
                "visible_column": section.tone_visible_label,
            }
        )
        for internal_column, visible_column in section.binary_labels.items():
            rows.append(
                {
                    "section": section_name,
                    "internal_column": internal_column,
                    "visible_column": visible_column,
                }
            )

    return pd.DataFrame(rows)


def build_annotation_dataframes(
    sampled_df: pd.DataFrame,
    study: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for _, row in sampled_df.iterrows():
        out_row: dict[str, Any] = {"Participant Code": row["participant_code"]}
        for section_name in study.section_order:
            section = study.sections[section_name]
            out_row[section.text_label] = row[section.source_column]
            out_row[section.tone_visible_label] = ""
            for visible_label in section.binary_labels.values():
                out_row[visible_label] = ""
        rows.append(out_row)
        mapping_rows.append(
            {
                "participant_code": row["participant_code"],
                study.id_column: row[study.id_column],
                "true_stratum": row.get(study.stratify_column, np.nan),
            }
        )

    annotation_df = pd.DataFrame(rows)
    mapping_df = pd.DataFrame(mapping_rows)
    column_map_df = build_visible_column_map(study)
    sampled_private_df = sampled_df.copy()
    return annotation_df, mapping_df, column_map_df, sampled_private_df


def create_annotation_frames(
    source_df: pd.DataFrame,
    study: StudyConfig,
    n_total: int | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    filtered = filter_source_data(source_df, study)
    sampled = proportional_stratified_sample(
        df=filtered,
        stratify_column=study.stratify_column,
        n_total=int(n_total or study.sampling["n_total"]),
        random_state=int(random_state or study.sampling["random_state"]),
    )
    sampled = assign_participant_codes(sampled, study)
    annotation_df, mapping_df, column_map_df, sampled_private_df = build_annotation_dataframes(sampled, study)

    summary = {
        "n_input": int(len(source_df)),
        "n_filtered": int(len(filtered)),
        "n_sampled": int(len(sampled_private_df)),
        "strata_distribution_filtered": filtered[study.stratify_column].value_counts().to_dict(),
        "strata_distribution_sampled": sampled_private_df[study.stratify_column].value_counts().to_dict(),
    }
    return annotation_df, mapping_df, column_map_df, sampled_private_df, summary
