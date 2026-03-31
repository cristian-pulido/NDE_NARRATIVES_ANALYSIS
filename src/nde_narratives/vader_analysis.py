from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import PathsConfig, StudyConfig
from .io_utils import read_tabular_file
from .prompting import load_batch_source, resolve_survey_source_path
from .sampling import apply_dataset_row_filters, is_meaningful_text


VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05
VADER_SUBDIR = "vader_sentiment"
DEFAULT_SCORES_FILENAME = "vader_sentiment_scores.csv"
DEFAULT_REPORT_FILENAME = "vader_report.md"
DEFAULT_SUMMARY_FILENAME = "vader_summary.json"




def vader_score_columns(include_text: bool = False) -> list[str]:
    columns = [
        "participant_code",
        "section",
        "source_column",
        "neg",
        "neu",
        "pos",
        "compound",
        "vader_label",
    ]
    if include_text:
        columns.append("text")
    return columns

def derive_vader_label(compound: float) -> str:
    if compound >= VADER_POSITIVE_THRESHOLD:
        return "positive"
    if compound <= VADER_NEGATIVE_THRESHOLD:
        return "negative"
    return "mixed"


def default_vader_output_dir(paths: PathsConfig) -> Path:
    return paths.evaluation_output_dir / VADER_SUBDIR


def default_vader_scores_path(paths: PathsConfig) -> Path:
    return default_vader_output_dir(paths) / DEFAULT_SCORES_FILENAME


def _filter_description(study: StudyConfig, all_records: bool, quality_values: list[str] | None) -> str:
    if all_records:
        return "All source rows without configured quality/drop/strata filters"
    if quality_values is not None:
        return f"Rows filtered to quality values: {', '.join(quality_values)}"
    configured = study.dataset.get("quality_values_to_use") or []
    if configured:
        return f"Rows filtered to configured quality values: {', '.join(configured)}"
    return "Rows filtered with the study defaults"


def _prepare_source_rows(
    source_df: pd.DataFrame,
    study: StudyConfig,
    *,
    all_records: bool = False,
    quality_values: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if all_records:
        prepared = source_df.copy()
    else:
        prepared = apply_dataset_row_filters(source_df, study, quality_values=quality_values)
    if limit is not None:
        prepared = prepared.head(limit).copy()
    return prepared


def build_vader_scores(
    source_df: pd.DataFrame,
    study: StudyConfig,
    *,
    all_records: bool = False,
    quality_values: list[str] | None = None,
    limit: int | None = None,
    include_text: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    analyzer = SentimentIntensityAnalyzer()
    prepared = _prepare_source_rows(
        source_df,
        study,
        all_records=all_records,
        quality_values=quality_values,
        limit=limit,
    )

    rows: list[dict[str, Any]] = []
    for section_name in study.section_order:
        section = study.sections[section_name]
        if section.source_column not in prepared.columns:
            raise ValueError(f"Source data is missing text column: {section.source_column}")

        section_df = prepared.loc[prepared[section.source_column].apply(is_meaningful_text)].copy()
        for _, record in section_df.iterrows():
            scores = analyzer.polarity_scores(str(record[section.source_column]))
            row = {
                study.id_column: record[study.id_column],
                "participant_code": record.get("participant_code", ""),
                "section": section_name,
                "source_column": section.source_column,
                "neg": float(scores["neg"]),
                "neu": float(scores["neu"]),
                "pos": float(scores["pos"]),
                "compound": float(scores["compound"]),
                "vader_label": derive_vader_label(float(scores["compound"])),
            }
            if include_text:
                row["text"] = str(record[section.source_column])
            rows.append(row)

    scores_df = pd.DataFrame(rows, columns=[study.id_column] + vader_score_columns(include_text=include_text))
    summary = {
        "n_input": int(len(source_df)),
        "n_rows_after_filters": int(len(prepared)),
        "filter_description": _filter_description(study, all_records, quality_values),
        "sections": summarize_vader_scores(scores_df),
    }
    return scores_df, summary


def summarize_vader_scores(scores_df: pd.DataFrame) -> dict[str, Any]:
    if scores_df.empty:
        return {}

    section_summary: dict[str, Any] = {}
    for section_name, group in scores_df.groupby("section"):
        counts = {label: int(count) for label, count in group["vader_label"].value_counts().to_dict().items()}
        section_summary[section_name] = {
            "n": int(len(group)),
            "compound_mean": float(group["compound"].mean()),
            "compound_median": float(group["compound"].median()),
            "compound_std": float(group["compound"].std(ddof=0)),
            "compound_min": float(group["compound"].min()),
            "compound_max": float(group["compound"].max()),
            "label_counts": counts,
        }
    return section_summary


def _plot_section_distribution(section_name: str, group: pd.DataFrame, figure_path: Path) -> None:
    color_map = {"positive": "#2E8B57", "mixed": "#C47F17", "negative": "#B23A48"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ordered_labels = [label for label in ("positive", "mixed", "negative") if label in set(group["vader_label"])]
    hist_data = [group.loc[group["vader_label"] == label, "compound"] for label in ordered_labels]

    if hist_data:
        axes[0].hist(
            hist_data,
            bins=12,
            stacked=True,
            color=[color_map[label] for label in ordered_labels],
            label=ordered_labels,
            alpha=0.85,
        )
        axes[0].legend(frameon=False)
    axes[0].axvline(0.0, color="#2B2D42", linewidth=1.2, linestyle="--")
    axes[0].set_title(f"Compound distribution: {section_name}")
    axes[0].set_xlabel("VADER compound")
    axes[0].set_ylabel("Count")
    axes[0].set_facecolor("#F7F4EA")

    counts = group["vader_label"].value_counts().reindex(["positive", "mixed", "negative"], fill_value=0)
    axes[1].bar(counts.index, counts.values, color=[color_map[label] for label in counts.index], width=0.6)
    axes[1].set_title(f"Label counts (n={len(group)})")
    axes[1].set_ylabel("Rows")
    axes[1].set_facecolor("#F7F4EA")

    fig.patch.set_facecolor("#FFFDF8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def write_vader_figures(scores_df: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    figure_paths: dict[str, str] = {}
    for section_name, group in scores_df.groupby("section"):
        figure_path = figures_dir / f"{section_name}_distribution.png"
        _plot_section_distribution(section_name, group, figure_path)
        figure_paths[section_name] = str(figure_path)
    return figure_paths


def _report_lines(
    study: StudyConfig,
    summary: dict[str, Any],
    figure_paths: dict[str, str],
    output_dir: Path,
) -> list[str]:
    lines = [
        "# VADER Sensitivity Analysis",
        "",
        "## Objective",
        "",
        "Provide a first-layer sentiment sensitivity analysis across each narrative text column using VADER.",
        "",
        "## Methodology",
        "",
        f"- Source columns: {', '.join(study.text_columns().values())}",
        f"- Filter applied: {summary['filter_description']}",
        "- VADER scoring: sentiment is computed on the full text with the lexicon-and-rules model; it does not require stopword removal or simple word-score averaging.",
        (
            "- Label mapping used in this report: `compound >= 0.05 -> positive`, "
            "`compound <= -0.05 -> negative`, otherwise `mixed`"
        ),
        "- Output format: reusable CSV scores keyed by source ID, section-level PNG distributions, and this Markdown report.",
        "",
        "## Results",
        "",
    ]

    if not summary["sections"]:
        lines.extend(["No eligible text rows were found for the configured sections.", ""])
        return lines

    for section_name in study.section_order:
        if section_name not in summary["sections"]:
            continue
        section_summary = summary["sections"][section_name]
        figure_path = Path(figure_paths[section_name])
        rel_path = figure_path.relative_to(output_dir)
        lines.extend(
            [
                f"### {section_name.title()}",
                "",
                f"- N: {section_summary['n']}",
                f"- Compound mean: {section_summary['compound_mean']:.4f}",
                f"- Compound median: {section_summary['compound_median']:.4f}",
                f"- Compound std: {section_summary['compound_std']:.4f}",
                f"- Compound range: {section_summary['compound_min']:.4f} to {section_summary['compound_max']:.4f}",
                "",
                "| Label | Count |",
                "| --- | ---: |",
            ]
        )
        for label in ("positive", "mixed", "negative"):
            lines.append(f"| {label} | {section_summary['label_counts'].get(label, 0)} |")
        lines.extend(["", f"![{section_name} distribution]({rel_path.as_posix()})", ""])

    return lines


def write_vader_report(
    study: StudyConfig,
    summary: dict[str, Any],
    figure_paths: dict[str, str],
    output_dir: Path,
) -> Path:
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text("\n".join(_report_lines(study, summary, figure_paths, output_dir)), encoding="utf-8")
    return report_path


def run_vader_sensitivity(
    study: StudyConfig,
    paths: PathsConfig,
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    all_records: bool = False,
    quality_values: list[str] | None = None,
    limit: int | None = None,
    include_text: bool = False,
    source_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    source_mode = "provided_dataframe"
    if source_df is not None:
        source = source_df
        prefiltered = False
    else:
        explicit_input = Path(input_path) if input_path is not None else None
        if explicit_input is not None:
            source = read_tabular_file(explicit_input)
            source_mode = "explicit_input"
            prefiltered = False
        else:
            resolved_source_path = resolve_survey_source_path(paths, None)
            if resolved_source_path != Path(paths.survey_csv):
                source = load_batch_source(
                    study=study,
                    paths=paths,
                    source="survey",
                    all_records=all_records,
                    min_valid_sections=(3 if not all_records else None),
                )
                source_mode = "translated" if resolved_source_path.name == "translated_dataset.csv" else "preprocessed_cleaned"
                prefiltered = True
            else:
                source = read_tabular_file(Path(paths.survey_csv))
                source_mode = "survey_original"
                prefiltered = False

    resolved_output_dir = Path(output_dir or default_vader_output_dir(paths))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = resolved_output_dir / "figures"

    scores_df, summary = build_vader_scores(
        source,
        study,
        all_records=True if prefiltered else all_records,
        quality_values=None if prefiltered else quality_values,
        limit=limit,
        include_text=include_text,
    )
    summary["source_mode"] = source_mode

    scores_path = resolved_output_dir / DEFAULT_SCORES_FILENAME
    summary_path = resolved_output_dir / DEFAULT_SUMMARY_FILENAME
    scores_df.to_csv(scores_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_paths = write_vader_figures(scores_df, figures_dir) if not scores_df.empty else {}
    report_path = write_vader_report(study, summary, figure_paths, resolved_output_dir)

    written = {
        "scores_file": str(scores_path),
        "summary_file": str(summary_path),
        "report_file": str(report_path),
        "figures_dir": str(figures_dir),
    }
    return scores_df, summary, written


def load_vader_scores(path: Path, study: StudyConfig) -> pd.DataFrame:
    scores_df = read_tabular_file(path)
    required = {study.id_column, "section", "vader_label", "compound", "neg", "neu", "pos"}
    missing = sorted(required - set(scores_df.columns))
    if missing:
        raise ValueError(f"VADER scores file is missing required columns: {missing}")
    if "participant_code" not in scores_df.columns:
        scores_df["participant_code"] = ""
    return scores_df


def vader_scores_to_tone_predictions(
    scores_df: pd.DataFrame,
    sampled_private_df: pd.DataFrame,
    study: StudyConfig,
) -> pd.DataFrame:
    required = {"participant_code", study.id_column}
    missing = sorted(required - set(sampled_private_df.columns))
    if missing:
        raise ValueError(f"Sampled private workbook is missing required columns for VADER matching: {missing}")

    matched = sampled_private_df[["participant_code", study.id_column]].merge(
        scores_df[[study.id_column, "section", "vader_label"]],
        on=study.id_column,
        how="left",
    )
    pivot = matched.pivot_table(index="participant_code", columns="section", values="vader_label", aggfunc="first")
    out = pivot.reset_index()

    rename_map = {section_name: study.sections[section_name].tone_internal_column for section_name in study.section_order}
    out = out.rename(columns=rename_map)

    expected_columns = ["participant_code"] + study.tone_columns()
    for column in study.tone_columns():
        if column not in out.columns:
            raise ValueError(f"VADER scores did not provide labels for section tone column: {column}")
        if out[column].isna().any():
            raise ValueError(f"VADER scores are missing values needed to compare section tone column: {column}")

    return out[expected_columns].copy()
