from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt

from .config import StudyConfig


ALIGNMENT_REPORT_FILENAME = "alignment_report.md"
ALIGNMENT_FIGURES_SUBDIR = Path("figures") / "alignment"
ALIGNMENT_LONG_FILENAME = "alignment_metrics_long.csv"


def _comparison_label(comparison: str) -> str:
    return comparison.replace("_vs_", " vs ").replace("_", " ").title()


def _field_group(field: str) -> str:
    return "tone" if field.endswith("_tone") else "binary"


def _field_display_label(field: str, study: StudyConfig) -> str:
    return study.internal_to_visible_annotation_columns().get(field, field)


def _interpret_alignment(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "moderate"
    if value >= 0.25:
        return "limited"
    return "low"


def _interpret_kappa(value: float) -> str:
    if value >= 0.6:
        return "substantial agreement"
    if value >= 0.4:
        return "moderate agreement"
    if value >= 0.2:
        return "fair agreement"
    if value >= 0.0:
        return "slight agreement"
    return "agreement below chance expectation"


def build_alignment_long_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metrics_df.iterrows():
        for metric_name in ("accuracy", "cohen_kappa", "macro_f1"):
            rows.append(
                {
                    "comparison": row["comparison"],
                    "comparison_label": _comparison_label(str(row["comparison"])),
                    "field": row["field"],
                    "field_group": _field_group(str(row["field"])),
                    "n": int(row["n"]),
                    "metric": metric_name,
                    "value": float(row[metric_name]),
                }
            )
    return pd.DataFrame(rows)


def _style_axes(ax) -> None:
    ax.set_facecolor("#F7F4EA")
    ax.grid(axis="x", color="#DDD6C8", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_comparison_summary(metrics_df: pd.DataFrame, figure_path: Path) -> None:
    summary_df = (
        metrics_df.groupby("comparison", as_index=False)[["accuracy", "cohen_kappa", "macro_f1"]]
        .mean()
        .sort_values("cohen_kappa", ascending=True)
    )
    labels = [_comparison_label(value) for value in summary_df["comparison"]]
    y = list(range(len(summary_df)))
    height = 0.22

    fig, ax = plt.subplots(figsize=(10, max(4.5, 1.2 * len(summary_df))))
    ax.barh([value - height for value in y], summary_df["accuracy"], height=height, color="#2A6F97", label="Accuracy")
    ax.barh(y, summary_df["cohen_kappa"], height=height, color="#C1666B", label="Cohen kappa")
    ax.barh([value + height for value in y], summary_df["macro_f1"], height=height, color="#6C9A8B", label="Macro F1")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean metric value")
    ax.set_title("Alignment summary by comparison")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_metric_heatmap(metrics_df: pd.DataFrame, study: StudyConfig, metric_name: str, figure_path: Path) -> None:
    pivot = metrics_df.pivot(index="field", columns="comparison", values=metric_name)
    pivot = pivot.loc[sorted(pivot.index)]
    display_columns = list(pivot.columns)
    matrix = pivot.to_numpy(dtype=float)

    fig_width = max(7.5, 1.5 * len(display_columns))
    fig_height = max(5.0, 0.45 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=min(0.0, float(pd.DataFrame(matrix).min().min())), vmax=1.0)
    ax.set_xticks(range(len(display_columns)))
    ax.set_xticklabels([_comparison_label(value) for value in display_columns], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_field_display_label(field, study) for field in pivot.index])
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by field and comparison")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not math.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="#1F2933", fontsize=8)

    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label(metric_name.replace("_", " ").title())
    fig.patch.set_facecolor("#FFFDF8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_tone_alignment(metrics_df: pd.DataFrame, study: StudyConfig, figure_path: Path) -> None:
    tone_df = metrics_df[metrics_df["field"].str.endswith("_tone")].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    comparisons = list(dict.fromkeys(tone_df["comparison"].tolist()))
    fields = list(dict.fromkeys(tone_df["field"].tolist()))
    x = list(range(len(fields)))
    width = 0.8 / max(1, len(comparisons))
    colors = ["#2A6F97", "#C1666B", "#6C9A8B", "#B08968", "#7A4EAB"]

    for index, comparison in enumerate(comparisons):
        subset = tone_df[tone_df["comparison"] == comparison].set_index("field")
        values = [float(subset.loc[field, "cohen_kappa"]) for field in fields]
        offsets = [value - 0.4 + width / 2 + index * width for value in x]
        ax.bar(offsets, values, width=width, label=_comparison_label(comparison), color=colors[index % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([_field_display_label(field, study) for field in fields])
    ax.set_ylabel("Cohen kappa")
    ax.set_title("Tone alignment by section")
    ax.axhline(0.0, color="#2B2D42", linewidth=1.0, linestyle="--")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def write_alignment_figures(metrics_df: pd.DataFrame, study: StudyConfig, figures_dir: Path) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "comparison_summary.png"
    accuracy_heatmap_path = figures_dir / "accuracy_heatmap.png"
    kappa_heatmap_path = figures_dir / "cohen_kappa_heatmap.png"
    tone_path = figures_dir / "tone_alignment.png"

    plot_comparison_summary(metrics_df, summary_path)
    plot_metric_heatmap(metrics_df, study, "accuracy", accuracy_heatmap_path)
    plot_metric_heatmap(metrics_df, study, "cohen_kappa", kappa_heatmap_path)
    plot_tone_alignment(metrics_df, study, tone_path)

    return {
        "comparison_summary": str(summary_path),
        "accuracy_heatmap": str(accuracy_heatmap_path),
        "cohen_kappa_heatmap": str(kappa_heatmap_path),
        "tone_alignment": str(tone_path),
    }


def _available_sources(metrics_df: pd.DataFrame) -> list[str]:
    comparisons = set(metrics_df["comparison"])
    sources = ["human annotations", "questionnaire-derived labels", "VADER tone labels"]
    if any(comparison.startswith("human_vs_llm") or comparison.startswith("llm_") for comparison in comparisons):
        sources.append("LLM predictions")
    return sources


def _coverage_lines(coverage: dict[str, Any]) -> list[str]:
    lines = [
        "## Evaluation Coverage",
        "",
        f"- Sampled rows available in private mapping: {coverage['n_sampled_total']}",
        f"- Rows still present in the human workbook: {coverage['n_human_rows_total']}",
        f"- Fully annotated human rows included in metrics: {coverage['n_human_evaluable']}",
        f"- Human rows skipped because all labels were blank: {coverage['n_skipped_unannotated']}",
        "- Evaluation is computed against what humans actually completed; fully blank rows are skipped and partially completed rows fail validation.",
        "",
    ]
    removed_rows = coverage["n_sampled_total"] - coverage["n_human_rows_total"]
    if removed_rows > 0:
        lines.extend([
            f"- Sampled rows removed from the workbook before evaluation: {removed_rows}",
            "",
        ])
    return lines


def _ranking_lines(comparison_summary: dict[str, Any]) -> list[str]:
    ranking = sorted(
        comparison_summary.items(),
        key=lambda item: (item[1]["cohen_kappa_mean"], item[1]["accuracy_mean"], item[1]["macro_f1_mean"]),
        reverse=True,
    )
    lines = ["## Global Results", "", "| Comparison | Fields | Mean Accuracy | Mean Kappa | Mean Macro F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for comparison, values in ranking:
        lines.append(
            f"| {_comparison_label(comparison)} | {values['fields']} | {values['accuracy_mean']:.3f} | {values['cohen_kappa_mean']:.3f} | {values['macro_f1_mean']:.3f} |"
        )
    lines.extend(["", "### Alignment Ranking", ""])
    for index, (comparison, values) in enumerate(ranking, start=1):
        lines.append(
            f"{index}. {_comparison_label(comparison)}: {_interpret_kappa(values['cohen_kappa_mean'])} with {_interpret_alignment(values['accuracy_mean'])} mean alignment (accuracy {values['accuracy_mean']:.3f})."
        )
    lines.append("")
    return lines


def _field_result_lines(metrics_df: pd.DataFrame, study: StudyConfig) -> list[str]:
    lines = ["## Field-Level Results", ""]
    for comparison in metrics_df["comparison"].drop_duplicates().tolist():
        subset = metrics_df[metrics_df["comparison"] == comparison].copy().sort_values("field")
        lines.extend([f"### {_comparison_label(comparison)}", "", "| Field | Type | N | Accuracy | Kappa | Macro F1 |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for _, row in subset.iterrows():
            lines.append(
                f"| {_field_display_label(str(row['field']), study)} | {_field_group(str(row['field']))} | {int(row['n'])} | {float(row['accuracy']):.3f} | {float(row['cohen_kappa']):.3f} | {float(row['macro_f1']):.3f} |"
            )
        best_row = subset.sort_values(["cohen_kappa", "accuracy"], ascending=False).iloc[0]
        worst_row = subset.sort_values(["cohen_kappa", "accuracy"], ascending=True).iloc[0]
        lines.extend(
            [
                "",
                f"Best aligned field: `{_field_display_label(str(best_row['field']), study)}` ({_interpret_kappa(float(best_row['cohen_kappa']))}, accuracy {float(best_row['accuracy']):.3f}).",
                f"Lowest aligned field: `{_field_display_label(str(worst_row['field']), study)}` ({_interpret_kappa(float(worst_row['cohen_kappa']))}, accuracy {float(worst_row['accuracy']):.3f}).",
                "",
            ]
        )
    return lines


def _tone_result_lines(metrics_df: pd.DataFrame, study: StudyConfig) -> list[str]:
    tone_df = metrics_df[metrics_df["field"].str.endswith("_tone")].copy()
    lines = ["## Narrative Section Results", ""]
    for _, row in tone_df.sort_values(["comparison", "field"]).iterrows():
        lines.append(
            f"- {_comparison_label(str(row['comparison']))} on `{_field_display_label(str(row['field']), study)}`: accuracy {float(row['accuracy']):.3f}, kappa {float(row['cohen_kappa']):.3f}, macro F1 {float(row['macro_f1']):.3f}."
        )
    lines.append("")
    return lines


def _interpretation_lines(metrics_df: pd.DataFrame, comparison_summary: dict[str, Any]) -> list[str]:
    ranking = sorted(comparison_summary.items(), key=lambda item: item[1]["cohen_kappa_mean"], reverse=True)
    best_name, best_values = ranking[0]
    worst_name, worst_values = ranking[-1]
    tone_df = metrics_df[metrics_df["field"].str.endswith("_tone")]
    binary_df = metrics_df[~metrics_df["field"].str.endswith("_tone")]
    tone_kappa = float(tone_df["cohen_kappa"].mean()) if not tone_df.empty else float("nan")
    binary_kappa = float(binary_df["cohen_kappa"].mean()) if not binary_df.empty else float("nan")

    lines = [
        "## Interpretation",
        "",
        f"The strongest overall alignment in this run was {_comparison_label(best_name)} with {_interpret_kappa(best_values['cohen_kappa_mean'])}.",
        f"The weakest overall alignment was {_comparison_label(worst_name)} with {_interpret_kappa(worst_values['cohen_kappa_mean'])}.",
    ]
    if not math.isnan(tone_kappa) and not math.isnan(binary_kappa):
        relation = "higher" if tone_kappa > binary_kappa else "lower"
        lines.append(
            f"Average tone-field agreement was {relation} than binary-field agreement (mean kappa {tone_kappa:.3f} vs {binary_kappa:.3f})."
        )
    if (metrics_df["n"] < metrics_df["n"].max()).any():
        lines.append("Some fields were evaluated with fewer complete observations than others; this should be considered when comparing alignment magnitudes.")
    lines.append(
        "These results should be read as alignment between sources rather than proof that one source is the definitive target."
    )
    lines.append("")
    return lines


def _llm_section_lines(metrics_df: pd.DataFrame) -> list[str]:
    comparisons = set(metrics_df["comparison"])
    lines = ["## LLM Integration Readiness", ""]
    llm_present = any(comparison.startswith("human_vs_llm") or comparison.startswith("llm_") for comparison in comparisons)
    if llm_present:
        lines.append("LLM-derived comparisons were available in this run and have been integrated into the same reporting structure.")
    else:
        lines.append("LLM-derived comparisons were not available in this run. The report structure already reserves space for them, so future LLM outputs can be added without changing the format.")
    lines.append("")
    return lines


def write_alignment_report(
    study: StudyConfig,
    metrics_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    figure_paths: dict[str, str],
    vader_summary: dict[str, Any] | None = None,
) -> Path:
    report_path = output_dir / ALIGNMENT_REPORT_FILENAME
    sources = ", ".join(_available_sources(metrics_df))
    coverage = summary["coverage"]
    comparison_summary = summary["comparisons"]
    lines = [
        "# Alignment Report Across Sources",
        "",
        "## Objective",
        "",
        "Quantify how strongly the available annotation and derived-signal sources align with one another across NDE narratives, rather than treating any one source as a definitive ground truth.",
        "",
        "## Methodology",
        "",
        f"- Sources included in this run: {sources}",
        f"- Narrative tone sections: {', '.join(_field_display_label(field, study) for field in study.tone_columns())}",
        f"- Binary questionnaire-derived fields: {len(study.binary_columns())}",
        "- Metrics reported: accuracy, Cohen kappa, and macro F1.",
        "- VADER is applied with its lexicon-and-rules model and then discretized into positive, negative, and mixed labels using the standard compound thresholds.",
        "- The evaluation runs on the subset of participant codes that humans fully annotated; fully blank human rows are skipped and partially completed rows fail validation.",
        "",
    ]
    lines.extend(_coverage_lines(coverage))
    if vader_summary:
        lines.extend(
            [
                "### VADER Context",
                "",
                f"- Rows contributing to VADER analysis after filtering: {vader_summary.get('n_rows_after_filters', 'n/a')}",
                f"- VADER filter description: {vader_summary.get('filter_description', 'n/a')}",
                "",
            ]
        )

    lines.extend(_ranking_lines(comparison_summary))
    lines.extend(
        [
            "### Figures",
            "",
            f"![Alignment summary]({Path(figure_paths['comparison_summary']).relative_to(output_dir).as_posix()})",
            "",
            f"![Accuracy heatmap]({Path(figure_paths['accuracy_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![Kappa heatmap]({Path(figure_paths['cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![Tone alignment]({Path(figure_paths['tone_alignment']).relative_to(output_dir).as_posix()})",
            "",
        ]
    )
    lines.extend(_field_result_lines(metrics_df, study))
    lines.extend(_tone_result_lines(metrics_df, study))
    lines.extend(_interpretation_lines(metrics_df, comparison_summary))
    lines.extend(_llm_section_lines(metrics_df))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_alignment_outputs(
    study: StudyConfig,
    metrics_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    vader_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    figures_dir = output_dir / ALIGNMENT_FIGURES_SUBDIR
    figure_paths = write_alignment_figures(metrics_df, study, figures_dir)
    long_df = build_alignment_long_table(metrics_df)
    long_path = output_dir / ALIGNMENT_LONG_FILENAME
    long_df.to_csv(long_path, index=False)
    report_path = write_alignment_report(study, metrics_df, summary, output_dir, figure_paths, vader_summary=vader_summary)
    return {
        "alignment_report_file": str(report_path),
        "alignment_metrics_long_file": str(long_path),
        "alignment_figures_dir": str(figures_dir),
        **{f"figure_{name}": path for name, path in figure_paths.items()},
    }
