from __future__ import annotations

import math
from pathlib import Path
from textwrap import fill
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
    base, _, detail = comparison.partition(":")
    label = base.replace("_vs_", " vs ").replace("_", " ").title()
    if detail:
        return f"{label} ({detail})"
    return label


def _wrap_label(label: str, width: int = 24) -> str:
    return fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def _field_group(field: str) -> str:
    return "tone" if field.endswith("_tone") else "binary"


def _field_display_label(field: str, study: StudyConfig) -> str:
    return study.internal_to_visible_annotation_columns().get(field, field)


def _field_bucket(field: str) -> str:
    if field.endswith("_tone"):
        return "tone"
    if field.startswith("m8_"):
        return "m8"
    if field.startswith("m9_"):
        return "m9"
    return "other"


def _field_bucket_title(bucket: str) -> str:
    titles = {
        "tone": "Tone",
        "m8": "M8 — NDE-C (Content of the Near-Death Experience Scale)",
        "m9": "M9 — NDE-MCQ (Impact of the NDE on Moral Cognition)",
        "other": "Other",
    }
    return titles.get(bucket, bucket.replace("_", " ").title())


def _field_bucket_order(study: StudyConfig) -> dict[str, list[str]]:
    experience_fields = list(study.sections["experience"].binary_labels.keys())
    aftereffects_fields = list(study.sections["aftereffects"].binary_labels.keys())
    return {
        "tone": list(study.tone_columns()),
        "m8": [field for field in experience_fields if field.startswith("m8_")],
        "m9": [field for field in aftereffects_fields if field.startswith("m9_")],
    }


def _comparison_sort_key(comparison: str) -> tuple[int, str]:
    if comparison.startswith("human_reference_vs_llm"):
        return (0, comparison)
    if comparison == "human_reference_vs_questionnaire":
        return (1, comparison)
    if comparison == "human_reference_vs_vader":
        return (2, comparison)
    return (3, comparison)


def _interpret_alignment(value: float) -> str:
    if math.isnan(value):
        return "unavailable"
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "moderate"
    if value >= 0.25:
        return "limited"
    return "low"


def _interpret_kappa(value: float) -> str:
    if math.isnan(value):
        return "agreement unavailable"
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
                    "value": float(row[metric_name]) if not pd.isna(row[metric_name]) else float("nan"),
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
        .mean(numeric_only=True)
        .sort_values("cohen_kappa", ascending=True, na_position="first")
    )
    labels = [_wrap_label(_comparison_label(value), width=28) for value in summary_df["comparison"]]
    y = list(range(len(summary_df)))
    height = 0.22

    fig, ax = plt.subplots(figsize=(10, max(4.5, 1.2 * len(summary_df))))
    ax.barh([value - height for value in y], summary_df["accuracy"].fillna(0.0), height=height, color="#2A6F97", label="Accuracy")
    ax.barh(y, summary_df["cohen_kappa"].fillna(0.0), height=height, color="#C1666B", label="Cohen kappa")
    ax.barh([value + height for value in y], summary_df["macro_f1"].fillna(0.0), height=height, color="#6C9A8B", label="Macro F1")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean metric value")
    ax.set_title("Alignment summary by comparison")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    study: StudyConfig,
    metric_name: str,
    figure_path: Path,
    bucket: str,
) -> None:
    bucket_fields = _field_bucket_order(study).get(bucket, [])
    bucket_df = metrics_df[metrics_df["field"].map(_field_bucket) == bucket].copy()
    if bucket_fields:
        bucket_df = bucket_df[bucket_df["field"].isin(bucket_fields)]
    comparisons = sorted(bucket_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key)
    if bucket_fields:
        pivot = bucket_df.pivot(index="comparison", columns="field", values=metric_name)
        pivot = pivot.reindex(index=comparisons, columns=bucket_fields)
    else:
        pivot = bucket_df.pivot(index="comparison", columns="field", values=metric_name)
        pivot = pivot.reindex(index=comparisons)
    display_columns = list(pivot.columns)
    matrix = pivot.to_numpy(dtype=float)

    longest_x_label = max((_field_display_label(field, study) for field in display_columns), key=len, default="")
    longest_y_label = max((_comparison_label(comparison) for comparison in pivot.index), key=len, default="")
    fig_width = max(9.5, 1.6 * max(1, len(display_columns)), 7.0 + 0.07 * len(longest_y_label))
    fig_height = max(5.6, 1.05 * max(1, len(pivot.index)) + 2.4, 4.8 + 0.03 * len(longest_x_label))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    min_value = min(0.0, float(pd.DataFrame(matrix).min().min())) if matrix.size else 0.0
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=min_value, vmax=1.0)
    ax.set_xticks(range(len(display_columns)))
    ax.set_xticklabels([_wrap_label(_field_display_label(field, study), width=18) for field in display_columns], rotation=18, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_wrap_label(_comparison_label(comparison), width=26) for comparison in pivot.index])
    ax.set_xlabel("Items")
    ax.set_ylabel("Comparison")
    ax.set_title(_wrap_label(f"{metric_name.replace('_', ' ').title()} heatmap for {_field_bucket_title(bucket)}", width=52))
    ax.tick_params(axis="y", pad=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not math.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="#1F2933", fontsize=8)

    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label(metric_name.replace("_", " ").title())
    fig.patch.set_facecolor("#FFFDF8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tone_alignment(metrics_df: pd.DataFrame, study: StudyConfig, figure_path: Path) -> None:
    tone_fields = _field_bucket_order(study)["tone"]
    tone_df = metrics_df[metrics_df["field"].isin(tone_fields)].copy()
    longest_legend = max((_comparison_label(comparison) for comparison in comparisons), key=len, default="")
    fig_width = max(10.5, 8.5 + 0.06 * len(longest_legend))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    comparisons = sorted(tone_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key)
    fields = tone_fields
    x = list(range(len(fields)))
    width = 0.8 / max(1, len(comparisons))
    colors = ["#2A6F97", "#C1666B", "#6C9A8B", "#B08968", "#7A4EAB"]

    for index, comparison in enumerate(comparisons):
        subset = tone_df[tone_df["comparison"] == comparison].set_index("field")
        values = [float(subset.loc[field, "cohen_kappa"]) if field in subset.index else 0.0 for field in fields]
        offsets = [value - 0.4 + width / 2 + index * width for value in x]
        ax.bar(offsets, values, width=width, label=_wrap_label(_comparison_label(comparison), width=30), color=colors[index % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([_wrap_label(_field_display_label(field, study), width=18) for field in fields])
    ax.set_ylabel("Cohen kappa")
    ax.set_title("Tone alignment by section")
    ax.axhline(0.0, color="#2B2D42", linewidth=1.0, linestyle="--")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.4)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_alignment_figures(metrics_df: pd.DataFrame, study: StudyConfig, figures_dir: Path) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "comparison_summary.png"
    tone_path = figures_dir / "tone_alignment.png"
    figure_paths = {
        "comparison_summary": str(summary_path),
        "tone_accuracy_heatmap": str(figures_dir / "tone_accuracy_heatmap.png"),
        "tone_cohen_kappa_heatmap": str(figures_dir / "tone_cohen_kappa_heatmap.png"),
        "m8_accuracy_heatmap": str(figures_dir / "m8_accuracy_heatmap.png"),
        "m8_cohen_kappa_heatmap": str(figures_dir / "m8_cohen_kappa_heatmap.png"),
        "m9_accuracy_heatmap": str(figures_dir / "m9_accuracy_heatmap.png"),
        "m9_cohen_kappa_heatmap": str(figures_dir / "m9_cohen_kappa_heatmap.png"),
        "tone_alignment": str(tone_path),
    }

    plot_comparison_summary(metrics_df, summary_path)
    for bucket in ("tone", "m8", "m9"):
        plot_metric_heatmap(metrics_df, study, "accuracy", Path(figure_paths[f"{bucket}_accuracy_heatmap"]), bucket)
        plot_metric_heatmap(metrics_df, study, "cohen_kappa", Path(figure_paths[f"{bucket}_cohen_kappa_heatmap"]), bucket)
    plot_tone_alignment(metrics_df, study, tone_path)

    return figure_paths


def _coverage_lines(summary: dict[str, Any]) -> list[str]:
    coverage = summary["coverage"]
    adjudication = summary["adjudication"]
    return [
        "## Evaluation Coverage",
        "",
        f"- Sampled rows available in private mapping: {coverage['n_sampled_total']}",
        f"- Valid human annotation artifacts: {coverage['n_valid_human_artifacts']}",
        f"- Rejected human annotation artifacts: {coverage['n_rejected_human_artifacts']}",
        f"- Participants carried into the majority human reference: {coverage['n_reference_participants']}",
        f"- Valid LLM artifacts evaluated: {coverage['n_valid_llm_artifacts']}",
        f"- Rejected LLM artifacts: {coverage['n_rejected_llm_artifacts']}",
        f"- Unresolved field/participant pairs after majority voting: {adjudication['n_unresolved_field_participant_pairs']}",
        "- The reference set is computed by field-level majority vote across valid human artifacts. Unresolved ties remain blank and are excluded from the affected field metrics.",
        "",
    ]


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
        lines.append("")
    return lines


def _human_agreement_lines(summary: dict[str, Any], output_dir: Path) -> list[str]:
    accepted = summary["human_artifacts"]["accepted"]
    rejected = summary["human_artifacts"]["rejected"]
    lines = ["## Human Annotation Agreement", ""]
    if accepted:
        lines.append("Valid annotators included in the majority reference:")
        lines.append("")
        for artifact in accepted:
            lines.append(f"- `{artifact['annotator_id']}` from `{Path(artifact['artifact_path']).name}`")
        lines.append("")
    if rejected:
        lines.append("Rejected human artifacts:")
        lines.append("")
        for artifact in rejected:
            lines.append(f"- `{Path(artifact['artifact_path']).name}`: {artifact['reason']}")
        lines.append("")
    lines.append(f"Pairwise agreement details: `{Path(output_dir / 'human_agreement_pairwise.csv').name}`")
    lines.append(f"Agreement summary by field: `{Path(output_dir / 'human_agreement_summary.csv').name}`")
    lines.append("")
    return lines


def _llm_lines(summary: dict[str, Any]) -> list[str]:
    accepted = summary["llm_artifacts"]["accepted"]
    rejected = summary["llm_artifacts"]["rejected"]
    lines = ["## LLM Experiments", ""]
    if accepted:
        lines.append("Accepted experiments evaluated against the human majority reference:")
        lines.append("")
        for artifact in accepted:
            descriptor = artifact["artifact_id"]
            if artifact.get("prompt_variant"):
                descriptor += f", prompt={artifact['prompt_variant']}"
            if artifact.get("model_variant"):
                descriptor += f", model={artifact['model_variant']}"
            lines.append(f"- `{descriptor}`")
        lines.append("")
    else:
        lines.append("No valid LLM artifacts were available for this run.")
        lines.append("")
    if rejected:
        lines.append("Rejected LLM artifacts:")
        lines.append("")
        for artifact in rejected:
            lines.append(f"- `{Path(artifact['artifact_path']).name}`: {artifact['reason']}")
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
    comparison_summary = summary["comparisons"]
    lines = [
        "# Alignment Report Across Sources",
        "",
        "## Objective",
        "",
        "Quantify alignment across a majority-vote human reference, questionnaire-derived labels, VADER tone labels, and any accepted LLM experiments.",
        "",
        "## Methodology",
        "",
        "- Human reference: field-level majority vote across valid human annotation artifacts.",
        "- Unresolved ties remain blank and are excluded only from the affected field metrics.",
        "- Metrics reported: accuracy, Cohen kappa, and macro F1.",
        "- VADER is applied with its lexicon-and-rules model and then discretized into positive, negative, and mixed labels using the standard compound thresholds.",
        "",
    ]
    lines.extend(_coverage_lines(summary))
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
            "#### Tone",
            "",
            f"![Tone accuracy heatmap]({Path(figure_paths['tone_accuracy_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![Tone kappa heatmap]({Path(figure_paths['tone_cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![Tone alignment]({Path(figure_paths['tone_alignment']).relative_to(output_dir).as_posix()})",
            "",
            "#### M8",
            "",
            f"![M8 accuracy heatmap]({Path(figure_paths['m8_accuracy_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![M8 kappa heatmap]({Path(figure_paths['m8_cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            "#### M9",
            "",
            f"![M9 accuracy heatmap]({Path(figure_paths['m9_accuracy_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![M9 kappa heatmap]({Path(figure_paths['m9_cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
            "",
        ]
    )
    lines.extend(_field_result_lines(metrics_df, study))
    lines.extend(_human_agreement_lines(summary, output_dir))
    lines.extend(_llm_lines(summary))
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
