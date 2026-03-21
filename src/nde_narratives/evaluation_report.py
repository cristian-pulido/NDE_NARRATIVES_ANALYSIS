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
ALIGNMENT_QUESTIONNAIRE_REPORT_FILENAME = "alignment_report_questionnaire.md"
ALIGNMENT_FIGURES_SUBDIR = Path("figures") / "alignment"
ALIGNMENT_LONG_FILENAME = "alignment_metrics_long.csv"
ALIGNMENT_FAMILY_FILENAME = "alignment_family_metrics.csv"


def _comparison_label(comparison: str) -> str:
    base, _, detail = comparison.partition(":")
    label = base.replace("_vs_", " vs ").replace("_", " ").title()
    if detail:
        return f"{label} ({detail})"
    return label


def _wrap_label(label: str, width: int = 24) -> str:
    return fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def _bar_label_y(value: float, min_offset: float = 0.01, scale: float = 0.04) -> float:
    return value + max(min_offset, abs(value) * scale)


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


def _family_sort_key(bucket: str) -> tuple[int, str]:
    order = {"tone": 0, "m8": 1, "m9": 2, "other": 3}
    return (order.get(bucket, 99), bucket)


def _field_bucket_order(study: StudyConfig) -> dict[str, list[str]]:
    experience_fields = list(study.sections["experience"].binary_labels.keys())
    aftereffects_fields = list(study.sections["aftereffects"].binary_labels.keys())
    return {
        "tone": list(study.tone_columns()),
        "m8": [field for field in experience_fields if field.startswith("m8_")],
        "m9": [field for field in aftereffects_fields if field.startswith("m9_")],
    }


def _comparison_sort_key(comparison: str) -> tuple[int, str]:
    if comparison.startswith("questionnaire_vs_llm"):
        return (0, comparison)
    if comparison == "questionnaire_vs_vader":
        return (1, comparison)
    if comparison.startswith("human_reference_vs_llm"):
        return (2, comparison)
    if comparison == "human_reference_vs_questionnaire":
        return (3, comparison)
    if comparison == "human_reference_vs_vader":
        return (4, comparison)
    if comparison.startswith("vader_vs_llm"):
        return (5, comparison)
    if comparison.startswith("llm_vs_llm"):
        return (6, comparison)
    return (7, comparison)


def _comparison_tab(comparison: str) -> str:
    if comparison.startswith("questionnaire_vs_") and (
        "llm" in comparison or "vader" in comparison or "automated" in comparison
    ):
        return "questionnaire_vs_automated"
    if comparison.startswith("human_reference_vs_"):
        return "human_vs_all"
    return "other"


def _select_top_comparisons_for_figure(
    metrics_df: pd.DataFrame,
    scope_prefix: str,
    baseline_comparisons: list[str],
    top_n: int = 3,
    ranking_metric: str = "macro_f1",
) -> list[str]:
    """Select comparisons for figure display: always include baselines, then top N LLMs by ranking metric.

    Args:
        metrics_df: Full metrics dataframe.
        scope_prefix: Scope prefix (e.g., "questionnaire_vs_" or "human_reference_vs_").
        baseline_comparisons: List of baseline comparison names to always include.
        top_n: Number of top LLM comparisons to include.
        ranking_metric: Metric to rank LLMs by ("accuracy", "cohen_kappa", or "macro_f1").

    Returns:
        List of comparison names to display in figures.
    """
    scoped_df = metrics_df[metrics_df["comparison"].astype(str).str.startswith(scope_prefix)].copy()
    if scoped_df.empty:
        return baseline_comparisons

    # Get unique comparisons
    all_comparisons = scoped_df["comparison"].drop_duplicates().tolist()

    # Separate baselines from LLMs
    baseline_set = set(baseline_comparisons)
    baselines_present = [c for c in baseline_comparisons if c in all_comparisons]
    llm_comparisons = [c for c in all_comparisons if c not in baseline_set]

    # Compute mean ranking metric per comparison
    comparison_means = (
        scoped_df.groupby("comparison", as_index=False)[[ranking_metric]]
        .mean(numeric_only=True)
        .sort_values(ranking_metric, ascending=False)
    )

    # Select top N LLMs
    top_llms = comparison_means.head(top_n)["comparison"].tolist()

    # Combine baselines + top LLMs
    selected = baselines_present + top_llms
    return selected


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
        for metric_name in (
            "accuracy",
            "cohen_kappa",
            "macro_f1",
            "precision_yes",
            "recall_yes",
            "f1_yes",
            "prevalence_reference_yes",
            "prevalence_candidate_yes",
            "prevalence_gap_yes",
        ):
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


def _comparison_subset(metrics_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return metrics_df[metrics_df["comparison"].astype(str).str.startswith(prefix)].copy()


def _questionnaire_automated_subset(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df[metrics_df["comparison"].astype(str).str.startswith("questionnaire_vs_")].copy()


def _style_axes(ax) -> None:
    ax.set_facecolor("#F7F4EA")
    ax.grid(axis="x", color="#DDD6C8", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _save_figure(fig, figure_path: Path, dpi: int = 300, export_pdf: bool = False) -> list[str]:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    written = [str(figure_path)]
    if export_pdf:
        pdf_path = figure_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        written.append(str(pdf_path))
    plt.close(fig)
    return written


def plot_comparison_summary(metrics_df: pd.DataFrame, figure_path: Path, dpi: int = 300, export_pdf: bool = False, scope_prefix: str | None = None, baseline_comparisons: list[str] | None = None, top_n: int = 3) -> list[str]:
    """Plot comparison summary figure, optionally filtered to baselines + top N LLMs by macro_f1.

    Args:
        metrics_df: Full metrics dataframe.
        figure_path: Output figure path.
        dpi: Figure DPI.
        export_pdf: Whether to also export PDF.
        scope_prefix: Scope prefix for filtering (e.g., "questionnaire_vs_" or "human_reference_vs_").
        baseline_comparisons: Baseline comparisons to always include.
        top_n: Number of top LLMs to include.
    """
    # Filter to scope if provided
    if scope_prefix:
        filtered_df = metrics_df[metrics_df["comparison"].astype(str).str.startswith(scope_prefix)].copy()
    else:
        filtered_df = metrics_df.copy()

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        selected_comparisons = _select_top_comparisons_for_figure(
            filtered_df, scope_prefix, baseline_comparisons, top_n=top_n, ranking_metric="macro_f1"
        )
        filtered_df = filtered_df[filtered_df["comparison"].isin(selected_comparisons)]

    summary_df = (
        filtered_df.groupby("comparison", as_index=False)[["accuracy", "cohen_kappa", "macro_f1"]]
        .mean(numeric_only=True)
        .sort_values("macro_f1", ascending=False, na_position="last")
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
    # Move legend to upper left to avoid overlapping with bars
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def build_family_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "comparison",
                "comparison_label",
                "family",
                "family_label",
                "field_count",
                "n_mean",
                "n_min",
                "n_max",
                "accuracy_mean",
                "cohen_kappa_mean",
                "macro_f1_mean",
                "precision_yes_mean",
                "recall_yes_mean",
                "f1_yes_mean",
                "prevalence_reference_yes_mean",
                "prevalence_candidate_yes_mean",
                "prevalence_gap_yes_mean",
            ]
        )
    family_df = metrics_df.copy()
    family_df["family"] = family_df["field"].map(lambda value: _field_bucket(str(value)))
    grouped = (
        family_df.groupby(["comparison", "family"], as_index=False)
        .agg(
            field_count=("field", "count"),
            n_mean=("n", "mean"),
            n_min=("n", "min"),
            n_max=("n", "max"),
            accuracy_mean=("accuracy", "mean"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            precision_yes_mean=("precision_yes", "mean"),
            recall_yes_mean=("recall_yes", "mean"),
            f1_yes_mean=("f1_yes", "mean"),
            prevalence_reference_yes_mean=("prevalence_reference_yes", "mean"),
            prevalence_candidate_yes_mean=("prevalence_candidate_yes", "mean"),
            prevalence_gap_yes_mean=("prevalence_gap_yes", "mean"),
        )
        .copy()
    )
    grouped["comparison_label"] = grouped["comparison"].map(lambda value: _comparison_label(str(value)))
    grouped["family_label"] = grouped["family"].map(lambda value: _field_bucket_title(str(value)))
    return grouped.sort_values(
        ["comparison", "family"],
        key=lambda col: col.map(_comparison_sort_key) if col.name == "comparison" else col.map(_family_sort_key),
    )


def plot_family_summary(
    family_df: pd.DataFrame,
    figure_path: Path,
    metric_name: str = "cohen_kappa_mean",
    title: str = "Family-level alignment summary",
    dpi: int = 300,
    export_pdf: bool = False,
    scope_prefix: str | None = None,
    baseline_comparisons: list[str] | None = None,
    top_n: int = 3,
) -> list[str]:
    """Plot family summary figure, optionally filtered to baselines + top N LLMs by macro_f1.

    Args:
        family_df: Family-level summary dataframe.
        figure_path: Output figure path.
        metric_name: Metric to plot ("cohen_kappa_mean" or "recall_yes_mean").
        title: Figure title.
        dpi: Figure DPI.
        export_pdf: Whether to also export PDF.
        scope_prefix: Scope prefix for filtering.
        baseline_comparisons: Baseline comparisons to always include.
        top_n: Number of top LLMs to include.
    """
    plot_df = family_df.copy()
    plot_df = plot_df[plot_df["family"].isin(["tone", "m8", "m9"])]
    families = sorted(plot_df["family"].drop_duplicates().tolist(), key=_family_sort_key)
    recall_families = [
        family
        for family in families
        if not plot_df.loc[plot_df["family"] == family, "recall_yes_mean"].isna().all()
    ]

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        # Compute mean macro_f1 per comparison for ranking
        comparison_means = (
            plot_df.groupby("comparison", as_index=False)["macro_f1_mean"]
            .mean(numeric_only=True)
            .sort_values("macro_f1_mean", ascending=False)
        )
        baseline_set = set(baseline_comparisons)
        baselines_present = [c for c in baseline_comparisons if c in comparison_means["comparison"].tolist()]
        llm_comparisons = [c for c in comparison_means["comparison"].tolist() if c not in baseline_set]
        top_llms = comparison_means.head(top_n)["comparison"].tolist()
        selected_comparisons = baselines_present + top_llms
        plot_df = plot_df[plot_df["comparison"].isin(selected_comparisons)]

    comparisons = sorted(plot_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0))
    x = list(range(len(families)))
    recall_x = list(range(len(recall_families)))
    width = 0.82 / max(1, len(comparisons))
    colors = ["#355070", "#6D597A", "#B56576", "#2A9D8F", "#457B9D"]
    for index, comparison in enumerate(comparisons):
        subset = plot_df[plot_df["comparison"] == comparison].set_index("family")
        values = [float(subset.loc[family, metric_name]) if family in subset.index else float("nan") for family in families]
        recall_values = [float(subset.loc[family, "recall_yes_mean"]) if family in subset.index else float("nan") for family in recall_families]
        offsets = [value - 0.41 + width / 2 + index * width for value in x]
        recall_offsets = [value - 0.41 + width / 2 + index * width for value in recall_x]
        axes[0].bar(
            offsets,
            [float("nan") if math.isnan(value) else value for value in values],
            width=width,
            label=_wrap_label(_comparison_label(comparison), width=28),
            color=colors[index % len(colors)],
            edgecolor="#F8F5F0",
            linewidth=0.8,
        )
        for offset, value in zip(offsets, values, strict=False):
            if not math.isnan(value):
                axes[0].text(
                    offset,
                    _bar_label_y(value, min_offset=0.008, scale=0.06),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2B2D42",
                    clip_on=False,
                )
        if recall_families:
            axes[1].bar(
                recall_offsets,
                [float("nan") if math.isnan(value) else value for value in recall_values],
                width=width,
                color=colors[index % len(colors)],
                edgecolor="#F8F5F0",
                linewidth=0.8,
            )
            for offset, value in zip(recall_offsets, recall_values, strict=False):
                if not math.isnan(value):
                    axes[1].text(
                        offset,
                        _bar_label_y(value, min_offset=0.012, scale=0.05),
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#2B2D42",
                        clip_on=False,
                    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_wrap_label(_field_bucket_title(family), width=18) for family in families])
    _style_axes(axes[0])
    axes[1].set_xticks(recall_x)
    axes[1].set_xticklabels([_wrap_label(_field_bucket_title(family), width=18) for family in recall_families])
    _style_axes(axes[1])
    axes[0].tick_params(axis="x", labelsize=9)
    axes[1].tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("Mean Cohen kappa")
    axes[0].set_title(title)
    axes[0].axhline(0.0, color="#2B2D42", linewidth=1.0, linestyle="--")
    left_max = float(pd.Series(plot_df[metric_name]).max()) if not plot_df.empty else 0.0
    axes[0].set_ylim(0.0, max(0.42, _bar_label_y(left_max, min_offset=0.02, scale=0.12)))
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[1].set_ylabel("Mean recall for 'yes'")
    axes[1].set_title("Positive-class recovery by binary family")
    right_max = float(plot_df[plot_df["family"].isin(recall_families)]["recall_yes_mean"].max()) if recall_families else 0.0
    axes[1].set_ylim(0.0, min(1.0, max(0.8, _bar_label_y(right_max, min_offset=0.04, scale=0.08))))
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_single_field_metric_summary(
    metrics_df: pd.DataFrame,
    field: str,
    study: StudyConfig,
    figure_path: Path,
    title: str,
    dpi: int = 300,
    export_pdf: bool = False,
    scope_prefix: str | None = None,
    baseline_comparisons: list[str] | None = None,
    top_n: int = 3,
) -> list[str]:
    """Plot single field metric summary, optionally filtered to baselines + top N LLMs by macro_f1.

    Args:
        metrics_df: Full metrics dataframe.
        field: Field name to filter on.
        study: Study configuration.
        figure_path: Output figure path.
        title: Figure title.
        dpi: Figure DPI.
        export_pdf: Whether to also export PDF.
        scope_prefix: Scope prefix for filtering (e.g., "questionnaire_vs_" or "human_reference_vs_").
        baseline_comparisons: Baseline comparisons to always include.
        top_n: Number of top LLMs to include.
    """
    field_df = metrics_df[metrics_df["field"] == field].copy()

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        selected_comparisons = _select_top_comparisons_for_figure(
            field_df, scope_prefix, baseline_comparisons, top_n=top_n, ranking_metric="macro_f1"
        )
        field_df = field_df[field_df["comparison"].isin(selected_comparisons)]

    comparisons = sorted(field_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key)
    labels = [_wrap_label(_comparison_label(comparison), width=24) for comparison in comparisons]
    x = list(range(len(comparisons)))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8.5, 1.9 * len(comparisons) + 2.8), 6.1))
    metric_specs = [
        ("accuracy", "Accuracy", "#2A6F97", -width),
        ("cohen_kappa", "Cohen kappa", "#C1666B", 0.0),
        ("macro_f1", "Macro F1", "#6C9A8B", width),
    ]
    for metric_name, metric_label, color, offset in metric_specs:
        values = [
            float(field_df.loc[field_df["comparison"] == comparison, metric_name].iloc[0]) if comparison in set(field_df["comparison"]) else float("nan")
            for comparison in comparisons
        ]
        positions = [value + offset for value in x]
        ax.bar(positions, [0.0 if math.isnan(value) else value for value in values], width=width, label=metric_label, color=color)
        for position, value in zip(positions, values, strict=False):
            if not math.isnan(value):
                ax.text(
                    position,
                    _bar_label_y(value, min_offset=0.012, scale=0.05),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2B2D42",
                    clip_on=False,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.axhline(0.0, color="#2B2D42", linewidth=1.0, linestyle="--")
    max_value = max((float(field_df[metric].max()) for metric, _, _, _ in metric_specs if metric in field_df.columns), default=0.0)
    ax.set_ylim(0.0, min(1.0, max(0.75, _bar_label_y(max_value, min_offset=0.04, scale=0.08))))
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def _figure_basename(scope_prefix: str) -> str:
    return "human" if scope_prefix == "human_reference_vs_" else "questionnaire"


def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    study: StudyConfig,
    metric_name: str,
    figure_path: Path,
    bucket: str,
    dpi: int = 300,
    export_pdf: bool = False,
    scope_prefix: str | None = None,
    baseline_comparisons: list[str] | None = None,
    top_n: int = 3,
) -> list[str]:
    """Plot metric heatmap, optionally filtered to baselines + top N LLMs by macro_f1.

    Args:
        metrics_df: Full metrics dataframe.
        study: Study configuration.
        metric_name: Metric name to plot ("accuracy" or "cohen_kappa").
        figure_path: Output figure path.
        bucket: Bucket name ("tone", "m8", or "m9").
        dpi: Figure DPI.
        export_pdf: Whether to also export PDF.
        scope_prefix: Scope prefix for filtering.
        baseline_comparisons: Baseline comparisons to always include.
        top_n: Number of top LLMs to include.
    """
    bucket_fields = _field_bucket_order(study).get(bucket, [])
    bucket_df = metrics_df[metrics_df["field"].map(_field_bucket) == bucket].copy()
    if bucket_fields:
        bucket_df = bucket_df[bucket_df["field"].isin(bucket_fields)]

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        selected_comparisons = _select_top_comparisons_for_figure(
            bucket_df, scope_prefix, baseline_comparisons, top_n=top_n, ranking_metric="macro_f1"
        )
        bucket_df = bucket_df[bucket_df["comparison"].isin(selected_comparisons)]

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
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_tone_alignment(metrics_df: pd.DataFrame, study: StudyConfig, figure_path: Path, dpi: int = 300, export_pdf: bool = False, scope_prefix: str | None = None, baseline_comparisons: list[str] | None = None, top_n: int = 3) -> list[str]:
    """Plot tone alignment figure, optionally filtered to baselines + top N LLMs by macro_f1.

    Args:
        metrics_df: Full metrics dataframe.
        study: Study configuration.
        figure_path: Output figure path.
        dpi: Figure DPI.
        export_pdf: Whether to also export PDF.
        scope_prefix: Scope prefix for filtering.
        baseline_comparisons: Baseline comparisons to always include.
        top_n: Number of top LLMs to include.
    """
    tone_fields = _field_bucket_order(study)["tone"]
    tone_df = metrics_df[metrics_df["field"].isin(tone_fields)].copy()

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        selected_comparisons = _select_top_comparisons_for_figure(
            tone_df, scope_prefix, baseline_comparisons, top_n=top_n, ranking_metric="macro_f1"
        )
        tone_df = tone_df[tone_df["comparison"].isin(selected_comparisons)]

    comparisons = sorted(tone_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key)
    longest_legend = max((_comparison_label(comparison) for comparison in comparisons), key=len, default="")
    fig_width = max(10.5, 8.5 + 0.06 * len(longest_legend))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
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
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def write_alignment_figures(
    metrics_df: pd.DataFrame,
    study: StudyConfig,
    figures_dir: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    scoped_paths: dict[str, str] = {}
    for scope_prefix in ("human_reference_vs_", "questionnaire_vs_"):
        scoped_metrics_df = _comparison_subset(metrics_df, scope_prefix)
        if scoped_metrics_df.empty:
            continue
        family_df = build_family_summary_table(scoped_metrics_df)
        base = _figure_basename(scope_prefix)
        summary_path = figures_dir / f"{base}_comparison_summary.png"
        family_summary_path = figures_dir / f"{base}_family_summary.png"
        path_map = {
            f"{base}_comparison_summary": str(summary_path),
            f"{base}_family_summary": str(family_summary_path),
            f"{base}_tone_summary": str(figures_dir / f"{base}_tone_summary.png"),
            f"{base}_tone_accuracy_heatmap": str(figures_dir / f"{base}_tone_accuracy_heatmap.png"),
            f"{base}_tone_cohen_kappa_heatmap": str(figures_dir / f"{base}_tone_cohen_kappa_heatmap.png"),
            f"{base}_m8_accuracy_heatmap": str(figures_dir / f"{base}_m8_accuracy_heatmap.png"),
            f"{base}_m8_cohen_kappa_heatmap": str(figures_dir / f"{base}_m8_cohen_kappa_heatmap.png"),
            f"{base}_m9_accuracy_heatmap": str(figures_dir / f"{base}_m9_accuracy_heatmap.png"),
            f"{base}_m9_cohen_kappa_heatmap": str(figures_dir / f"{base}_m9_cohen_kappa_heatmap.png"),
        }
        # Select comparisons for figure: baselines + top 3 LLMs by macro_f1
        if scope_prefix == "questionnaire_vs_":
            baseline_comparisons = ["questionnaire_vs_vader"]
        elif scope_prefix == "human_reference_vs_":
            baseline_comparisons = ["human_reference_vs_questionnaire", "human_reference_vs_vader"]
        else:
            baseline_comparisons = []
        plot_comparison_summary(
            scoped_metrics_df,
            summary_path,
            dpi=dpi,
            export_pdf=export_pdf,
            scope_prefix=scope_prefix,
            baseline_comparisons=baseline_comparisons,
            top_n=3,
        )
        scope_title = "Human vs all — family-level alignment summary" if scope_prefix == "human_reference_vs_" else "Questionnaire vs automated — family-level alignment summary"
        plot_family_summary(
            family_df,
            family_summary_path,
            title=scope_title,
            dpi=dpi,
            export_pdf=export_pdf,
            scope_prefix=scope_prefix,
            baseline_comparisons=baseline_comparisons,
            top_n=3,
        )
        tone_summary_title = "Human vs all — tone summary" if scope_prefix == "human_reference_vs_" else "Questionnaire vs automated — Experience Tone summary"
        # Select comparisons for tone figure: baselines + top 3 LLMs by macro_f1
        if scope_prefix == "questionnaire_vs_":
            tone_baseline_comparisons = ["questionnaire_vs_vader"]
        elif scope_prefix == "human_reference_vs_":
            tone_baseline_comparisons = ["human_reference_vs_questionnaire", "human_reference_vs_vader"]
        else:
            tone_baseline_comparisons = []
        plot_single_field_metric_summary(
            scoped_metrics_df,
            field=study.sections["experience"].tone_internal_column,
            study=study,
            figure_path=Path(path_map[f"{base}_tone_summary"]),
            title=tone_summary_title,
            dpi=dpi,
            export_pdf=export_pdf,
            scope_prefix=scope_prefix,
            baseline_comparisons=tone_baseline_comparisons,
            top_n=3,
        )
        # Select comparisons for heatmap figures: baselines + top 3 LLMs by macro_f1
        if scope_prefix == "questionnaire_vs_":
            heatmap_baseline_comparisons = ["questionnaire_vs_vader"]
        elif scope_prefix == "human_reference_vs_":
            heatmap_baseline_comparisons = ["human_reference_vs_questionnaire", "human_reference_vs_vader"]
        else:
            heatmap_baseline_comparisons = []
        for bucket in ("tone", "m8", "m9"):
            plot_metric_heatmap(
                scoped_metrics_df,
                study,
                "accuracy",
                Path(path_map[f"{base}_{bucket}_accuracy_heatmap"]),
                bucket,
                dpi=dpi,
                export_pdf=export_pdf,
                scope_prefix=scope_prefix,
                baseline_comparisons=heatmap_baseline_comparisons,
                top_n=3,
            )
            plot_metric_heatmap(
                scoped_metrics_df,
                study,
                "cohen_kappa",
                Path(path_map[f"{base}_{bucket}_cohen_kappa_heatmap"]),
                bucket,
                dpi=dpi,
                export_pdf=export_pdf,
                scope_prefix=scope_prefix,
                baseline_comparisons=heatmap_baseline_comparisons,
                top_n=3,
            )
        if scope_prefix == "human_reference_vs_":
            tone_path = figures_dir / f"{base}_tone_alignment.png"
            path_map[f"{base}_tone_alignment"] = str(tone_path)
            # Select comparisons for tone alignment figure: baselines + top 3 LLMs by macro_f1
            tone_alignment_baseline_comparisons = ["human_reference_vs_questionnaire", "human_reference_vs_vader"]
            plot_tone_alignment(
                scoped_metrics_df,
                study,
                tone_path,
                dpi=dpi,
                export_pdf=export_pdf,
                scope_prefix=scope_prefix,
                baseline_comparisons=tone_alignment_baseline_comparisons,
                top_n=3,
            )
        scoped_paths.update(path_map)

    return scoped_paths


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
    # Sort by macro_f1 descending, then by comparison order for baselines
    ranking = sorted(
        comparison_summary.items(),
        key=lambda item: (item[1]["macro_f1_mean"], item[1]["accuracy_mean"], item[1]["cohen_kappa_mean"]),
        reverse=True,
    )
    lines = ["## Global Results", "", "| Comparison | Fields | Mean Accuracy | Mean Kappa | Mean Macro F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for comparison, values in ranking:
        lines.append(
            f"| {_comparison_label(comparison)} | {values['fields']} | {values['accuracy_mean']:.3f} | {values['cohen_kappa_mean']:.3f} | {values['macro_f1_mean']:.3f} |"
        )
    lines.append("")
    return lines


def _summary_table_lines(metrics_df: pd.DataFrame, title: str) -> list[str]:
    if metrics_df.empty:
        return [f"## {title}", "", "- No comparisons were available in this run.", ""]
    # Compute mean metrics and field count per comparison
    grouped = (
        metrics_df.groupby("comparison", as_index=False)
        .agg(
            fields=("field", "count"),
            accuracy_mean=("accuracy", "mean"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
        )
    )
    # Sort by macro_f1 descending
    grouped = grouped.sort_values("macro_f1_mean", ascending=False, na_position="last")
    lines = [f"## {title}", "", "| Comparison | Fields | Mean Accuracy | Mean Kappa | Mean Macro F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, row in grouped.iterrows():
        comparison = row["comparison"]
        lines.append(
            f"| {_comparison_label(comparison)} | {int(row['fields'])} | {float(row['accuracy_mean']):.3f} | {float(row['cohen_kappa_mean']):.3f} | {float(row['macro_f1_mean']):.3f} |"
        )
    lines.append("")
    return lines


def _family_summary_lines(family_df: pd.DataFrame, title: str) -> list[str]:
    lines = [f"## {title}", ""]
    if family_df.empty:
        lines.extend(["- No family-level comparisons were available in this run.", ""])
        return lines
    lines.extend([
        "| Comparison | Family | Fields | Mean Kappa | Mean Macro F1 | Recall Yes | Prevalence Gap | Mean N |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    # Sort by macro_f1 descending, then by comparison order, then by family order
    family_df = family_df.copy()
    family_df["_sort_key"] = family_df["comparison"].map(_comparison_sort_key)
    family_df = family_df.sort_values(["_sort_key", "macro_f1_mean", "family"], ascending=[True, False, False])
    family_df = family_df.drop(columns=["_sort_key"])
    for _, row in family_df.iterrows():
        lines.append(
            f"| {_comparison_label(str(row['comparison']))} | {str(row['family_label'])} | {int(row['field_count'])} | {float(row['cohen_kappa_mean']):.3f} | {float(row['macro_f1_mean']):.3f} | {float(row['recall_yes_mean']):.3f} | {float(row['prevalence_gap_yes_mean']):.3f} | {float(row['n_mean']):.1f} |"
        )
    lines.append("")
    return lines


def _questionnaire_interpretation_lines(family_df: pd.DataFrame) -> list[str]:
    lines = ["## Interpretation For The Manuscript", ""]
    llm_family_df = family_df[family_df["comparison"].astype(str).str.startswith("questionnaire_vs_llm")].copy()
    if llm_family_df.empty:
        lines.extend(["- Questionnaire-based family summaries were unavailable in this run.", ""])
        return lines
    family_means = (
        llm_family_df.groupby("family", as_index=False)[["cohen_kappa_mean", "macro_f1_mean", "recall_yes_mean", "prevalence_gap_yes_mean"]]
        .mean(numeric_only=True)
        .sort_values("family", key=lambda s: s.map(_family_sort_key))
    )
    m8 = family_means[family_means["family"] == "m8"]
    m9 = family_means[family_means["family"] == "m9"]
    if not m8.empty and not m9.empty:
        m8_kappa = float(m8.iloc[0]["cohen_kappa_mean"])
        m9_kappa = float(m9.iloc[0]["cohen_kappa_mean"])
        m8_recall = float(m8.iloc[0]["recall_yes_mean"])
        m9_recall = float(m9.iloc[0]["recall_yes_mean"])
        kappa_relation = (
            f"M8 shows stronger agreement than M9 (mean family kappa {m8_kappa:.3f} vs {m9_kappa:.3f})"
            if m8_kappa > m9_kappa
            else f"M9 shows stronger agreement than M8 (mean family kappa {m9_kappa:.3f} vs {m8_kappa:.3f})"
            if m9_kappa > m8_kappa
            else f"M8 and M9 show matched agreement at the family level (mean family kappa {m8_kappa:.3f} vs {m9_kappa:.3f})"
        )
        recall_relation = (
            f"Positive-class recovery is weaker for M9 than for M8 (mean recall for `yes`: {m9_recall:.3f} vs {m8_recall:.3f})"
            if (not pd.isna(m8_recall) and not pd.isna(m9_recall) and m8_recall > m9_recall)
            else f"Positive-class recovery is stronger for M9 than for M8 (mean recall for `yes`: {m9_recall:.3f} vs {m8_recall:.3f})"
            if (not pd.isna(m8_recall) and not pd.isna(m9_recall) and m9_recall > m8_recall)
            else f"Positive-class recovery is matched or unavailable across M8 and M9 (mean recall for `yes`: {m8_recall:.3f} vs {m9_recall:.3f})"
        )
        if m8_kappa > m9_kappa:
            interpretation_tail = "- This pattern is consistent with the prompt design: the system marks `yes` only when the construct is explicitly verbalized in the narrative, so lower M9 alignment is compatible with weaker narrative explicitness rather than a pure model-quality failure."
        elif m9_kappa > m8_kappa:
            interpretation_tail = "- In this run, the questionnaire-based family pattern does not support the usual expectation that M9 is less recoverable than M8, so the result should be interpreted cautiously and checked against item-level detail and prevalence structure."
        else:
            interpretation_tail = "- In this run, the questionnaire-based family comparison does not separate M8 and M9 on agreement, so the interpretation should rely more heavily on item-level detail, recall patterns, and prevalence structure."
        lines.extend(
            [
                f"- Across questionnaire-vs-LLM comparisons, {kappa_relation}.",
                f"- {recall_relation}.",
                interpretation_tail,
                "- The most article-ready interpretation is therefore that lower scores reflect both model limitations and systematic differences in textual recoverability, with reflective aftereffects appearing less recoverable than concrete experiential features.",
                "",
            ]
        )
        return lines
    lines.extend(["- The current run did not contain both M8 and M9 family summaries for interpretation.", ""])
    return lines


def _comparison_scope_lines(summary: dict[str, Any], metrics_df: pd.DataFrame, title: str) -> list[str]:
    scopes = summary.get("comparison_scopes", {})
    comparisons = sorted(metrics_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key)
    lines = [f"### {title}", ""]
    if not comparisons:
        lines.extend(["- No overlap-based comparisons were available in this run.", ""])
        return lines
    for comparison in comparisons:
        scope = scopes.get(comparison, {})
        lines.append(
            f"- {_comparison_label(comparison)}: overlap range across fields = {scope.get('min_overlap_n', 0)}-{scope.get('max_overlap_n', 0)} participants; fields evaluated = {scope.get('fields', 0)}"
        )
    lines.append("")
    return lines


def _comparison_navigation_lines(metrics_df: pd.DataFrame) -> list[str]:
    comparisons = sorted(metrics_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key)
    groups: dict[str, list[str]] = {"human_vs_all": [], "questionnaire_vs_automated": [], "other": []}
    for comparison in comparisons:
        groups[_comparison_tab(comparison)].append(comparison)

    lines = [
        "## Interactive Navigation",
        "",
        "The report uses Markdown + native HTML (`<details>` and anchor links) so navigation is preserved when rendered as HTML, in exported notebooks, and in shared Markdown viewers.",
        "",
        "### Highlighted tabs",
        "",
        "<details open>",
        "<summary><strong>Human vs All</strong></summary>",
        "",
    ]
    if groups["human_vs_all"]:
        for comparison in groups["human_vs_all"]:
            lines.append(f"- [{_comparison_label(comparison)}](#comparison-{comparison})")
    else:
        lines.append("- No human-vs-all comparisons were available in this run.")
    lines.extend(["", "</details>", "", "<details open>", "<summary><strong>Questionnaire vs Automated</strong></summary>", ""])
    if groups["questionnaire_vs_automated"]:
        for comparison in groups["questionnaire_vs_automated"]:
            lines.append(f"- [{_comparison_label(comparison)}](#comparison-{comparison})")
    else:
        lines.append("- No questionnaire-vs-automated comparisons were available in this run.")
    lines.extend(["", "</details>", "", "<details>", "<summary><strong>Other comparisons</strong></summary>", ""])
    if groups["other"]:
        for comparison in groups["other"]:
            lines.append(f"- [{_comparison_label(comparison)}](#comparison-{comparison})")
    else:
        lines.append("- No additional comparisons.")
    lines.extend(["", "</details>", ""])
    return lines


def _metric_toggle_lines(metrics_df: pd.DataFrame) -> list[str]:
    grouped = (
        metrics_df.groupby("comparison", as_index=False)[["accuracy", "cohen_kappa", "macro_f1"]]
        .mean(numeric_only=True)
        .copy()
    )
    grouped["error_rate"] = 1.0 - grouped["accuracy"]

    lines = [
        "## Quick Metric Controls",
        "",
        "Jump between metric-focused views:",
        "",
        "- [Agreement (Cohen kappa)](#metric-agreement-cohen-kappa)",
        "- [Correlation proxy (Macro F1)](#metric-correlation-proxy-macro-f1)",
        "- [Accuracy](#metric-accuracy)",
        "- [Error rate](#metric-error-rate)",
        "",
    ]

    metric_sections = [
        ("metric-agreement-cohen-kappa", "Agreement (Cohen kappa)", "cohen_kappa"),
        ("metric-correlation-proxy-macro-f1", "Correlation proxy (Macro F1)", "macro_f1"),
        ("metric-accuracy", "Accuracy", "accuracy"),
        ("metric-error-rate", "Error rate", "error_rate"),
    ]
    for anchor, title, column in metric_sections:
        section = grouped.sort_values(column, ascending=False, na_position="last")
        lines.extend(
            [
                f"### <a id=\"{anchor}\"></a>{title}",
                "",
                "| Comparison | Mean value |",
                "| --- | ---: |",
            ]
        )
        for _, row in section.iterrows():
            lines.append(f"| {_comparison_label(str(row['comparison']))} | {float(row[column]):.3f} |")
        lines.extend(["", "[Back to metric controls](#quick-metric-controls)", ""])
    return lines


def _field_result_lines(metrics_df: pd.DataFrame, study: StudyConfig) -> list[str]:
    lines = ["## Item-Level Detail", "", "The tables below retain the fine-grained item evidence after the general and family-level summaries.", ""]
    # Sort comparisons by macro_f1 descending
    comparison_means = (
        metrics_df.groupby("comparison", as_index=False)["macro_f1"]
        .mean(numeric_only=True)
        .sort_values("macro_f1", ascending=False, na_position="last")
    )
    for comparison in comparison_means["comparison"].tolist():
        subset = metrics_df[metrics_df["comparison"] == comparison].copy().sort_values("field")
        accordion_open = _comparison_tab(comparison) in {"human_vs_all", "questionnaire_vs_automated"}
        lines.append("<details open>" if accordion_open else "<details>")
        lines.append("")
        lines.append(f"<summary><strong><a id=\"comparison-{comparison}\"></a>{_comparison_label(comparison)}</strong></summary>")
        lines.append("")
        lines.extend([f"### {_comparison_label(comparison)}", "", "| Field | Type | N | Accuracy | Kappa | Macro F1 |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for _, row in subset.iterrows():
            lines.append(
                f"| {_field_display_label(str(row['field']), study)} | {_field_group(str(row['field']))} | {int(row['n'])} | {float(row['accuracy']):.3f} | {float(row['cohen_kappa']):.3f} | {float(row['macro_f1']):.3f} |"
            )
        lines.extend(["", "</details>", ""])
    return lines


def _compact_field_result_lines(metrics_df: pd.DataFrame, study: StudyConfig, heading: str) -> list[str]:
    lines = [f"## {heading}", ""]
    if metrics_df.empty:
        lines.extend(["- No comparisons were available in this run.", ""])
        return lines
    # Sort comparisons by macro_f1 descending
    comparison_means = (
        metrics_df.groupby("comparison", as_index=False)["macro_f1"]
        .mean(numeric_only=True)
        .sort_values("macro_f1", ascending=False, na_position="last")
    )
    for comparison in comparison_means["comparison"].tolist():
        subset = metrics_df[metrics_df["comparison"] == comparison].copy().sort_values(["field"])
        lines.extend([
            f"### <a id=\"comparison-{comparison}\"></a>{_comparison_label(comparison)}",
            "",
            "| Field | Type | N | Accuracy | Kappa | Macro F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for _, row in subset.iterrows():
            lines.append(
                f"| {_field_display_label(str(row['field']), study)} | {_field_group(str(row['field']))} | {int(row['n'])} | {float(row['accuracy']):.3f} | {float(row['cohen_kappa']):.3f} | {float(row['macro_f1']):.3f} |"
            )
        lines.append("")
    return lines


def write_alignment_report_for_scope(
    study: StudyConfig,
    metrics_df: pd.DataFrame,
    family_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    figure_paths: dict[str, str],
    report_filename: str,
    title: str,
    objective: str,
    include_support_sections: bool,
    figure_prefix: str,
    vader_summary: dict[str, Any] | None = None,
) -> Path:
    report_path = output_dir / report_filename
    lines = [
        f"# {title}",
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Methodology",
        "",
        "- Human reference: field-level majority vote across valid human annotation artifacts.",
        "- Unresolved ties remain blank and are excluded only from the affected field metrics.",
        "- Metrics reported: accuracy, Cohen kappa, and macro F1.",
        "- Families are operationalized as Tone, M8, and M9 to support article-oriented interpretation before item-level detail.",
        "",
    ]
    lines.extend(_coverage_lines(summary))
    if vader_summary and report_filename == ALIGNMENT_REPORT_FILENAME:
        lines.extend(
            [
                "### VADER Context",
                "",
                f"- Rows contributing to VADER analysis after filtering: {vader_summary.get('n_rows_after_filters', 'n/a')}",
                f"- VADER filter description: {vader_summary.get('filter_description', 'n/a')}",
                "",
            ]
        )
    # Figures first, then tables
    lines.extend(
        [
            "## Figures",
            "",
            "The figures below show baseline comparisons plus the top 3 LLMs ranked by macro F1.",
            "",
            f"![Overall comparison summary]({Path(figure_paths[f'{figure_prefix}_comparison_summary']).relative_to(output_dir).as_posix()})",
            "",
            f"![Family-level summary]({Path(figure_paths[f'{figure_prefix}_family_summary']).relative_to(output_dir).as_posix()})",
            "",
        ]
    )
    lines.extend(_summary_table_lines(metrics_df, "General Results"))
    lines.extend(_family_summary_lines(family_df, "Family-Level Results"))
    if report_filename == ALIGNMENT_QUESTIONNAIRE_REPORT_FILENAME:
        lines.extend(_questionnaire_interpretation_lines(family_df))
    tone_lines = ["#### Tone", "", f"![Tone summary]({Path(figure_paths[f'{figure_prefix}_tone_summary']).relative_to(output_dir).as_posix()})", ""]
    if figure_prefix == "human":
        tone_lines.extend(
            [
                f"![Tone accuracy heatmap]({Path(figure_paths[f'{figure_prefix}_tone_accuracy_heatmap']).relative_to(output_dir).as_posix()})",
                "",
                f"![Tone alignment]({Path(figure_paths[f'{figure_prefix}_tone_alignment']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
    else:
        tone_lines.extend(
            [
                "In the questionnaire scope, only Experience Tone is available, so the tone section is summarized in a single three-metric figure instead of redundant heatmaps.",
                "",
            ]
        )
    # Detailed heatmaps (also filtered to baselines + top 3 LLMs)
    lines.extend(
        [
            "## Detailed Heatmaps",
            "",
            "These heatmaps show the same filtered comparisons: baseline(s) plus top 3 LLMs ranked by macro F1.",
            "",
            *tone_lines,
            "#### M8",
            "",
            f"![M8 accuracy heatmap]({Path(figure_paths[f'{figure_prefix}_m8_accuracy_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![M8 kappa heatmap]({Path(figure_paths[f'{figure_prefix}_m8_cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            "#### M9",
            "",
            f"![M9 accuracy heatmap]({Path(figure_paths[f'{figure_prefix}_m9_accuracy_heatmap']).relative_to(output_dir).as_posix()})",
            "",
            f"![M9 kappa heatmap]({Path(figure_paths[f'{figure_prefix}_m9_cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
            "",
        ]
    )
    lines.extend(_field_result_lines(metrics_df, study))
    if include_support_sections:
        lines.extend(_human_agreement_lines(summary, output_dir))
        lines.extend(_llm_lines(summary))
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


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
    primary_metrics_df = _comparison_subset(metrics_df, "human_reference_vs_")
    family_df = build_family_summary_table(primary_metrics_df)
    return write_alignment_report_for_scope(
        study=study,
        metrics_df=primary_metrics_df,
        family_df=family_df,
        summary=summary,
        output_dir=output_dir,
        figure_paths=figure_paths,
        report_filename=ALIGNMENT_REPORT_FILENAME,
        title="Alignment Report — Human vs All",
        objective="Quantify alignment across the majority-vote human reference, questionnaire-derived labels, VADER tone labels, and accepted LLM experiments, with the report organized from global results to families and then individual items.",
        include_support_sections=True,
        figure_prefix="human",
        vader_summary=vader_summary,
    )


def write_questionnaire_alignment_report(
    study: StudyConfig,
    metrics_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    figure_paths: dict[str, str],
) -> Path:
    questionnaire_metrics_df = _questionnaire_automated_subset(metrics_df)
    family_df = build_family_summary_table(questionnaire_metrics_df)
    return write_alignment_report_for_scope(
        study=study,
        metrics_df=questionnaire_metrics_df,
        family_df=family_df,
        summary=summary,
        output_dir=output_dir,
        figure_paths=figure_paths,
        report_filename=ALIGNMENT_QUESTIONNAIRE_REPORT_FILENAME,
        title="Alignment Report — Questionnaire vs Automated",
        objective="Summarize questionnaire-based comparisons in a separate article-supporting report so the main human-centered report remains clearer and more interpretable.",
        include_support_sections=False,
        figure_prefix="questionnaire",
        vader_summary=None,
    )


def write_alignment_outputs(
    study: StudyConfig,
    metrics_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    vader_summary: dict[str, Any] | None = None,
    figure_dpi: int = 300,
    export_figures_pdf: bool = False,
) -> dict[str, str]:
    figures_dir = output_dir / ALIGNMENT_FIGURES_SUBDIR
    figure_paths = write_alignment_figures(metrics_df, study, figures_dir, dpi=figure_dpi, export_pdf=export_figures_pdf)
    long_df = build_alignment_long_table(metrics_df)
    family_df = build_family_summary_table(metrics_df)
    long_path = output_dir / ALIGNMENT_LONG_FILENAME
    family_path = output_dir / ALIGNMENT_FAMILY_FILENAME
    long_df.to_csv(long_path, index=False)
    family_df.to_csv(family_path, index=False)
    report_path = write_alignment_report(study, metrics_df, summary, output_dir, figure_paths, vader_summary=vader_summary)
    questionnaire_report_path = write_questionnaire_alignment_report(study, metrics_df, summary, output_dir, figure_paths)
    return {
        "alignment_report_file": str(report_path),
        "alignment_report_questionnaire_file": str(questionnaire_report_path),
        "alignment_metrics_long_file": str(long_path),
        "alignment_family_metrics_file": str(family_path),
        "alignment_figures_dir": str(figures_dir),
        **{f"figure_{name}": path for name, path in figure_paths.items()},
    }
