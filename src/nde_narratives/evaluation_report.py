from __future__ import annotations

import math
import re
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

# Constants for bar label positioning
MIN_OFFSET_FOR_BAR_LABEL = 0.01
SCALE_FACTOR_FOR_BAR_LABEL = 0.04

# Constants for family sort order
UNKNOWN_FAMILY_SORT_ORDER = 99


def _comparison_label(comparison: str) -> str:
    base, _, detail = comparison.partition(":")
    label = base.replace("_vs_", " vs ").replace("_", " ").title()
    if detail:
        return f"{label} ({_presentation_model_name(detail)})"
    return label


def _presentation_model_name(identifier: str) -> str:
    def _normalize_version(version_digits: str) -> str:
        if not version_digits:
            return ""
        if len(version_digits) == 1:
            return version_digits
        major = version_digits[0]
        minor = str(int(version_digits[1:]))
        return f"{major}.{minor}"

    def _extract_size(token: str) -> str | None:
        match = re.fullmatch(r"(\d+)(?:b)?", token)
        if not match:
            return None
        return f"{int(match.group(1))}B"

    def _family_label(name: str) -> str:
        labels = {
            "qwen": "Qwen",
            "gemma": "Gemma",
            "llama": "Llama",
            "claude": "Claude",
            "ministral": "Ministral",
            "mistral": "Mistral",
            "deepseek": "DeepSeek",
            "nemotron": "Nemotron",
        }
        return labels.get(name, name.title())

    raw = str(identifier or "").strip()
    if not raw:
        return raw
    if raw.lower() == "vader":
        return "VADER"

    normalized = raw.replace("-", "_").replace(":", "_").replace("/", "_")
    normalized = normalized.strip("_ ").lower()
    normalized = normalized.replace("__run_", "__").replace("__run-", "__")
    if "__" in normalized:
        normalized = normalized.split("__", 1)[0]
    normalized = re.sub(r"_+", "_", normalized)

    alias_map = {
        "deepseek_r1_32": "DeepSeek-R1 32B",
        "gemma3_27": "Gemma 3 27B",
        "gemma4_26": "Gemma 4 26B",
        "gemma4_31": "Gemma 4 31B",
        "llama31_8": "Llama 3.1 8B",
        "ministral3_14": "Ministral 3 14B",
        "nemotron_3_nano": "Nemotron-3 Nano 30B",
        "qwen35_9": "Qwen 3.5 9B",
        "qwen35_27": "Qwen 3.5 27B",
        "qwen35_35": "Qwen 3.5 35B",
        "qwen36_27": "Qwen 3.6 27B",
        "qwen36_35": "Qwen 3.6 35B",
        "qwen3_32": "Qwen 3 32B",
        "claude3_haiku": "Claude 3 Haiku",
        "claude35_sonnet": "Claude 3.5 Sonnet",
        "llama3_70b": "Llama 3 70B",
        "mistral_large3": "Mistral Large 3",
        "qwen3_32b": "Qwen 3 32B",
    }
    if normalized in alias_map:
        return alias_map[normalized]

    match = re.fullmatch(r"([a-z]+)(\d+)(?:_(.+))?", normalized)
    if match:
        family, version_digits, remainder = match.groups()
        family_text = _family_label(family)
        version_text = _normalize_version(version_digits)
        if remainder:
            remainder_tokens = [token for token in remainder.split("_") if token]
            size_text = _extract_size(remainder_tokens[0]) if remainder_tokens else None
            if size_text:
                tail_tokens = [token.title() for token in remainder_tokens[1:]]
                tail_text = f" {' '.join(tail_tokens)}" if tail_tokens else ""
                return f"{family_text} {version_text} {size_text}{tail_text}".strip()
            remainder_text = " ".join(token.title() for token in remainder_tokens)
            return f"{family_text} {version_text} {remainder_text}".strip()
        return f"{family_text} {version_text}".strip()

    tokens = [token for token in normalized.split("_") if token]
    if tokens:
        head = tokens[0]
        head_match = re.fullmatch(r"([a-z]+)(\d+)", head)
        if head_match and len(tokens) >= 2:
            family, version_digits = head_match.groups()
            family_text = _family_label(family)
            version_text = _normalize_version(version_digits)
            size_text = _extract_size(tokens[1])
            if size_text:
                tail_tokens = [token.title() for token in tokens[2:]]
                tail_text = f" {' '.join(tail_tokens)}" if tail_tokens else ""
                return f"{family_text} {version_text} {size_text}{tail_text}".strip()

    fallback = normalized.replace("_", " ").strip()
    return fallback.title()


def _wrap_label(label: str, width: int = 24) -> str:
    return fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def _format_pct(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def _bar_label_y(
    value: float,
    min_offset: float = MIN_OFFSET_FOR_BAR_LABEL,
    scale: float = SCALE_FACTOR_FOR_BAR_LABEL,
) -> float:
    return value + max(min_offset, abs(value) * scale)


def _field_group(field: str) -> str:
    return "tone" if field.endswith("_tone") else "binary"


def _field_display_label(field: str, study: StudyConfig) -> str:
    return study.internal_to_visible_annotation_columns().get(field, field)


def _field_bucket(field: str, study: StudyConfig) -> str:
    if field.endswith("_tone"):
        return "tone"
    if field in study.sections["experience"].binary_labels:
        return "m8"
    if field in study.sections["aftereffects"].binary_labels:
        return "m9"
    return "other"


def _field_bucket_title(bucket: str) -> str:
    titles = {
        "tone": "Tone",
        "m8": "NDE-C (Content of the Near-Death Experience Scale)",
        "m9": "LCI-R (Long-term Changes Inventory-Revised)",
        "other": "Other",
    }
    return titles.get(bucket, bucket.replace("_", " ").title())


def _family_sort_key(bucket: str) -> tuple[int, str]:
    order = {"tone": 0, "m8": 1, "m9": 2, "other": 3}
    return (order.get(bucket, UNKNOWN_FAMILY_SORT_ORDER), bucket)


def _field_bucket_order(study: StudyConfig) -> dict[str, list[str]]:
    experience_fields = list(study.sections["experience"].binary_labels.keys())
    aftereffects_fields = list(study.sections["aftereffects"].binary_labels.keys())
    return {
        "tone": list(study.tone_columns()),
        "m8": experience_fields,
        "m9": aftereffects_fields,
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


def _select_top_comparisons_by_pareto(
    comparison_means: pd.DataFrame,
    *,
    top_n: int,
    metric_x: str,
    metric_y: str,
) -> list[str]:
    if comparison_means.empty:
        return []

    work = comparison_means[["comparison", metric_x, metric_y]].copy()
    work = work[work[metric_x].notna() & work[metric_y].notna()].reset_index(drop=True)
    if work.empty:
        return []

    selected: list[str] = []
    remaining = work.copy()
    while not remaining.empty and len(selected) < top_n:
        idx = remaining.index.to_list()
        frontier: list[int] = []
        for i in idx:
            i_x = float(remaining.at[i, metric_x])
            i_y = float(remaining.at[i, metric_y])
            dominated = False
            for j in idx:
                if i == j:
                    continue
                j_x = float(remaining.at[j, metric_x])
                j_y = float(remaining.at[j, metric_y])
                if (j_x >= i_x and j_y >= i_y) and (j_x > i_x or j_y > i_y):
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)

        front_df = remaining.loc[frontier].copy()
        span_x = float(front_df[metric_x].max() - front_df[metric_x].min())
        span_y = float(front_df[metric_y].max() - front_df[metric_y].min())
        front_df["_x_norm"] = 0.5 if span_x == 0 else (front_df[metric_x] - float(front_df[metric_x].min())) / span_x
        front_df["_y_norm"] = 0.5 if span_y == 0 else (front_df[metric_y] - float(front_df[metric_y].min())) / span_y
        front_df["_balance"] = front_df[["_x_norm", "_y_norm"]].min(axis=1)
        front_df = front_df.sort_values(["_balance", metric_x, metric_y], ascending=[False, False, False], na_position="last")
        selected.extend(front_df["comparison"].tolist())
        remaining = remaining.drop(index=frontier)

    return selected[:top_n]


def _select_top_comparisons_for_figure(
    metrics_df: pd.DataFrame,
    scope_prefix: str,
    baseline_comparisons: list[str],
    top_n: int = 3,
    ranking_metric: str = "macro_f1",
) -> list[str]:
    """Select comparisons for figure display: always include baselines, then top N LLMs by Pareto ranking.

    Args:
        metrics_df: Full metrics dataframe.
        scope_prefix: Scope prefix (e.g., "questionnaire_vs_" or "human_reference_vs_").
        baseline_comparisons: List of baseline comparison names to always include.
        top_n: Number of top LLM comparisons to include.
        ranking_metric: Retained for API compatibility; TOP selection uses macro_f1 and cohen_kappa jointly.

    Returns:
        List of comparison names to display in figures.
    """
    scoped_df = metrics_df[
        metrics_df["comparison"].astype(str).str.startswith(scope_prefix)
    ].copy()
    if scoped_df.empty:
        return baseline_comparisons

    # Get unique comparisons
    all_comparisons = scoped_df["comparison"].drop_duplicates().tolist()

    # Separate baselines from LLMs
    baseline_set = set(baseline_comparisons)
    baselines_present = [c for c in baseline_comparisons if c in all_comparisons]
    llm_comparisons = [c for c in all_comparisons if c not in baseline_set]

    if "macro_f1" not in scoped_df.columns or "cohen_kappa" not in scoped_df.columns:
        return baselines_present
    comparison_means = scoped_df.groupby("comparison", as_index=False)[["macro_f1", "cohen_kappa"]].mean(numeric_only=True)

    # Select top N LLMs by Pareto-front ranking on macro_f1 + cohen_kappa.
    llm_means = comparison_means[comparison_means["comparison"].isin(llm_comparisons)].copy()
    top_llms = _select_top_comparisons_by_pareto(
        llm_means,
        top_n=top_n,
        metric_x="macro_f1",
        metric_y="cohen_kappa",
    )

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
                    "value": float(row[metric_name])
                    if not pd.isna(row[metric_name])
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _comparison_subset(metrics_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return metrics_df[
        metrics_df["comparison"].astype(str).str.startswith(prefix)
    ].copy()


def _questionnaire_automated_subset(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df[
        metrics_df["comparison"].astype(str).str.startswith("questionnaire_vs_")
    ].copy()


def _style_axes(ax) -> None:
    ax.set_facecolor("#F7F4EA")
    ax.grid(axis="x", color="#DDD6C8", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _legend_title_with_count(base: str, count: int) -> str:
    return f"{base} ({count})"


def _save_figure(
    fig, figure_path: Path, dpi: int = 300, export_pdf: bool = False
) -> list[str]:
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


def _questionnaire_contradiction_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload = summary.get("questionnaire_contradictions", {})
    return payload if isinstance(payload, dict) else {}


def _questionnaire_tone_label_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload = summary.get("questionnaire_tone_label_analysis", {})
    return payload if isinstance(payload, dict) else {}


def _comparison_rank_map(payload: dict[str, Any]) -> dict[str, int]:
    selected = payload.get("selected_comparisons", [])
    if not isinstance(selected, list):
        return {}
    return {str(comparison): index for index, comparison in enumerate(selected)}


def _sort_rows_by_selected_order(
    rows: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    rank_map = _comparison_rank_map(payload)
    if not rank_map:
        return rows
    return sorted(
        rows,
        key=lambda row: (
            rank_map.get(str(row.get("comparison", "")), 999),
            str(row.get("comparison", "")),
        ),
    )


def plot_questionnaire_tone_confusion_matrix(
    tone_payload: dict[str, Any],
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    confusion_rows = tone_payload.get("confusion", [])
    if not isinstance(confusion_rows, list) or not confusion_rows:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No tone confusion data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    labels = tone_payload.get("labels", [])
    if not isinstance(labels, list) or not labels:
        labels = sorted(
            {
                str(row.get("questionnaire_label", ""))
                for row in confusion_rows
                if str(row.get("questionnaire_label", ""))
            }
        )

    confusion_df = pd.DataFrame(confusion_rows)
    if confusion_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No tone confusion data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    preferred_order = ["positive", "negative", "mixed", "neutral"]
    labels = sorted(
        {str(label).lower() for label in labels if str(label).strip()},
        key=lambda value: (
            preferred_order.index(value)
            if value in preferred_order
            else len(preferred_order)
        ),
    )

    # Keep only observed questionnaire rows and observed automated columns.
    row_labels = [
        label
        for label in labels
        if int(
            confusion_df.loc[
                confusion_df["questionnaire_label"] == label, "count"
            ].sum()
        )
        > 0
    ]
    col_labels = [
        label
        for label in labels
        if int(
            confusion_df.loc[confusion_df["candidate_label"] == label, "count"].sum()
        )
        > 0
    ]
    if not row_labels:
        row_labels = labels
    if not col_labels:
        col_labels = labels

    per_label_df = pd.DataFrame(tone_payload.get("per_label", []))

    selected_order = _comparison_rank_map(tone_payload)
    comparisons = confusion_df["comparison"].drop_duplicates().tolist()
    if selected_order:
        comparisons = [
            comparison
            for comparison in comparisons
            if str(comparison) in selected_order
        ]
    comparisons = sorted(
        comparisons,
        key=lambda value: (
            selected_order.get(str(value), 999),
            _comparison_sort_key(str(value)),
        ),
    )
    n_panels = max(1, len(comparisons))
    n_cols = 1 if n_panels == 1 else 2
    n_rows = int(math.ceil(n_panels / n_cols))
    fig = plt.figure(figsize=(6.2 * n_cols + 2.0, 4.8 * n_rows + 0.8))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[*([1.0] * n_cols), 0.055],
        wspace=0.12,
        hspace=0.24,
    )
    axes_list = [
        fig.add_subplot(gs[row, col]) for row in range(n_rows) for col in range(n_cols)
    ]
    cax = fig.add_subplot(gs[:, n_cols])

    used_axes = axes_list[:n_panels]
    for panel_index, (ax, comparison) in enumerate(
        zip(used_axes, comparisons, strict=False)
    ):
        subset = confusion_df[confusion_df["comparison"] == comparison].copy()
        rate_matrix = subset.pivot(
            index="questionnaire_label", columns="candidate_label", values="row_rate"
        )
        rate_matrix = rate_matrix.reindex(index=row_labels, columns=col_labels)
        count_matrix = subset.pivot(
            index="questionnaire_label", columns="candidate_label", values="count"
        )
        count_matrix = count_matrix.reindex(index=row_labels, columns=col_labels)
        values = rate_matrix.to_numpy(dtype=float)
        counts = count_matrix.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="YlGnBu", aspect="equal", vmin=0.0, vmax=1.0)
        panel_row = panel_index // n_cols
        panel_col = panel_index % n_cols
        is_bottom_row = panel_row == (n_rows - 1)
        is_left_col = panel_col == 0
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(
            [_wrap_label(str(label).title(), width=10) for label in col_labels],
            rotation=0,
            ha="center",
            fontsize=15,
        )
        ax.set_yticklabels(
            [_wrap_label(str(label).title(), width=10) for label in row_labels],
            fontsize=15,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        if not is_left_col:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        comparison_key = str(comparison)
        if comparison_key == "questionnaire_vs_vader":
            panel_label = "VADER"
        elif comparison_key.startswith("questionnaire_vs_llm:"):
            panel_label = _presentation_model_name(comparison_key.split(":", 1)[1])
        else:
            panel_label = _comparison_label(comparison_key)
        title_text = _wrap_label(panel_label, width=28)

        if not per_label_df.empty:
            comparison_rows = per_label_df[
                per_label_df["comparison"] == comparison
            ].copy()
            label_f1_map = {
                str(row["label"]).lower(): float(row["f1"])
                for _, row in comparison_rows.iterrows()
            }

            def _fmt_f1(label: str) -> str:
                value = label_f1_map.get(label)
                if value is None or math.isnan(value):
                    return "n/a"
                return f"{value:.2f}"

            title_text = (
                f"{title_text}\n"
                f"F1 pos={_fmt_f1('positive')} · neg={_fmt_f1('negative')} · "
                f"mix={_fmt_f1('mixed')}"
            )

        ax.set_title(title_text, fontsize=15, pad=6)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if not math.isnan(value):
                    count = counts[i, j]
                    count_text = f"{int(count)}" if not math.isnan(count) else "?"
                    ax.text(
                        j,
                        i,
                        f"{count_text}\n({_format_pct(value)})",
                        ha="center",
                        va="center",
                        color="#FFFFFF" if float(value) >= 0.50 else "#1F2933",
                        fontsize=16,
                    )

    for extra_ax in axes_list[n_panels:]:
        extra_ax.axis("off")

    fig.text(
        0.02,
        0.5,
        "Questionnaire label",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=16,
    )
    fig.text(
        0.50,
        0.02,
        "Automated label",
        va="center",
        ha="center",
        fontsize=15,
    )
    cbar = fig.colorbar(image, cax=cax)
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cbar.set_label("Row-normalized rate", fontsize=17)
    cbar.ax.tick_params(labelsize=15)
    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.10, right=0.985, hspace=0.26)
    fig.patch.set_facecolor("#FFFFFF")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_questionnaire_tone_flow(
    tone_payload: dict[str, Any],
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    """Render an alluvial-style flow chart for questionnaire→automated tone transitions."""
    confusion_rows = tone_payload.get("confusion", [])
    if not isinstance(confusion_rows, list) or not confusion_rows:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No tone flow data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    confusion_df = pd.DataFrame(confusion_rows)
    if confusion_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No tone flow data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    labels = tone_payload.get("labels", [])
    if not isinstance(labels, list) or not labels:
        labels = sorted(
            {
                str(row.get("questionnaire_label", ""))
                for row in confusion_rows
                if str(row.get("questionnaire_label", ""))
            }
        )

    row_labels = [
        label
        for label in labels
        if int(
            confusion_df.loc[
                confusion_df["questionnaire_label"] == label, "count"
            ].sum()
        )
        > 0
    ]
    col_labels = [
        label
        for label in labels
        if int(
            confusion_df.loc[confusion_df["candidate_label"] == label, "count"].sum()
        )
        > 0
    ]
    if not row_labels:
        row_labels = labels
    if not col_labels:
        col_labels = labels

    selected_order = _comparison_rank_map(tone_payload)
    comparisons = confusion_df["comparison"].drop_duplicates().tolist()
    comparisons = sorted(
        comparisons,
        key=lambda value: (
            selected_order.get(str(value), 999),
            _comparison_sort_key(str(value)),
        ),
    )
    if not comparisons:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No tone flow data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    n_panels = len(comparisons)
    n_cols = min(2, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10.2 * n_cols, 3.9 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    left_y = {
        label: 1.0 - (index + 1) / (len(row_labels) + 1)
        for index, label in enumerate(row_labels)
    }
    right_y = {
        label: 1.0 - (index + 1) / (len(col_labels) + 1)
        for index, label in enumerate(col_labels)
    }
    flow_palette = {
        "positive": "#2A9D8F",
        "negative": "#B23A48",
        "mixed": "#6D597A",
        "neutral": "#457B9D",
    }

    for panel_index, (ax, comparison) in enumerate(
        zip(axes_flat, comparisons, strict=False)
    ):
        subset = confusion_df[confusion_df["comparison"] == comparison].copy()
        subset = subset[
            (subset["questionnaire_label"].isin(row_labels))
            & (subset["candidate_label"].isin(col_labels))
        ]
        max_count = float(subset["count"].max()) if not subset.empty else 1.0
        max_count = max(1.0, max_count)

        for label in row_labels:
            y = left_y[label]
            ax.scatter(
                [0.10],
                [y],
                s=190,
                color=flow_palette.get(str(label), "#355070"),
                edgecolors="#F8F5F0",
                linewidths=0.8,
                zorder=4,
            )
            ax.text(
                0.06,
                y,
                str(label).title(),
                ha="right",
                va="center",
                fontsize=9,
                color="#2B2D42",
            )
        for label in col_labels:
            y = right_y[label]
            ax.scatter(
                [0.90],
                [y],
                s=190,
                color=flow_palette.get(str(label), "#355070"),
                edgecolors="#F8F5F0",
                linewidths=0.8,
                zorder=4,
            )
            ax.text(
                0.94,
                y,
                str(label).title(),
                ha="left",
                va="center",
                fontsize=9,
                color="#2B2D42",
            )

        visible_rows: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            source = str(row["questionnaire_label"])
            target = str(row["candidate_label"])
            count = float(row.get("count", 0.0))
            if count <= 0 or source not in left_y or target not in right_y:
                continue
            rate = float(row.get("row_rate", float("nan")))
            lw = 1.2 + 10.0 * (count / max_count)
            color = flow_palette.get(source, "#355070")
            alpha = 0.25 + 0.55 * (count / max_count)
            ax.plot(
                [0.10, 0.90],
                [left_y[source], right_y[target]],
                color=color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=2,
            )
            visible_rows.append(
                {
                    "source": source,
                    "target": target,
                    "count": count,
                    "rate": rate,
                    "y_mid": (left_y[source] + right_y[target]) / 2,
                }
            )

        visible_rows = sorted(
            visible_rows, key=lambda item: item["count"], reverse=True
        )
        for label_index, item in enumerate(visible_rows[:4]):
            if pd.isna(item["rate"]) or float(item["rate"]) < 0.18:
                continue
            jitter = ((label_index % 2) * 0.018) - 0.009
            y_text = min(0.96, max(0.06, float(item["y_mid"]) + jitter))
            ax.text(
                0.52,
                y_text,
                f"{str(item['source']).title()}→{str(item['target']).title()}: {int(item['count'])} ({_format_pct(float(item['rate']))})",
                ha="center",
                va="center",
                fontsize=7.6,
                color="#1F2933",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "#FFFDF8",
                    "edgecolor": "#DDD6C8",
                    "alpha": 0.88,
                },
                zorder=5,
            )

        ax.text(
            0.10,
            1.02,
            "Questionnaire",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2B2D42",
        )
        ax.text(
            0.90,
            1.02,
            "Automated",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2B2D42",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.06)
        ax.axis("off")
        ax.set_title(
            _wrap_label(_comparison_label(str(comparison)), width=30), fontsize=10
        )

    for extra_ax in axes_flat[n_panels:]:
        extra_ax.axis("off")

    fig.suptitle(
        "Experience Tone flow (questionnaire → automated)", fontsize=13, y=0.985
    )
    fig.subplots_adjust(
        top=0.90, bottom=0.08, left=0.03, right=0.98, wspace=0.18, hspace=0.24
    )
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def _questionnaire_tone_confusion_table_lines(payload: dict[str, Any]) -> list[str]:
    confusion_rows = payload.get("confusion", [])
    if not isinstance(confusion_rows, list) or not confusion_rows:
        return []

    confusion_df = pd.DataFrame(confusion_rows)
    if confusion_df.empty:
        return []

    labels = payload.get("labels", [])
    if not isinstance(labels, list) or not labels:
        labels = sorted(
            {
                str(row.get("questionnaire_label", ""))
                for row in confusion_rows
                if str(row.get("questionnaire_label", ""))
            }
        )

    row_labels = [
        label
        for label in labels
        if int(
            confusion_df.loc[
                confusion_df["questionnaire_label"] == label, "count"
            ].sum()
        )
        > 0
    ]
    col_labels = [
        label
        for label in labels
        if int(
            confusion_df.loc[confusion_df["candidate_label"] == label, "count"].sum()
        )
        > 0
    ]
    if not row_labels:
        row_labels = labels
    if not col_labels:
        col_labels = labels

    selected_order = _comparison_rank_map(payload)
    comparisons = confusion_df["comparison"].drop_duplicates().tolist()
    comparisons = sorted(
        comparisons,
        key=lambda value: (
            selected_order.get(str(value), 999),
            _comparison_sort_key(str(value)),
        ),
    )

    lines: list[str] = ["#### Tone Confusion Table (count + row %)", ""]
    for comparison in comparisons:
        subset = confusion_df[confusion_df["comparison"] == comparison].copy()
        rate_matrix = subset.pivot(
            index="questionnaire_label", columns="candidate_label", values="row_rate"
        )
        rate_matrix = rate_matrix.reindex(index=row_labels, columns=col_labels)
        count_matrix = subset.pivot(
            index="questionnaire_label", columns="candidate_label", values="count"
        )
        count_matrix = count_matrix.reindex(index=row_labels, columns=col_labels)

        lines.extend(
            [
                f"**{_comparison_label(str(comparison))}**",
                "",
                "| Questionnaire \\ Automated | "
                + " | ".join(str(label).title() for label in col_labels)
                + " |",
                "| --- | " + " | ".join(["---:"] * len(col_labels)) + " |",
            ]
        )

        for row_label in row_labels:
            row_cells: list[str] = []
            for col_label in col_labels:
                count = count_matrix.loc[row_label, col_label]
                rate = rate_matrix.loc[row_label, col_label]
                if pd.isna(count) and pd.isna(rate):
                    row_cells.append("n/a")
                else:
                    count_text = str(int(count)) if pd.notna(count) else "?"
                    row_cells.append(f"{count_text} ({_format_pct(float(rate))})")
            lines.append(
                f"| {str(row_label).title()} | " + " | ".join(row_cells) + " |"
            )
        lines.append("")

    return lines


def plot_questionnaire_contradiction_overview(
    contradiction_payload: dict[str, Any],
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    overview = pd.DataFrame(contradiction_payload.get("overview", []))
    if overview.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No contradiction data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    overview = overview.copy()
    overview["comparison_label"] = overview["comparison"].map(_comparison_label)
    overview = overview.sort_values("contradiction_rate", ascending=False)
    labels = [
        _wrap_label(value, width=26) for value in overview["comparison_label"].tolist()
    ]
    x = list(range(len(overview)))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, max(4.8, 1.1 * len(overview) + 2.6)))
    bars = axes[0].barh(
        x,
        overview["n_contradictions"],
        color="#B23A48",
        edgecolor="#F8F5F0",
        linewidth=0.8,
    )
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Contradictory rows (count)")
    axes[0].set_title("Strict polarity contradictions")
    _style_axes(axes[0])
    for bar, value in zip(bars, overview["n_contradictions"], strict=False):
        axes[0].text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            ha="left",
            fontsize=8,
            color="#2B2D42",
        )

    rates = overview["contradiction_rate"].fillna(0.0).astype(float)
    bars_rate = axes[1].barh(
        x, rates, color="#2A6F97", edgecolor="#F8F5F0", linewidth=0.8
    )
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0.0, max(0.25, float(rates.max()) * 1.2 if len(rates) else 0.25))
    axes[1].set_xlabel("Contradiction rate")
    axes[1].set_title("Rate among positive/negative questionnaire rows")
    _style_axes(axes[1])
    for bar, value in zip(bars_rate, rates, strict=False):
        axes[1].text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left",
            fontsize=8,
            color="#2B2D42",
        )

    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def _draw_wordcloud_like(ax, terms: list[dict[str, Any]]) -> None:
    ax.set_facecolor("#F7F4EA")
    ax.axis("off")
    if not terms:
        ax.text(
            0.5,
            0.5,
            "No unigram evidence terms available",
            ha="center",
            va="center",
            fontsize=12,
            color="#2B2D42",
        )
        return

    top_terms = terms[:45]
    max_count = max(int(item.get("count", 1)) for item in top_terms)
    palette = ["#355070", "#6D597A", "#B56576", "#2A9D8F", "#457B9D", "#B08968"]

    def _box_for(
        x: float, y: float, text: str, size: float
    ) -> tuple[float, float, float, float]:
        # Approximate text bbox in axes coordinates.
        width = min(0.46, max(0.055, 0.0019 * size * max(3, len(text))))
        height = min(0.10, max(0.028, 0.0033 * size))
        return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)

    def _overlaps(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        margin: float = 0.008,
    ) -> bool:
        return not (
            a[2] + margin < b[0]
            or a[0] - margin > b[2]
            or a[3] + margin < b[1]
            or a[1] - margin > b[3]
        )

    placed_boxes: list[tuple[float, float, float, float]] = []
    golden = math.pi * (3 - math.sqrt(5))

    for index, item in enumerate(top_terms):
        term = str(item.get("term") or item.get("ngram") or "").strip()
        if not term:
            continue
        count = int(item.get("count", 1))
        rank_factor = count / max_count
        size = 11 + (22 * rank_factor)

        placed = False
        for step in range(1, 900):
            radius = 0.01 + 0.46 * math.sqrt(step / 900)
            angle = step * golden
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            if x < 0.08 or x > 0.92 or y < 0.10 or y > 0.90:
                continue

            box = _box_for(x, y, term, size)
            if box[0] < 0.03 or box[2] > 0.97 or box[1] < 0.04 or box[3] > 0.96:
                continue
            if any(_overlaps(box, other) for other in placed_boxes):
                continue

            ax.text(
                x,
                y,
                term,
                fontsize=size,
                rotation=0,
                ha="center",
                va="center",
                color=palette[index % len(palette)],
                alpha=0.94,
            )
            placed_boxes.append(box)
            placed = True
            break

        if not placed:
            # Safe fallback near lower band to avoid hard failure.
            fallback_x = 0.08 + 0.84 * ((index % 10) / 9 if 9 else 0)
            fallback_y = 0.08 + 0.05 * (index // 10)
            ax.text(
                fallback_x,
                fallback_y,
                term,
                fontsize=max(9, min(13, size * 0.55)),
                rotation=0,
                ha="center",
                va="center",
                color=palette[index % len(palette)],
                alpha=0.85,
            )


def plot_questionnaire_unigram_wordcloud(
    contradiction_payload: dict[str, Any],
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    terms_by_direction = contradiction_payload.get("ngrams", {}).get(
        "wordcloud_terms_by_direction", {}
    )
    if isinstance(terms_by_direction, dict) and terms_by_direction:
        qpos_lneg = terms_by_direction.get("q_positive_llm_negative", [])
        qneg_lpos = terms_by_direction.get("q_negative_llm_positive", [])
        fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.8))
        _draw_wordcloud_like(axes[0], qpos_lneg if isinstance(qpos_lneg, list) else [])
        axes[0].set_title("Questionnaire POSITIVE → LLM NEGATIVE")
        _draw_wordcloud_like(axes[1], qneg_lpos if isinstance(qneg_lpos, list) else [])
        axes[1].set_title("Questionnaire NEGATIVE → LLM POSITIVE")
        fig.suptitle("Contradiction evidence word cloud (unigrams)")
    else:
        terms = contradiction_payload.get("ngrams", {}).get("wordcloud_terms", [])
        fig, ax = plt.subplots(figsize=(12.5, 6.8))
        _draw_wordcloud_like(ax, terms if isinstance(terms, list) else [])
        ax.set_title("Contradiction evidence word cloud (unigrams)")
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_questionnaire_ngram_panels(
    contradiction_payload: dict[str, Any],
    figure_path: Path,
    ngram_key: str,
    title: str,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    per_comparison = contradiction_payload.get("ngrams", {}).get("per_comparison", [])
    if not isinstance(per_comparison, list):
        per_comparison = []
    selected = per_comparison[:3]

    if not selected:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.text(
            0.5,
            0.5,
            "No n-gram evidence data available",
            ha="center",
            va="center",
            fontsize=12,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    fig, axes = plt.subplots(len(selected), 1, figsize=(12.5, 3.5 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, row in zip(axes, selected, strict=False):
        direction_block = row.get("by_direction", {}) if isinstance(row, dict) else {}
        qpos_lneg_items = (
            direction_block.get("q_positive_llm_negative", {}).get(ngram_key, [])
            if isinstance(direction_block, dict)
            else []
        )
        qneg_lpos_items = (
            direction_block.get("q_negative_llm_positive", {}).get(ngram_key, [])
            if isinstance(direction_block, dict)
            else []
        )

        if not isinstance(qpos_lneg_items, list):
            qpos_lneg_items = []
        if not isinstance(qneg_lpos_items, list):
            qneg_lpos_items = []

        tagged_items: list[tuple[str, int, str]] = []
        tagged_items.extend(
            [
                (
                    f"Q+→L- | {str(item.get('term') or item.get('ngram') or '')}",
                    int(item.get("count", 0)),
                    "#B23A48",
                )
                for item in qpos_lneg_items[:6]
            ]
        )
        tagged_items.extend(
            [
                (
                    f"Q-→L+ | {str(item.get('term') or item.get('ngram') or '')}",
                    int(item.get("count", 0)),
                    "#2A6F97",
                )
                for item in qneg_lpos_items[:6]
            ]
        )

        if not tagged_items:
            fallback_items = row.get(ngram_key, []) if isinstance(row, dict) else []
            if not isinstance(fallback_items, list):
                fallback_items = []
            tagged_items = [
                (
                    str(item.get("term") or item.get("ngram") or ""),
                    int(item.get("count", 0)),
                    "#6C9A8B",
                )
                for item in fallback_items[:10]
            ]

        labels = [_wrap_label(item[0], width=34) for item in tagged_items]
        values = [item[1] for item in tagged_items]
        colors = [item[2] for item in tagged_items]
        y = list(range(len(tagged_items)))
        ax.barh(y, values, color=colors, edgecolor="#F8F5F0", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(
            _wrap_label(_comparison_label(str(row.get("comparison", ""))), width=46),
            fontsize=10,
        )
        _style_axes(ax)
        for index, value in enumerate(values):
            ax.text(
                value + 0.1,
                index,
                str(value),
                va="center",
                ha="left",
                fontsize=8,
                color="#2B2D42",
            )

    fig.suptitle(title, fontsize=13)
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def write_questionnaire_contradiction_figures(
    summary: dict[str, Any],
    figures_dir: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> dict[str, str]:
    contradiction_payload = _questionnaire_contradiction_payload(summary)
    if not contradiction_payload:
        return {}

    overview_path = figures_dir / "questionnaire_contradiction_overview.png"
    wordcloud_path = figures_dir / "questionnaire_contradiction_unigram_wordcloud.png"
    bigrams_path = figures_dir / "questionnaire_contradiction_bigrams.png"
    trigrams_path = figures_dir / "questionnaire_contradiction_trigrams.png"

    plot_questionnaire_contradiction_overview(
        contradiction_payload, overview_path, dpi=dpi, export_pdf=export_pdf
    )
    plot_questionnaire_unigram_wordcloud(
        contradiction_payload, wordcloud_path, dpi=dpi, export_pdf=export_pdf
    )
    plot_questionnaire_ngram_panels(
        contradiction_payload,
        bigrams_path,
        ngram_key="bigrams",
        title="Top contradiction evidence bigrams",
        dpi=dpi,
        export_pdf=export_pdf,
    )
    plot_questionnaire_ngram_panels(
        contradiction_payload,
        trigrams_path,
        ngram_key="trigrams",
        title="Top contradiction evidence trigrams",
        dpi=dpi,
        export_pdf=export_pdf,
    )

    return {
        "questionnaire_contradiction_overview": str(overview_path),
        "questionnaire_contradiction_unigram_wordcloud": str(wordcloud_path),
        "questionnaire_contradiction_bigrams": str(bigrams_path),
        "questionnaire_contradiction_trigrams": str(trigrams_path),
    }


def write_questionnaire_tone_label_figures(
    summary: dict[str, Any],
    figures_dir: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> dict[str, str]:
    tone_payload = _questionnaire_tone_label_payload(summary)
    if not tone_payload:
        return {}

    confusion_path = figures_dir / "questionnaire_tone_confusion_matrix.png"
    plot_questionnaire_tone_confusion_matrix(
        tone_payload, confusion_path, dpi=dpi, export_pdf=export_pdf
    )
    return {
        "questionnaire_tone_confusion_matrix": str(confusion_path),
    }


def plot_questionnaire_family_tradeoff_map(
    family_df: pd.DataFrame,
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    """Main questionnaire figure in one panel: NDE-C + LCI-R + Tone."""
    if family_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No family summary data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    source_df = family_df[
        family_df["comparison"].astype(str).str.startswith("questionnaire_vs_llm:")
    ].copy()
    if source_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No questionnaire-vs-LLM family data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    rank_df = source_df.copy().rename(
        columns={"cohen_kappa_mean": "cohen_kappa", "macro_f1_mean": "macro_f1"}
    )
    comparisons = _select_top_comparisons_for_figure(
        rank_df,
        "questionnaire_vs_",
        baseline_comparisons=[],
        top_n=10,
        ranking_metric="macro_f1",
    )
    selected_llm = [
        comparison
        for comparison in comparisons
        if comparison.startswith("questionnaire_vs_llm:")
    ]
    plot_df = source_df[source_df["comparison"].isin(selected_llm)].copy()
    plot_df = plot_df[plot_df["family"].isin(["m8", "m9", "tone"])].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No family slices available for scatter figure",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFDF8")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    fig, ax = plt.subplots(figsize=(13.6, 8.2))

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]
    marker_map = {
        comparison: marker_cycle[index % len(marker_cycle)]
        for index, comparison in enumerate(selected_llm)
    }
    legend_handles = []
    legend_labels = []

    family_colors = {
        "m8": "#2A9D8F",
        "m9": "#E76F51",
        "tone": "#4C78A8",
    }
    family_labels = {
        "m8": "NDE-C (Content of the Near-Death Experience Scale)",
        "m9": "LCI-R (Long-term Changes Inventory-Revised)",
        "tone": "Experience Tone",
    }
    for _, row in plot_df.iterrows():
        comparison_name = str(row["comparison"])
        family = str(row["family"])
        x = (
            float(row["cohen_kappa_mean"])
            if pd.notna(row["cohen_kappa_mean"])
            else float("nan")
        )
        y = (
            float(row["macro_f1_mean"])
            if pd.notna(row["macro_f1_mean"])
            else float("nan")
        )
        if math.isnan(x) or math.isnan(y):
            continue
        if family == "tone":
            size = 250
        else:
            recall_yes = (
                float(row["recall_yes_mean"])
                if pd.notna(row["recall_yes_mean"])
                else 0.35
            )
            size = 180 + 720 * max(0.0, min(1.0, recall_yes))
        marker = marker_map.get(comparison_name, "o")
        ax.scatter(
            [x],
            [y],
            s=size,
            color=family_colors.get(family, "#6B7280"),
            marker=marker,
            alpha=0.84,
            edgecolors="#F8F5F0",
            linewidths=0.9,
        )

        if comparison_name not in legend_labels:
            handle = plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor="#9AA0A6",
                markeredgecolor="#2B2D42",
                markersize=8.5,
                linewidth=0,
            )
            legend_handles.append(handle)
            legend_labels.append(comparison_name)

    # Add VADER baseline for tone reference
    vader_tone = family_df[
        (family_df["comparison"] == "questionnaire_vs_vader")
        & (family_df["family"] == "tone")
    ]
    vader_kappa = (
        float(vader_tone.iloc[0]["cohen_kappa_mean"])
        if not vader_tone.empty and pd.notna(vader_tone.iloc[0]["cohen_kappa_mean"])
        else float("nan")
    )
    vader_macro_f1 = (
        float(vader_tone.iloc[0]["macro_f1_mean"])
        if not vader_tone.empty and pd.notna(vader_tone.iloc[0]["macro_f1_mean"])
        else float("nan")
    )
    if not math.isnan(vader_kappa):
        ax.axvline(
            vader_kappa, color="#5E6472", linestyle="--", linewidth=1.0, alpha=0.85
        )
    if not math.isnan(vader_macro_f1):
        ax.axhline(
            vader_macro_f1, color="#5E6472", linestyle=":", linewidth=1.0, alpha=0.85
        )
    if not math.isnan(vader_kappa) and not math.isnan(vader_macro_f1):
        ax.scatter(
            [vader_kappa],
            [vader_macro_f1],
            s=220,
            color="#5E6472",
            marker="*",
            edgecolors="#F8F5F0",
            linewidths=0.8,
            zorder=4,
        )

    all_xy = plot_df[["cohen_kappa_mean", "macro_f1_mean"]].dropna()
    x_min = float(all_xy["cohen_kappa_mean"].min())
    x_max = float(all_xy["cohen_kappa_mean"].max())
    y_min = float(all_xy["macro_f1_mean"].min())
    y_max = float(all_xy["macro_f1_mean"].max())
    if not math.isnan(vader_kappa):
        x_min = min(x_min, vader_kappa)
        x_max = max(x_max, vader_kappa)
    if not math.isnan(vader_macro_f1):
        y_min = min(y_min, vader_macro_f1)
        y_max = max(y_max, vader_macro_f1)
    x_pad = max(0.010, (x_max - x_min) * 0.12)
    y_pad = max(0.012, (y_max - y_min) * 0.12)

    ax.set_xlabel("Mean Cohen kappa", fontsize=15)
    ax.set_ylabel("Mean Macro F1", fontsize=15)
    ax.set_xlim(max(0.00, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.22, y_min - y_pad), y_max + y_pad)
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="both", color="#E5E7EB", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=14)

    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=family_colors["m8"],
            markeredgecolor=family_colors["m8"],
            markersize=10,
            linewidth=0,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=family_colors["m9"],
            markeredgecolor=family_colors["m9"],
            markersize=10,
            linewidth=0,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=family_colors["tone"],
            markeredgecolor=family_colors["tone"],
            markersize=10,
            linewidth=0,
        ),
    ]
    family_legend = ax.legend(
        family_handles,
        [
            "NDE-C (Content of the Near-Death Experience Scale)",
            "LCI-R (Long-term Changes Inventory-Revised)",
            "Experience Tone",
        ],
        loc="upper left",
        frameon=False,
        title="Family grouping",
        title_fontsize=14,
        fontsize=13.5,
    )
    ax.add_artist(family_legend)

    size_legend_sizes = [0.30, 0.55, 0.80]
    size_handles = [
        plt.scatter(
            [],
            [],
            s=180 + 720 * value,
            color="#BFC5CE",
            edgecolors="#2B2D42",
            linewidths=0.6,
        )
        for value in size_legend_sizes
    ]
    size_legend = ax.legend(
        size_handles,
        [f"Recall yes ≈ {value:.2f}" for value in size_legend_sizes],
        loc="lower right",
        frameon=False,
        fontsize=13,
        title="Bubble size (recall yes)",
        title_fontsize=14,
        labelspacing=0.8,
        handletextpad=0.9,
        borderaxespad=1.0,
    )
    ax.add_artist(size_legend)

    if not math.isnan(vader_kappa) and not math.isnan(vader_macro_f1):
        vader_handle = plt.Line2D(
            [0],
            [0],
            marker="*",
            color="#5E6472",
            markersize=13,
            linewidth=1.0,
            linestyle="--",
        )
        ax.legend(
            [vader_handle],
            [f"VADER baseline (κ={vader_kappa:.2f}, F1={vader_macro_f1:.2f})"],
            loc="upper right",
            frameon=False,
            fontsize=13,
        )

    if legend_handles:
        ordered_labels = [
            _presentation_model_name(name.split(":", 1)[1])
            if ":" in name
            else _presentation_model_name(name)
            for name in legend_labels
        ]
        fig.legend(
            legend_handles,
            ordered_labels,
            title=_legend_title_with_count("Models", len(ordered_labels)),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=5,
            frameon=False,
            fontsize=14,
            title_fontsize=15,
        )

    fig.suptitle(
        "Questionnaire vs automated — family-level agreement landscape", fontsize=17
    )
    fig.subplots_adjust(bottom=0.17, top=0.92, left=0.07, right=0.99)
    fig.patch.set_facecolor("#FFFFFF")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_questionnaire_extraction_item_scatter(
    metrics_df: pd.DataFrame,
    study: StudyConfig,
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    """Scatter view for questionnaire extraction items (NDE-C + LCI-R).

    X axis: Cohen kappa
    Y axis: Macro F1
    Color: model (top 3 LLMs)
    Marker: item
    """
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No item-level extraction data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    m8_fields_full = _field_bucket_order(study)["m8"]
    m9_fields_full = _field_bucket_order(study)["m9"]
    extraction_fields = m8_fields_full + m9_fields_full
    plot_df = metrics_df[metrics_df["field"].isin(extraction_fields)].copy()
    plot_df = plot_df[
        plot_df["comparison"].astype(str).str.startswith("questionnaire_vs_llm:")
    ].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No questionnaire-vs-LLM extraction rows available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    selected_models = _select_top_comparisons_for_figure(
        plot_df,
        "questionnaire_vs_",
        baseline_comparisons=[],
        top_n=3,
        ranking_metric="macro_f1",
    )
    selected_models = [
        value
        for value in selected_models
        if str(value).startswith("questionnaire_vs_llm:")
    ]
    plot_df = plot_df[plot_df["comparison"].isin(selected_models)].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No selected extraction rows available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    model_palette = ["#1D3557", "#2A9D8F", "#E76F51"]
    model_color = {
        model: model_palette[index % len(model_palette)]
        for index, model in enumerate(selected_models)
    }

    observed_fields = set(plot_df["field"].astype(str))
    m8_fields = [field for field in m8_fields_full if field in observed_fields]
    m9_fields = [field for field in m9_fields_full if field in observed_fields]

    # Keep item families visually distinct:
    # - NDE-C: mostly circular/polygonal markers.
    # - LCI-R: mostly triangular markers.
    marker_map: dict[str, Any] = {}
    nde_c_markers: list[Any] = [
        "o",
        "8",
        "h",
        "H",
        "p",
        "D",
        "d",
        "X",
        "P",
        "s",
    ]
    lci_r_markers: list[Any] = [
        "^",
        "v",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        (3, 0, 0),
        (3, 0, 30),
    ]
    for index, field in enumerate(m8_fields):
        if index < len(nde_c_markers):
            marker_map[field] = nde_c_markers[index]
        else:
            marker_map[field] = f"${index + 1}$"
    for index, field in enumerate(m9_fields):
        if index < len(lci_r_markers):
            marker_map[field] = lci_r_markers[index]
        else:
            marker_map[field] = f"$T{index + 1}$"

    fig, ax = plt.subplots(figsize=(13.4, 8.4))

    x_min, x_max = 0.00, 0.75
    y_min, y_max = 0.22, 0.90

    for _, row in plot_df.iterrows():
        model = str(row["comparison"])
        field = str(row["field"])
        x_value = float(row["cohen_kappa"])
        y_value = float(row["macro_f1"])
        ax.scatter(
            x_value,
            y_value,
            s=130,
            color=model_color.get(model, "#457B9D"),
            marker=marker_map.get(field, "o"),
            edgecolors="#FFFFFF",
            linewidths=0.9,
            alpha=0.95,
        )

    # Data-driven partition: split points into 2 metric groups via lightweight 2-means,
    # then shade the two regions separated by the centroid bisector.
    metric_points = (
        plot_df[["cohen_kappa", "macro_f1"]].dropna().astype(float).to_numpy().tolist()
    )
    if len(metric_points) >= 4:

        def _score(point: list[float]) -> float:
            return float(point[0]) + float(point[1])

        low_center = list(min(metric_points, key=_score))
        high_center = list(max(metric_points, key=_score))

        centers = [low_center, high_center]
        for _ in range(20):
            clusters = [[], []]
            for point in metric_points:
                d0 = (point[0] - centers[0][0]) ** 2 + (point[1] - centers[0][1]) ** 2
                d1 = (point[0] - centers[1][0]) ** 2 + (point[1] - centers[1][1]) ** 2
                clusters[0 if d0 <= d1 else 1].append(point)

            new_centers: list[list[float]] = []
            for index, cluster in enumerate(clusters):
                if cluster:
                    new_centers.append(
                        [
                            sum(point[0] for point in cluster) / len(cluster),
                            sum(point[1] for point in cluster) / len(cluster),
                        ]
                    )
                else:
                    new_centers.append(centers[index])

            if (
                max(
                    abs(new_centers[0][0] - centers[0][0])
                    + abs(new_centers[0][1] - centers[0][1]),
                    abs(new_centers[1][0] - centers[1][0])
                    + abs(new_centers[1][1] - centers[1][1]),
                )
                < 1e-5
            ):
                centers = new_centers
                break
            centers = new_centers

        c0, c1 = centers[0], centers[1]
        score0 = c0[0] + c0[1]
        score1 = c1[0] + c1[1]

        a = 2.0 * (c1[0] - c0[0])
        b = 2.0 * (c1[1] - c0[1])
        c = (c1[0] ** 2 + c1[1] ** 2) - (c0[0] ** 2 + c0[1] ** 2)

        x_grid = [x_min + (x_max - x_min) * step / 200.0 for step in range(201)]
        if abs(b) > 1e-8:
            y_line = [(c - a * x_value) / b for x_value in x_grid]
            y_line = [min(y_max, max(y_min, value)) for value in y_line]

            low_region_color = "#FEE2E2" if score0 < score1 else "#DBEAFE"
            high_region_color = "#DBEAFE" if score0 < score1 else "#FEE2E2"
            ax.fill_between(
                x_grid,
                [y_min] * len(x_grid),
                y_line,
                color=low_region_color,
                alpha=0.14,
                zorder=0,
            )
            ax.fill_between(
                x_grid,
                y_line,
                [y_max] * len(x_grid),
                color=high_region_color,
                alpha=0.14,
                zorder=0,
            )
            ax.plot(
                x_grid, y_line, color="#4B5563", linestyle="--", linewidth=1.2, zorder=1
            )
        elif abs(a) > 1e-8:
            x_cut = c / a
            x_cut = min(x_max, max(x_min, x_cut))
            left_color = "#FEE2E2" if score0 < score1 else "#DBEAFE"
            right_color = "#DBEAFE" if score0 < score1 else "#FEE2E2"
            ax.axvspan(x_min, x_cut, color=left_color, alpha=0.14, zorder=0)
            ax.axvspan(x_cut, x_max, color=right_color, alpha=0.14, zorder=0)
            ax.axvline(x_cut, color="#4B5563", linestyle="--", linewidth=1.2, zorder=1)

    ax.axvline(0.20, color="#9CA3AF", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axvline(0.40, color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.text(
        0.202,
        0.985,
        "κ=0.20",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=13,
        color="#6B7280",
    )
    ax.text(
        0.402,
        0.985,
        "κ=0.40",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=13,
        color="#4B5563",
    )

    ax.set_xlabel("Cohen kappa", fontsize=15)
    ax.set_ylabel("Macro F1", fontsize=15)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    _style_axes(ax)
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(axis="both", labelsize=14)

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=model_color[model],
            markeredgecolor=model_color[model],
            markersize=8,
            linewidth=0,
        )
        for model in selected_models
    ]
    model_labels = [
        _presentation_model_name(str(model).split(":", 1)[1])
        if ":" in str(model)
        else _presentation_model_name(str(model))
        for model in selected_models
    ]
    model_legend = ax.legend(
        model_handles,
        model_labels,
        title=_legend_title_with_count("Top models", len(model_labels)),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=13,
        title_fontsize=14,
    )
    ax.add_artist(model_legend)

    nde_c_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[field],
            color="#374151",
            markerfacecolor="#374151",
            linestyle="",
            markersize=7,
        )
        for field in m8_fields
    ]
    nde_c_labels = [_field_display_label(field, study) for field in m8_fields]
    nde_c_legend = fig.legend(
        nde_c_handles,
        nde_c_labels,
        title="NDE-C items",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.70, 0.55),
        fontsize=12.0,
        title_fontsize=13.5,
    )
    nde_c_legend.set_in_layout(False)

    nde_mcq_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[field],
            color="#374151",
            markerfacecolor="#374151",
            linestyle="",
            markersize=7,
        )
        for field in m9_fields
    ]
    nde_mcq_labels = [_field_display_label(field, study) for field in m9_fields]
    nde_mcq_legend = fig.legend(
        nde_mcq_handles,
        nde_mcq_labels,
        title="LCI-R items",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.125),
        ncol=2,
        fontsize=12.5,
        title_fontsize=14.5,
    )
    nde_mcq_legend.set_in_layout(False)

    fig.patch.set_facecolor("#FFFFFF")
    fig.subplots_adjust(right=0.98, top=0.97, bottom=0.20)
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_questionnaire_family_gap_slope(
    family_df: pd.DataFrame,
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
) -> list[str]:
    """Slope chart contrasting NDE-C vs LCI-R macro F1 for selected questionnaire comparisons."""
    if family_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No family summary data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFDF8")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    plot_df = family_df[
        family_df["comparison"].astype(str).str.startswith("questionnaire_vs_")
    ].copy()
    plot_df = plot_df[plot_df["family"].isin(["m8", "m9"])].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No NDE-C/LCI-R data available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFDF8")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    comparisons = _select_top_comparisons_for_figure(
        plot_df.rename(
            columns={"cohen_kappa_mean": "cohen_kappa", "macro_f1_mean": "macro_f1"}
        ),
        "questionnaire_vs_",
        baseline_comparisons=["questionnaire_vs_vader"],
        top_n=4,
        ranking_metric="macro_f1",
    )
    plot_df = plot_df[plot_df["comparison"].isin(comparisons)]

    pivot = plot_df.pivot(index="comparison", columns="family", values="macro_f1_mean")
    pivot = pivot.dropna(subset=["m8", "m9"], how="any")
    if pivot.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(
            0.5,
            0.5,
            "No paired NDE-C/LCI-R values available",
            ha="center",
            va="center",
            fontsize=13,
            color="#2B2D42",
        )
        ax.axis("off")
        fig.patch.set_facecolor("#FFFDF8")
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    pivot = pivot.sort_values("m8", ascending=False)
    fig, ax = plt.subplots(figsize=(10.6, max(5.8, 0.7 * len(pivot) + 3.2)))
    x_left, x_right = 0.0, 1.0
    palette = ["#355070", "#6D597A", "#2A9D8F", "#B56576", "#457B9D", "#B08968"]

    for index, (comparison, row) in enumerate(pivot.iterrows()):
        left = float(row["m8"])
        right = float(row["m9"])
        color = palette[index % len(palette)]
        ax.plot([x_left, x_right], [left, right], color=color, linewidth=2.2, alpha=0.9)
        ax.scatter(
            [x_left, x_right],
            [left, right],
            s=58,
            color=color,
            edgecolors="#F8F5F0",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            x_left - 0.03,
            left,
            _comparison_label(str(comparison)).replace("Questionnaire Vs ", ""),
            ha="right",
            va="center",
            fontsize=8.3,
            color="#2B2D42",
        )
        ax.text(
            x_right + 0.03,
            right,
            f"Δ={right - left:+.2f}",
            ha="left",
            va="center",
            fontsize=8.1,
            color="#2B2D42",
        )

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["NDE-C", "LCI-R"])
    ax.set_xlim(-0.20, 1.22)
    ax.set_ylim(0.25, 0.90)
    ax.set_ylabel("Mean Macro F1")
    ax.set_title("Family gap per model: NDE-C vs LCI-R")
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_comparison_summary(
    metrics_df: pd.DataFrame,
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
    scope_prefix: str | None = None,
    baseline_comparisons: list[str] | None = None,
    top_n: int = 3,
) -> list[str]:
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
        filtered_df = metrics_df[
            metrics_df["comparison"].astype(str).str.startswith(scope_prefix)
        ].copy()
    else:
        filtered_df = metrics_df.copy()

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        selected_comparisons = _select_top_comparisons_for_figure(
            filtered_df,
            scope_prefix,
            baseline_comparisons,
            top_n=top_n,
            ranking_metric="macro_f1",
        )
        filtered_df = filtered_df[filtered_df["comparison"].isin(selected_comparisons)]

    summary_df = (
        filtered_df.groupby("comparison", as_index=False)[
            ["accuracy", "cohen_kappa", "macro_f1"]
        ]
        .mean(numeric_only=True)
        .sort_values("macro_f1", ascending=False, na_position="last")
    )
    labels = [
        _wrap_label(_comparison_label(value), width=28)
        for value in summary_df["comparison"]
    ]
    y = list(range(len(summary_df)))
    height = 0.22

    fig, ax = plt.subplots(figsize=(10, max(4.5, 1.2 * len(summary_df))))
    ax.barh(
        [value - height for value in y],
        summary_df["accuracy"].fillna(0.0),
        height=height,
        color="#2A6F97",
        label="Accuracy",
    )
    ax.barh(
        y,
        summary_df["cohen_kappa"].fillna(0.0),
        height=height,
        color="#C1666B",
        label="Cohen kappa",
    )
    ax.barh(
        [value + height for value in y],
        summary_df["macro_f1"].fillna(0.0),
        height=height,
        color="#6C9A8B",
        label="Macro F1",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean metric value")
    ax.set_title("Alignment summary by comparison")
    # Move legend to upper left to avoid overlapping with bars
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)
    _style_axes(ax)
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def build_family_summary_table(metrics_df: pd.DataFrame, study: StudyConfig) -> pd.DataFrame:
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
                "accuracy_std",
                "cohen_kappa_mean",
                "cohen_kappa_std",
                "macro_f1_mean",
                "macro_f1_std",
                "precision_yes_mean",
                "recall_yes_mean",
                "recall_yes_std",
                "f1_yes_mean",
                "prevalence_reference_yes_mean",
                "prevalence_candidate_yes_mean",
                "prevalence_gap_yes_mean",
                "prevalence_gap_yes_std",
            ]
        )
    family_df = metrics_df.copy()
    # ensure study is available for bucketing
    family_df["family"] = family_df["field"].map(
        lambda value: _field_bucket(str(value), study)
    )
    grouped = (
        family_df.groupby(["comparison", "family"], as_index=False)
        .agg(
            field_count=("field", "count"),
            n_mean=("n", "mean"),
            n_min=("n", "min"),
            n_max=("n", "max"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
            cohen_kappa_std=("cohen_kappa", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            precision_yes_mean=("precision_yes", "mean"),
            recall_yes_mean=("recall_yes", "mean"),
            recall_yes_std=("recall_yes", "std"),
            f1_yes_mean=("f1_yes", "mean"),
            prevalence_reference_yes_mean=("prevalence_reference_yes", "mean"),
            prevalence_candidate_yes_mean=("prevalence_candidate_yes", "mean"),
            prevalence_gap_yes_mean=("prevalence_gap_yes", "mean"),
            prevalence_gap_yes_std=("prevalence_gap_yes", "std"),
        )
        .copy()
    )
    grouped["comparison_label"] = grouped["comparison"].map(
        lambda value: _comparison_label(str(value))
    )
    grouped["family_label"] = grouped["family"].map(
        lambda value: _field_bucket_title(str(value))
    )
    return grouped.sort_values(
        ["comparison", "family"],
        key=lambda col: (
            col.map(_comparison_sort_key)
            if col.name == "comparison"
            else col.map(_family_sort_key)
        ),
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
    """Plot family summary figure, optionally filtered to baselines + top N LLMs by macro_f1 and kappa.

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
    families = sorted(
        plot_df["family"].drop_duplicates().tolist(), key=_family_sort_key
    )
    recall_families = [
        family
        for family in families
        if not plot_df.loc[plot_df["family"] == family, "recall_yes_mean"].isna().all()
    ]

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        # Compute ranking using Pareto fronts on macro F1 and Cohen kappa.
        ranking_columns = ["macro_f1_mean", "cohen_kappa_mean"]
        comparison_means = plot_df.groupby("comparison", as_index=False)[ranking_columns].mean(numeric_only=True)
        baseline_set = set(baseline_comparisons)
        baselines_present = [
            c
            for c in baseline_comparisons
            if c in comparison_means["comparison"].tolist()
        ]
        llm_comparisons = [c for c in comparison_means["comparison"].tolist() if c not in baseline_set]
        llm_means = comparison_means[comparison_means["comparison"].isin(llm_comparisons)].copy()
        top_llms = _select_top_comparisons_by_pareto(
            llm_means,
            top_n=top_n,
            metric_x="macro_f1_mean",
            metric_y="cohen_kappa_mean",
        )
        selected_comparisons = baselines_present + top_llms
        plot_df = plot_df[plot_df["comparison"].isin(selected_comparisons)]

    comparisons = sorted(
        plot_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0))
    x = list(range(len(families)))
    recall_x = list(range(len(recall_families)))
    width = 0.82 / max(1, len(comparisons))
    colors = ["#355070", "#6D597A", "#B56576", "#2A9D8F", "#457B9D"]
    for index, comparison in enumerate(comparisons):
        subset = plot_df[plot_df["comparison"] == comparison].set_index("family")
        values = [
            float(subset.loc[family, metric_name])
            if family in subset.index
            else float("nan")
            for family in families
        ]
        recall_values = [
            float(subset.loc[family, "recall_yes_mean"])
            if family in subset.index
            else float("nan")
            for family in recall_families
        ]
        offsets = [value - 0.41 + width / 2 + index * width for value in x]
        recall_offsets = [
            value - 0.41 + width / 2 + index * width for value in recall_x
        ]
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
                [
                    float("nan") if math.isnan(value) else value
                    for value in recall_values
                ],
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
    axes[0].set_xticklabels(
        [_wrap_label(_field_bucket_title(family), width=18) for family in families]
    )
    _style_axes(axes[0])
    axes[1].set_xticks(recall_x)
    axes[1].set_xticklabels(
        [
            _wrap_label(_field_bucket_title(family), width=18)
            for family in recall_families
        ]
    )
    _style_axes(axes[1])
    axes[0].tick_params(axis="x", labelsize=9)
    axes[1].tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("Mean Cohen kappa")
    axes[0].set_title(title)
    axes[0].axhline(0.0, color="#2B2D42", linewidth=1.0, linestyle="--")
    left_max = (
        float(pd.Series(plot_df[metric_name]).max()) if not plot_df.empty else 0.0
    )
    axes[0].set_ylim(
        0.0, max(0.42, _bar_label_y(left_max, min_offset=0.02, scale=0.12))
    )
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[1].set_ylabel("Mean recall for 'yes'")
    axes[1].set_title("Positive-class recovery by binary family")
    right_max = (
        float(plot_df[plot_df["family"].isin(recall_families)]["recall_yes_mean"].max())
        if recall_families
        else 0.0
    )
    axes[1].set_ylim(
        0.0, min(1.0, max(0.8, _bar_label_y(right_max, min_offset=0.04, scale=0.08)))
    )
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
            field_df,
            scope_prefix,
            baseline_comparisons,
            top_n=top_n,
            ranking_metric="macro_f1",
        )
        field_df = field_df[field_df["comparison"].isin(selected_comparisons)]

    comparisons = sorted(
        field_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key
    )
    labels = [
        _wrap_label(_comparison_label(comparison), width=24)
        for comparison in comparisons
    ]
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
            float(
                field_df.loc[field_df["comparison"] == comparison, metric_name].iloc[0]
            )
            if comparison in set(field_df["comparison"])
            else float("nan")
            for comparison in comparisons
        ]
        positions = [value + offset for value in x]
        ax.bar(
            positions,
            [0.0 if math.isnan(value) else value for value in values],
            width=width,
            label=metric_label,
            color=color,
        )
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
    max_value = max(
        (
            float(field_df[metric].max())
            for metric, _, _, _ in metric_specs
            if metric in field_df.columns
        ),
        default=0.0,
    )
    ax.set_ylim(
        0.0, min(1.0, max(0.75, _bar_label_y(max_value, min_offset=0.04, scale=0.08)))
    )
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
        metric_name: Metric name to plot ("macro_f1" or "cohen_kappa").
        figure_path: Output figure path.
        bucket: Bucket name ("tone", "m8", or "m9").
        dpi: Figure DPI.
        export_pdf: Whether to also export PDF.
        scope_prefix: Scope prefix for filtering.
        baseline_comparisons: Baseline comparisons to always include.
        top_n: Number of top LLMs to include.
    """
    bucket_fields = _field_bucket_order(study).get(bucket, [])
    bucket_df = metrics_df[metrics_df["field"].map(lambda v: _field_bucket(v, study)) == bucket].copy()
    if bucket_fields:
        bucket_df = bucket_df[bucket_df["field"].isin(bucket_fields)]

    # Select comparisons for figure if filtering is enabled
    if scope_prefix and baseline_comparisons:
        selected_comparisons = _select_top_comparisons_for_figure(
            bucket_df,
            scope_prefix,
            baseline_comparisons,
            top_n=top_n,
            ranking_metric="macro_f1",
        )
        bucket_df = bucket_df[bucket_df["comparison"].isin(selected_comparisons)]

    comparisons = sorted(
        bucket_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key
    )
    if bucket_fields:
        pivot = bucket_df.pivot(index="comparison", columns="field", values=metric_name)
        pivot = pivot.reindex(index=comparisons, columns=bucket_fields)
        n_pivot = bucket_df.pivot(index="comparison", columns="field", values="n")
        n_pivot = n_pivot.reindex(index=comparisons, columns=bucket_fields)
    else:
        pivot = bucket_df.pivot(index="comparison", columns="field", values=metric_name)
        pivot = pivot.reindex(index=comparisons)
        n_pivot = bucket_df.pivot(index="comparison", columns="field", values="n")
        n_pivot = n_pivot.reindex(index=comparisons)
    display_columns = list(pivot.columns)
    matrix = pivot.to_numpy(dtype=float)
    n_matrix = n_pivot.to_numpy(dtype=float)

    longest_x_label = max(
        (_field_display_label(field, study) for field in display_columns),
        key=len,
        default="",
    )
    longest_y_label = max(
        (_comparison_label(comparison) for comparison in pivot.index),
        key=len,
        default="",
    )
    fig_width = max(
        9.5, 1.6 * max(1, len(display_columns)), 7.0 + 0.07 * len(longest_y_label)
    )
    fig_height = max(
        5.6, 1.05 * max(1, len(pivot.index)) + 2.4, 4.8 + 0.03 * len(longest_x_label)
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    min_value = (
        min(0.0, float(pd.DataFrame(matrix).min().min())) if matrix.size else 0.0
    )
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=min_value, vmax=1.0)
    ax.set_xticks(range(len(display_columns)))
    ax.set_xticklabels(
        [
            _wrap_label(_field_display_label(field, study), width=18)
            for field in display_columns
        ],
        rotation=18,
        ha="right",
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [
            _wrap_label(_comparison_label(comparison), width=26)
            for comparison in pivot.index
        ]
    )
    ax.set_xlabel("Items")
    ax.set_ylabel("Comparison")
    ax.set_title(
        _wrap_label(
            f"{metric_name.replace('_', ' ').title()} heatmap for {_field_bucket_title(bucket)}",
            width=52,
        )
    )
    ax.tick_params(axis="y", pad=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not math.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="#1F2933",
                    fontsize=9,
                )

    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label(metric_name.replace("_", " ").title())
    fig.patch.set_facecolor("#FFFDF8")
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def plot_tone_alignment(
    metrics_df: pd.DataFrame,
    study: StudyConfig,
    figure_path: Path,
    dpi: int = 300,
    export_pdf: bool = False,
    scope_prefix: str | None = None,
    baseline_comparisons: list[str] | None = None,
    top_n: int = 3,
) -> list[str]:
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
            tone_df,
            scope_prefix,
            baseline_comparisons,
            top_n=top_n,
            ranking_metric="macro_f1",
        )
        tone_df = tone_df[tone_df["comparison"].isin(selected_comparisons)]

    comparisons = sorted(
        tone_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key
    )
    longest_legend = max(
        (_comparison_label(comparison) for comparison in comparisons),
        key=len,
        default="",
    )
    fig_width = max(10.5, 8.5 + 0.06 * len(longest_legend))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    fields = tone_fields
    x = list(range(len(fields)))
    width = 0.8 / max(1, len(comparisons))
    colors = ["#2A6F97", "#C1666B", "#6C9A8B", "#B08968", "#7A4EAB"]

    for index, comparison in enumerate(comparisons):
        subset = tone_df[tone_df["comparison"] == comparison].set_index("field")
        values = [
            float(subset.loc[field, "cohen_kappa"]) if field in subset.index else 0.0
            for field in fields
        ]
        offsets = [value - 0.4 + width / 2 + index * width for value in x]
        ax.bar(
            offsets,
            values,
            width=width,
            label=_wrap_label(_comparison_label(comparison), width=30),
            color=colors[index % len(colors)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_wrap_label(_field_display_label(field, study), width=18) for field in fields]
    )
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
        family_df = build_family_summary_table(scoped_metrics_df, study)
        base = _figure_basename(scope_prefix)
        summary_path = figures_dir / f"{base}_comparison_summary.png"
        family_summary_path = figures_dir / f"{base}_family_summary.png"
        path_map = {
            f"{base}_comparison_summary": str(summary_path),
            f"{base}_family_summary": str(family_summary_path),
            f"{base}_tone_macro_f1_heatmap": str(
                figures_dir / f"{base}_tone_macro_f1_heatmap.png"
            ),
            f"{base}_tone_cohen_kappa_heatmap": str(
                figures_dir / f"{base}_tone_cohen_kappa_heatmap.png"
            ),
            f"{base}_nde_c_macro_f1_heatmap": str(
                figures_dir / f"{base}_nde_c_macro_f1_heatmap.png"
            ),
            f"{base}_nde_c_cohen_kappa_heatmap": str(
                figures_dir / f"{base}_nde_c_cohen_kappa_heatmap.png"
            ),
            f"{base}_lci_r_macro_f1_heatmap": str(
                figures_dir / f"{base}_lci_r_macro_f1_heatmap.png"
            ),
            f"{base}_lci_r_cohen_kappa_heatmap": str(
                figures_dir / f"{base}_lci_r_cohen_kappa_heatmap.png"
            ),
        }
        if scope_prefix == "questionnaire_vs_":
            path_map.update(
                {
                    f"{base}_family_tradeoff_map": str(
                        figures_dir / f"{base}_family_tradeoff_map.png"
                    ),
                    f"{base}_extraction_item_scatter": str(
                        figures_dir / f"{base}_extraction_item_scatter.png"
                    ),
                }
            )
        else:
            path_map[f"{base}_tone_summary"] = str(
                figures_dir / f"{base}_tone_summary.png"
            )
        # Select comparisons for figure: baselines + top 3 LLMs by macro_f1
        if scope_prefix == "questionnaire_vs_":
            baseline_comparisons = ["questionnaire_vs_vader"]
        elif scope_prefix == "human_reference_vs_":
            baseline_comparisons = [
                "human_reference_vs_questionnaire",
                "human_reference_vs_vader",
            ]
        else:
            baseline_comparisons = []
        if scope_prefix == "human_reference_vs_":
            plot_comparison_summary(
                scoped_metrics_df,
                summary_path,
                dpi=dpi,
                export_pdf=export_pdf,
                scope_prefix=scope_prefix,
                baseline_comparisons=baseline_comparisons,
                top_n=3,
            )
        scope_title = (
            "Human vs all — family-level alignment summary"
            if scope_prefix == "human_reference_vs_"
            else "Questionnaire vs automated — family-level alignment summary"
        )
        if scope_prefix == "human_reference_vs_":
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
        if scope_prefix == "questionnaire_vs_":
            plot_questionnaire_family_tradeoff_map(
                family_df,
                Path(path_map[f"{base}_family_tradeoff_map"]),
                dpi=dpi,
                export_pdf=export_pdf,
            )
            plot_questionnaire_extraction_item_scatter(
                scoped_metrics_df,
                study,
                Path(path_map[f"{base}_extraction_item_scatter"]),
                dpi=dpi,
                export_pdf=export_pdf,
            )
        tone_summary_title = (
            "Human vs all — tone summary"
            if scope_prefix == "human_reference_vs_"
            else "Questionnaire vs automated — Experience Tone summary"
        )
        # Select comparisons for tone figure: baselines + top 3 LLMs by macro_f1
        if scope_prefix == "questionnaire_vs_":
            tone_baseline_comparisons = ["questionnaire_vs_vader"]
        elif scope_prefix == "human_reference_vs_":
            tone_baseline_comparisons = [
                "human_reference_vs_questionnaire",
                "human_reference_vs_vader",
            ]
        else:
            tone_baseline_comparisons = []
        if scope_prefix == "human_reference_vs_":
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
            heatmap_baseline_comparisons = [
                "human_reference_vs_questionnaire",
                "human_reference_vs_vader",
            ]
        else:
            heatmap_baseline_comparisons = []
        bucket_to_public_key = {
            "tone": "tone",
            "m8": "nde_c",
            "m9": "lci_r",
        }
        for bucket in ("tone", "m8", "m9"):
            public_key = bucket_to_public_key[bucket]
            plot_metric_heatmap(
                scoped_metrics_df,
                study,
                "macro_f1",
                Path(path_map[f"{base}_{public_key}_macro_f1_heatmap"]),
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
                Path(path_map[f"{base}_{public_key}_cohen_kappa_heatmap"]),
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
            tone_alignment_baseline_comparisons = [
                "human_reference_vs_questionnaire",
                "human_reference_vs_vader",
            ]
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
    lines = [
        "## Evaluation Coverage",
        "",
        f"- Total NDE records that passed preprocessing with the 3 sections: {coverage.get('n_preprocessed_three_sections_total', coverage['n_sampled_total'])}",
        f"- Total sampled records available in private mapping: {coverage['n_sampled_total']}",
        f"- Records retained in the majority-vote human reference: {coverage['n_reference_participants']}",
        f"- Valid human annotation artifacts: {coverage['n_valid_human_artifacts']}",
        f"- Rejected human annotation artifacts: {coverage['n_rejected_human_artifacts']}",
        f"- Valid LLM artifacts evaluated: {coverage['n_valid_llm_artifacts']}",
        f"- Rejected LLM artifacts: {coverage['n_rejected_llm_artifacts']}",
        f"- Unresolved field/participant pairs after majority voting: {adjudication['n_unresolved_field_participant_pairs']}",
        "- The reference set is computed by field-level majority vote across valid human artifacts. Unresolved ties remain blank and are excluded from the affected field metrics.",
    ]

    accepted = summary.get("llm_artifacts", {}).get("accepted", [])
    if isinstance(accepted, list) and accepted:
        lines.extend(
            [
                "- Valid LLMs included in this report:",
                *[
                    f"  - `{_presentation_model_name(str(artifact.get('artifact_id', 'unknown')))}`"
                    for artifact in accepted
                ],
            ]
        )
    else:
        lines.append("- Valid LLMs included in this report: none")

    lines.append("")
    return lines


def _ranking_lines(comparison_summary: dict[str, Any]) -> list[str]:
    # Sort by macro_f1 descending, then by comparison order for baselines
    ranking = sorted(
        comparison_summary.items(),
        key=lambda item: (
            item[1]["macro_f1_mean"],
            item[1]["accuracy_mean"],
            item[1]["cohen_kappa_mean"],
        ),
        reverse=True,
    )
    lines = [
        "## Global Results",
        "",
        "| Comparison | Fields | Mean Accuracy | Mean Kappa | Mean Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
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
    grouped = metrics_df.groupby("comparison", as_index=False).agg(
        fields=("field", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        cohen_kappa_mean=("cohen_kappa", "mean"),
        cohen_kappa_std=("cohen_kappa", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    # Sort by macro_f1 descending
    grouped = grouped.sort_values("macro_f1_mean", ascending=False, na_position="last")
    lines = [
        f"## {title}",
        "",
        "| Comparison | Fields | Mean Accuracy | SD Accuracy | Mean Kappa | SD Kappa | Mean Macro F1 | SD Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in grouped.iterrows():
        comparison = row["comparison"]
        lines.append(
            f"| {_comparison_label(comparison)} | {int(row['fields'])} | {float(row['accuracy_mean']):.3f} | {float(row['accuracy_std']):.3f} | {float(row['cohen_kappa_mean']):.3f} | {float(row['cohen_kappa_std']):.3f} | {float(row['macro_f1_mean']):.3f} | {float(row['macro_f1_std']):.3f} |"
        )
    lines.append("")
    return lines


def _family_summary_lines(family_df: pd.DataFrame, title: str) -> list[str]:
    lines = [f"## {title}", ""]
    if family_df.empty:
        lines.extend(["- No family-level comparisons were available in this run.", ""])
        return lines
    lines.extend(
        [
            "| Comparison | Family | Fields | Mean Kappa | SD Kappa | Mean Macro F1 | SD Macro F1 | Recall Yes | SD Recall Yes | Prevalence Gap | SD Prevalence Gap | Mean N |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    # Sort by macro_f1 descending, then by comparison order, then by family order
    family_df = family_df.copy()
    family_df["_sort_key"] = family_df["comparison"].map(_comparison_sort_key)
    family_df = family_df.sort_values(
        ["_sort_key", "macro_f1_mean", "family"], ascending=[True, False, False]
    )
    family_df = family_df.drop(columns=["_sort_key"])
    for _, row in family_df.iterrows():
        lines.append(
            f"| {_comparison_label(str(row['comparison']))} | {str(row['family_label'])} | {int(row['field_count'])} | {float(row['cohen_kappa_mean']):.3f} | {float(row['cohen_kappa_std']):.3f} | {float(row['macro_f1_mean']):.3f} | {float(row['macro_f1_std']):.3f} | {float(row['recall_yes_mean']):.3f} | {float(row['recall_yes_std']):.3f} | {float(row['prevalence_gap_yes_mean']):.3f} | {float(row['prevalence_gap_yes_std']):.3f} | {float(row['n_mean']):.1f} |"
        )
    lines.append("")
    return lines


def _questionnaire_interpretation_lines(family_df: pd.DataFrame) -> list[str]:
    lines = ["## Interpretation For The Manuscript", ""]
    llm_family_df = family_df[
        family_df["comparison"].astype(str).str.startswith("questionnaire_vs_llm")
    ].copy()
    if llm_family_df.empty:
        lines.extend(
            ["- Questionnaire-based family summaries were unavailable in this run.", ""]
        )
        return lines
    family_means = (
        llm_family_df.groupby("family", as_index=False)
        .agg(
            cohen_kappa_mean=("cohen_kappa_mean", "mean"),
            cohen_kappa_std=("cohen_kappa_mean", "std"),
            macro_f1_mean=("macro_f1_mean", "mean"),
            macro_f1_std=("macro_f1_mean", "std"),
            recall_yes_mean=("recall_yes_mean", "mean"),
            recall_yes_std=("recall_yes_mean", "std"),
            prevalence_gap_yes_mean=("prevalence_gap_yes_mean", "mean"),
            prevalence_gap_yes_std=("prevalence_gap_yes_mean", "std"),
        )
        .sort_values("family", key=lambda s: s.map(_family_sort_key))
    )
    m8 = family_means[family_means["family"] == "m8"]
    m9 = family_means[family_means["family"] == "m9"]
    if not m8.empty and not m9.empty:
        m8_kappa = float(m8.iloc[0]["cohen_kappa_mean"])
        m8_kappa_std = float(m8.iloc[0]["cohen_kappa_std"])
        m9_kappa = float(m9.iloc[0]["cohen_kappa_mean"])
        m9_kappa_std = float(m9.iloc[0]["cohen_kappa_std"])
        m8_recall = float(m8.iloc[0]["recall_yes_mean"])
        m8_recall_std = float(m8.iloc[0]["recall_yes_std"])
        m9_recall = float(m9.iloc[0]["recall_yes_mean"])
        m9_recall_std = float(m9.iloc[0]["recall_yes_std"])
        kappa_relation = (
            f"NDE-C shows stronger agreement than LCI-R (mean family kappa {m8_kappa:.3f} ± {m8_kappa_std:.3f} vs {m9_kappa:.3f} ± {m9_kappa_std:.3f})"
            if m8_kappa > m9_kappa
            else f"LCI-R shows stronger agreement than NDE-C (mean family kappa {m9_kappa:.3f} ± {m9_kappa_std:.3f} vs {m8_kappa:.3f} ± {m8_kappa_std:.3f})"
            if m9_kappa > m8_kappa
            else f"NDE-C and LCI-R show matched agreement at the family level (mean family kappa {m8_kappa:.3f} ± {m8_kappa_std:.3f} vs {m9_kappa:.3f} ± {m9_kappa_std:.3f})"
        )
        recall_relation = (
            f"Positive-class recovery is weaker for LCI-R than for NDE-C (mean recall for `yes`: {m9_recall:.3f} ± {m9_recall_std:.3f} vs {m8_recall:.3f} ± {m8_recall_std:.3f})"
            if (
                not pd.isna(m8_recall)
                and not pd.isna(m9_recall)
                and m8_recall > m9_recall
            )
            else f"Positive-class recovery is stronger for LCI-R than for NDE-C (mean recall for `yes`: {m9_recall:.3f} ± {m9_recall_std:.3f} vs {m8_recall:.3f} ± {m8_recall_std:.3f})"
            if (
                not pd.isna(m8_recall)
                and not pd.isna(m9_recall)
                and m9_recall > m8_recall
            )
            else f"Positive-class recovery is matched or unavailable across NDE-C and LCI-R (mean recall for `yes`: {m8_recall:.3f} ± {m8_recall_std:.3f} vs {m9_recall:.3f} ± {m9_recall_std:.3f})"
        )
        if m8_kappa > m9_kappa:
            interpretation_tail = "- This pattern is consistent with the prompt design: the system marks `yes` only when the construct is explicitly verbalized in the narrative, so lower LCI-R alignment is compatible with weaker narrative explicitness rather than a pure model-quality failure."
        elif m9_kappa > m8_kappa:
            interpretation_tail = "- In this run, the questionnaire-based family pattern does not support the usual expectation that LCI-R is less recoverable than NDE-C, so the result should be interpreted cautiously and checked against item-level detail and prevalence structure."
        else:
            interpretation_tail = "- In this run, the questionnaire-based family comparison does not separate NDE-C and LCI-R on agreement, so the interpretation should rely more heavily on item-level detail, recall patterns, and prevalence structure."
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
    lines.extend(
        [
            "- The current run did not contain both NDE-C and LCI-R family summaries for interpretation.",
            "",
        ]
    )
    return lines


def _comparison_scope_lines(
    summary: dict[str, Any], metrics_df: pd.DataFrame, title: str
) -> list[str]:
    scopes = summary.get("comparison_scopes", {})
    comparisons = sorted(
        metrics_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key
    )
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
    comparisons = sorted(
        metrics_df["comparison"].drop_duplicates().tolist(), key=_comparison_sort_key
    )
    groups: dict[str, list[str]] = {
        "human_vs_all": [],
        "questionnaire_vs_automated": [],
        "other": [],
    }
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
            lines.append(
                f"- [{_comparison_label(comparison)}](#comparison-{comparison})"
            )
    else:
        lines.append("- No human-vs-all comparisons were available in this run.")
    lines.extend(
        [
            "",
            "</details>",
            "",
            "<details open>",
            "<summary><strong>Questionnaire vs Automated</strong></summary>",
            "",
        ]
    )
    if groups["questionnaire_vs_automated"]:
        for comparison in groups["questionnaire_vs_automated"]:
            lines.append(
                f"- [{_comparison_label(comparison)}](#comparison-{comparison})"
            )
    else:
        lines.append(
            "- No questionnaire-vs-automated comparisons were available in this run."
        )
    lines.extend(
        [
            "",
            "</details>",
            "",
            "<details>",
            "<summary><strong>Other comparisons</strong></summary>",
            "",
        ]
    )
    if groups["other"]:
        for comparison in groups["other"]:
            lines.append(
                f"- [{_comparison_label(comparison)}](#comparison-{comparison})"
            )
    else:
        lines.append("- No additional comparisons.")
    lines.extend(["", "</details>", ""])
    return lines


def _metric_toggle_lines(metrics_df: pd.DataFrame) -> list[str]:
    grouped = (
        metrics_df.groupby("comparison", as_index=False)[
            ["accuracy", "cohen_kappa", "macro_f1"]
        ]
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
        (
            "metric-correlation-proxy-macro-f1",
            "Correlation proxy (Macro F1)",
            "macro_f1",
        ),
        ("metric-accuracy", "Accuracy", "accuracy"),
        ("metric-error-rate", "Error rate", "error_rate"),
    ]
    for anchor, title, column in metric_sections:
        section = grouped.sort_values(column, ascending=False, na_position="last")
        lines.extend(
            [
                f'### <a id="{anchor}"></a>{title}',
                "",
                "| Comparison | Mean value |",
                "| --- | ---: |",
            ]
        )
        for _, row in section.iterrows():
            lines.append(
                f"| {_comparison_label(str(row['comparison']))} | {float(row[column]):.3f} |"
            )
        lines.extend(["", "[Back to metric controls](#quick-metric-controls)", ""])
    return lines


def _field_result_lines(metrics_df: pd.DataFrame, study: StudyConfig) -> list[str]:
    lines = [
        "## Item-Level Detail",
        "",
        "The tables below retain the fine-grained item evidence after the general and family-level summaries.",
        "",
    ]
    # Sort comparisons by macro_f1 descending
    comparison_means = (
        metrics_df.groupby("comparison", as_index=False)["macro_f1"]
        .mean(numeric_only=True)
        .sort_values("macro_f1", ascending=False, na_position="last")
    )
    for comparison in comparison_means["comparison"].tolist():
        subset = (
            metrics_df[metrics_df["comparison"] == comparison]
            .copy()
            .sort_values("field")
        )
        accordion_open = _comparison_tab(comparison) in {
            "human_vs_all",
            "questionnaire_vs_automated",
        }
        lines.append("<details open>" if accordion_open else "<details>")
        lines.append("")
        lines.append(
            f'<summary><strong><a id="comparison-{comparison}"></a>{_comparison_label(comparison)}</strong></summary>'
        )
        lines.append("")
        lines.extend(
            [
                f"### {_comparison_label(comparison)}",
                "",
                "| Field | Type | N | Accuracy | Kappa | Macro F1 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in subset.iterrows():
            lines.append(
                f"| {_field_display_label(str(row['field']), study)} | {_field_group(str(row['field']))} | {int(row['n'])} | {float(row['accuracy']):.3f} | {float(row['cohen_kappa']):.3f} | {float(row['macro_f1']):.3f} |"
            )
        lines.extend(["", "</details>", ""])
    return lines


def _questionnaire_contradiction_lines(
    summary: dict[str, Any], output_dir: Path, figure_paths: dict[str, str]
) -> list[str]:
    payload = _questionnaire_contradiction_payload(summary)
    lines = ["## Contradiction-Focused Qualitative Analysis", ""]
    if not payload:
        lines.extend(["- Contradiction analysis was not available in this run.", ""])
        return lines

    lines.extend(
        [
            "### Scope and Definition",
            "",
            f"- {payload.get('definition', 'Strict polarity contradiction definition not available.')}",
            f"- Top LLMs selected for qualitative analysis: {', '.join(_presentation_model_name(value) for value in payload.get('selected_llm_artifact_ids', [])) or 'none'}",
            "- VADER is retained only as a quantitative baseline because it does not provide extractive evidence spans.",
            "",
            "### Quantitative Contradiction Overview",
            "",
            "| Comparison | Type | N total | N contradictions | Rate | Evidence spans |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    overview = payload.get("overview", [])
    if isinstance(overview, list) and overview:
        for row in overview:
            lines.append(
                f"| {_comparison_label(str(row.get('comparison', '')))} | {str(row.get('source_kind', 'n/a')).upper()} | {int(row.get('n_total', 0))} | {int(row.get('n_contradictions', 0))} | {float(row.get('contradiction_rate', 0.0)):.1%} | {'yes' if bool(row.get('evidence_available', False)) else 'no'} |"
            )
    else:
        lines.append("| n/a | n/a | 0 | 0 | 0.0% | no |")
    lines.append("")

    if "questionnaire_contradiction_overview" in figure_paths:
        lines.extend(
            [
                "### Visual Summaries",
                "",
                f"![Contradiction overview]({Path(figure_paths['questionnaire_contradiction_overview']).relative_to(output_dir).as_posix()})",
                "",
                f"![Unigram contradiction word cloud]({Path(figure_paths['questionnaire_contradiction_unigram_wordcloud']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )

    lines.extend(["### Curated Contradictory Evidence Examples", ""])
    examples = payload.get("examples", [])
    if isinstance(examples, list) and examples:
        lines.extend(
            [
                "| Comparison | Participant | Questionnaire | Automated | Evidence excerpt |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in examples:
            excerpt = str(row.get("evidence_text", "")).replace("\n", " ").strip()
            if len(excerpt) > 180:
                excerpt = f"{excerpt[:177]}..."
            direction = str(row.get("direction", "")).strip()
            direction_label = (
                "Q+→LLM-"
                if direction == "q_positive_llm_negative"
                else "Q-→LLM+"
                if direction == "q_negative_llm_positive"
                else "n/a"
            )
            lines.append(
                f"| {_comparison_label(str(row.get('comparison', '')))} ({direction_label}) | {str(row.get('participant_code', ''))} | {str(row.get('questionnaire_label', ''))} | {str(row.get('candidate_label', ''))} | {excerpt or 'n/a'} |"
            )
    else:
        lines.append(
            "- No contradiction examples were available for the selected LLM comparisons."
        )
    lines.append("")

    lines.extend(["### N-gram Findings", ""])
    if "questionnaire_contradiction_bigrams" in figure_paths:
        lines.extend(
            [
                f"![Contradiction bigrams]({Path(figure_paths['questionnaire_contradiction_bigrams']).relative_to(output_dir).as_posix()})",
                "",
                f"![Contradiction trigrams]({Path(figure_paths['questionnaire_contradiction_trigrams']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
    else:
        lines.append("- N-gram visual summaries were not available in this run.")
        lines.append("")

    notes = payload.get("notes", [])
    if isinstance(notes, list) and notes:
        lines.append("### Interpretation Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return lines


def _questionnaire_neutral_focus_lines(
    summary: dict[str, Any], output_dir: Path, figure_paths: dict[str, str]
) -> list[str]:
    payload = _questionnaire_tone_label_payload(summary)
    lines = ["## Neutral Focus In Experience Tone", ""]
    if not payload:
        lines.extend(
            ["- Neutral-label diagnostics were not available in this run.", ""]
        )
        return lines

    per_label = payload.get("per_label", [])
    if not isinstance(per_label, list) or not per_label:
        lines.extend(
            ["- Neutral-label diagnostics were not available in this run.", ""]
        )
        return lines

    rows = _sort_rows_by_selected_order(
        [row for row in per_label if str(row.get("label", "")).lower() == "neutral"],
        payload,
    )
    if rows:
        lines.extend(
            [
                "This section isolates the **neutral** label so it is not implicitly absorbed into mixed interpretations.",
                "",
                "| Comparison | Neutral support (Q) | Neutral predicted (A) | Precision (neutral) | Recall (neutral) | F1 (neutral) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            precision = float(row.get("precision", float("nan")))
            recall = float(row.get("recall", float("nan")))
            f1 = float(row.get("f1", float("nan")))
            lines.append(
                f"| {_comparison_label(str(row.get('comparison', '')))} | {int(row.get('support_n', 0))} | {int(row.get('predicted_n', 0))} | {precision:.3f} | {recall:.3f} | {f1:.3f} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "- Neutral-label diagnostics were unavailable for the selected comparisons.",
                "",
            ]
        )

    if "questionnaire_tone_confusion_matrix" in figure_paths:
        lines.extend(
            [
                "### Tone Confusion Matrix",
                "",
                "![Experience Tone confusion matrix]"
                f"({Path(figure_paths['questionnaire_tone_confusion_matrix']).relative_to(output_dir).as_posix()})",
                "",
                "The matrix is row-normalized by questionnaire label and reports both row percentage and absolute count for exact inspection.",
                "",
            ]
        )
        lines.extend(_questionnaire_tone_confusion_table_lines(payload))
    return lines


def _compact_field_result_lines(
    metrics_df: pd.DataFrame, study: StudyConfig, heading: str
) -> list[str]:
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
        subset = (
            metrics_df[metrics_df["comparison"] == comparison]
            .copy()
            .sort_values(["field"])
        )
        lines.extend(
            [
                f'### <a id="comparison-{comparison}"></a>{_comparison_label(comparison)}',
                "",
                "| Field | Type | N | Accuracy | Kappa | Macro F1 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in subset.iterrows():
            lines.append(
                f"| {_field_display_label(str(row['field']), study)} | {_field_group(str(row['field']))} | {int(row['n'])} | {float(row['accuracy']):.3f} | {float(row['cohen_kappa']):.3f} | {float(row['macro_f1']):.3f} |"
            )
        lines.append("")
    return lines


def _human_hypothesis_lines(
    metrics_df: pd.DataFrame, family_df: pd.DataFrame
) -> list[str]:
    lines = ["## Hypothesis Verdict", ""]
    if metrics_df.empty:
        lines.extend(
            [
                "- Hypothesis evaluation was unavailable because no metrics were produced.",
                "",
            ]
        )
        return lines

    grouped = (
        metrics_df.groupby("comparison", as_index=False)["macro_f1"]
        .mean(numeric_only=True)
        .rename(columns={"macro_f1": "macro_f1_mean"})
    )
    questionnaire_row = grouped[
        grouped["comparison"] == "human_reference_vs_questionnaire"
    ]
    if questionnaire_row.empty:
        lines.extend(
            [
                "- Hypothesis evaluation was unavailable because `Human Reference vs Questionnaire` is missing.",
                "",
            ]
        )
        return lines

    questionnaire_macro_f1 = float(questionnaire_row.iloc[0]["macro_f1_mean"])
    automated_rows = grouped[
        grouped["comparison"].astype(str).str.startswith("human_reference_vs_llm:")
        | (grouped["comparison"] == "human_reference_vs_vader")
    ].copy()
    automated_rows = automated_rows.sort_values(
        "macro_f1_mean", ascending=False, na_position="last"
    )

    if automated_rows.empty:
        lines.extend(
            [
                "- Hypothesis evaluation was unavailable because no automated comparisons were found.",
                "",
            ]
        )
        return lines

    automated_rows["delta_vs_questionnaire"] = (
        automated_rows["macro_f1_mean"] - questionnaire_macro_f1
    )
    automated_rows["supports_hypothesis"] = automated_rows["delta_vs_questionnaire"] > 0

    n_support = int(automated_rows["supports_hypothesis"].sum())
    n_total = int(len(automated_rows))
    if n_support == n_total:
        verdict = "Supported"
    elif n_support == 0:
        verdict = "Rejected"
    else:
        verdict = "Partially supported"

    lines.extend(
        [
            f"- **Verdict:** {verdict}.",
            f"- Criterion: global macro F1 compared against `Human Reference vs Questionnaire` ({questionnaire_macro_f1:.3f}), plus family-level support consistency.",
            "",
            "### Global Evidence (Macro F1)",
            "",
            "| Comparison | Mean Macro F1 | Δ vs Human vs Questionnaire | Supports hypothesis |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for _, row in automated_rows.iterrows():
        lines.append(
            f"| {_comparison_label(str(row['comparison']))} | {float(row['macro_f1_mean']):.3f} | {float(row['delta_vs_questionnaire']):+.3f} | {'yes' if bool(row['supports_hypothesis']) else 'no'} |"
        )
    lines.append("")

    if family_df.empty:
        lines.extend(["- Family-level support was unavailable in this run.", ""])
        return lines

    family_scope = family_df[
        family_df["comparison"].astype(str).str.startswith("human_reference_vs_")
    ].copy()
    questionnaire_family = family_scope[
        family_scope["comparison"] == "human_reference_vs_questionnaire"
    ]
    automated_family = family_scope[
        family_scope["comparison"].astype(str).str.startswith("human_reference_vs_llm:")
        | (family_scope["comparison"] == "human_reference_vs_vader")
    ].copy()

    if questionnaire_family.empty or automated_family.empty:
        lines.extend(["- Family-level support was unavailable in this run.", ""])
        return lines

    baseline_by_family = {
        str(row["family"]): float(row["macro_f1_mean"])
        for _, row in questionnaire_family.iterrows()
    }
    family_rows: list[dict[str, Any]] = []
    for _, row in automated_family.iterrows():
        family_key = str(row["family"])
        if family_key not in baseline_by_family:
            continue
        baseline = baseline_by_family[family_key]
        current = float(row["macro_f1_mean"])
        delta = current - baseline
        family_rows.append(
            {
                "comparison": str(row["comparison"]),
                "family_label": str(row["family_label"]),
                "macro_f1_mean": current,
                "delta": delta,
                "supports": delta > 0,
            }
        )

    if family_rows:
        lines.extend(
            [
                "### Family-Level Support (Macro F1)",
                "",
                "| Comparison | Family | Mean Macro F1 | Δ vs Human vs Questionnaire (same family) | Supports hypothesis |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in sorted(
            family_rows,
            key=lambda item: (
                _comparison_sort_key(item["comparison"]),
                item["family_label"],
            ),
        ):
            lines.append(
                f"| {_comparison_label(row['comparison'])} | {row['family_label']} | {row['macro_f1_mean']:.3f} | {row['delta']:+.3f} | {'yes' if row['supports'] else 'no'} |"
            )
        lines.append("")

    vader_row = automated_rows[
        automated_rows["comparison"] == "human_reference_vs_vader"
    ]
    if not vader_row.empty and float(vader_row.iloc[0]["delta_vs_questionnaire"]) <= 0:
        lines.append(
            "- Interpretation guardrail: VADER underperforming questionnaire should be treated as method-class heterogeneity, not direct evidence against LLM-based alignment gains."
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
    objective_context = (
        "This report summarizes questionnaire-vs-automated agreement at global, family, and item level, "
        "and makes class-distribution behavior explicit in matrices to support interpretation of where alignment succeeds or fails."
        if report_filename == ALIGNMENT_QUESTIONNAIRE_REPORT_FILENAME
        else "This report prioritizes questionnaire-vs-automated alignment and preserves record-level support tables for manuscript traceability."
    )
    lines = [
        f"# {title}",
        "",
        "## Objective",
        "",
        objective,
        "",
        objective_context,
        "",
        "## Methodology",
        "",
        "- Human reference: field-level majority vote across valid human annotation artifacts.",
        "- Unresolved ties remain blank and are excluded only from the affected field metrics.",
        "- Metrics reported: accuracy, Cohen kappa, and macro F1.",
        "- Families are operationalized as Tone, NDE-C, and LCI-R to support article-oriented interpretation before item-level detail.",
        "- Coverage explicitly reports available sampled records, retained reference records, and the complete set of valid LLM artifacts used in evaluation.",
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
    # Paper-first figure placement for human report.
    if report_filename == ALIGNMENT_REPORT_FILENAME:
        lines.extend(
            [
                "## Paper Figures (Lead)",
                "",
                "These lead figures are placed first for manuscript-facing reading order.",
                "",
                f"![Family-level summary]({Path(figure_paths[f'{figure_prefix}_family_summary']).relative_to(output_dir).as_posix()})",
                "",
                f"![Tone summary]({Path(figure_paths[f'{figure_prefix}_tone_summary']).relative_to(output_dir).as_posix()})",
                "",
                f"![Tone alignment]({Path(figure_paths[f'{figure_prefix}_tone_alignment']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
        lines.extend(_human_hypothesis_lines(metrics_df, family_df))
        lines.extend(
            [
                "## Extended Figures",
                "",
                f"![Overall comparison summary]({Path(figure_paths[f'{figure_prefix}_comparison_summary']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Main Figures (Narrative)",
                "",
                "These lead figures prioritize integrated interpretation before item-level detail. Model names are standardized for readability, and captions carry the explanatory context instead of duplicating chart titles.",
                "",
                f"![Family tradeoff map]({Path(figure_paths.get(f'{figure_prefix}_family_tradeoff_map', figure_paths[f'{figure_prefix}_family_summary'])).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
    lines.extend(_summary_table_lines(metrics_df, "General Results"))
    complete_case_summary = summary.get("comparisons_complete_case", {})
    if isinstance(complete_case_summary, dict) and complete_case_summary:
        lines.extend(
            [
                "## Complete-Case Sensitivity",
                "",
                "This sensitivity section re-computes metrics using only participant rows with non-missing values across all shared fields in each comparison.",
                "",
                "| Comparison | Fields | Mean Accuracy | Mean Kappa | Mean Macro F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison, values in sorted(
            complete_case_summary.items(),
            key=lambda item: str(item[0]),
        ):
            lines.append(
                f"| {_comparison_label(comparison)} | {values['fields']} | {values['accuracy_mean']:.3f} | {values['cohen_kappa_mean']:.3f} | {values['macro_f1_mean']:.3f} |"
            )
        lines.append("")
    lines.extend(_family_summary_lines(family_df, "Family-Level Results"))
    if report_filename == ALIGNMENT_QUESTIONNAIRE_REPORT_FILENAME:
        lines.extend(_questionnaire_interpretation_lines(family_df))
    tone_lines = ["#### Tone Label Confusion and Per-Label F1", ""]
    if figure_prefix == "human":
        tone_lines.extend(
            [
                f"![Tone summary]({Path(figure_paths[f'{figure_prefix}_tone_summary']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
        tone_lines.extend(
            [
                f"![Tone macro F1 heatmap]({Path(figure_paths[f'{figure_prefix}_tone_macro_f1_heatmap']).relative_to(output_dir).as_posix()})",
                "",
                f"![Tone alignment]({Path(figure_paths[f'{figure_prefix}_tone_alignment']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
    else:
        tone_lines.extend(
            [
                f"![Tone confusion matrix]({Path(figure_paths['questionnaire_tone_confusion_matrix']).relative_to(output_dir).as_posix()})",
                "",
                "Rows are questionnaire labels and columns are automated labels. Each panel reports row-normalized percentages together with counts, and the per-panel subtitle includes per-label F1 for compact comparison.",
                "",
            ]
        )
    # Detailed item-level panels (also filtered to baselines + top 3 LLMs)
    if figure_prefix == "questionnaire":
        lines.extend(
            [
                "## Item-Level Structure (Extended Figures)",
                "",
                "This section integrates NDE-C and LCI-R extraction items in one scatter figure (x = kappa, y = macro F1), with standardized model names, larger markers, and compact legends for article readability.",
                "",
                *tone_lines,
                "#### NDE-C + LCI-R (Integrated Extraction View)",
                "",
                f"![NDE-C + LCI-R extraction scatter]({Path(figure_paths[f'{figure_prefix}_extraction_item_scatter']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Item-Level Structure (Extended Figures)",
                "",
                "These panels keep macro F1 and Cohen kappa detail after the narrative figures, preserving item-level transparency for manuscript appendix use.",
                "",
                *tone_lines,
                "#### NDE-C",
                "",
                f"![NDE-C macro F1 heatmap]({Path(figure_paths[f'{figure_prefix}_nde_c_macro_f1_heatmap']).relative_to(output_dir).as_posix()})",
                "",
                f"![NDE-C kappa heatmap]({Path(figure_paths[f'{figure_prefix}_nde_c_cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
                "",
                "#### LCI-R",
                "",
                f"![LCI-R macro F1 heatmap]({Path(figure_paths[f'{figure_prefix}_lci_r_macro_f1_heatmap']).relative_to(output_dir).as_posix()})",
                "",
                f"![LCI-R kappa heatmap]({Path(figure_paths[f'{figure_prefix}_lci_r_cohen_kappa_heatmap']).relative_to(output_dir).as_posix()})",
                "",
            ]
        )
    if report_filename == ALIGNMENT_QUESTIONNAIRE_REPORT_FILENAME:
        lines.extend(
            _questionnaire_contradiction_lines(summary, output_dir, figure_paths)
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
            lines.append(
                f"- `{artifact['annotator_id']}` from `{Path(artifact['artifact_path']).name}`"
            )
        lines.append("")
    if rejected:
        lines.append("Rejected human artifacts:")
        lines.append("")
        for artifact in rejected:
            lines.append(
                f"- `{Path(artifact['artifact_path']).name}`: {artifact['reason']}"
            )
        lines.append("")
    lines.append(
        f"Pairwise agreement details: `{Path(output_dir / 'human_agreement_pairwise.csv').name}`"
    )
    lines.append(
        f"Agreement summary by field: `{Path(output_dir / 'human_agreement_summary.csv').name}`"
    )
    lines.append("")
    return lines


def _llm_lines(summary: dict[str, Any]) -> list[str]:
    accepted = summary["llm_artifacts"]["accepted"]
    rejected = summary["llm_artifacts"]["rejected"]
    lines = ["## LLM Experiments", ""]
    if accepted:
        lines.append(
            "Accepted experiments evaluated against the human majority reference:"
        )
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
            lines.append(
                f"- `{Path(artifact['artifact_path']).name}`: {artifact['reason']}"
            )
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
    family_df = build_family_summary_table(primary_metrics_df, study)
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
    family_df = build_family_summary_table(questionnaire_metrics_df, study)
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
    figure_paths = write_alignment_figures(
        metrics_df, study, figures_dir, dpi=figure_dpi, export_pdf=export_figures_pdf
    )
    figure_paths.update(
        write_questionnaire_tone_label_figures(
            summary,
            figures_dir,
            dpi=figure_dpi,
            export_pdf=export_figures_pdf,
        )
    )
    figure_paths.update(
        write_questionnaire_contradiction_figures(
            summary,
            figures_dir,
            dpi=figure_dpi,
            export_pdf=export_figures_pdf,
        )
    )
    long_df = build_alignment_long_table(metrics_df)
    family_df = build_family_summary_table(metrics_df, study)
    long_path = output_dir / ALIGNMENT_LONG_FILENAME
    family_path = output_dir / ALIGNMENT_FAMILY_FILENAME
    long_df.to_csv(long_path, index=False)
    family_df.to_csv(family_path, index=False)
    has_human_scope = not _comparison_subset(metrics_df, "human_reference_vs_").empty
    written: dict[str, str] = {
        "alignment_metrics_long_file": str(long_path),
        "alignment_family_metrics_file": str(family_path),
        "alignment_figures_dir": str(figures_dir),
        **{f"figure_{name}": path for name, path in figure_paths.items()},
    }
    if has_human_scope:
        report_path = write_alignment_report(
            study,
            metrics_df,
            summary,
            output_dir,
            figure_paths,
            vader_summary=vader_summary,
        )
        written["alignment_report_file"] = str(report_path)
    questionnaire_report_path = write_questionnaire_alignment_report(
        study, metrics_df, summary, output_dir, figure_paths
    )
    written["alignment_report_questionnaire_file"] = str(questionnaire_report_path)
    return written
