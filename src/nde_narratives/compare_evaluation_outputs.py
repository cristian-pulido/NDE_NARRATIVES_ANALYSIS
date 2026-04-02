from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    path: Path


def _save_figure_png_pdf(fig: plt.Figure, path: Path, *, dpi: int = 220) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _escape_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"

    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(col) for col in df.columns]
        header_row = "| " + " | ".join(headers) + " |"
        divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_rows = []
        for row in df.itertuples(index=False, name=None):
            body_rows.append(
                "| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |"
            )
        return "\n".join([header_row, divider_row, *body_rows])


def _parse_comparison_key(comparison: str) -> tuple[str, str | None]:
    if ":" in comparison:
        scope, model_key = comparison.split(":", 1)
        return scope, model_key
    return comparison, None


def _load_condition_outputs(spec: ConditionSpec) -> dict[str, Any]:
    root = spec.path
    evaluation_metrics = pd.read_csv(root / "evaluation_metrics.csv")
    family_metrics = pd.read_csv(root / "alignment_family_metrics.csv")
    with (root / "evaluation_summary.json").open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    evaluation_metrics["condition"] = spec.name
    evaluation_metrics[["scope", "model_key"]] = evaluation_metrics["comparison"].apply(
        lambda value: pd.Series(_parse_comparison_key(str(value)))
    )

    family_metrics["condition"] = spec.name
    family_metrics[["scope", "model_key"]] = family_metrics["comparison"].apply(
        lambda value: pd.Series(_parse_comparison_key(str(value)))
    )

    summary_rows: list[dict[str, Any]] = []
    comparisons = summary.get("comparisons", {})
    for key, values in comparisons.items():
        scope, model_key = _parse_comparison_key(key)
        summary_rows.append(
            {
                "condition": spec.name,
                "comparison": key,
                "scope": scope,
                "model_key": model_key,
                "fields": values.get("fields"),
                "accuracy_mean": values.get("accuracy_mean"),
                "cohen_kappa_mean": values.get("cohen_kappa_mean"),
                "macro_f1_mean": values.get("macro_f1_mean"),
            }
        )

    coverage = summary.get("coverage", {})
    coverage_row = {
        "condition": spec.name,
        "n_preprocessed_three_sections_total": coverage.get("n_preprocessed_three_sections_total"),
        "n_sampled_total": coverage.get("n_sampled_total"),
        "n_reference_participants": coverage.get("n_reference_participants"),
        "n_valid_llm_artifacts": coverage.get("n_valid_llm_artifacts"),
        "n_rejected_llm_artifacts": coverage.get("n_rejected_llm_artifacts"),
    }

    return {
        "evaluation_metrics": evaluation_metrics,
        "family_metrics": family_metrics,
        "summary_rows": pd.DataFrame(summary_rows),
        "coverage_row": coverage_row,
    }


def _figure_family_dumbbell(
    *,
    baseline_contrast_df: pd.DataFrame,
    baseline: str,
    path: Path,
) -> None:
    if baseline_contrast_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No baseline-family shared model support available for dumbbell figure.",
            ha="center",
            va="center",
            fontsize=11,
        )
        _save_figure_png_pdf(fig, path, dpi=220)
        plt.close(fig)
        return

    def _family_short(label: Any) -> str:
        text = str(label)
        lower = text.lower()
        if "tone" in lower:
            return "Tone"
        if "nde-c" in lower or "content of the near-death experience" in lower:
            return "NDE-C"
        if "nde-mcq" in lower or "moral cognition" in lower:
            return "NDE-MCQ"
        return text

    def _condition_short(name: Any) -> str:
        text = str(name).replace("_", " ").strip()
        return text

    def _compute_xlim(values: pd.Series, *, bounded: tuple[float, float] | None = None) -> tuple[float, float]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return (0.0, 1.0) if bounded is not None else (-0.5, 0.5)
        vmin = float(numeric.min())
        vmax = float(numeric.max())
        spread = max(vmax - vmin, 1e-9)
        pad = max(0.015, spread * 0.10)
        left = vmin - pad
        right = vmax + pad

        if bounded is not None:
            lo, hi = bounded
            left = max(lo, left)
            right = min(hi, right)
            min_width = 0.10
            if right - left < min_width:
                center = (left + right) / 2.0
                left = max(lo, center - min_width / 2.0)
                right = min(hi, center + min_width / 2.0)
                if right - left < min_width:
                    # Edge case when centered expansion hits both bounds.
                    if abs(left - lo) < 1e-9:
                        right = min(hi, lo + min_width)
                    else:
                        left = max(lo, hi - min_width)
        return left, right

    plot_df = baseline_contrast_df.copy()
    plot_df["family_short"] = plot_df["family_label"].apply(_family_short)
    plot_df["condition_short"] = plot_df["condition"].apply(_condition_short)
    plot_df["label_raw"] = plot_df["family_short"]
    plot_df = plot_df.sort_values(["family_label", "condition"]).reset_index(drop=True)

    n_rows = len(plot_df)
    n_conditions = int(plot_df["condition"].nunique()) if "condition" in plot_df.columns else 1
    max_label_len = max((len(str(value)) for value in plot_df["label_raw"].tolist()), default=18)

    # Adaptive wrapping and typography so the same code scales from 2 to many conditions.
    wrap_width = 16 if n_conditions <= 3 else 18 if n_conditions <= 5 else 20
    plot_df["label"] = plot_df["label_raw"].apply(
        lambda value: fill(str(value), width=wrap_width, break_long_words=False, break_on_hyphens=False)
    )

    # Shrink whitespace for small comparisons, expand only when rows genuinely increase.
    row_height = 0.31 if n_rows <= 12 else 0.29 if n_rows <= 22 else 0.27
    height = max(3.9, row_height * n_rows + 1.55)

    # Left margin needs to follow wrapped label complexity.
    fig_width = 9.2 if max_label_len <= 18 else 9.8 if max_label_len <= 30 else 10.4

    title_fs = 12 if n_rows <= 22 else 11
    label_fs = 11
    tick_fs = 9 if n_rows <= 18 else 8
    ytick_fs = 9 if n_rows <= 14 else 8 if n_rows <= 24 else 7
    marker_size = 34 if n_rows <= 22 else 30

    y = list(range(len(plot_df)))

    # Condition-specific color mapping so each pipeline is visually distinct.
    unique_conditions = list(plot_df["condition"].dropna().unique())
    palette = [
        "#E45756",
        "#72B7B2",
        "#F58518",
        "#54A24B",
        "#B279A2",
        "#EECA3B",
        "#4C78A8",
        "#FF9DA6",
    ]
    condition_colors = {cond: palette[idx % len(palette)] for idx, cond in enumerate(unique_conditions)}

    macro_xlim = _compute_xlim(
        pd.concat([plot_df["baseline_macro_f1"], plot_df["condition_macro_f1"]], axis=0),
        bounded=(0.0, 1.0),
    )
    kappa_xlim = _compute_xlim(
        pd.concat([plot_df["baseline_cohen_kappa"], plot_df["condition_cohen_kappa"]], axis=0),
        bounded=(-1.0, 1.0),
    )

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, height), sharey=True, constrained_layout=True)

    # Panel A: Macro F1
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        c = condition_colors.get(row["condition"], "#E45756")
        axes[0].hlines(
            y[idx],
            row["baseline_macro_f1"],
            row["condition_macro_f1"],
            color=c,
            linewidth=1.6,
            alpha=0.55,
        )
        axes[0].scatter(row["baseline_macro_f1"], y[idx], color="#5B5B5B", s=marker_size, zorder=3)
        axes[0].scatter(row["condition_macro_f1"], y[idx], color=c, s=marker_size, zorder=3)
    axes[0].set_title("Family Macro F1", fontsize=title_fs)
    axes[0].set_xlabel("Macro F1", fontsize=label_fs)
    axes[0].set_xlim(*macro_xlim)
    axes[0].grid(axis="x", alpha=0.25)

    # Panel B: Cohen's kappa
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        c = condition_colors.get(row["condition"], "#E45756")
        axes[1].hlines(
            y[idx],
            row["baseline_cohen_kappa"],
            row["condition_cohen_kappa"],
            color=c,
            linewidth=1.6,
            alpha=0.55,
        )
        axes[1].scatter(row["baseline_cohen_kappa"], y[idx], color="#5B5B5B", s=marker_size, zorder=3)
        axes[1].scatter(row["condition_cohen_kappa"], y[idx], color=c, s=marker_size, zorder=3)
    axes[1].axvline(0.0, linestyle="--", linewidth=1.0, color="#666", alpha=0.7)
    axes[1].set_title("Family Cohen's kappa", fontsize=title_fs)
    axes[1].set_xlabel("Cohen's kappa", fontsize=label_fs)
    axes[1].set_xlim(*kappa_xlim)
    axes[1].grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["label"])
    axes[0].invert_yaxis()
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(plot_df["label"])

    for axis in axes:
        axis.tick_params(axis="x", labelsize=tick_fs)
        axis.tick_params(axis="y", labelsize=ytick_fs)

    # Compact but readable placement; avoids large blank area on small matrices.
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.02)

    # Compact legend: baseline plus one entry per condition.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#5B5B5B", markersize=6, label=f"{baseline} (baseline)"),
    ]
    legend_handles.extend(
        [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=_condition_short(cond))
            for cond, color in condition_colors.items()
        ]
    )
    axes[0].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7 if n_rows <= 24 else 6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.3,
        handletextpad=0.4,
    )
    _save_figure_png_pdf(fig, path, dpi=220)
    plt.close(fig)


def _pairwise_item_deltas(
    all_metrics_df: pd.DataFrame,
    *,
    focus_scope: str,
    conditions: list[str],
) -> pd.DataFrame:
    df = all_metrics_df.copy()
    df = df[df["scope"] == focus_scope]
    df = df[df["model_variant"].notna()]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "condition_a",
                "condition_b",
                "model_variant",
                "field",
                "n_a",
                "n_b",
                "macro_f1_a",
                "macro_f1_b",
                "delta_macro_f1",
                "cohen_kappa_a",
                "cohen_kappa_b",
                "delta_cohen_kappa",
            ]
        )

    rows: list[pd.DataFrame] = []
    for condition_a, condition_b in combinations(conditions, 2):
        left = (
            df[df["condition"] == condition_a][["model_variant", "field", "n", "macro_f1", "cohen_kappa"]]
            .rename(columns={"n": "n_a", "macro_f1": "macro_f1_a", "cohen_kappa": "cohen_kappa_a"})
            .drop_duplicates(subset=["model_variant", "field"], keep="last")
        )
        right = (
            df[df["condition"] == condition_b][["model_variant", "field", "n", "macro_f1", "cohen_kappa"]]
            .rename(columns={"n": "n_b", "macro_f1": "macro_f1_b", "cohen_kappa": "cohen_kappa_b"})
            .drop_duplicates(subset=["model_variant", "field"], keep="last")
        )
        merged = left.merge(right, on=["model_variant", "field"], how="inner")
        if merged.empty:
            continue
        merged["condition_a"] = condition_a
        merged["condition_b"] = condition_b
        merged["delta_macro_f1"] = merged["macro_f1_b"] - merged["macro_f1_a"]
        merged["delta_cohen_kappa"] = merged["cohen_kappa_b"] - merged["cohen_kappa_a"]
        rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "condition_a",
                "condition_b",
                "model_variant",
                "field",
                "n_a",
                "n_b",
                "macro_f1_a",
                "macro_f1_b",
                "delta_macro_f1",
                "cohen_kappa_a",
                "cohen_kappa_b",
                "delta_cohen_kappa",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["condition_a", "condition_b", "model_variant", "field"]).reset_index(drop=True)


def _pairwise_family_deltas(
    all_family_df: pd.DataFrame,
    *,
    focus_scope: str,
    conditions: list[str],
) -> pd.DataFrame:
    df = all_family_df.copy()
    df = df[(df["scope"] == focus_scope) & (df["model_key"].notna())]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "condition_a",
                "condition_b",
                "model_key",
                "family_label",
                "macro_f1_a",
                "macro_f1_b",
                "delta_macro_f1",
                "cohen_kappa_a",
                "cohen_kappa_b",
                "delta_cohen_kappa",
                "n_a",
                "n_b",
            ]
        )

    rows: list[pd.DataFrame] = []
    for condition_a, condition_b in combinations(conditions, 2):
        left = (
            df[df["condition"] == condition_a][
                ["model_key", "family_label", "macro_f1_mean", "cohen_kappa_mean", "n_mean"]
            ]
            .rename(
                columns={
                    "macro_f1_mean": "macro_f1_a",
                    "cohen_kappa_mean": "cohen_kappa_a",
                    "n_mean": "n_a",
                }
            )
            .drop_duplicates(subset=["model_key", "family_label"], keep="last")
        )
        right = (
            df[df["condition"] == condition_b][
                ["model_key", "family_label", "macro_f1_mean", "cohen_kappa_mean", "n_mean"]
            ]
            .rename(
                columns={
                    "macro_f1_mean": "macro_f1_b",
                    "cohen_kappa_mean": "cohen_kappa_b",
                    "n_mean": "n_b",
                }
            )
            .drop_duplicates(subset=["model_key", "family_label"], keep="last")
        )
        merged = left.merge(right, on=["model_key", "family_label"], how="inner")
        if merged.empty:
            continue
        merged["condition_a"] = condition_a
        merged["condition_b"] = condition_b
        merged["delta_macro_f1"] = merged["macro_f1_b"] - merged["macro_f1_a"]
        merged["delta_cohen_kappa"] = merged["cohen_kappa_b"] - merged["cohen_kappa_a"]
        rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "condition_a",
                "condition_b",
                "model_key",
                "family_label",
                "macro_f1_a",
                "macro_f1_b",
                "delta_macro_f1",
                "cohen_kappa_a",
                "cohen_kappa_b",
                "delta_cohen_kappa",
                "n_a",
                "n_b",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["condition_a", "condition_b", "model_key", "family_label"]).reset_index(drop=True)


def _pairwise_global_deltas(global_df: pd.DataFrame) -> pd.DataFrame:
    if global_df.empty:
        return pd.DataFrame(
            columns=[
                "condition_a",
                "condition_b",
                "macro_f1_mean_a",
                "macro_f1_mean_b",
                "delta_macro_f1",
                "cohen_kappa_mean_a",
                "cohen_kappa_mean_b",
                "delta_cohen_kappa",
            ]
        )

    rows: list[dict[str, Any]] = []
    indexed = global_df.set_index("condition")
    for condition_a, condition_b in combinations(indexed.index.tolist(), 2):
        a = indexed.loc[condition_a]
        b = indexed.loc[condition_b]
        rows.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "macro_f1_mean_a": a.get("macro_f1_mean"),
                "macro_f1_mean_b": b.get("macro_f1_mean"),
                "delta_macro_f1": b.get("macro_f1_mean") - a.get("macro_f1_mean"),
                "cohen_kappa_mean_a": a.get("cohen_kappa_mean"),
                "cohen_kappa_mean_b": b.get("cohen_kappa_mean"),
                "delta_cohen_kappa": b.get("cohen_kappa_mean") - a.get("cohen_kappa_mean"),
            }
        )

    return pd.DataFrame(rows)


def _baseline_family_contrast(
    *,
    all_family_df: pd.DataFrame,
    focus_scope: str,
    baseline: str,
    conditions: list[str],
) -> pd.DataFrame:
    df = all_family_df.copy()
    df = df[(df["scope"] == focus_scope) & (df["model_key"].notna())]
    if df.empty or baseline not in conditions:
        return pd.DataFrame(
            columns=[
                "condition",
                "family_label",
                "shared_models",
                "baseline_macro_f1",
                "condition_macro_f1",
                "delta_macro_f1",
                "baseline_cohen_kappa",
                "condition_cohen_kappa",
                "delta_cohen_kappa",
            ]
        )

    baseline_df = (
        df[df["condition"] == baseline][["model_key", "family_label", "macro_f1_mean", "cohen_kappa_mean"]]
        .rename(columns={"macro_f1_mean": "baseline_macro_f1", "cohen_kappa_mean": "baseline_cohen_kappa"})
        .drop_duplicates(subset=["model_key", "family_label"], keep="last")
    )

    rows: list[pd.DataFrame] = []
    for condition in conditions:
        if condition == baseline:
            continue
        condition_df = (
            df[df["condition"] == condition][["model_key", "family_label", "macro_f1_mean", "cohen_kappa_mean"]]
            .rename(columns={"macro_f1_mean": "condition_macro_f1", "cohen_kappa_mean": "condition_cohen_kappa"})
            .drop_duplicates(subset=["model_key", "family_label"], keep="last")
        )
        merged = baseline_df.merge(condition_df, on=["model_key", "family_label"], how="inner")
        if merged.empty:
            continue
        grouped = (
            merged.groupby("family_label", as_index=False)
            .agg(
                shared_models=("model_key", "nunique"),
                baseline_macro_f1=("baseline_macro_f1", "mean"),
                condition_macro_f1=("condition_macro_f1", "mean"),
                baseline_cohen_kappa=("baseline_cohen_kappa", "mean"),
                condition_cohen_kappa=("condition_cohen_kappa", "mean"),
            )
            .assign(condition=condition)
        )
        grouped["delta_macro_f1"] = grouped["condition_macro_f1"] - grouped["baseline_macro_f1"]
        grouped["delta_cohen_kappa"] = grouped["condition_cohen_kappa"] - grouped["baseline_cohen_kappa"]
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(
            columns=[
                "condition",
                "family_label",
                "shared_models",
                "baseline_macro_f1",
                "condition_macro_f1",
                "delta_macro_f1",
                "baseline_cohen_kappa",
                "condition_cohen_kappa",
                "delta_cohen_kappa",
            ]
        )

    return pd.concat(rows, ignore_index=True).sort_values(["family_label", "condition"]).reset_index(drop=True)


def _write_report(
    *,
    report_path: Path,
    title: str,
    baseline: str,
    focus_scope: str,
    conditions: list[ConditionSpec],
    coverage_df: pd.DataFrame,
    global_df: pd.DataFrame,
    pairwise_global_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    pairwise_item_df: pd.DataFrame,
    pairwise_family_df: pd.DataFrame,
    baseline_family_contrast_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Compare evaluation outputs across preprocessing conditions and quantify how each condition changes alignment outcomes, with tables and figures suitable for manuscript appendix material."
    )
    lines.append("")
    lines.append("## Conditions")
    lines.append("")
    for spec in conditions:
        lines.append(f"- **{spec.name}**: `{spec.path}`")
    lines.append("")
    lines.append("## Methodological guardrails")
    lines.append("")
    lines.append("- Two views are reported: (1) all-available summaries and (2) fair-intersection paired deltas.")
    lines.append(
        "- Pairwise deltas are computed on shared keys per condition pair (model+field for item-level and model+family for family-level) to avoid confounding from unequal model sets or missing outputs."
    )
    lines.append("- The report remains fully pairwise, while the narrative emphasis remains baseline vs translate-run style contrasts.")
    lines.append("")
    lines.append(f"**Baseline condition:** `{baseline}`")
    lines.append("")
    lines.append(f"**Primary scope analyzed:** `{focus_scope}`")
    lines.append("")
    lines.append("## Coverage summary")
    lines.append("")
    lines.append(_dataframe_to_markdown(coverage_df))
    lines.append("")
    lines.append("## Primary figure (family dumbbell, dual metrics)")
    lines.append("")
    lines.append(
        "Each row compares baseline vs one condition within a family, with paired points connected by a dumbbell segment. "
        "Panel A reports Macro F1 and Panel B reports Cohen's kappa."
    )
    lines.append("")
    lines.append("![Family dumbbell dual metric](figures/family_dumbbell_dual_metric.png)")
    lines.append("[Family dumbbell dual metric (PDF)](figures/family_dumbbell_dual_metric.pdf)")
    lines.append("")
    lines.append("## Global comparison summary (all-available)")
    lines.append("")
    lines.append(_dataframe_to_markdown(global_df))
    lines.append("")
    lines.append("## Global pairwise deltas (condition B − condition A)")
    lines.append("")
    if pairwise_global_df.empty:
        lines.append("No global pairwise deltas were available.")
    else:
        lines.append(_dataframe_to_markdown(pairwise_global_df))
    lines.append("")
    lines.append("## Family summary (all-available)")
    lines.append("")
    if family_summary_df.empty:
        lines.append("No family summary rows were available.")
    else:
        lines.append(_dataframe_to_markdown(family_summary_df))
    lines.append("")
    lines.append("## Family pairwise deltas (paired fair-intersection)")
    lines.append("")
    if pairwise_family_df.empty:
        lines.append("No shared model+family support was available across condition pairs.")
    else:
        family_pair_summary = (
            pairwise_family_df.groupby(["condition_a", "condition_b", "family_label"], as_index=False)
            .agg(
                shared_models=("model_key", "nunique"),
                delta_macro_f1_mean=("delta_macro_f1", "mean"),
                delta_cohen_kappa_mean=("delta_cohen_kappa", "mean"),
            )
            .sort_values(["condition_a", "condition_b", "family_label"])
        )
        lines.append(_dataframe_to_markdown(family_pair_summary))
    lines.append("")
    lines.append("## Item pairwise deltas (paired fair-intersection)")
    lines.append("")
    if pairwise_item_df.empty:
        lines.append("No shared model+field support was available across condition pairs.")
    else:
        item_pair_summary = (
            pairwise_item_df.groupby(["condition_a", "condition_b"], as_index=False)
            .agg(
                shared_model_fields=("field", "count"),
                delta_macro_f1_mean=("delta_macro_f1", "mean"),
                delta_cohen_kappa_mean=("delta_cohen_kappa", "mean"),
            )
            .sort_values(["condition_a", "condition_b"])
        )
        lines.append(_dataframe_to_markdown(item_pair_summary))
    lines.append("")
    lines.append("## Baseline-focused family contrast")
    lines.append("")
    if baseline_family_contrast_df.empty:
        lines.append("No baseline family contrast rows were available.")
    else:
        lines.append(_dataframe_to_markdown(baseline_family_contrast_df))
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "- Prioritize pairwise fair-intersection deltas for methodological claims when conditions differ in sample size, accepted models, or valid sections."
    )
    lines.append(
        "- Read Macro F1 and Cohen's kappa together: F1 tracks label balance-sensitive agreement, while kappa discounts chance agreement and is stricter under class imbalance."
    )
    lines.append(
        "- A positive pairwise delta means condition B outperforms condition A for the same shared support; a negative delta indicates degradation."
    )
    lines.append(
        "- Baseline-vs-translate-run interpretations should be drawn from the baseline-focused family contrast table and dumbbell figure, while retaining all-pairs tables for methodological completeness."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_evaluation_outputs(
    *,
    conditions: list[ConditionSpec],
    output_dir: Path,
    baseline: str | None = None,
    title: str = "Preprocessing Effect Comparison Report",
    focus_scope: str = "questionnaire_vs_llm",
    metric: str = "macro_f1",
) -> dict[str, str]:
    if len(conditions) < 2:
        raise ValueError("At least two conditions are required for comparison.")
    condition_names = [spec.name for spec in conditions]
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("Condition names must be unique.")

    baseline_name = baseline or conditions[0].name
    if baseline_name not in condition_names:
        raise ValueError(f"Baseline condition '{baseline_name}' is not in declared conditions: {condition_names}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    loaded = [_load_condition_outputs(spec) for spec in conditions]
    all_metrics_df = pd.concat([entry["evaluation_metrics"] for entry in loaded], ignore_index=True)
    all_family_df = pd.concat([entry["family_metrics"] for entry in loaded], ignore_index=True)
    all_summary_df = pd.concat([entry["summary_rows"] for entry in loaded], ignore_index=True)
    coverage_df = pd.DataFrame([entry["coverage_row"] for entry in loaded]).sort_values("condition")

    metric_col = metric
    if metric_col not in all_metrics_df.columns:
        raise ValueError(f"Metric '{metric_col}' is not available in evaluation_metrics.csv")

    global_df = (
        all_summary_df[all_summary_df["scope"] == focus_scope]
        .groupby("condition", as_index=False)[["accuracy_mean", "cohen_kappa_mean", "macro_f1_mean"]]
        .mean()
        .sort_values("condition")
    )

    family_summary_df = (
        all_family_df[all_family_df["scope"] == focus_scope]
        .groupby("condition", as_index=False)[["accuracy_mean", "cohen_kappa_mean", "macro_f1_mean", "field_count", "n_mean"]]
        .mean()
        .sort_values("condition")
    )

    pairwise_item_df = _pairwise_item_deltas(
        all_metrics_df,
        focus_scope=focus_scope,
        conditions=condition_names,
    )
    pairwise_family_df = _pairwise_family_deltas(
        all_family_df,
        focus_scope=focus_scope,
        conditions=condition_names,
    )
    pairwise_global_df = _pairwise_global_deltas(global_df)
    baseline_family_contrast_df = _baseline_family_contrast(
        all_family_df=all_family_df,
        focus_scope=focus_scope,
        baseline=baseline_name,
        conditions=condition_names,
    )

    coverage_csv = output_dir / "coverage_comparison.csv"
    global_csv = output_dir / "global_comparison_summary.csv"
    family_csv = output_dir / "family_scope_summary.csv"
    pairwise_global_csv = output_dir / "pairwise_global_deltas.csv"
    pairwise_item_csv = output_dir / "pairwise_model_field_deltas.csv"
    pairwise_family_csv = output_dir / "pairwise_model_family_deltas.csv"
    baseline_family_csv = output_dir / "baseline_family_contrast.csv"
    report_md = output_dir / "preprocessing_effect_report.md"

    coverage_df.to_csv(coverage_csv, index=False)
    global_df.to_csv(global_csv, index=False)
    family_summary_df.to_csv(family_csv, index=False)
    pairwise_global_df.to_csv(pairwise_global_csv, index=False)
    pairwise_item_df.to_csv(pairwise_item_csv, index=False)
    pairwise_family_df.to_csv(pairwise_family_csv, index=False)
    baseline_family_contrast_df.to_csv(baseline_family_csv, index=False)

    dumbbell_fig = figures_dir / "family_dumbbell_dual_metric.png"
    _figure_family_dumbbell(
        baseline_contrast_df=baseline_family_contrast_df,
        baseline=baseline_name,
        path=dumbbell_fig,
    )

    _write_report(
        report_path=report_md,
        title=title,
        baseline=baseline_name,
        focus_scope=focus_scope,
        conditions=conditions,
        coverage_df=coverage_df,
        global_df=global_df,
        pairwise_global_df=pairwise_global_df,
        family_summary_df=family_summary_df,
        pairwise_item_df=pairwise_item_df,
        pairwise_family_df=pairwise_family_df,
        baseline_family_contrast_df=baseline_family_contrast_df,
    )

    return {
        "report_file": str(report_md),
        "coverage_csv": str(coverage_csv),
        "global_csv": str(global_csv),
        "family_csv": str(family_csv),
        "pairwise_global_csv": str(pairwise_global_csv),
        "pairwise_item_csv": str(pairwise_item_csv),
        "pairwise_family_csv": str(pairwise_family_csv),
        "baseline_family_csv": str(baseline_family_csv),
        "dumbbell_figure": str(dumbbell_fig),
        "dumbbell_figure_pdf": str(dumbbell_fig.with_suffix(".pdf")),
    }


def parse_condition_argument(value: str) -> ConditionSpec:
    match = re.match(r"^([^=]+)=(.+)$", value)
    if not match:
        raise ValueError("Condition must follow NAME=PATH format.")
    name, raw_path = match.group(1).strip(), match.group(2).strip()
    if not name:
        raise ValueError("Condition name cannot be empty.")
    path = Path(raw_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Condition path does not exist: {path}")
    return ConditionSpec(name=name, path=path)


def load_conditions_from_config(config_path: Path) -> tuple[list[ConditionSpec], dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list) or len(raw_conditions) < 2:
        raise ValueError("Config must contain at least two entries under 'conditions'.")

    conditions: list[ConditionSpec] = []
    for item in raw_conditions:
        if not isinstance(item, dict):
            raise ValueError("Each config condition must be an object with keys 'name' and 'path'.")
        name = str(item.get("name", "")).strip()
        raw_path = str(item.get("path", "")).strip()
        if not name or not raw_path:
            raise ValueError("Each config condition requires non-empty 'name' and 'path'.")
        condition_path = Path(raw_path).resolve()
        if not condition_path.exists():
            raise FileNotFoundError(f"Condition path does not exist: {condition_path}")
        conditions.append(ConditionSpec(name=name, path=condition_path))

    return conditions, payload
