from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    path: Path


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


def _figure_pipeline_performance(
    *,
    coverage_df: pd.DataFrame,
    global_df: pd.DataFrame,
    family_delta_df: pd.DataFrame,
    baseline: str,
    path: Path,
) -> None:
    cov = coverage_df.copy().sort_values("condition").reset_index(drop=True)
    glob = global_df.copy().sort_values("condition").reset_index(drop=True)

    baseline_records = cov.loc[cov["condition"] == baseline, "n_preprocessed_three_sections_total"]
    baseline_llm = cov.loc[cov["condition"] == baseline, "n_valid_llm_artifacts"]
    base_records = float(baseline_records.iloc[0]) if not baseline_records.empty else float("nan")
    base_llm = float(baseline_llm.iloc[0]) if not baseline_llm.empty else float("nan")

    cov["records_pct_vs_baseline"] = (
        (cov["n_preprocessed_three_sections_total"] / base_records) * 100.0 if pd.notna(base_records) and base_records else float("nan")
    )
    cov["llm_pct_vs_baseline"] = (
        (cov["n_valid_llm_artifacts"] / base_llm) * 100.0 if pd.notna(base_llm) and base_llm else float("nan")
    )

    # Family delta aggregation for a compact inference-effect trace on panel B.
    if family_delta_df.empty:
        family_delta_mean = pd.DataFrame(columns=["condition", "mean_family_delta"])
    else:
        family_delta_mean = (
            family_delta_df.groupby("condition", as_index=False)["delta_vs_baseline"]
            .mean()
            .rename(columns={"delta_vs_baseline": "mean_family_delta"})
        )

    glob = glob.merge(family_delta_mean, on="condition", how="left")
    glob["mean_family_delta"] = glob["mean_family_delta"].fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Panel A: Pipeline effect (% vs baseline)
    x = list(range(len(cov)))
    w = 0.38
    axes[0].bar(
        [v - (w / 2.0) for v in x],
        cov["records_pct_vs_baseline"],
        width=w,
        color="#4C78A8",
        label="Usable records (% vs baseline)",
    )
    axes[0].bar(
        [v + (w / 2.0) for v in x],
        cov["llm_pct_vs_baseline"],
        width=w,
        color="#F58518",
        label="Valid LLM artifacts (% vs baseline)",
    )
    axes[0].axhline(100.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cov["condition"], rotation=30, ha="right")
    axes[0].set_ylabel("% of baseline")
    axes[0].set_title("Panel A — Pipeline coverage effect")
    axes[0].legend(loc="upper right", fontsize=8)

    # Panel B: Performance + mean family deltas
    x2 = list(range(len(glob)))
    axes[1].bar(x2, glob["macro_f1_mean"], color="#54A24B", alpha=0.9, label="Global Macro F1")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(glob["condition"], rotation=30, ha="right")
    axes[1].set_ylabel("Macro F1 (all-available summary)")
    axes[1].set_title("Panel B — Inference outcome effect")

    ax2 = axes[1].twinx()
    ax2.plot(
        x2,
        glob["mean_family_delta"],
        color="#E45756",
        marker="o",
        linewidth=2.0,
        label="Mean family Δ vs baseline",
    )
    ax2.axhline(0.0, color="#E45756", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("Mean family Δ Macro F1 vs baseline")

    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    axes[1].legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", fontsize=8)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def _shared_item_deltas(
    all_metrics_df: pd.DataFrame,
    *,
    focus_scope: str,
    metric_col: str,
    baseline: str,
    conditions: list[str],
) -> pd.DataFrame:
    df = all_metrics_df.copy()
    df = df[df["scope"] == focus_scope]
    df = df[df[metric_col].notna()]
    df = df[df["model_variant"].notna()]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "model_variant",
                "field",
                "condition",
                "value",
                "n",
                "baseline_value",
                "delta_vs_baseline",
            ]
        )

    key_counts = (
        df[["model_variant", "field", "condition"]]
        .drop_duplicates()
        .groupby(["model_variant", "field"], as_index=False)["condition"]
        .nunique()
        .rename(columns={"condition": "n_conditions"})
    )
    shared_keys = key_counts[key_counts["n_conditions"] == len(conditions)][["model_variant", "field"]]
    if shared_keys.empty:
        return pd.DataFrame(
            columns=[
                "model_variant",
                "field",
                "condition",
                "value",
                "n",
                "baseline_value",
                "delta_vs_baseline",
            ]
        )

    shared_df = df.merge(shared_keys, on=["model_variant", "field"], how="inner")
    shared_df = shared_df[["model_variant", "field", "condition", metric_col, "n"]].rename(columns={metric_col: "value"})

    baseline_map = (
        shared_df[shared_df["condition"] == baseline][["model_variant", "field", "value"]]
        .rename(columns={"value": "baseline_value"})
        .drop_duplicates()
    )
    shared_df = shared_df.merge(baseline_map, on=["model_variant", "field"], how="left")
    shared_df["delta_vs_baseline"] = shared_df["value"] - shared_df["baseline_value"]
    return shared_df.sort_values(["model_variant", "field", "condition"]).reset_index(drop=True)


def _shared_family_deltas(
    all_family_df: pd.DataFrame,
    *,
    focus_scope: str,
    baseline: str,
    conditions: list[str],
) -> pd.DataFrame:
    df = all_family_df.copy()
    df = df[(df["scope"] == focus_scope) & (df["macro_f1_mean"].notna()) & (df["model_key"].notna())]
    if df.empty:
        return pd.DataFrame(
            columns=["model_key", "family_label", "condition", "macro_f1_mean", "baseline_value", "delta_vs_baseline"]
        )

    key_counts = (
        df[["model_key", "family_label", "condition"]]
        .drop_duplicates()
        .groupby(["model_key", "family_label"], as_index=False)["condition"]
        .nunique()
        .rename(columns={"condition": "n_conditions"})
    )
    shared_keys = key_counts[key_counts["n_conditions"] == len(conditions)][["model_key", "family_label"]]
    if shared_keys.empty:
        return pd.DataFrame(
            columns=["model_key", "family_label", "condition", "macro_f1_mean", "baseline_value", "delta_vs_baseline"]
        )

    shared_df = df.merge(shared_keys, on=["model_key", "family_label"], how="inner")
    baseline_map = (
        shared_df[shared_df["condition"] == baseline][["model_key", "family_label", "macro_f1_mean"]]
        .rename(columns={"macro_f1_mean": "baseline_value"})
        .drop_duplicates()
    )
    shared_df = shared_df.merge(baseline_map, on=["model_key", "family_label"], how="left")
    shared_df["delta_vs_baseline"] = shared_df["macro_f1_mean"] - shared_df["baseline_value"]
    return shared_df.sort_values(["model_key", "family_label", "condition"]).reset_index(drop=True)


def _write_report(
    *,
    report_path: Path,
    title: str,
    baseline: str,
    focus_scope: str,
    conditions: list[ConditionSpec],
    coverage_df: pd.DataFrame,
    global_df: pd.DataFrame,
    item_delta_df: pd.DataFrame,
    family_delta_df: pd.DataFrame,
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
    lines.append("- Two views are reported: (1) all-available summaries and (2) fair-intersection deltas.")
    lines.append(
        "- Fair-intersection deltas only compare model/field (or model/family) keys present in **all** conditions to avoid confounding from missing models or differing field coverage."
    )
    lines.append("- Deltas are computed against the baseline condition shown below.")
    lines.append("")
    lines.append(f"**Baseline condition:** `{baseline}`")
    lines.append("")
    lines.append(f"**Primary scope analyzed:** `{focus_scope}`")
    lines.append("")
    lines.append("## Coverage summary")
    lines.append("")
    lines.append(_dataframe_to_markdown(coverage_df))
    lines.append("")
    lines.append("## Combined figure (pipeline vs inference)")
    lines.append("")
    lines.append(
        "Panel A shows coverage change induced by preprocessing (usable records and valid LLM artifacts, normalized to baseline = 100%). "
        "Panel B shows outcome change (global Macro F1 bars) and mean family-level deltas vs baseline (line)."
    )
    lines.append("")
    lines.append("![Pipeline vs inference composite](figures/pipeline_performance_composite.png)")
    lines.append("")
    lines.append("## Global comparison summary")
    lines.append("")
    lines.append(_dataframe_to_markdown(global_df))
    lines.append("")
    lines.append("## Fair-intersection deltas (model x field)")
    lines.append("")
    if item_delta_df.empty:
        lines.append("No shared model+field keys were available across all conditions for the selected scope.")
    else:
        agg_item = (
            item_delta_df.groupby("condition", as_index=False)["delta_vs_baseline"]
            .mean()
            .rename(columns={"delta_vs_baseline": "mean_delta_macro_f1"})
        )
        lines.append(_dataframe_to_markdown(agg_item))
    lines.append("")
    lines.append("## Fair-intersection deltas (model x family)")
    lines.append("")
    if family_delta_df.empty:
        lines.append("No shared model+family keys were available across all conditions for the selected scope.")
    else:
        agg_family = (
            family_delta_df.groupby(["family_label", "condition"], as_index=False)["delta_vs_baseline"]
            .mean()
        )
        lines.append(_dataframe_to_markdown(agg_family))
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "- Pipeline effects are read first (Panel A): large shifts in usable rows or artifact validity indicate preprocessing-induced evaluation frame changes."
    )
    lines.append(
        "- Inference effects are read second (Panel B): differences in Macro F1 and family deltas after controlling for baseline indicate model-behavior changes under the available evidence frame."
    )
    lines.append(
        "- Positive delta means the condition improved Macro F1 relative to baseline under fair intersection constraints."
    )
    lines.append(
        "- Negative delta means the condition reduced Macro F1 relative to baseline for the same shared model-field support."
    )
    lines.append(
        "- When coverage differs strongly between conditions, prioritize fair-intersection deltas over all-available summaries for causal interpretation."
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

    item_delta_df = _shared_item_deltas(
        all_metrics_df,
        focus_scope=focus_scope,
        metric_col=metric_col,
        baseline=baseline_name,
        conditions=condition_names,
    )
    family_delta_df = _shared_family_deltas(
        all_family_df,
        focus_scope=focus_scope,
        baseline=baseline_name,
        conditions=condition_names,
    )

    coverage_csv = output_dir / "coverage_comparison.csv"
    global_csv = output_dir / "global_comparison_summary.csv"
    family_csv = output_dir / "family_comparison_summary.csv"
    item_delta_csv = output_dir / "shared_model_field_deltas.csv"
    family_delta_csv = output_dir / "shared_model_family_deltas.csv"
    report_md = output_dir / "preprocessing_effect_report.md"

    coverage_df.to_csv(coverage_csv, index=False)
    global_df.to_csv(global_csv, index=False)
    all_family_df.to_csv(family_csv, index=False)
    item_delta_df.to_csv(item_delta_csv, index=False)
    family_delta_df.to_csv(family_delta_csv, index=False)

    composite_fig = figures_dir / "pipeline_performance_composite.png"
    if not global_df.empty:
        _figure_pipeline_performance(
            coverage_df=coverage_df,
            global_df=global_df,
            family_delta_df=family_delta_df,
            baseline=baseline_name,
            path=composite_fig,
        )

    _write_report(
        report_path=report_md,
        title=title,
        baseline=baseline_name,
        focus_scope=focus_scope,
        conditions=conditions,
        coverage_df=coverage_df,
        global_df=global_df,
        item_delta_df=item_delta_df,
        family_delta_df=family_delta_df,
    )

    return {
        "report_file": str(report_md),
        "coverage_csv": str(coverage_csv),
        "global_csv": str(global_csv),
        "family_csv": str(family_csv),
        "item_delta_csv": str(item_delta_csv),
        "family_delta_csv": str(family_delta_csv),
        "composite_figure": str(composite_fig),
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
