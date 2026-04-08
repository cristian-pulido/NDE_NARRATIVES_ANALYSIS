from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_scope_model(comparison: str) -> tuple[str, str | None]:
    if ":" in comparison:
        scope, model_key = comparison.split(":", 1)
        return scope, model_key
    return comparison, None


def _field_family(field: str) -> str:
    if field.endswith("_tone"):
        return "tone"
    if field.startswith("m8_"):
        return "m8"
    if field.startswith("m9_"):
        return "m9"
    return "other"


def _family_label(family: str) -> str:
    labels = {
        "tone": "Tone",
        "m8": "NDE-C",
        "m9": "NDE-MCQ",
        "other": "Other",
    }
    return labels.get(family, family)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence_level: float,
) -> dict[str, float]:
    clean_values = values[np.isfinite(values)]
    if clean_values.size == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "std_bootstrap": float("nan"),
        }
    point_estimate = float(clean_values.mean())
    if clean_values.size == 1:
        return {
            "mean": point_estimate,
            "ci_low": point_estimate,
            "ci_high": point_estimate,
            "std_bootstrap": 0.0,
        }

    sample_idx = rng.integers(
        low=0,
        high=clean_values.size,
        size=(int(n_bootstrap), clean_values.size),
    )
    sample_means = clean_values[sample_idx].mean(axis=1)
    alpha = 1.0 - confidence_level
    ci_low = float(np.quantile(sample_means, alpha / 2.0))
    ci_high = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    std_bootstrap = float(np.std(sample_means, ddof=1))
    return {
        "mean": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std_bootstrap": std_bootstrap,
    }


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(column) for column in df.columns]
        header_line = "| " + " | ".join(headers) + " |"
        divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines = [header_line, divider_line]
        for row in df.itertuples(index=False, name=None):
            values = ["" if pd.isna(value) else str(value) for value in row]
            escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append("| " + " | ".join(escaped) + " |")
        return "\n".join(lines)


def _compute_uncertainty_table(
    metrics_df: pd.DataFrame,
    *,
    group_cols: list[str],
    level: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence_level: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = metrics_df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = dict(zip(group_cols, keys, strict=True))

        kappa_stats = _bootstrap_mean_ci(
            group["cohen_kappa"].to_numpy(dtype=float),
            rng=rng,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        macro_f1_stats = _bootstrap_mean_ci(
            group["macro_f1"].to_numpy(dtype=float),
            rng=rng,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )

        rows.append(
            {
                "level": level,
                "n_rows": int(len(group)),
                **key_payload,
                "cohen_kappa_mean": kappa_stats["mean"],
                "cohen_kappa_ci_low": kappa_stats["ci_low"],
                "cohen_kappa_ci_high": kappa_stats["ci_high"],
                "cohen_kappa_bootstrap_std": kappa_stats["std_bootstrap"],
                "macro_f1_mean": macro_f1_stats["mean"],
                "macro_f1_ci_low": macro_f1_stats["ci_low"],
                "macro_f1_ci_high": macro_f1_stats["ci_high"],
                "macro_f1_bootstrap_std": macro_f1_stats["std_bootstrap"],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "level",
                *group_cols,
                "n_rows",
                "cohen_kappa_mean",
                "cohen_kappa_ci_low",
                "cohen_kappa_ci_high",
                "cohen_kappa_bootstrap_std",
                "macro_f1_mean",
                "macro_f1_ci_low",
                "macro_f1_ci_high",
                "macro_f1_bootstrap_std",
            ]
        )
    return pd.DataFrame(rows)


def _save_figure(
    fig: plt.Figure, path: Path, *, dpi: int, export_pdf: bool
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    written = {str(path.stem): str(path)}
    if export_pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        written[f"{path.stem}_pdf"] = str(pdf_path)
    plt.close(fig)
    return written


def _plot_scope_ci(
    scope_df: pd.DataFrame, figure_path: Path, *, dpi: int, export_pdf: bool
) -> dict[str, str]:
    plot_df = scope_df.copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No scope-level uncertainty rows available",
            ha="center",
            va="center",
        )
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    plot_df = plot_df.sort_values("macro_f1_mean", ascending=False).reset_index(
        drop=True
    )
    y_pos = np.arange(len(plot_df))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, max(4.8, 0.75 * len(plot_df) + 2.6)), sharey=True
    )

    f1_lower = plot_df["macro_f1_mean"] - plot_df["macro_f1_ci_low"]
    f1_upper = plot_df["macro_f1_ci_high"] - plot_df["macro_f1_mean"]
    ax1.errorbar(
        x=plot_df["macro_f1_mean"],
        y=y_pos,
        xerr=[f1_lower, f1_upper],
        fmt="o",
        color="#1F6F8B",
        ecolor="#1F6F8B",
        capsize=3,
        linewidth=1.3,
    )
    ax1.set_xlabel("Macro F1 mean (bootstrap CI)")
    ax1.set_ylabel("Scope")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_df["scope"].astype(str).tolist())
    ax1.grid(axis="x", alpha=0.25)
    ax1.set_xlim(
        0.0, min(1.0, max(0.05, float(plot_df["macro_f1_ci_high"].max()) + 0.05))
    )

    kappa_lower = plot_df["cohen_kappa_mean"] - plot_df["cohen_kappa_ci_low"]
    kappa_upper = plot_df["cohen_kappa_ci_high"] - plot_df["cohen_kappa_mean"]
    ax2.errorbar(
        x=plot_df["cohen_kappa_mean"],
        y=y_pos,
        xerr=[kappa_lower, kappa_upper],
        fmt="o",
        color="#B23A48",
        ecolor="#B23A48",
        capsize=3,
        linewidth=1.3,
    )
    ax2.axvline(0.0, color="#444444", linestyle="--", linewidth=0.9, alpha=0.8)
    ax2.set_xlabel("Cohen kappa mean (bootstrap CI)")
    ax2.grid(axis="x", alpha=0.25)
    min_kappa = float(plot_df["cohen_kappa_ci_low"].min())
    max_kappa = float(plot_df["cohen_kappa_ci_high"].max())
    pad = max(0.03, 0.1 * (max_kappa - min_kappa if max_kappa > min_kappa else 1.0))
    ax2.set_xlim(max(-1.0, min_kappa - pad), min(1.0, max_kappa + pad))

    fig.suptitle("Bootstrap uncertainty by evaluation scope", fontsize=13)
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def _plot_questionnaire_family_ci(
    comparison_family_df: pd.DataFrame,
    figure_path: Path,
    *,
    dpi: int,
    export_pdf: bool,
) -> dict[str, str]:
    plot_df = comparison_family_df.copy()
    plot_df = plot_df[
        (plot_df["scope"] == "questionnaire_vs_llm")
        & (plot_df["family"].isin(["tone", "m8", "m9"]))
    ]
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No questionnaire_vs_llm family uncertainty rows available",
            ha="center",
            va="center",
        )
        return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)

    plot_df = plot_df.sort_values(
        ["family", "macro_f1_mean"], ascending=[True, False]
    ).reset_index(drop=True)
    labels = [
        f"{_family_label(str(row.family))} | {row.model_key}"
        for row in plot_df.itertuples()
    ]
    y_pos = np.arange(len(plot_df))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, max(5.2, 0.4 * len(plot_df) + 2.8)), sharey=True
    )

    f1_lower = plot_df["macro_f1_mean"] - plot_df["macro_f1_ci_low"]
    f1_upper = plot_df["macro_f1_ci_high"] - plot_df["macro_f1_mean"]
    ax1.errorbar(
        x=plot_df["macro_f1_mean"],
        y=y_pos,
        xerr=[f1_lower, f1_upper],
        fmt="o",
        color="#2A9D8F",
        ecolor="#2A9D8F",
        capsize=2.8,
        linewidth=1.2,
    )
    ax1.set_xlabel("Macro F1 mean (bootstrap CI)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.grid(axis="x", alpha=0.25)
    ax1.set_xlim(
        0.0, min(1.0, max(0.05, float(plot_df["macro_f1_ci_high"].max()) + 0.05))
    )

    kappa_lower = plot_df["cohen_kappa_mean"] - plot_df["cohen_kappa_ci_low"]
    kappa_upper = plot_df["cohen_kappa_ci_high"] - plot_df["cohen_kappa_mean"]
    ax2.errorbar(
        x=plot_df["cohen_kappa_mean"],
        y=y_pos,
        xerr=[kappa_lower, kappa_upper],
        fmt="o",
        color="#E76F51",
        ecolor="#E76F51",
        capsize=2.8,
        linewidth=1.2,
    )
    ax2.axvline(0.0, color="#444444", linestyle="--", linewidth=0.9, alpha=0.8)
    ax2.set_xlabel("Cohen kappa mean (bootstrap CI)")
    ax2.grid(axis="x", alpha=0.25)
    min_kappa = float(plot_df["cohen_kappa_ci_low"].min())
    max_kappa = float(plot_df["cohen_kappa_ci_high"].max())
    pad = max(0.03, 0.1 * (max_kappa - min_kappa if max_kappa > min_kappa else 1.0))
    ax2.set_xlim(max(-1.0, min_kappa - pad), min(1.0, max_kappa + pad))

    fig.suptitle("Questionnaire vs LLM uncertainty by family/model", fontsize=13)
    return _save_figure(fig, figure_path, dpi=dpi, export_pdf=export_pdf)


def _write_uncertainty_report(
    *,
    report_path: Path,
    input_dir: Path,
    output_dir: Path,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
    scope_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    scope_family_df: pd.DataFrame,
    comparison_family_df: pd.DataFrame,
) -> None:
    ci_pct = int(round(confidence_level * 100))
    lines: list[str] = [
        "# Bootstrap Uncertainty Report",
        "",
        "## Objective",
        "",
        "Estimate uncertainty intervals for average Cohen kappa and Macro F1 metrics using bootstrap resampling over evaluation metric rows.",
        "",
        "## Inputs And Parameters",
        "",
        f"- Input evaluation directory: `{input_dir}`",
        f"- Output uncertainty directory: `{output_dir}`",
        f"- Bootstrap replicates: `{n_bootstrap}`",
        f"- Confidence level: `{confidence_level:.3f}` ({ci_pct}% CI)",
        f"- Random seed: `{random_seed}`",
        "",
        "## Scope-Level Uncertainty",
        "",
        _dataframe_to_markdown(
            scope_df[
                [
                    "scope",
                    "n_rows",
                    "cohen_kappa_mean",
                    "cohen_kappa_ci_low",
                    "cohen_kappa_ci_high",
                    "macro_f1_mean",
                    "macro_f1_ci_low",
                    "macro_f1_ci_high",
                ]
            ].sort_values("macro_f1_mean", ascending=False)
        ),
        "",
        "![Scope uncertainty](figures/uncertainty_scope_ci.png)",
        "",
        "## Questionnaire Vs LLM Family Uncertainty",
        "",
        _dataframe_to_markdown(
            comparison_family_df[
                comparison_family_df["scope"] == "questionnaire_vs_llm"
            ][
                [
                    "comparison",
                    "family",
                    "n_rows",
                    "cohen_kappa_mean",
                    "cohen_kappa_ci_low",
                    "cohen_kappa_ci_high",
                    "macro_f1_mean",
                    "macro_f1_ci_low",
                    "macro_f1_ci_high",
                ]
            ].sort_values(["family", "macro_f1_mean"], ascending=[True, False])
        ),
        "",
        "![Questionnaire family uncertainty](figures/questionnaire_llm_family_ci.png)",
        "",
        "## Comparison-Level Snapshot",
        "",
        _dataframe_to_markdown(
            comparison_df[
                [
                    "comparison",
                    "scope",
                    "n_rows",
                    "cohen_kappa_mean",
                    "cohen_kappa_ci_low",
                    "cohen_kappa_ci_high",
                    "macro_f1_mean",
                    "macro_f1_ci_low",
                    "macro_f1_ci_high",
                ]
            ].sort_values(["scope", "macro_f1_mean"], ascending=[True, False])
        ),
        "",
        "## Scope-Family Slice",
        "",
        _dataframe_to_markdown(
            scope_family_df[
                [
                    "scope",
                    "family",
                    "n_rows",
                    "cohen_kappa_mean",
                    "cohen_kappa_ci_low",
                    "cohen_kappa_ci_high",
                    "macro_f1_mean",
                    "macro_f1_ci_low",
                    "macro_f1_ci_high",
                ]
            ].sort_values(["scope", "family"])
        ),
        "",
        "## Interpretation Notes",
        "",
        "- Wider intervals indicate less stable average performance for that grouping under row-level resampling.",
        "- Comparisons whose CIs overlap strongly should not be interpreted as clearly separated without additional evidence.",
        "- The questionnaire-vs-LLM slice is the primary lens for this report, while other scopes are retained for context.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_uncertainty_analysis(
    *,
    input_dir: Path,
    output_dir: Path | None = None,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    figure_dpi: int = 300,
    export_figures_pdf: bool = False,
) -> dict[str, str]:
    resolved_input_dir = Path(input_dir).resolve()
    if not resolved_input_dir.exists():
        raise FileNotFoundError(
            f"Evaluation input directory not found: {resolved_input_dir}"
        )
    if not resolved_input_dir.is_dir():
        raise NotADirectoryError(
            f"Evaluation input path is not a directory: {resolved_input_dir}"
        )
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100 for stable intervals")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    metrics_path = resolved_input_dir / "evaluation_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing required file: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    required_columns = {"comparison", "field", "cohen_kappa", "macro_f1"}
    missing = sorted(required_columns - set(metrics_df.columns))
    if missing:
        raise ValueError(
            f"evaluation_metrics.csv is missing required columns: {missing}"
        )

    prepared = metrics_df.copy()
    prepared["comparison"] = prepared["comparison"].astype(str)
    prepared["field"] = prepared["field"].astype(str)
    prepared[["scope", "model_key"]] = prepared["comparison"].apply(
        lambda value: pd.Series(_parse_scope_model(str(value)))
    )
    prepared["family"] = prepared["field"].map(lambda value: _field_family(str(value)))
    prepared["cohen_kappa"] = prepared["cohen_kappa"].map(_safe_float)
    prepared["macro_f1"] = prepared["macro_f1"].map(_safe_float)
    prepared = prepared[
        prepared["cohen_kappa"].notna() & prepared["macro_f1"].notna()
    ].copy()
    if prepared.empty:
        raise ValueError("No valid rows with both cohen_kappa and macro_f1 were found.")

    out_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (resolved_input_dir / "uncertainty").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=int(random_seed))
    comparison_df = (
        _compute_uncertainty_table(
            prepared,
            group_cols=["comparison", "scope", "model_key"],
            level="comparison",
            rng=rng,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        .sort_values(["scope", "comparison"])
        .reset_index(drop=True)
    )
    scope_df = (
        _compute_uncertainty_table(
            prepared,
            group_cols=["scope"],
            level="scope",
            rng=rng,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        .sort_values(["scope"])
        .reset_index(drop=True)
    )
    scope_family_df = (
        _compute_uncertainty_table(
            prepared,
            group_cols=["scope", "family"],
            level="scope_family",
            rng=rng,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        .sort_values(["scope", "family"])
        .reset_index(drop=True)
    )
    comparison_family_df = (
        _compute_uncertainty_table(
            prepared,
            group_cols=["comparison", "scope", "model_key", "family"],
            level="comparison_family",
            rng=rng,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        .sort_values(["scope", "family", "comparison"])
        .reset_index(drop=True)
    )

    comparison_path = out_dir / "uncertainty_comparison.csv"
    scope_path = out_dir / "uncertainty_scope.csv"
    scope_family_path = out_dir / "uncertainty_scope_family.csv"
    comparison_family_path = out_dir / "uncertainty_comparison_family.csv"
    report_path = out_dir / "uncertainty_report.md"

    comparison_df.to_csv(comparison_path, index=False)
    scope_df.to_csv(scope_path, index=False)
    scope_family_df.to_csv(scope_family_path, index=False)
    comparison_family_df.to_csv(comparison_family_path, index=False)

    scope_figure = figures_dir / "uncertainty_scope_ci.png"
    family_figure = figures_dir / "questionnaire_llm_family_ci.png"
    written_figures = {}
    written_figures.update(
        _plot_scope_ci(
            scope_df, scope_figure, dpi=figure_dpi, export_pdf=export_figures_pdf
        )
    )
    written_figures.update(
        _plot_questionnaire_family_ci(
            comparison_family_df,
            family_figure,
            dpi=figure_dpi,
            export_pdf=export_figures_pdf,
        )
    )

    _write_uncertainty_report(
        report_path=report_path,
        input_dir=resolved_input_dir,
        output_dir=out_dir,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
        scope_df=scope_df,
        comparison_df=comparison_df,
        scope_family_df=scope_family_df,
        comparison_family_df=comparison_family_df,
    )

    ci_pct = int(round(confidence_level * 100))
    return {
        "input_dir": str(resolved_input_dir),
        "output_dir": str(out_dir),
        "report_file": str(report_path),
        "comparison_uncertainty_file": str(comparison_path),
        "scope_uncertainty_file": str(scope_path),
        "scope_family_uncertainty_file": str(scope_family_path),
        "comparison_family_uncertainty_file": str(comparison_family_path),
        "scope_figure": str(scope_figure),
        "questionnaire_family_figure": str(family_figure),
        "bootstrap_samples": str(int(n_bootstrap)),
        "confidence_interval": f"{ci_pct}%",
        **{f"figure_{key}": value for key, value in written_figures.items()},
    }
