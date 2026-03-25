from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .config import BenchmarkConfig, BenchmarkExperimentConfig, BenchmarkRuntimeConfig, PathsConfig
from .constants import PROJECT_ROOT
from .llm.ollama import OllamaProvider
from .llm.types import LLMRequest


BENCHMARK_LABELS = ("negative", "neutral", "positive")
DEFAULT_BENCHMARK_PROMPT = "sentiment_prompt.md"
LABEL_PRIORITY = ("negative", "neutral", "positive", "mixed")


@dataclass(frozen=True)
class BenchmarkArtifactPaths:
    raw_file: Path
    processed_file: Path
    manifest_file: Path


def _now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _map_amazon_label_to_sentiment(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in BENCHMARK_LABELS:
            return normalized
    try:
        label_value = int(value)
    except (TypeError, ValueError):
        return None

    # Amazon stars: 1..5
    if 1 <= label_value <= 5:
        if label_value <= 2:
            return "negative"
        if label_value == 3:
            return "neutral"
        return "positive"

    # Common preprocessed class ids: 0..2
    if label_value == 0:
        return "negative"
    if label_value == 1:
        return "neutral"
    if label_value == 2:
        return "positive"
    return None


def _map_imdb_label_to_sentiment(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"negative", "positive"}:
            return normalized
    try:
        label_value = int(value)
    except (TypeError, ValueError):
        return None
    if label_value == 0:
        return "negative"
    if label_value == 1:
        return "positive"
    return None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_dataset_display_name(dataset_name: object, *, input_path: object | None = None) -> str:
    raw_name = str(dataset_name or "").strip()
    normalized = raw_name.lower()
    if raw_name and normalized not in {"n/a", "none", "nan", "user_provided_csv"}:
        return raw_name

    if input_path:
        stem = Path(str(input_path)).stem.strip()
        if stem:
            return stem

    return raw_name or "dataset"


def _resolve_local_dataset_name(benchmark: BenchmarkConfig, dataset_path: Path) -> str:
    configured_name = str(getattr(benchmark.dataset, "dataset_name", "") or "").strip()
    # Keep explicit custom names, but avoid leaking the default external benchmark label for local CSVs.
    if configured_name and configured_name != "amazon_reviews_multi":
        return configured_name
    csv_stem = dataset_path.stem.strip()
    if csv_stem:
        return csv_stem
    return configured_name or "user_provided_csv"


def _normalize_model_family(artifact_id: object, source: object) -> str:
    source_name = str(source or "").strip().lower()
    if source_name == "vader":
        return "vader"

    model_id = str(artifact_id or "").strip()
    base = model_id.split("__", 1)[0] if model_id else "unknown"
    base = re.sub(r"_[0-9]+(?:\.[0-9]+)?$", "", base)
    return base or "unknown"


def _plot_benchmark_scatter(
    metrics_df: pd.DataFrame,
    *,
    figure_path: Path,
    per_label_df: pd.DataFrame | None = None,
    dpi: int = 300,
) -> Path | None:
    if metrics_df.empty:
        return None

    plot_df = metrics_df.copy()
    plot_df["macro_f1"] = pd.to_numeric(plot_df["macro_f1"], errors="coerce")
    plot_df["cohen_kappa"] = pd.to_numeric(plot_df["cohen_kappa"], errors="coerce")
    plot_df = plot_df.dropna(subset=["macro_f1", "cohen_kappa"])
    if plot_df.empty:
        return None

    plot_df["dataset_display"] = [
        _normalize_dataset_display_name(row.get("dataset_name"), input_path=row.get("input_path"))
        for _, row in plot_df.iterrows()
    ]
    plot_df["model_family"] = [
        _normalize_model_family(row.get("artifact_id"), row.get("source"))
        for _, row in plot_df.iterrows()
    ]

    families = sorted(plot_df["model_family"].astype(str).unique().tolist())
    datasets = sorted(plot_df["dataset_display"].astype(str).unique().tolist())

    family_cmap = plt.get_cmap("tab20")
    family_colors = {family: family_cmap(index % 20) for index, family in enumerate(families)}
    dataset_cmap = plt.get_cmap("Set2")
    dataset_edge_colors = {dataset: dataset_cmap(index % 8) for index, dataset in enumerate(datasets)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    dataset_markers = {dataset: marker_cycle[index % len(marker_cycle)] for index, dataset in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in plot_df.iterrows():
        family = str(row["model_family"])
        dataset = str(row["dataset_display"])
        x_value = float(row["macro_f1"])
        y_value = float(row["cohen_kappa"])
        ax.scatter(
            x_value,
            y_value,
            c=[family_colors[family]],
            marker=dataset_markers[dataset],
            s=130,
            alpha=0.88,
            edgecolors=[dataset_edge_colors[dataset]],
            linewidths=1.2,
            zorder=3,
        )

        if per_label_df is not None and not per_label_df.empty:
            label_subset = per_label_df[
                (per_label_df["artifact_id"].astype(str) == str(row.get("artifact_id", "")))
                & (per_label_df["source"].astype(str) == str(row.get("source", "")))
                & (per_label_df["dataset_name"].astype(str) == str(row.get("dataset_name", "")))
                & (per_label_df["dataset_config"].astype(str) == str(row.get("dataset_config", "")))
            ].copy()
            if not label_subset.empty:
                emoji_by_label = {"positive": "😀", "neutral": "😐", "negative": "☹️"}
                label_subset["label"] = label_subset["label"].astype(str).str.lower()
                f1_by_label = {
                    str(item["label"]): float(item["f1"])
                    for _, item in label_subset.iterrows()
                    if str(item.get("label", "")).lower() in emoji_by_label
                }
                ordered_labels = [label for label in ("positive", "neutral", "negative") if label in f1_by_label]
                emoji_offsets = {"positive": -6, "neutral": 0, "negative": 6}
                emoji_colors = {"positive": "#1B9E3E", "neutral": "#6E6E6E", "negative": "#C62828"}
                for idx, label in enumerate(ordered_labels):
                    f1_value = min(1.0, max(0.0, float(f1_by_label[label])))
                    alpha = 0.12 + (0.88 * f1_value)
                    ax.annotate(
                        emoji_by_label[label],
                        (x_value, y_value),
                        xytext=(emoji_offsets.get(label, idx * 6), 0),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=emoji_colors.get(label, "#333333"),
                        alpha=alpha,
                        zorder=4,
                    )

    best_row = plot_df.sort_values(["macro_f1", "cohen_kappa"], ascending=False).iloc[0]
    worst_row = plot_df.sort_values(["macro_f1", "cohen_kappa"], ascending=True).iloc[0]
    for labeled_row in (best_row, worst_row):
        ax.annotate(
            str(labeled_row.get("artifact_id", "n/a")),
            (float(labeled_row["macro_f1"]), float(labeled_row["cohen_kappa"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=family_colors[family],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=family,
        )
        for family in families
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=dataset_markers[dataset],
            linestyle="None",
            markerfacecolor="#f5f5f5",
            markeredgecolor=dataset_edge_colors[dataset],
            markeredgewidth=1.3,
            markersize=8,
            label=dataset,
        )
        for dataset in datasets
    ]

    family_legend = ax.legend(handles=family_handles, title="Model family", loc="lower right")
    ax.add_artist(family_legend)
    ax.legend(handles=dataset_handles, title="Dataset", loc="upper left")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Macro F1")
    ax.set_ylabel("Cohen kappa")
    ax.set_title("Benchmark performance: macro_f1 vs cohen_kappa")
    ax.grid(True, alpha=0.25)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _dataset_to_frame(dataset: Any, *, text_column: str, label_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in dataset:
        text = str(row.get(text_column, "")).strip()
        label = _map_amazon_label_to_sentiment(row.get(label_column))
        if not text or label is None:
            continue
        raw_label = row.get(label_column)
        star_rating: int | None = None
        try:
            star_rating = int(raw_label)  # optional metadata when available
        except (TypeError, ValueError):
            star_rating = None
        rows.append(
            {
                "text": text,
                "gold_label": label,
                "star_rating": star_rating,
            }
        )
    return pd.DataFrame(rows, columns=["text", "gold_label", "star_rating"])


def _dataset_to_frame_imdb(dataset: Any, *, text_column: str, label_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in dataset:
        text = str(row.get(text_column, "")).strip()
        label = _map_imdb_label_to_sentiment(row.get(label_column))
        if not text or label is None:
            continue
        raw_label = row.get(label_column)
        class_id: int | None = None
        try:
            class_id = int(raw_label)
        except (TypeError, ValueError):
            class_id = None
        rows.append(
            {
                "text": text,
                "gold_label": label,
                "class_id": class_id,
            }
        )
    return pd.DataFrame(rows, columns=["text", "gold_label", "class_id"])


def download_and_prepare_amazon_benchmark(
    paths: PathsConfig,
    benchmark: BenchmarkConfig,
    *,
    max_rows: int | None = None,
    output_raw_dir: Path | None = None,
    output_processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, BenchmarkArtifactPaths, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by CLI users without dependency
        raise RuntimeError(
            "Missing optional dependency 'datasets'. Install dependencies with: pip install -e .[dev]"
        ) from exc

    dataset_cfg = benchmark.dataset
    effective_max_rows = int(max_rows if max_rows is not None else dataset_cfg.max_rows)
    if effective_max_rows < 1:
        raise ValueError("max_rows must be >= 1")

    def _load(primary_name: str, primary_config: str | None, primary_split: str) -> Any:
        if primary_config in {None, "", "none", "null"}:
            return load_dataset(primary_name, split=primary_split)
        return load_dataset(primary_name, primary_config, split=primary_split)

    used_dataset_name = dataset_cfg.dataset_name
    used_dataset_config = dataset_cfg.dataset_config
    used_split = dataset_cfg.split
    used_text_column = dataset_cfg.text_column
    used_label_column = dataset_cfg.label_column

    try:
        dataset = _load(dataset_cfg.dataset_name, dataset_cfg.dataset_config, dataset_cfg.split)
        frame = _dataset_to_frame(dataset, text_column=dataset_cfg.text_column, label_column=dataset_cfg.label_column)
    except Exception as exc:
        message = str(exc).lower()
        fallback_candidates = [
            ("SetFit/amazon_reviews_multi_en", None, "train", "text", "label"),
            ("mteb/amazon_reviews_multi", "en", "test", "text", "label"),
        ]
        if "defunct" not in message and "no longer accessible" not in message:
            raise
        frame = pd.DataFrame()
        last_error = exc
        for candidate_name, candidate_config, candidate_split, candidate_text_column, candidate_label_column in fallback_candidates:
            try:
                dataset = _load(candidate_name, candidate_config, candidate_split)
                candidate_frame = _dataset_to_frame(dataset, text_column=candidate_text_column, label_column=candidate_label_column)
                if candidate_frame.empty:
                    continue
                frame = candidate_frame
                used_dataset_name = candidate_name
                used_dataset_config = candidate_config or ""
                used_split = candidate_split
                used_text_column = candidate_text_column
                used_label_column = candidate_label_column
                break
            except Exception as fallback_exc:  # noqa: BLE001
                last_error = fallback_exc
                continue
        if frame.empty:
            raise RuntimeError(
                "Could not download an Amazon benchmark dataset. "
                "Use benchmark-run --dataset-path with a local normalized CSV as fallback."
            ) from last_error

    if frame.empty:
        raise ValueError("No valid Amazon benchmark rows were extracted from the selected dataset configuration.")

    sampled = (
        frame.sample(n=min(effective_max_rows, len(frame)), random_state=dataset_cfg.random_state, replace=False)
        .reset_index(drop=True)
        .copy()
    )
    sampled.insert(0, "record_id", [f"amazon_{index + 1:07d}" for index in range(len(sampled))])
    sampled.insert(1, "split", used_split)
    sampled.insert(2, "source_dataset", used_dataset_name)
    sampled.insert(3, "source_config", used_dataset_config)

    raw_dir = _ensure_dir(Path(output_raw_dir or paths.benchmark_raw_dir or (paths.evaluation_output_dir / "benchmark_raw")))
    processed_dir = _ensure_dir(
        Path(output_processed_dir or paths.benchmark_processed_dir or (paths.evaluation_output_dir / "benchmark_processed"))
    )

    raw_file = raw_dir / "amazon_reviews_multi_raw.jsonl"
    processed_file = processed_dir / "amazon_reviews_multi_normalized.csv"
    manifest_file = processed_dir / "amazon_reviews_multi_manifest.json"

    sampled.to_json(raw_file, orient="records", lines=True, force_ascii=False)
    sampled.to_csv(processed_file, index=False)

    summary = {
        "dataset_name": used_dataset_name,
        "dataset_config": used_dataset_config,
        "split": used_split,
        "text_column": used_text_column,
        "label_column": used_label_column,
        "label_mapping_description": _infer_label_mapping_description(sampled["star_rating"]),
        "rows_total_after_cleaning": int(len(frame)),
        "rows_written": int(len(sampled)),
        "class_distribution": {label: int(count) for label, count in sampled["gold_label"].value_counts().to_dict().items()},
        "generated_at": _now_utc_iso(),
        "raw_file": str(raw_file),
        "processed_file": str(processed_file),
    }
    manifest_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return sampled, BenchmarkArtifactPaths(raw_file=raw_file, processed_file=processed_file, manifest_file=manifest_file), summary


def download_and_prepare_imdb_benchmark(
    paths: PathsConfig,
    benchmark: BenchmarkConfig,
    *,
    max_rows: int | None = None,
    output_raw_dir: Path | None = None,
    output_processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, BenchmarkArtifactPaths, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by CLI users without dependency
        raise RuntimeError(
            "Missing optional dependency 'datasets'. Install dependencies with: pip install -e .[dev]"
        ) from exc

    dataset_cfg = benchmark.dataset
    effective_max_rows = int(max_rows if max_rows is not None else dataset_cfg.max_rows)
    if effective_max_rows < 1:
        raise ValueError("max_rows must be >= 1")

    def _load(primary_name: str, primary_config: str | None, primary_split: str) -> Any:
        if primary_config in {None, "", "none", "null"}:
            return load_dataset(primary_name, split=primary_split)
        return load_dataset(primary_name, primary_config, split=primary_split)

    used_dataset_name = dataset_cfg.dataset_name
    used_dataset_config = dataset_cfg.dataset_config
    used_split = dataset_cfg.split
    used_text_column = dataset_cfg.text_column
    used_label_column = dataset_cfg.label_column

    dataset = _load(dataset_cfg.dataset_name, dataset_cfg.dataset_config, dataset_cfg.split)
    frame = _dataset_to_frame_imdb(dataset, text_column=dataset_cfg.text_column, label_column=dataset_cfg.label_column)
    if frame.empty:
        raise ValueError("No valid IMDB benchmark rows were extracted from the selected dataset configuration.")

    sampled = (
        frame.sample(n=min(effective_max_rows, len(frame)), random_state=dataset_cfg.random_state, replace=False)
        .reset_index(drop=True)
        .copy()
    )
    sampled.insert(0, "record_id", [f"imdb_{index + 1:07d}" for index in range(len(sampled))])
    sampled.insert(1, "split", used_split)
    sampled.insert(2, "source_dataset", used_dataset_name)
    sampled.insert(3, "source_config", used_dataset_config)

    raw_dir = _ensure_dir(Path(output_raw_dir or paths.benchmark_raw_dir or (paths.evaluation_output_dir / "benchmark_raw")))
    processed_dir = _ensure_dir(
        Path(output_processed_dir or paths.benchmark_processed_dir or (paths.evaluation_output_dir / "benchmark_processed"))
    )

    raw_file = raw_dir / "imdb_raw.jsonl"
    processed_file = processed_dir / "imdb_normalized.csv"
    manifest_file = processed_dir / "imdb_manifest.json"

    sampled.to_json(raw_file, orient="records", lines=True, force_ascii=False)
    sampled.to_csv(processed_file, index=False)

    summary = {
        "dataset_name": used_dataset_name,
        "dataset_config": used_dataset_config,
        "split": used_split,
        "text_column": used_text_column,
        "label_column": used_label_column,
        "label_mapping_description": "Binary class-id mapping used: 0 negative, 1 positive",
        "rows_total_after_cleaning": int(len(frame)),
        "rows_written": int(len(sampled)),
        "class_distribution": {label: int(count) for label, count in sampled["gold_label"].value_counts().to_dict().items()},
        "generated_at": _now_utc_iso(),
        "raw_file": str(raw_file),
        "processed_file": str(processed_file),
    }
    manifest_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return sampled, BenchmarkArtifactPaths(raw_file=raw_file, processed_file=processed_file, manifest_file=manifest_file), summary


def download_and_prepare_benchmark_dataset(
    paths: PathsConfig,
    benchmark: BenchmarkConfig,
    *,
    max_rows: int | None = None,
    output_raw_dir: Path | None = None,
    output_processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, BenchmarkArtifactPaths, dict[str, Any]]:
    dataset_name = str(benchmark.dataset.dataset_name).lower()
    if "imdb" in dataset_name:
        return download_and_prepare_imdb_benchmark(
            paths=paths,
            benchmark=benchmark,
            max_rows=max_rows,
            output_raw_dir=output_raw_dir,
            output_processed_dir=output_processed_dir,
        )
    return download_and_prepare_amazon_benchmark(
        paths=paths,
        benchmark=benchmark,
        max_rows=max_rows,
        output_raw_dir=output_raw_dir,
        output_processed_dir=output_processed_dir,
    )


def _default_prompt_path(project_root: Path) -> Path:
    return project_root / "prompts" / "benchmark" / DEFAULT_BENCHMARK_PROMPT


def resolve_benchmark_prompt_path(paths: PathsConfig, prompt_variant: str | None = None) -> Path:
    project_root = PROJECT_ROOT
    default_prompt = _default_prompt_path(project_root)
    if prompt_variant:
        variants_root = Path(paths.benchmark_prompt_variants_dir or (paths.data_dir or paths.path.parent) / "benchmark" / "prompt_variants")
        candidate = variants_root / prompt_variant / DEFAULT_BENCHMARK_PROMPT
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Benchmark prompt variant file not found: {candidate}")
    if not default_prompt.exists():
        raise FileNotFoundError(f"Default benchmark prompt not found: {default_prompt}")
    return default_prompt


def _parse_llm_label(raw_text: str, *, labels: tuple[str, ...]) -> str:
    normalized = raw_text.strip()
    if not normalized:
        raise ValueError("Empty LLM response")
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        label = str(payload.get("label", "")).strip().lower()
        if label in labels:
            return label

    lowered = normalized.lower()
    for label in labels:
        if label in lowered:
            return label
    raise ValueError(f"Could not parse benchmark label from model response: {normalized[:120]}")


def _label_guidance_line(label: str) -> str:
    if label == "negative":
        return "- negative: wording is predominantly unfavorable, critical, distressing, or adverse in tone."
    if label == "neutral":
        return "- neutral: wording is mostly factual/descriptive with little explicit emotional polarity."
    if label == "positive":
        return "- positive: wording is predominantly favorable, appreciative, relieved, or supportive in tone."
    if label == "mixed":
        return "- mixed: explicit positive and negative wording are both present and near-balanced."
    return f"- {label}: use only if the wording clearly matches this label."


def _build_prompt(template: str, text: str, *, labels: tuple[str, ...]) -> str:
    labels_csv = ", ".join(labels)
    labels_schema = "|".join(labels)
    guidance_block = "\n".join(_label_guidance_line(label) for label in labels)

    prompt = template
    prompt = prompt.replace("{{labels_csv}}", labels_csv)
    prompt = prompt.replace("{{labels_schema}}", labels_schema)
    prompt = prompt.replace("{{label_guidance}}", guidance_block)
    prompt = prompt.replace("{{text}}", text)

    if "{{labels_csv}}" not in template and "{{labels_schema}}" not in template:
        dynamic_header = (
            "Task context: classify writing tone (not inferred event severity).\n"
            f"Allowed labels: {labels_csv}.\n"
            f"Return strict JSON with this schema: {{\"label\": \"{labels_schema}\"}}\n\n"
        )
        prompt = dynamic_header + prompt
    return prompt


def _infer_active_labels(frame: pd.DataFrame) -> tuple[str, ...]:
    labels_in_data = {str(value).strip().lower() for value in frame.get("gold_label", pd.Series(dtype=str)).dropna().tolist()}
    ordered = [label for label in LABEL_PRIORITY if label in labels_in_data]
    extras = sorted(label for label in labels_in_data if label not in LABEL_PRIORITY)
    labels = tuple(ordered + extras)
    if not labels:
        return BENCHMARK_LABELS
    return labels


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _infer_label_mapping_description(values: pd.Series) -> str:
    non_null = values.dropna()
    if non_null.empty:
        return "Labels already normalized in source dataset"

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        min_value = int(numeric.min())
        max_value = int(numeric.max())
        unique_values = set(int(value) for value in numeric.unique().tolist())
        if min_value >= 1 and max_value <= 5:
            return "Star mapping used: 1-2 negative, 3 neutral, 4-5 positive"
        if unique_values.issubset({0, 1, 2}):
            return "Class-id mapping used: 0 negative, 1 neutral, 2 positive"
        if unique_values.issubset({0, 1}):
            return "Binary class-id mapping used: 0 negative, 1 positive"

    return "Text labels mapped directly when matching negative/neutral/positive"


def _cohen_kappa(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = BENCHMARK_LABELS) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    p0 = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == predicted) / n
    expected_counts = {label: 0 for label in labels}
    predicted_counts = {label: 0 for label in labels}
    for label in y_true:
        expected_counts[label] = expected_counts.get(label, 0) + 1
    for label in y_pred:
        predicted_counts[label] = predicted_counts.get(label, 0) + 1
    pe = sum((expected_counts[label] / n) * (predicted_counts[label] / n) for label in labels)
    return _safe_div((p0 - pe), (1.0 - pe)) if pe < 1.0 else 0.0


def compute_metrics(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = BENCHMARK_LABELS) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "cohen_kappa": 0.0,
        }

    per_label: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for label in labels:
        tp = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == label and predicted == label)
        fp = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected != label and predicted == label)
        fn = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == label and predicted != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(sum(1 for item in y_true if item == label)),
        }
        f1_scores.append(f1)

    confusion_rows: list[dict[str, Any]] = []
    for expected in labels:
        for predicted in labels:
            confusion_rows.append(
                {
                    "gold_label": expected,
                    "predicted_label": predicted,
                    "count": int(
                        sum(1 for y_t, y_p in zip(y_true, y_pred, strict=True) if y_t == expected and y_p == predicted)
                    ),
                }
            )

    return {
        "n": n,
        "accuracy": _safe_div(sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == predicted), n),
        "macro_f1": _safe_div(sum(f1_scores), len(labels)),
        "cohen_kappa": _cohen_kappa(y_true, y_pred, labels=labels),
        "per_label": per_label,
        "confusion": confusion_rows,
    }


def _build_provider(runtime: BenchmarkRuntimeConfig) -> OllamaProvider:
    provider_name = runtime.provider.lower().strip()
    if provider_name != "ollama":
        raise ValueError(f"Unsupported benchmark runtime provider: {runtime.provider}")
    return OllamaProvider(base_url=runtime.base_url, timeout_seconds=runtime.timeout_seconds, temperature=runtime.temperature)


def _vader_predictions(frame: pd.DataFrame, *, labels: tuple[str, ...]) -> pd.DataFrame:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

        analyzer = SentimentIntensityAnalyzer()

        def compound_score(text: str) -> float:
            return float(analyzer.polarity_scores(text)["compound"])

    except ImportError:

        def compound_score(text: str) -> float:
            tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
            pos_words = {"good", "great", "excellent", "amazing", "love", "perfect", "helpful", "fantastic"}
            neg_words = {"bad", "poor", "terrible", "awful", "hate", "refund", "disappointing", "waste"}
            pos_hits = len(tokens.intersection(pos_words))
            neg_hits = len(tokens.intersection(neg_words))
            if pos_hits == neg_hits:
                return 0.0
            return 0.2 if pos_hits > neg_hits else -0.2

    def to_label(compound: float) -> str:
        if "neutral" in labels:
            if compound >= 0.05:
                return "positive"
            if compound <= -0.05:
                return "negative"
            return "neutral"
        return "positive" if compound >= 0 else "negative"

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        compound = float(compound_score(str(row["text"])))
        rows.append(
            {
                "record_id": str(row["record_id"]),
                "gold_label": str(row["gold_label"]),
                "predicted_label": to_label(compound),
                "compound": compound,
                "model": "vader",
            }
        )
    return pd.DataFrame(rows, columns=["record_id", "gold_label", "predicted_label", "compound", "model"])


def _run_llm_predictions(
    frame: pd.DataFrame,
    runtime: BenchmarkRuntimeConfig,
    experiment: BenchmarkExperimentConfig,
    prompt_template: str,
    *,
    output_path: Path | None = None,
    existing_df: pd.DataFrame | None = None,
    labels: tuple[str, ...] = BENCHMARK_LABELS,
) -> tuple[pd.DataFrame, int, int]:
    provider = _build_provider(runtime)
    model_name = str(experiment.model or "")
    if not model_name:
        raise ValueError(f"Benchmark experiment {experiment.experiment_id} has no model configured")

    existing_rows: list[dict[str, Any]] = []
    if existing_df is not None and not existing_df.empty:
        existing_rows = existing_df.to_dict(orient="records")
    existing_ids = {str(row.get("record_id")) for row in existing_rows}

    rows: list[dict[str, Any]] = list(existing_rows)
    n_new = 0
    for _, row in frame.iterrows():
        record_id = str(row["record_id"])
        if record_id in existing_ids:
            continue
        prompt = _build_prompt(prompt_template, str(row["text"]), labels=labels)
        request = LLMRequest(
            participant_code=record_id,
            section="benchmark",
            prompt=prompt,
            response_schema={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": list(labels),
                    }
                },
                "required": ["label"],
            },
            model=model_name,
            temperature=experiment.temperature,
            metadata={},
        )
        response = provider.generate_structured(request)
        predicted_label = _parse_llm_label(response.raw_text, labels=labels)
        rows.append(
            {
                "record_id": record_id,
                "gold_label": str(row["gold_label"]),
                "predicted_label": predicted_label,
                "model": str(experiment.model_variant or model_name),
                "experiment_id": experiment.experiment_id,
                "run_id": experiment.run_id,
                "artifact_id": f"{experiment.experiment_id}__{experiment.run_id}" if experiment.run_id else experiment.experiment_id,
            }
        )
        n_new += 1
        if output_path is not None:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    predictions = pd.DataFrame(rows)
    if output_path is not None and not predictions.empty:
        predictions.to_csv(output_path, index=False)
    return predictions, len(existing_rows), n_new


def run_benchmark_pipeline(
    paths: PathsConfig,
    benchmark: BenchmarkConfig,
    *,
    dataset_path: Path | None = None,
    run_output_dir: Path | None = None,
    artifact_prefix: str | None = None,
    prompt_variant: str | None = None,
    max_rows: int | None = None,
    resume: bool = True,
    from_scratch: bool = False,
) -> dict[str, Any]:
    if from_scratch:
        resume = True

    dataset_metadata: dict[str, Any]
    if dataset_path is not None:
        frame = pd.read_csv(dataset_path)
        missing = [column for column in ("record_id", "text", "gold_label") if column not in frame.columns]
        if missing:
            raise ValueError(f"Benchmark dataset is missing required columns: {missing}")
        local_dataset_name = _resolve_local_dataset_name(benchmark, Path(dataset_path))
        dataset_metadata = {
            "source_kind": "local_csv",
            "dataset_name": local_dataset_name,
            "dataset_config": "",
            "split": "custom",
            "text_column": "text",
            "label_column": "gold_label",
            "label_mapping_description": "Dataset already normalized to gold_label.",
            "input_path": str(Path(dataset_path).resolve()),
        }
    else:
        frame, _, download_summary = download_and_prepare_amazon_benchmark(paths, benchmark, max_rows=max_rows)
        dataset_metadata = {
            "source_kind": "downloaded",
            "dataset_name": download_summary.get("dataset_name"),
            "dataset_config": download_summary.get("dataset_config"),
            "split": download_summary.get("split"),
            "text_column": download_summary.get("text_column"),
            "label_column": download_summary.get("label_column"),
            "label_mapping_description": download_summary.get("label_mapping_description"),
        }

    active_labels = _infer_active_labels(frame)

    run_root = _ensure_dir(Path(run_output_dir or paths.benchmark_runs_dir or (paths.evaluation_output_dir / "benchmark_runs")))
    artifact_name = str(artifact_prefix or "amazon_baseline").strip() or "amazon_baseline"
    if resume:
        artifact_dir = run_root / artifact_name
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        artifact_dir = run_root / f"{artifact_name}__{run_id}"

    if from_scratch and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir = _ensure_dir(artifact_dir)

    prompt_path = resolve_benchmark_prompt_path(paths, prompt_variant=prompt_variant)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    predictions_dir = _ensure_dir(artifact_dir / "predictions")
    metrics_dir = _ensure_dir(artifact_dir / "metrics")

    vader_predictions_path = predictions_dir / "vader_predictions.csv"
    if resume and vader_predictions_path.exists() and not from_scratch:
        vader_df = pd.read_csv(vader_predictions_path)
    else:
        vader_df = _vader_predictions(frame, labels=active_labels)
        vader_df.to_csv(vader_predictions_path, index=False)

    model_metrics_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []

    vader_metrics = compute_metrics(vader_df["gold_label"].tolist(), vader_df["predicted_label"].tolist(), labels=active_labels)
    model_metrics_rows.append(
        {
            "source": "vader",
            "artifact_id": "vader",
            "n": int(vader_metrics["n"]),
            "accuracy": float(vader_metrics["accuracy"]),
            "macro_f1": float(vader_metrics["macro_f1"]),
            "cohen_kappa": float(vader_metrics["cohen_kappa"]),
        }
    )
    for row in vader_metrics.get("confusion", []):
        confusion_rows.append({"source": "vader", "artifact_id": "vader", **row})
    for label, details in dict(vader_metrics.get("per_label", {})).items():
        per_label_rows.append({"source": "vader", "artifact_id": "vader", "label": label, **details})

    enabled_experiments = [experiment for experiment in benchmark.experiments if experiment.enabled]
    llm_progress: list[dict[str, Any]] = []
    for experiment in enabled_experiments:
        artifact_id = f"{experiment.experiment_id}__{experiment.run_id}" if experiment.run_id else experiment.experiment_id
        prediction_path = predictions_dir / f"{artifact_id}_predictions.csv"
        existing_df: pd.DataFrame | None = None
        if resume and prediction_path.exists() and not from_scratch:
            existing_df = pd.read_csv(prediction_path)

        prediction_df, n_reused, n_new = _run_llm_predictions(
            frame,
            benchmark.runtime,
            experiment,
            prompt_template,
            output_path=prediction_path,
            existing_df=existing_df,
            labels=active_labels,
        )
        artifact_id = str(prediction_df["artifact_id"].iloc[0]) if not prediction_df.empty else experiment.experiment_id
        if not prediction_df.empty:
            prediction_df.to_csv(prediction_path, index=False)

        llm_progress.append(
            {
                "artifact_id": artifact_id,
                "reused_rows": int(n_reused),
                "new_rows": int(n_new),
                "total_rows": int(len(prediction_df)),
            }
        )

        metrics = compute_metrics(
            prediction_df["gold_label"].tolist(),
            prediction_df["predicted_label"].tolist(),
            labels=active_labels,
        )
        model_metrics_rows.append(
            {
                "source": "llm",
                "artifact_id": artifact_id,
                "experiment_id": experiment.experiment_id,
                "model_variant": experiment.model_variant or experiment.model,
                "n": int(metrics["n"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "cohen_kappa": float(metrics["cohen_kappa"]),
            }
        )
        for row in metrics.get("confusion", []):
            confusion_rows.append({"source": "llm", "artifact_id": artifact_id, **row})
        for label, details in dict(metrics.get("per_label", {})).items():
            per_label_rows.append({"source": "llm", "artifact_id": artifact_id, "label": label, **details})

    metrics_df = pd.DataFrame(model_metrics_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    per_label_df = pd.DataFrame(per_label_rows)

    metrics_path = metrics_dir / "benchmark_metrics.csv"
    confusion_path = metrics_dir / "benchmark_confusion.csv"
    per_label_path = metrics_dir / "benchmark_per_label.csv"
    metrics_df.to_csv(metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)

    run_summary = {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": _now_utc_iso(),
        "resume_enabled": bool(resume),
        "from_scratch": bool(from_scratch),
        "dataset_rows": int(len(frame)),
        "prompt_path": str(prompt_path),
        "prompt_variant": prompt_variant,
        "labels": list(active_labels),
        "runtime": benchmark.runtime.to_dict(),
        "dataset": dataset_metadata,
        "experiments": [experiment.to_dict() for experiment in enabled_experiments],
        "llm_progress": llm_progress,
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "predictions_dir": str(predictions_dir),
            "metrics_file": str(metrics_path),
            "confusion_file": str(confusion_path),
            "per_label_file": str(per_label_path),
        },
    }
    summary_path = artifact_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    return {
        "rows": int(len(frame)),
        "summary": run_summary,
        "artifact_dir": str(artifact_dir),
        "summary_file": str(summary_path),
        "metrics_file": str(metrics_path),
        "confusion_file": str(confusion_path),
        "per_label_file": str(per_label_path),
    }


def write_benchmark_report(
    run_summary_path: Path,
    *,
    output_dir: Path,
    nde_metrics_path: Path | None = None,
    comparison_run_summaries: list[Path] | None = None,
) -> Path:
    def _load_summary_with_metrics(path: Path, *, role: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        loaded_summary = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded_metrics = pd.read_csv(Path(loaded_summary["artifacts"]["metrics_file"]))
        loaded_per_label = pd.read_csv(Path(loaded_summary["artifacts"]["per_label_file"]))
        input_path = loaded_summary.get("dataset", {}).get("input_path")
        dataset_name = _normalize_dataset_display_name(
            loaded_summary.get("dataset", {}).get("dataset_name", "n/a"),
            input_path=input_path,
        )
        dataset_config = str(loaded_summary.get("dataset", {}).get("dataset_config", ""))
        loaded_metrics = loaded_metrics.copy()
        loaded_per_label = loaded_per_label.copy()
        loaded_metrics["dataset_name"] = dataset_name
        loaded_metrics["dataset_config"] = dataset_config
        loaded_metrics["input_path"] = str(input_path or "")
        loaded_metrics["summary_role"] = role
        loaded_per_label["dataset_name"] = dataset_name
        loaded_per_label["dataset_config"] = dataset_config
        loaded_per_label["input_path"] = str(input_path or "")
        loaded_per_label["summary_role"] = role
        return loaded_summary, loaded_metrics, loaded_per_label

    def _quality_band(macro_f1: float, kappa: float) -> str:
        if macro_f1 >= 0.70 and kappa >= 0.60:
            return "strong"
        if macro_f1 >= 0.60 and kappa >= 0.45:
            return "adequate"
        if macro_f1 >= 0.50 and kappa >= 0.30:
            return "mixed"
        return "limited"

    summary, primary_metrics_df, primary_per_label_df = _load_summary_with_metrics(Path(run_summary_path), role="primary")
    metrics_df = primary_metrics_df.copy()
    per_label_df = primary_per_label_df.copy()
    if comparison_run_summaries:
        for comparison_summary_path in comparison_run_summaries:
            _, comparison_metrics_df, comparison_per_label_df = _load_summary_with_metrics(
                Path(comparison_summary_path),
                role="comparison",
            )
            metrics_df = pd.concat([metrics_df, comparison_metrics_df], ignore_index=True)
            per_label_df = pd.concat([per_label_df, comparison_per_label_df], ignore_index=True)

    report_dir = _ensure_dir(output_dir)
    report_path = report_dir / "benchmark_report.md"
    figure_path = report_dir / "figures" / "benchmark_macro_f1_vs_kappa.png"
    written_figure_path = _plot_benchmark_scatter(metrics_df, figure_path=figure_path, per_label_df=per_label_df)

    source_dataset_display = _normalize_dataset_display_name(
        summary.get("dataset", {}).get("dataset_name", "n/a"),
        input_path=summary.get("dataset", {}).get("input_path"),
    )

    source_datasets = sorted(
        {
            f"{name}:{cfg}" if str(cfg).strip() else str(name)
            for name, cfg in zip(metrics_df["dataset_name"], metrics_df["dataset_config"], strict=False)
        }
    )

    lines = [
        "# Benchmark Baseline Report",
        "",
        "## Source",
        f"- Dataset: {source_dataset_display}",
        f"- Dataset config: {summary['dataset']['dataset_config']}",
        f"- Split: {summary['dataset']['split']}",
        f"- Source kind: {summary['dataset'].get('source_kind', 'n/a')}",
        f"- Rows evaluated: {summary['dataset_rows']}",
        f"- Compared datasets in report: {', '.join(source_datasets) if source_datasets else 'n/a'}",
        "",
        "## Methodology",
        f"- Label mapping: {summary['dataset'].get('label_mapping_description', 'n/a')}",
        "- Models evaluated: VADER + configured LLM experiments.",
        "- Metrics: Accuracy, Macro F1, Cohen Kappa.",
        "",
        "## Prompts",
        f"- Prompt path: {summary['prompt_path']}",
        f"- Prompt variant: {summary.get('prompt_variant') or 'default'}",
        "",
        "## Metrics",
        "| dataset | source | artifact_id | n | accuracy | macro_f1 | cohen_kappa |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        dataset_cell = str(row.get("dataset_name", "n/a"))
        dataset_cfg = str(row.get("dataset_config", "")).strip()
        if dataset_cfg:
            dataset_cell = f"{dataset_cell}:{dataset_cfg}"
        lines.append(
            f"| {dataset_cell} | {row.get('source', 'n/a')} | {row.get('artifact_id', 'n/a')} | {int(row.get('n', 0))} | {float(row.get('accuracy', 0.0)):.4f} | {float(row.get('macro_f1', 0.0)):.4f} | {float(row.get('cohen_kappa', 0.0)):.4f} |"
        )

    if written_figure_path is not None:
        lines.extend(
            [
                "",
                "## Visual Summary",
                "Scatter plot with macro_f1 (x) and cohen_kappa (y): marker fill color = model family; marker shape + border color = dataset.",
                "Emoji overlay per point: 😀 positive F1, 😐 neutral F1, ☹️ negative F1 (higher opacity = higher F1).",
                f"![Benchmark macro_f1 vs cohen_kappa]({written_figure_path.relative_to(report_dir).as_posix()})",
            ]
        )

    llm_metrics_df = metrics_df[metrics_df["source"].astype(str) == "llm"].copy()
    scoped_df = llm_metrics_df if not llm_metrics_df.empty else metrics_df
    top_row = scoped_df.sort_values(["macro_f1", "cohen_kappa"], ascending=False).iloc[0]
    bottom_row = scoped_df.sort_values(["macro_f1", "cohen_kappa"], ascending=True).iloc[0]

    aggregate_macro_f1 = float(scoped_df["macro_f1"].mean()) if not scoped_df.empty else 0.0
    aggregate_kappa = float(scoped_df["cohen_kappa"].mean()) if not scoped_df.empty else 0.0
    macro_f1_std = float(scoped_df["macro_f1"].std(ddof=0)) if not scoped_df.empty else 0.0
    kappa_std = float(scoped_df["cohen_kappa"].std(ddof=0)) if not scoped_df.empty else 0.0
    quality_band = _quality_band(aggregate_macro_f1, aggregate_kappa)

    top_macro_f1 = float(top_row.get("macro_f1", 0.0))
    top_kappa = float(top_row.get("cohen_kappa", 0.0))
    bottom_macro_f1 = float(bottom_row.get("macro_f1", 0.0))
    bottom_kappa = float(bottom_row.get("cohen_kappa", 0.0))

    interpretation_lines = [
        "## Interpretation",
        "### Evidence scale",
        "- **strong**: macro_f1 >= 0.70 and kappa >= 0.60",
        "- **adequate**: macro_f1 >= 0.60 and kappa >= 0.45",
        "- **mixed**: macro_f1 >= 0.50 and kappa >= 0.30",
        "- **limited**: below mixed thresholds",
        "",
        "### Aggregate reading",
        (
            f"- Aggregate performance across evaluated systems: macro_f1={aggregate_macro_f1:.4f} "
            f"(std={macro_f1_std:.4f}), kappa={aggregate_kappa:.4f} (std={kappa_std:.4f})."
        ),
        f"- Overall evidence level for this benchmark: **{quality_band}**.",
        "",
        "### Model spread (best and worst)",
        (
            f"- Best artifact in scope: {top_row.get('artifact_id', 'n/a')} "
            f"(dataset={top_row.get('dataset_name', 'n/a')}, macro_f1={top_macro_f1:.4f}, kappa={top_kappa:.4f})."
        ),
        (
            f"- Worst artifact in scope: {bottom_row.get('artifact_id', 'n/a')} "
            f"(dataset={bottom_row.get('dataset_name', 'n/a')}, macro_f1={bottom_macro_f1:.4f}, kappa={bottom_kappa:.4f})."
        ),
    ]

    vader_rows = metrics_df[metrics_df["source"].astype(str) == "vader"].copy()
    if not llm_metrics_df.empty and not vader_rows.empty:
        best_vader = vader_rows.sort_values(["macro_f1", "cohen_kappa"], ascending=False).iloc[0]
        delta_macro_f1 = top_macro_f1 - float(best_vader.get("macro_f1", 0.0))
        interpretation_lines.append(f"- Best LLM vs best VADER macro_f1 delta: {delta_macro_f1:+.4f}.")

    if metrics_df["dataset_name"].nunique() > 1:
        interpretation_lines.append("- Multi-dataset comparison enabled in this report:")
        for dataset_name, dataset_subset in metrics_df.groupby("dataset_name", dropna=False):
            dataset_best = dataset_subset.sort_values(["macro_f1", "cohen_kappa"], ascending=False).iloc[0]
            interpretation_lines.append(
                (
                    f"  - {dataset_name}: best={dataset_best.get('artifact_id', 'n/a')} "
                    f"(macro_f1={float(dataset_best.get('macro_f1', 0.0)):.4f}, "
                    f"kappa={float(dataset_best.get('cohen_kappa', 0.0)):.4f})."
                )
            )

    if nde_metrics_path is not None and Path(nde_metrics_path).exists():
        nde_metrics = pd.read_csv(nde_metrics_path)
        nde_scope = nde_metrics[nde_metrics["field"].astype(str).str.contains("_tone", na=False)].copy()
        if not nde_scope.empty and not metrics_df.empty:
            nde_by_comparison = (
                nde_scope.groupby("comparison", dropna=False)["macro_f1"].mean().sort_values(ascending=False).reset_index()
            )
            nde_best = nde_by_comparison.iloc[0]
            benchmark_best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
            interpretation_lines.extend(
                [
                    "",
                    "### Benchmark vs NDE",
                    "| scope | best_source | macro_f1 | accuracy | cohen_kappa |",
                    "|---|---|---:|---:|---:|",
                    (
                        f"| benchmark | {benchmark_best.get('artifact_id', 'n/a')} "
                        f"| {float(benchmark_best.get('macro_f1', 0.0)):.4f} "
                        f"| {float(benchmark_best.get('accuracy', 0.0)):.4f} "
                        f"| {float(benchmark_best.get('cohen_kappa', 0.0)):.4f} |"
                    ),
                    (
                        f"| nde_alignment | {nde_best.get('comparison', 'n/a')} "
                        f"| {float(nde_best.get('macro_f1', 0.0)):.4f} | n/a | n/a |"
                    ),
                    "",
                    f"- NDE metrics source: {Path(nde_metrics_path).resolve()}",
                ]
            )
        else:
            interpretation_lines.append("- NDE metrics file was provided, but no comparable tone rows were found.")

    lines.extend(
        [
            "",
            *interpretation_lines,
            "",
            "## Limitations",
            "- External datasets may differ from NDE narrative language and context.",
            "- Label-space mapping (binary or 3-class) can introduce ambiguity near class boundaries.",
            "- Threshold-based interpretation is heuristic and should be triangulated with qualitative error analysis.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
