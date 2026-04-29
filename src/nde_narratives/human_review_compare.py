from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

from .config import PathsConfig, StudyConfig
from .evaluation import cohen_kappa_score, macro_f1_score


HUMAN_TONE4 = {
    "context": "context_tone_4",
    "experience": "experience_tone_4",
    "aftereffects": "aftereffects_tone_4",
}
HUMAN_TONE3 = {
    "context": "context_tone_3",
    "experience": "experience_tone_3",
    "aftereffects": "aftereffects_tone_3",
}
HUMAN_TEXT = {
    "context": "context_narrative",
    "experience": "experience_narrative",
    "aftereffects": "aftereffects_narrative",
}

HUMAN_EXTRACTED_MAP = {
    "out_of_body": "outside_of_body_experience",
    "bright_light": "feeling_bright_light",
    "awareness": "feeling_awareness",
    "presence": "presence_encounter",
    "relived_past_events": "saw_relived_past_events",
    "time_perception": "time_perception_altered",
    "point_of_no_return": "border_point_of_no_return",
    "non_existence": "non_existence_feeling",
    "peace_wellbeing": "feeling_peace_wellbeing",
    "entered_gateway": "saw_entered_gateway",
    "fear_of_death": "fear_of_death",
    "inner_meaning": "inner_meaning_in_my_life",
    "compassion": "compassion_toward_others",
    "spiritual_feelings": "spiritual_feelings",
    "help_others": "desire_to_help_others",
    "personal_vulnerability": "personal_vulnerability",
    "material_goods": "interest_in_material_goods",
    "religion": "interest_in_religion",
    "understanding_myself": "understanding_myself",
    "social_justice": "social_justice_issues",
}

SEGMENT_LABELS = ["context", "experience", "aftereffects", "deleted"]


def _presentation_model_name(identifier: object) -> str:
    raw = str(identifier or "").strip()
    if not raw:
        return raw
    normalized = raw.replace("-", "_").replace(":", "_")
    if "__" in normalized:
        normalized = normalized.split("__", 1)[0]
    alias_map = {
        "deepseek_r1_32": "DeepSeek-R1 32B",
        "gemma3_27": "Gemma 3 27B",
        "llama31_8": "Llama 3.1 8B",
        "ministral3_14": "Ministral 3 14B",
        "nemotron_3_nano": "Nemotron-3 Nano 30B",
        "qwen35_9": "Qwen 3.5 9B",
        "qwen35_27": "Qwen 3.5 27B",
        "qwen35_35": "Qwen 3.5 35B",
        "qwen3_32": "Qwen 3 32B",
        "questionnaire": "Questionnaire",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    return normalized.replace("_", " ").title()


def _family_label(family: object) -> str:
    mapping = {
        "tone": "Experience Tone",
        "nde_c": "NDE-C",
        "lci_r": "LCI-R",
        "seg_unit": "SEG-UNIT",
    }
    return mapping.get(str(family), str(family))


def _comparison_label(comparison: object) -> str:
    text = str(comparison)
    if text == "human_vs_llm":
        return "Human vs LLM"
    if text == "human_vs_questionnaire":
        return "Human vs Questionnaire"
    if text == "human_vs_cleaned":
        return "Human vs Cleaned Dataset"
    return text


def _valid_comparison_label(comparison: object) -> str:
    text = str(comparison or "")
    if not text.startswith("human_vs_"):
        return text
    source = text.replace("human_vs_", "", 1)
    if source == "cleaned_dataset":
        return "Human vs Cleaned Dataset"
    return f"Human vs {_presentation_model_name(source)}"


def _valid_source_display(source: object) -> str:
    source_text = str(source or "")
    if source_text == "human":
        return "Human"
    if source_text == "cleaned_dataset":
        return "Cleaned Dataset"
    return _presentation_model_name(source_text)


def _is_blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " ")
    return _collapse_spaces(text)


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-zA-Z0-9À-ÿ']+", text.lower()))


def _split_units(text: object) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    work = re.sub(r"\s*[-*]+\s+", ". ", normalized)
    work = re.sub(r"([.!?;:])\s+", r"\1\n", work)
    chunks = [chunk.strip(" .;:-\t") for chunk in work.split("\n")]
    return [chunk for chunk in chunks if len(chunk) >= 3]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    if not compact:
        return set()
    padded = f" {compact} "
    if len(padded) < n:
        return {padded}
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def _soft_similarity(text_a: object, text_b: object) -> float:
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    tok_a = _tokenize(norm_a)
    tok_b = _tokenize(norm_b)
    tok_j = len(tok_a & tok_b) / len(tok_a | tok_b) if (tok_a or tok_b) else 0.0
    ch_a = _char_ngrams(norm_a)
    ch_b = _char_ngrams(norm_b)
    ch_overlap = len(ch_a & ch_b)
    ch_dice = (2 * ch_overlap / (len(ch_a) + len(ch_b))) if (ch_a or ch_b) else 0.0
    return 0.5 * tok_j + 0.5 * ch_dice


def _label_unit(
    unit_text: str,
    section_units: dict[str, list[str]],
    threshold: float = 0.33,
) -> tuple[str, float]:
    best_label = "deleted"
    best_score = 0.0
    for label in ("context", "experience", "aftereffects"):
        candidates = section_units.get(label, [])
        if not candidates:
            continue
        score = max(
            (_soft_similarity(unit_text, candidate) for candidate in candidates),
            default=0.0,
        )
        if score > best_score:
            best_score = score
            best_label = label
    if best_score < threshold:
        return "deleted", best_score
    return best_label, best_score


def _compute_unit_classification(
    source_df: pd.DataFrame,
    section_cols: dict[str, str],
    source_id_col: str = "response_id",
    threshold: float = 0.33,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in source_df.iterrows():
        response_id = int(row[source_id_col])
        full_text = " ".join(
            [
                normalize_text(row.get("nde_context")),
                normalize_text(row.get("nde_description")),
                normalize_text(row.get("nde_aftereffects")),
            ]
        ).strip()
        source_units = _split_units(full_text)
        target_units = {
            label: _split_units(row.get(column))
            for label, column in section_cols.items()
        }
        for unit_index, unit_text in enumerate(source_units):
            label, score = _label_unit(unit_text, target_units, threshold=threshold)
            rows.append(
                {
                    "response_id": response_id,
                    "unit_index": unit_index,
                    "unit_text": unit_text,
                    "label": label,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def _compute_unit_pair_metrics(
    unit_long: pd.DataFrame,
    reference_col: str,
    candidate_col: str,
    labels: list[str],
) -> dict[str, float]:
    usable = unit_long[["response_id", reference_col, candidate_col]].dropna()
    usable = usable[
        usable[reference_col].astype(str).isin(labels)
        & usable[candidate_col].astype(str).isin(labels)
    ].copy()
    if usable.empty:
        return {
            "n_units": 0.0,
            "accuracy_micro": float("nan"),
            "cohen_kappa_micro": float("nan"),
            "macro_f1_micro": float("nan"),
            "accuracy_macro_response": float("nan"),
            "cohen_kappa_macro_response": float("nan"),
            "macro_f1_macro_response": float("nan"),
        }

    y_true = usable[reference_col].astype(str)
    y_pred = usable[candidate_col].astype(str)
    accuracy_micro = float((y_true == y_pred).mean())
    kappa_micro = cohen_kappa_score(y_true, y_pred, labels)
    macro_micro = macro_f1_score(y_true, y_pred, labels)

    by_response: list[tuple[float, float, float]] = []
    for _, group in usable.groupby("response_id"):
        gt = group[reference_col].astype(str)
        gp = group[candidate_col].astype(str)
        if len(group) == 0:
            continue
        acc = float((gt == gp).mean())
        kap = cohen_kappa_score(gt, gp, labels)
        f1 = macro_f1_score(gt, gp, labels)
        by_response.append((acc, kap, f1))

    acc_macro = (
        float(pd.Series([x[0] for x in by_response]).mean())
        if by_response
        else float("nan")
    )
    kap_macro = (
        float(pd.Series([x[1] for x in by_response]).mean())
        if by_response
        else float("nan")
    )
    f1_macro = (
        float(pd.Series([x[2] for x in by_response]).mean())
        if by_response
        else float("nan")
    )
    return {
        "n_units": float(len(usable)),
        "accuracy_micro": accuracy_micro,
        "cohen_kappa_micro": kappa_micro,
        "macro_f1_micro": macro_micro,
        "accuracy_macro_response": acc_macro,
        "cohen_kappa_macro_response": kap_macro,
        "macro_f1_macro_response": f1_macro,
    }


def _normalize_tone(value: object) -> str | None:
    if _is_blank(value):
        return None
    text = _collapse_spaces(str(value)).lower()
    if text.startswith("pos"):
        return "positive"
    if text.startswith("neg"):
        return "negative"
    if text.startswith("mix"):
        return "mixed"
    if text.startswith("neu"):
        return "neutral"
    if text.startswith("nue"):
        return "neutral"
    return text


def _normalize_yes_no(value: object) -> str | None:
    if _is_blank(value):
        return None
    text = _collapse_spaces(str(value)).lower()
    if text in {"yes", "y", "true", "1", "si", "sí"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return None


def _map_questionnaire_binary(
    value: object,
    yes_values: list[str],
    no_values: list[str],
    na_values: list[str],
) -> str | None:
    if _is_blank(value):
        return None
    value_norm = _collapse_spaces(str(value)).lower()
    yes_norm = {_collapse_spaces(v).lower() for v in yes_values}
    no_norm = {_collapse_spaces(v).lower() for v in no_values}
    na_norm = {_collapse_spaces(v).lower() for v in na_values}
    if value_norm in yes_norm:
        return "yes"
    if value_norm in no_norm:
        return "no"
    if value_norm in na_norm:
        return None
    return None


def parse_human_md(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    key_map = {
        "response_id": "response_id",
        "context narrative": "context_narrative",
        "experience narrative": "experience_narrative",
        "aftereffects narrative": "aftereffects_narrative",
        "context tone (4)": "context_tone_4",
        "context tone (3)": "context_tone_3",
        "experience tone (4)": "experience_tone_4",
        "experience tone (3)": "experience_tone_3",
        "aftereffects tone (4)": "aftereffects_tone_4",
        "aftereffects tone (3)": "aftereffects_tone_3",
        "out-of-body sensation": "out_of_body",
        "bright light": "bright_light",
        "heightened awareness": "awareness",
        "altered time perception": "time_perception",
        "encounter with a presence": "presence",
        "relived past events": "relived_past_events",
        "border or point of no return": "point_of_no_return",
        "feeling of non-existence": "non_existence",
        "peace or wellbeing": "peace_wellbeing",
        "entered a gateway": "entered_gateway",
        "fear of death": "fear_of_death",
        "inner meaning in my life": "inner_meaning",
        "compassion toward others": "compassion",
        "spiritual feelings": "spiritual_feelings",
        "desire to help others": "help_others",
        "personal vulnerability": "personal_vulnerability",
        "interest in material goods": "material_goods",
        "interest in religion": "religion",
        "understanding myself": "understanding_myself",
        "social justice issues": "social_justice",
    }

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_key: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            current_key = None
            continue

        match = re.match(r"^([^:]+):\s*(.*)$", line)
        if match:
            raw_key = _collapse_spaces(match.group(1)).lower()
            value = match.group(2)
            if raw_key == "response_id":
                if current is not None:
                    records.append(current)
                current = {"response_id": value.strip()}
                current_key = "response_id"
                continue

            mapped = key_map.get(raw_key)
            if current is not None and mapped is not None:
                current[mapped] = value
                current_key = mapped
                continue

        if current is not None and current_key is not None:
            previous = str(current.get(current_key, ""))
            current[current_key] = f"{previous}\n{line}" if previous else line

    if current is not None:
        records.append(current)

    if not records:
        return pd.DataFrame(columns=["response_id"])

    df = pd.DataFrame(records)
    if "response_id" in df.columns:
        df["response_id"] = pd.to_numeric(df["response_id"], errors="coerce")
        df = df[df["response_id"].notna()].copy()
        df["response_id"] = df["response_id"].astype(int)

    for column in HUMAN_TEXT.values():
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].map(normalize_text)

    for column in HUMAN_TONE4.values():
        if column not in df.columns:
            df[column] = None
        df[column] = df[column].map(_normalize_tone)

    for column in HUMAN_TONE3.values():
        if column not in df.columns:
            df[column] = None
        df[column] = df[column].map(_normalize_tone)

    for column in HUMAN_EXTRACTED_MAP.keys():
        if column not in df.columns:
            df[column] = None
        df[column] = df[column].map(_normalize_yes_no)

    for human_column, internal_column in HUMAN_EXTRACTED_MAP.items():
        df[internal_column] = df[human_column]

    return df


def _first_non_blank(values: pd.Series) -> Any:
    for value in values.tolist():
        if not _is_blank(value):
            return value
    return None


def _load_llm_predictions(prediction_path: Path, study: StudyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            row = {
                "response_id": payload.get("response_id"),
                "context_tone": (payload.get("context") or {}).get("tone"),
                "experience_tone": (payload.get("experience") or {}).get("tone"),
                "aftereffects_tone": (payload.get("aftereffects") or {}).get("tone"),
                "context_text": " ".join(
                    (payload.get("context") or {}).get("evidence_segments") or []
                ),
                "experience_text": " ".join(
                    (payload.get("experience") or {}).get("evidence_segments") or []
                ),
                "aftereffects_text": " ".join(
                    (payload.get("aftereffects") or {}).get("evidence_segments") or []
                ),
            }
            experience_payload = payload.get("experience") or {}
            aftereffects_payload = payload.get("aftereffects") or {}
            for column in study.sections["experience"].binary_labels:
                row[column] = experience_payload.get(column)
            for column in study.sections["aftereffects"].binary_labels:
                row[column] = aftereffects_payload.get(column)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["response_id"] = pd.to_numeric(df["response_id"], errors="coerce")
    df = df[df["response_id"].notna()].copy()
    df["response_id"] = df["response_id"].astype(int)
    aggregated = df.groupby("response_id", as_index=False).agg(_first_non_blank)
    return aggregated


@dataclass
class LLMArtifact:
    artifact_id: str
    model_variant: str
    predictions_path: Path


def discover_default_llm_artifacts(llm_results_dir: Path) -> list[LLMArtifact]:
    artifacts: list[LLMArtifact] = []
    if not llm_results_dir.exists():
        return artifacts

    for candidate in sorted(
        path for path in llm_results_dir.iterdir() if path.is_dir()
    ):
        name = candidate.name
        if "ra1" in name.lower():
            continue
        manifest_path = candidate / "manifest.json"
        predictions_path = candidate / "predictions.jsonl"
        if not manifest_path.exists() or not predictions_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_id") or "")
        if run_id.lower() == "ra1":
            continue
        artifact_id = str(manifest.get("artifact_id") or name)
        model_variant = str(
            manifest.get("model_variant") or manifest.get("model") or artifact_id
        )
        artifacts.append(
            LLMArtifact(
                artifact_id=artifact_id,
                model_variant=model_variant,
                predictions_path=predictions_path,
            )
        )
    return artifacts


def _compute_section_validity(df: pd.DataFrame, prefix: str, study: StudyConfig) -> pd.Series:
    needed = {
        "context": [f"{prefix}context_tone"],
        "experience": [
            f"{prefix}experience_tone",
            *[f"{prefix}{field}" for field in study.sections["experience"].binary_labels],
        ],
        "aftereffects": [
            f"{prefix}aftereffects_tone",
            *[f"{prefix}{field}" for field in study.sections["aftereffects"].binary_labels],
        ],
    }
    counts = pd.Series(0, index=df.index, dtype=int)
    for fields in needed.values():
        local = pd.Series(True, index=df.index)
        for field in fields:
            local = local & df[field].map(lambda value: not _is_blank(value))
        counts = counts + local.astype(int)
    return counts


def _comparison_metrics(
    merged: pd.DataFrame,
    true_col: str,
    pred_col: str,
    labels: list[str],
) -> tuple[int, float, float]:
    valid = (~merged[true_col].map(_is_blank)) & (~merged[pred_col].map(_is_blank))
    y_true = merged.loc[valid, true_col].astype(str)
    y_pred = merged.loc[valid, pred_col].astype(str)
    n = int(len(y_true))
    if n == 0:
        return 0, float("nan"), float("nan")
    return (
        n,
        cohen_kappa_score(y_true, y_pred, labels),
        macro_f1_score(y_true, y_pred, labels),
    )


def _save_figure(fig: plt.Figure, path: Path, dpi: int, export_pdf: bool) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    written = [str(path)]
    if export_pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        written.append(str(pdf_path))
    plt.close(fig)
    return written


def _plot_family_metrics_combined(
    family_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    export_pdf: bool,
) -> list[str]:
    if family_df.empty:
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return _save_figure(fig, out_path, dpi=dpi, export_pdf=export_pdf)

    raw_df = family_df.copy()
    chart_df = (
        family_df.groupby(["comparison", "family"], as_index=False)[
            ["cohen_kappa", "macro_f1"]
        ]
        .mean(numeric_only=True)
        .copy()
    )
    chart_df["comparison_label"] = chart_df["comparison"].map(_comparison_label)
    chart_df["family_label"] = chart_df["family"].map(_family_label)

    # Visualization choice requested by user: show SEG-UNIT with the LLM category
    # in the combined family plot (position/color grouping), while keeping raw
    # tabular metrics unchanged.
    chart_df.loc[chart_df["family_label"] == "SEG-UNIT", "comparison_label"] = (
        "Human vs LLM"
    )

    # Combined family figure focuses on LLM aggregate and questionnaire comparison.
    source_order = ["Human vs LLM", "Human vs Questionnaire"]
    family_order = ["NDE-C", "LCI-R", "Experience Tone", "SEG-UNIT"]
    source_colors = {
        "Human vs LLM": "#457B9D",
        "Human vs Questionnaire": "#2A9D8F",
        "Human vs Cleaned Dataset": "#6D597A",
    }
    family_markers = {
        "NDE-C": "o",
        "LCI-R": "s",
        "Experience Tone": "^",
        "SEG-UNIT": "D",
    }
    family_offsets = {
        "NDE-C": -0.21,
        "LCI-R": -0.07,
        "Experience Tone": 0.07,
        "SEG-UNIT": 0.21,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=False)
    axis_label_fontsize = 13
    tick_label_fontsize = 12
    legend_fontsize = 12

    llm_ci = (
        raw_df[raw_df["comparison"] == "human_vs_llm"]
        .groupby("family", as_index=False)[["cohen_kappa", "macro_f1"]]
        .agg(["std", "count"])
    )
    ci_lookup: dict[tuple[str, str], float] = {}
    if not llm_ci.empty:
        llm_ci.columns = ["family", "kappa_std", "kappa_count", "f1_std", "f1_count"]
        for _, row in llm_ci.iterrows():
            family = str(row["family"])
            kappa_count = float(row["kappa_count"])
            f1_count = float(row["f1_count"])
            kappa_ci = (
                1.96 * float(row["kappa_std"]) / math.sqrt(kappa_count)
                if kappa_count > 1
                else 0.0
            )
            f1_ci = (
                1.96 * float(row["f1_std"]) / math.sqrt(f1_count)
                if f1_count > 1
                else 0.0
            )
            ci_lookup[("cohen_kappa", family)] = float(kappa_ci)
            ci_lookup[("macro_f1", family)] = float(f1_ci)
    source_positions = {
        label: float(idx) * 0.88 for idx, label in enumerate(source_order)
    }

    metric_panels = [
        ("cohen_kappa", "Cohen kappa"),
        ("macro_f1", "Macro-F1"),
    ]
    for panel_index, (metric_col, panel_title) in enumerate(metric_panels):
        ax = axes[panel_index]
        y_values: list[float] = []
        y_errors: list[float] = []
        for source_label in source_order:
            source_slice = chart_df[chart_df["comparison_label"] == source_label]
            for family_label in family_order:
                row = source_slice[source_slice["family_label"] == family_label]
                if row.empty:
                    continue
                x_value = source_positions[source_label] + family_offsets.get(
                    family_label, 0.0
                )
                y_value = float(row.iloc[0][metric_col])
                ax.scatter(
                    [x_value],
                    [y_value],
                    color=source_colors.get(source_label, "#6C757D"),
                    marker=family_markers.get(family_label, "o"),
                    s=95,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3,
                )
                y_values.append(y_value)
                if source_label == "Human vs LLM":
                    yerr = ci_lookup.get((metric_col, str(row.iloc[0]["family"])), 0.0)
                    if yerr > 0:
                        ax.errorbar(
                            [x_value],
                            [y_value],
                            yerr=[yerr],
                            fmt="none",
                            ecolor=source_colors.get(source_label, "#6C757D"),
                            elinewidth=1.2,
                            capsize=4,
                            zorder=2,
                        )
                    y_errors.append(float(yerr))
                else:
                    y_errors.append(0.0)
        ax.set_xticks([source_positions[label] for label in source_order])
        ax.set_xticklabels(
            source_order,
            rotation=0,
            ha="center",
            fontsize=tick_label_fontsize,
        )
        if source_positions:
            x_min = min(source_positions.values()) - 0.35
            x_max = max(source_positions.values()) + 0.35
            ax.set_xlim(x_min, x_max)
        if y_values:
            y_min = min(v - e for v, e in zip(y_values, y_errors, strict=False))
            y_max = max(v + e for v, e in zip(y_values, y_errors, strict=False))
            margin = max(0.03, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
            ax.set_ylim(max(0.0, y_min - margin), min(1.0, y_max + margin))
        ax.set_ylabel(panel_title, fontsize=axis_label_fontsize)
        ax.tick_params(axis="y", labelsize=tick_label_fontsize)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")

    present_sources = [
        label for label in source_order if label in set(chart_df["comparison_label"])
    ]
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=9,
            label=label,
        )
        for label, color in source_colors.items()
        if label in present_sources
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=family_markers[label],
            linestyle="",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=9,
            label=label,
        )
        for label in family_order
    ]
    combined_handles = [*color_handles, *family_handles]
    combined_labels = [h.get_label() for h in combined_handles]
    fig.legend(
        handles=combined_handles,
        labels=combined_labels,
        loc="upper center",
        frameon=False,
        fontsize=legend_fontsize,
        ncol=3,
        borderaxespad=0.3,
        handletextpad=0.5,
        labelspacing=0.4,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(bottom=0.20)
    fig.patch.set_facecolor("#FFFFFF")
    return _save_figure(fig, out_path, dpi=dpi, export_pdf=export_pdf)


def _plot_valid_sections(
    valid_df: pd.DataFrame, out_path: Path, dpi: int, export_pdf: bool
) -> list[str]:
    distribution = (
        valid_df.groupby(["source", "n_valid_sections"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    # Keep this figure focused on the direct preprocessing-vs-human validity check.
    # LLM validity comparisons remain available in the metrics tables.
    keep_sources = {"human", "cleaned_dataset"}
    distribution = distribution[distribution["source"].isin(keep_sources)].copy()

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for source, group in distribution.groupby("source"):
        display_name = _valid_source_display(source)
        linestyle = "--" if str(source) == "cleaned_dataset" else "-"
        ax.plot(
            group["n_valid_sections"],
            group["count"],
            marker="o",
            linestyle=linestyle,
            label=display_name,
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Valid section count")
    ax.set_ylabel("Rows")
    ax.set_title("Valid section distribution by source")
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.patch.set_facecolor("#FFFFFF")
    return _save_figure(fig, out_path, dpi=dpi, export_pdf=export_pdf)


def _dynamic_findings(
    llm_family: pd.DataFrame,
    questionnaire_family: pd.DataFrame,
    seg_unit_summary: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    if not llm_family.empty:
        ranking = (
            llm_family.groupby("model", as_index=False)[["macro_f1", "cohen_kappa"]]
            .mean(numeric_only=True)
            .sort_values(["macro_f1", "cohen_kappa"], ascending=False)
        )
        best = ranking.iloc[0]
        best_name = _presentation_model_name(best["model"])
        findings.append(
            f"Best LLM overall is `{best_name}` (macro-F1={best['macro_f1']:.3f}, kappa={best['cohen_kappa']:.3f})."
        )
        if len(ranking) > 1:
            worst = ranking.iloc[-1]
            worst_name = _presentation_model_name(worst["model"])
            findings.append(
                f"Lowest LLM overall is `{worst_name}` (macro-F1={worst['macro_f1']:.3f}, kappa={worst['cohen_kappa']:.3f})."
            )

    if not questionnaire_family.empty:
        tone_rows = questionnaire_family[questionnaire_family["family"] == "tone"]
        if not tone_rows.empty:
            tone = tone_rows.iloc[0]
            findings.append(
                "Questionnaire alignment (Tone(3) vs valence) reaches "
                f"macro-F1={tone['macro_f1']:.3f} and kappa={tone['cohen_kappa']:.3f}."
            )

    if not seg_unit_summary.empty:
        best_seg = seg_unit_summary.sort_values("macro_f1", ascending=False).iloc[0]
        findings.append(
            "Best SEG-UNIT alignment appears in "
            f"`{_presentation_model_name(best_seg['model'])}` (macro-F1={best_seg['macro_f1']:.3f}, kappa={best_seg['cohen_kappa']:.3f})."
        )
    return findings


def run_human_review_comparison(
    study: StudyConfig,
    paths: PathsConfig,
    human_md: Path,
    cleaned_dataset: Path,
    questionnaire_csv: Path,
    llm_results_dir: Path,
    output_dir: Path,
    figure_dpi: int = 300,
    export_figures_pdf: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.png", "*.pdf"):
        for stale in figures_dir.glob(ext):
            stale.unlink(missing_ok=True)

    human_df = parse_human_md(human_md)
    human_ids = set(human_df["response_id"].tolist())

    cleaned_df = pd.read_csv(cleaned_dataset)
    cleaned_df["response_id"] = pd.to_numeric(
        cleaned_df["response_id"], errors="coerce"
    )
    cleaned_df = cleaned_df[cleaned_df["response_id"].notna()].copy()
    cleaned_df["response_id"] = cleaned_df["response_id"].astype(int)
    cleaned_df = cleaned_df[cleaned_df["response_id"].isin(human_ids)].copy()

    questionnaire_df = pd.read_csv(questionnaire_csv)
    questionnaire_df["response_id"] = pd.to_numeric(
        questionnaire_df["response_id"], errors="coerce"
    )
    questionnaire_df = questionnaire_df[questionnaire_df["response_id"].notna()].copy()
    questionnaire_df["response_id"] = questionnaire_df["response_id"].astype(int)
    questionnaire_df = questionnaire_df[
        questionnaire_df["response_id"].isin(human_ids)
    ].copy()
    questionnaire_df["questionnaire_tone"] = questionnaire_df[
        study.stratify_column
    ].map(_normalize_tone)
    for column, source_column in study.questionnaire_column_map().items():
        if source_column in questionnaire_df.columns:
            block_name = study.questionnaire_block_for_column(column)
            block = study.questionnaire[block_name]
            questionnaire_df[column] = questionnaire_df[source_column].map(
                lambda value: _map_questionnaire_binary(
                    value,
                    block["yes_values"],
                    block["no_values"],
                    list(block.get("na_values", [])),
                )
            )

    base = human_df.merge(
        cleaned_df, on="response_id", how="left", suffixes=("", "_cleaned")
    )
    questionnaire_fields = list(study.binary_columns())
    q_subset = questionnaire_df[
        ["response_id", "questionnaire_tone", *questionnaire_fields]
    ].copy()
    q_subset = q_subset.rename(
        columns={field: f"{field}_questionnaire" for field in questionnaire_fields}
    )
    base = base.merge(q_subset, on="response_id", how="left")
    source_text_columns = ["nde_context", "nde_description", "nde_aftereffects"]
    if set(source_text_columns).issubset(set(questionnaire_df.columns)):
        source_text_subset = questionnaire_df[
            ["response_id", *source_text_columns]
        ].copy()
    else:
        source_text_subset = cleaned_df[
            ["response_id", "nde_context", "nde_description", "nde_aftereffects"]
        ].copy()
    source_text_subset = source_text_subset.rename(
        columns={
            "nde_context": "source_nde_context",
            "nde_description": "source_nde_description",
            "nde_aftereffects": "source_nde_aftereffects",
        }
    )
    base = base.merge(source_text_subset, on="response_id", how="left")

    for section in ("context", "experience", "aftereffects"):
        base[f"{section}_tone"] = base[HUMAN_TONE4[section]]
    for column in study.binary_columns():
        if column not in base.columns:
            base[column] = None

    unit_base = base[
        [
            "response_id",
            "source_nde_context",
            "source_nde_description",
            "source_nde_aftereffects",
            "context_narrative",
            "experience_narrative",
            "aftereffects_narrative",
            "nde_context",
            "nde_description",
            "nde_aftereffects",
        ]
    ].copy()
    unit_base = unit_base.rename(
        columns={
            "source_nde_context": "nde_context",
            "source_nde_description": "nde_description",
            "source_nde_aftereffects": "nde_aftereffects",
            "nde_context": "cleaned_context",
            "nde_description": "cleaned_experience",
            "nde_aftereffects": "cleaned_aftereffects",
        }
    )

    human_units = _compute_unit_classification(
        unit_base,
        section_cols={
            "context": "context_narrative",
            "experience": "experience_narrative",
            "aftereffects": "aftereffects_narrative",
        },
        threshold=0.33,
    ).rename(columns={"label": "human_unit_label", "score": "human_unit_score"})

    cleaned_units = _compute_unit_classification(
        unit_base,
        section_cols={
            "context": "cleaned_context",
            "experience": "cleaned_experience",
            "aftereffects": "cleaned_aftereffects",
        },
        threshold=0.33,
    ).rename(columns={"label": "cleaned_unit_label", "score": "cleaned_unit_score"})

    unit_long = human_units.merge(
        cleaned_units[
            [
                "response_id",
                "unit_index",
                "cleaned_unit_label",
                "cleaned_unit_score",
            ]
        ],
        on=["response_id", "unit_index"],
        how="left",
    )

    metrics_rows: list[dict[str, Any]] = []

    cleaned_unit_metrics = _compute_unit_pair_metrics(
        unit_long,
        reference_col="human_unit_label",
        candidate_col="cleaned_unit_label",
        labels=SEGMENT_LABELS,
    )
    metrics_rows.append(
        {
            "comparison": "human_vs_cleaned",
            "model": "cleaned_dataset",
            "family": "seg_unit",
            "field": "seg_unit_labels",
            "n": cleaned_unit_metrics["n_units"],
            "cohen_kappa": cleaned_unit_metrics["cohen_kappa_macro_response"],
            "macro_f1": cleaned_unit_metrics["macro_f1_macro_response"],
            "accuracy": cleaned_unit_metrics["accuracy_macro_response"],
        }
    )

    # Questionnaire comparisons: Experience Tone(3), NDE-C, and LCI-R
    questionnaire_work = base.copy()
    n, kappa, macro = _comparison_metrics(
        questionnaire_work,
        HUMAN_TONE3["experience"],
        "questionnaire_tone",
        study.tone_labels,
    )
    metrics_rows.append(
        {
            "comparison": "human_vs_questionnaire",
            "model": "questionnaire",
            "family": "tone",
            "field": "experience_tone",
            "n": n,
            "cohen_kappa": kappa,
            "macro_f1": macro,
        }
    )

    for field in study.sections["experience"].binary_labels:
        n, kappa, macro = _comparison_metrics(
            questionnaire_work,
            field,
            f"{field}_questionnaire",
            study.binary_labels,
        )
        metrics_rows.append(
            {
                "comparison": "human_vs_questionnaire",
                "model": "questionnaire",
                "family": "nde_c",
                "field": field,
                "n": n,
                "cohen_kappa": kappa,
                "macro_f1": macro,
            }
        )

    for field in study.sections["aftereffects"].binary_labels:
        n, kappa, macro = _comparison_metrics(
            questionnaire_work,
            field,
            f"{field}_questionnaire",
            study.binary_labels,
        )
        metrics_rows.append(
            {
                "comparison": "human_vs_questionnaire",
                "model": "questionnaire",
                "family": "lci_r",
                "field": field,
                "n": n,
                "cohen_kappa": kappa,
                "macro_f1": macro,
            }
        )

    llm_artifacts = discover_default_llm_artifacts(llm_results_dir)
    llm_valid_rows: list[dict[str, Any]] = []
    llm_metric_rows: list[dict[str, Any]] = []

    for artifact in llm_artifacts:
        llm_df = _load_llm_predictions(artifact.predictions_path, study)
        llm_df = llm_df[llm_df["response_id"].isin(human_ids)].copy()
        if llm_df.empty:
            continue
        merged = base.merge(llm_df, on="response_id", how="left", suffixes=("", "_llm"))

        # Tone is restricted to experience only for comparability with
        # questionnaire valence alignment.
        n, kappa, macro = _comparison_metrics(
            merged,
            HUMAN_TONE4["experience"],
            "experience_tone_llm",
            study.tone_labels,
        )
        llm_metric_rows.append(
            {
                "comparison": "human_vs_llm",
                "model": artifact.artifact_id,
                "model_variant": artifact.model_variant,
                "family": "tone",
                "field": "experience_tone",
                "n": n,
                "cohen_kappa": kappa,
                "macro_f1": macro,
            }
        )

        for field in study.sections["experience"].binary_labels:
            n, kappa, macro = _comparison_metrics(
                merged, field, f"{field}_llm", study.binary_labels
            )
            llm_metric_rows.append(
                {
                    "comparison": "human_vs_llm",
                    "model": artifact.artifact_id,
                    "model_variant": artifact.model_variant,
                    "family": "nde_c",
                    "field": field,
                    "n": n,
                    "cohen_kappa": kappa,
                    "macro_f1": macro,
                }
            )

        for field in study.sections["aftereffects"].binary_labels:
            n, kappa, macro = _comparison_metrics(
                merged,
                field,
                f"{field}_llm",
                study.binary_labels,
            )
            llm_metric_rows.append(
                {
                    "comparison": "human_vs_llm",
                    "model": artifact.artifact_id,
                    "model_variant": artifact.model_variant,
                    "family": "lci_r",
                    "field": field,
                    "n": n,
                    "cohen_kappa": kappa,
                    "macro_f1": macro,
                }
            )

        merged_llm_valid = merged.copy()
        llm_value_columns = [
            "context_tone_llm",
            "experience_tone_llm",
            "aftereffects_tone_llm",
            *[f"{field}_llm" for field in study.binary_columns()],
        ]
        for column in llm_value_columns:
            if column in merged_llm_valid.columns:
                if column.endswith("_tone_llm"):
                    merged_llm_valid[column] = merged_llm_valid[column].map(
                        _normalize_tone
                    )
                else:
                    merged_llm_valid[column] = merged_llm_valid[column].map(
                        _normalize_yes_no
                    )
        llm_counts = _compute_section_validity(merged_llm_valid, prefix="", study=study)
        rename_map = {
            "context_tone_llm": "llm_context_tone",
            "experience_tone_llm": "llm_experience_tone",
            "aftereffects_tone_llm": "llm_aftereffects_tone",
            **{f"{field}_llm": f"llm_{field}" for field in study.binary_columns()},
        }
        llm_counts = _compute_section_validity(
            merged_llm_valid.rename(columns=rename_map),
            prefix="llm_",
            study=study,
        )
        observed_columns = [
            column for column in llm_value_columns if column in merged_llm_valid.columns
        ]
        observed_llm = (
            merged_llm_valid[observed_columns].notna().any(axis=1)
            if observed_columns
            else pd.Series(False, index=merged_llm_valid.index)
        )
        llm_valid_frame = pd.DataFrame(
            {
                "response_id": merged_llm_valid["response_id"].astype(int),
                "n_valid_sections": pd.Series(llm_counts).astype(int),
                "observed": observed_llm.astype(bool),
            }
        )
        llm_valid_frame = llm_valid_frame[llm_valid_frame["observed"]].copy()
        llm_valid_rows.extend(
            [
                {
                    "response_id": int(row["response_id"]),
                    "source": artifact.artifact_id,
                    "n_valid_sections": int(row["n_valid_sections"]),
                }
                for _, row in llm_valid_frame.iterrows()
            ]
        )

    metrics_df = pd.DataFrame([*metrics_rows, *llm_metric_rows])
    family_metrics_df = metrics_df.groupby(
        ["comparison", "model", "family"], as_index=False
    )[["cohen_kappa", "macro_f1", "n"]].mean(numeric_only=True)

    valid_base = base.copy()
    valid_base["context_tone"] = valid_base[HUMAN_TONE4["context"]]
    valid_base["experience_tone"] = valid_base[HUMAN_TONE4["experience"]]
    valid_base["aftereffects_tone"] = valid_base[HUMAN_TONE4["aftereffects"]]
    valid_base["n_valid_sections_human"] = _compute_section_validity(
        valid_base, prefix="", study=study
    )
    valid_base["n_valid_sections_cleaned"] = pd.to_numeric(
        valid_base.get("n_valid_sections_cleaned"), errors="coerce"
    )

    valid_rows = [
        {
            "response_id": int(row["response_id"]),
            "source": "human",
            "n_valid_sections": int(row["n_valid_sections_human"]),
        }
        for _, row in valid_base.iterrows()
    ]
    valid_rows.extend(
        [
            {
                "response_id": int(row["response_id"]),
                "source": "cleaned_dataset",
                "n_valid_sections": int(row["n_valid_sections_cleaned"])
                if pd.notna(row["n_valid_sections_cleaned"])
                else 0,
            }
            for _, row in valid_base.iterrows()
        ]
    )
    valid_rows.extend(llm_valid_rows)
    valid_df = pd.DataFrame(valid_rows)

    valid_comparison_rows: list[dict[str, Any]] = []
    human_counts = valid_df[valid_df["source"] == "human"][
        ["response_id", "n_valid_sections"]
    ].rename(columns={"n_valid_sections": "human_n_valid_sections"})
    # Keep this agreement table focused on the intended preprocessing comparison.
    for source in ["cleaned_dataset"]:
        if source not in set(valid_df["source"]):
            continue
        candidate = valid_df[valid_df["source"] == source][
            ["response_id", "n_valid_sections"]
        ].rename(columns={"n_valid_sections": "candidate_n_valid_sections"})
        merged_valid = human_counts.merge(candidate, on="response_id", how="inner")
        exact = (
            float(
                (
                    merged_valid["human_n_valid_sections"]
                    == merged_valid["candidate_n_valid_sections"]
                ).mean()
            )
            if not merged_valid.empty
            else float("nan")
        )
        mean_abs = (
            float(
                (
                    merged_valid["human_n_valid_sections"]
                    - merged_valid["candidate_n_valid_sections"]
                )
                .abs()
                .mean()
            )
            if not merged_valid.empty
            else float("nan")
        )
        valid_comparison_rows.append(
            {
                "comparison": f"human_vs_{source}",
                "n": int(len(merged_valid)),
                "exact_match_rate": exact,
                "mean_absolute_error": mean_abs,
            }
        )
    valid_comparison_df = pd.DataFrame(valid_comparison_rows)

    unit_long_path = output_dir / "human_review_unit_alignment_long.csv"
    unit_metrics_path = output_dir / "human_review_unit_alignment_metrics.csv"
    metrics_path = output_dir / "human_review_metrics.csv"
    family_metrics_path = output_dir / "human_review_family_metrics.csv"
    valid_path = output_dir / "human_review_valid_sections.csv"
    valid_comparison_path = output_dir / "human_review_valid_sections_comparison.csv"

    unit_long.to_csv(unit_long_path, index=False)
    pd.DataFrame(
        [
            {
                "comparison": "human_vs_cleaned",
                "model": "cleaned_dataset",
                **cleaned_unit_metrics,
            }
        ]
        + [
            {
                "comparison": row.get("comparison"),
                "model": row.get("model"),
                "n_units": row.get("n"),
                "accuracy_macro_response": row.get("accuracy"),
                "cohen_kappa_macro_response": row.get("cohen_kappa"),
                "macro_f1_macro_response": row.get("macro_f1"),
            }
            for row in llm_metric_rows
            if row.get("family") == "seg_unit"
        ]
    ).to_csv(unit_metrics_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    family_metrics_df.to_csv(family_metrics_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    valid_comparison_df.to_csv(valid_comparison_path, index=False)

    figure_files: dict[str, str] = {}
    _plot_family_metrics_combined(
        family_metrics_df,
        figures_dir / "human_family_alignment_combined.png",
        figure_dpi,
        export_figures_pdf,
    )
    _plot_valid_sections(
        valid_df,
        figures_dir / "human_valid_sections_comparison.png",
        figure_dpi,
        export_figures_pdf,
    )
    for png in sorted(figures_dir.glob("*.png")):
        figure_files[png.stem] = str(png)

    seg_unit_summary = family_metrics_df[
        family_metrics_df["family"] == "seg_unit"
    ].copy()
    findings = _dynamic_findings(
        family_metrics_df[family_metrics_df["comparison"] == "human_vs_llm"],
        family_metrics_df[family_metrics_df["comparison"] == "human_vs_questionnaire"],
        seg_unit_summary,
    )

    family_report_df = family_metrics_df.copy()
    family_report_df["comparison_label"] = family_report_df["comparison"].map(
        _comparison_label
    )
    family_report_df["model_label"] = family_report_df["model"].map(
        _presentation_model_name
    )
    family_report_df["family_label"] = family_report_df["family"].map(_family_label)

    report_path = output_dir / "human_review_alignment_report.md"
    lines = [
        "# Human Review Alignment Report",
        "",
        "## Scope",
        f"- Human records parsed: **{len(human_df)}**",
        f"- IDs with cleaned segmentation: **{cleaned_df['response_id'].nunique()}**",
        f"- IDs with questionnaire rows: **{questionnaire_df['response_id'].nunique()}**",
        f"- Default-prompt LLM artifacts evaluated: **{len(llm_artifacts)}**",
        "- LLM selection excludes RA1 runs.",
        "",
        "## Main Figures",
        "",
        "![Human-family alignment combined](figures/human_family_alignment_combined.png)",
        "",
        "![Valid sections comparison](figures/human_valid_sections_comparison.png)",
        "",
        "## Unit-Classification Segmentation",
        "",
        "Source text is split into sentence-like units from `NDE_traslated.csv`. Each unit is classified as `context`, `experience`, `aftereffects`, or `deleted` according to each system output, then agreement is computed against the human review labels.",
        "",
        "| Comparison | Model | Accuracy (macro-by-response) | Kappa (macro-by-response) | Macro-F1 (macro-by-response) | n units |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    unit_metrics_df = pd.read_csv(unit_metrics_path)
    for _, row in unit_metrics_df.sort_values(["comparison", "model"]).iterrows():
        lines.append(
            f"| {_comparison_label(row['comparison'])} | {_presentation_model_name(row['model'])} | {float(row['accuracy_macro_response']):.3f} | {float(row['cohen_kappa_macro_response']):.3f} | {float(row['macro_f1_macro_response']):.3f} | {int(float(row['n_units']))} |"
        )

    lines.extend(
        [
            "",
            "## Family Alignment",
            "",
            "- `tone`: experience-only alignment (`Experience Tone (4)` vs LLM, `Experience Tone (3)` vs questionnaire valence).",
            "- `nde_c`: full NDE-C item set from the experience section.",
            "- `lci_r`: full LCI-R item set from the aftereffects section.",
            "- `seg_unit`: unit-classification segmentation agreement (`context/experience/aftereffects/deleted`).",
            "",
            "| Comparison | Model | Family | Mean Kappa | Mean Macro-F1 | Mean n |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in family_report_df.sort_values(
        ["comparison_label", "model_label", "family_label"]
    ).iterrows():
        lines.append(
            f"| {row['comparison_label']} | {row['model_label']} | {row['family_label']} | {row['cohen_kappa']:.3f} | {row['macro_f1']:.3f} | {row['n']:.1f} |"
        )

    family_agg = (
        family_report_df.groupby(["comparison_label", "family_label"], as_index=False)
        .agg(
            mean_kappa=("cohen_kappa", "mean"),
            sd_kappa=("cohen_kappa", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"),
            mean_n=("n", "mean"),
            sd_n=("n", "std"),
            n_models=("model_label", "nunique"),
        )
        .sort_values(["comparison_label", "family_label"])
    )

    def _fmt_or_dash(value: object, decimals: int = 3) -> str:
        try:
            if pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return "-"

    lines.extend(
        [
            "",
            "### Family Summary (Mean ± SD)",
            "",
            "Standard deviation is shown when more than one model contributes to the row.",
            "",
            "| Comparison | Family | Mean Kappa | SD Kappa | Mean Macro-F1 | SD Macro-F1 | Mean n | SD n | n models |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in family_agg.iterrows():
        lines.append(
            "| "
            + f"{row['comparison_label']} | {row['family_label']} | "
            + f"{_fmt_or_dash(row['mean_kappa'])} | {_fmt_or_dash(row['sd_kappa'])} | "
            + f"{_fmt_or_dash(row['mean_macro_f1'])} | {_fmt_or_dash(row['sd_macro_f1'])} | "
            + f"{_fmt_or_dash(row['mean_n'], 1)} | {_fmt_or_dash(row['sd_n'], 1)} | {int(row['n_models'])} |"
        )

    lines.extend(
        [
            "",
            "## Valid Section Count Agreement",
            "",
            "| Comparison | n | Exact match rate | Mean absolute error |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in valid_comparison_df.sort_values("comparison").iterrows():
        comparison_label = _valid_comparison_label(row["comparison"])
        lines.append(
            f"| {comparison_label} | {int(row['n'])} | {row['exact_match_rate']:.3f} | {row['mean_absolute_error']:.3f} |"
        )

    if findings:
        lines.extend(["", "## Dynamic Findings", ""])
        lines.extend([f"- {item}" for item in findings])

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/human_family_alignment_combined.png`",
            "- `figures/human_valid_sections_comparison.png`",
            "",
            "PDF copies are exported alongside PNG when enabled.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "coverage": {
            "n_human_records": int(len(human_df)),
            "n_cleaned_overlap": int(cleaned_df["response_id"].nunique()),
            "n_questionnaire_overlap": int(questionnaire_df["response_id"].nunique()),
            "n_llm_artifacts": int(len(llm_artifacts)),
        },
        "dynamic_findings": findings,
    }
    summary_path = output_dir / "human_review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    written = {
        "report_file": str(report_path),
        "summary_file": str(summary_path),
        "metrics_file": str(metrics_path),
        "family_metrics_file": str(family_metrics_path),
        "unit_alignment_long_file": str(unit_long_path),
        "unit_alignment_metrics_file": str(unit_metrics_path),
        "valid_sections_file": str(valid_path),
        "valid_sections_comparison_file": str(valid_comparison_path),
        **{f"figure_{key}": value for key, value in figure_files.items()},
    }
    return metrics_df, summary, written
