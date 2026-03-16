from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from .constants import PARTICIPANT_CODE_HEADER, PLACEHOLDER_PREFIX, PROJECT_ROOT


DEFAULT_PATH_LAYOUT = {
    "annotation_output_dir": "annotation_outputs",
    "human_annotations_dir": "human_annotations",
    "llm_batch_dir": "llm_batches",
    "llm_results_dir": "llm_outputs",
    "evaluation_output_dir": "evaluation_outputs",
    "prompt_variants_dir": "prompt_variants",
    "sampled_private_workbook": "annotation_outputs/nde_annotation_mapping_private.xlsx",
    "human_annotation_workbook": "human_annotations/nde_annotation_sample_completed.xlsx",
    "llm_predictions_path": "llm_outputs/nde_predictions.jsonl",
}


@dataclass(frozen=True)
class SectionConfig:
    name: str
    source_column: str
    text_label: str
    tone_internal_column: str
    tone_visible_label: str
    binary_labels: dict[str, str]


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    dataset: dict[str, Any]
    sampling: dict[str, Any]
    workflow: dict[str, Any]
    outputs: dict[str, str]
    labels: dict[str, list[str]]
    sections: dict[str, SectionConfig]
    questionnaire: dict[str, dict[str, Any]]

    @property
    def section_order(self) -> list[str]:
        return list(self.workflow["section_order"])

    @property
    def id_column(self) -> str:
        return str(self.dataset["id_column"])

    @property
    def stratify_column(self) -> str:
        return str(self.dataset["stratify_column"])

    @property
    def tone_labels(self) -> list[str]:
        return list(self.labels["tone"])

    @property
    def binary_labels(self) -> list[str]:
        return list(self.labels["binary"])

    def text_columns(self) -> dict[str, str]:
        return {name: section.source_column for name, section in self.sections.items()}

    def tone_columns(self) -> list[str]:
        return [self.sections[name].tone_internal_column for name in self.section_order]

    def binary_columns(self) -> list[str]:
        columns: list[str] = []
        for name in self.section_order:
            columns.extend(self.sections[name].binary_labels.keys())
        return columns

    def annotation_internal_columns(self) -> list[str]:
        return self.tone_columns() + self.binary_columns()

    def internal_to_visible_annotation_columns(self) -> dict[str, str]:
        mapping = {"participant_code": PARTICIPANT_CODE_HEADER}
        for name in self.section_order:
            section = self.sections[name]
            mapping[f"{name}_text"] = section.text_label
            mapping[section.tone_internal_column] = section.tone_visible_label
            mapping.update(section.binary_labels)
        return mapping

    def visible_to_internal_annotation_columns(self) -> dict[str, str]:
        return {visible: internal for internal, visible in self.internal_to_visible_annotation_columns().items()}

    def questionnaire_column_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for block_name in ("m8", "m9"):
            out.update(self.questionnaire[block_name]["columns"])
        return out

    def placeholder_questionnaire_columns(self) -> dict[str, str]:
        return {
            internal: source
            for internal, source in self.questionnaire_column_map().items()
            if str(source).startswith(PLACEHOLDER_PREFIX)
        }

    def required_source_columns(self) -> list[str]:
        columns = [self.id_column, self.stratify_column]
        columns.extend(self.text_columns().values())

        quality_col = self.dataset.get("quality_label_column")
        if quality_col:
            columns.append(str(quality_col))

        to_drop_col = self.dataset.get("to_drop_column")
        if to_drop_col:
            columns.append(str(to_drop_col))

        columns.extend(self.questionnaire_column_map().values())
        return columns

    def allowed_labels_for_column(self, column: str) -> list[str]:
        if column in self.tone_columns():
            return self.tone_labels
        if column in self.binary_columns():
            return self.binary_labels
        raise KeyError(f"Unknown normalized column: {column}")


@dataclass(frozen=True)
class PathsConfig:
    path: Path
    survey_csv: Path
    annotation_output_dir: Path
    llm_batch_dir: Path
    evaluation_output_dir: Path
    human_annotations_dir: Path
    llm_results_dir: Path
    sampled_private_workbook: Path
    human_annotation_workbook: Path
    llm_predictions_path: Path
    prompt_variants_dir: Path | None = None
    data_dir: Path | None = None


@dataclass(frozen=True)
class ExperimentMetadata:
    experiment_id: str
    prompt_variant: str | None = None
    run_id: str | None = None
    model_variant: str | None = None

    @property
    def artifact_id(self) -> str:
        if self.run_id:
            return f"{self.experiment_id}__{self.run_id}"
        return self.experiment_id

    def to_dict(self) -> dict[str, str]:
        data = {"experiment_id": self.experiment_id, "artifact_id": self.artifact_id}
        if self.prompt_variant:
            data["prompt_variant"] = self.prompt_variant
        if self.run_id:
            data["run_id"] = self.run_id
        if self.model_variant:
            data["model_variant"] = self.model_variant
        return data


def default_study_config_path() -> Path:
    return PROJECT_ROOT / "config" / "study.toml"


def default_paths_config_path() -> Path:
    return PROJECT_ROOT / "config" / "paths.local.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _path_with_default(paths: dict[str, Any], key: str, base_dir: Path, default_base: Path) -> Path:
    raw_value = paths.get(key)
    if raw_value:
        return _resolve_path(str(raw_value), base_dir)
    return (default_base / DEFAULT_PATH_LAYOUT[key]).resolve()


def load_study_config(path: str | Path | None = None) -> StudyConfig:
    resolved = Path(path or default_study_config_path()).resolve()
    raw = _load_toml(resolved)

    sections: dict[str, SectionConfig] = {}
    for name, section_raw in raw["sections"].items():
        sections[name] = SectionConfig(
            name=name,
            source_column=section_raw["source_column"],
            text_label=section_raw["text_label"],
            tone_internal_column=section_raw["tone_internal_column"],
            tone_visible_label=section_raw["tone_visible_label"],
            binary_labels=dict(section_raw.get("binary_labels", {})),
        )

    questionnaire = {
        "m8": {
            "yes_values": list(raw["questionnaire"]["m8"]["yes_values"]),
            "no_values": list(raw["questionnaire"]["m8"]["no_values"]),
            "columns": dict(raw["questionnaire"]["m8"]["columns"]),
        },
        "m9": {
            "yes_values": list(raw["questionnaire"]["m9"]["yes_values"]),
            "no_values": list(raw["questionnaire"]["m9"]["no_values"]),
            "columns": dict(raw["questionnaire"]["m9"]["columns"]),
        },
    }

    return StudyConfig(
        path=resolved,
        dataset=dict(raw["dataset"]),
        sampling=dict(raw["sampling"]),
        workflow=dict(raw["workflow"]),
        outputs=dict(raw["outputs"]),
        labels={key: list(value) for key, value in raw["labels"].items()},
        sections=sections,
        questionnaire=questionnaire,
    )


def load_paths_config(path: str | Path | None = None) -> PathsConfig:
    resolved = Path(path or default_paths_config_path()).resolve()
    raw = _load_toml(resolved)
    base_dir = resolved.parent
    paths = raw["paths"]

    data_dir_raw = paths.get("data_dir")
    default_base = _resolve_path(str(data_dir_raw), base_dir) if data_dir_raw else base_dir

    survey_csv = _resolve_path(paths["survey_csv"], base_dir)
    annotation_output_dir = _path_with_default(paths, "annotation_output_dir", base_dir, default_base)
    llm_batch_dir = _path_with_default(paths, "llm_batch_dir", base_dir, default_base)
    evaluation_output_dir = _path_with_default(paths, "evaluation_output_dir", base_dir, default_base)
    human_annotations_dir = _path_with_default(paths, "human_annotations_dir", base_dir, default_base)
    llm_results_dir = _path_with_default(paths, "llm_results_dir", base_dir, default_base)
    sampled_private_workbook = _path_with_default(paths, "sampled_private_workbook", base_dir, default_base)
    human_annotation_workbook = _path_with_default(paths, "human_annotation_workbook", base_dir, default_base)
    llm_predictions_path = _path_with_default(paths, "llm_predictions_path", base_dir, default_base)
    prompt_variants_dir = _path_with_default(paths, "prompt_variants_dir", base_dir, default_base)

    return PathsConfig(
        path=resolved,
        survey_csv=survey_csv,
        annotation_output_dir=annotation_output_dir,
        llm_batch_dir=llm_batch_dir,
        evaluation_output_dir=evaluation_output_dir,
        human_annotations_dir=human_annotations_dir,
        llm_results_dir=llm_results_dir,
        sampled_private_workbook=sampled_private_workbook,
        human_annotation_workbook=human_annotation_workbook,
        llm_predictions_path=llm_predictions_path,
        prompt_variants_dir=prompt_variants_dir,
        data_dir=default_base if data_dir_raw else None,
    )
