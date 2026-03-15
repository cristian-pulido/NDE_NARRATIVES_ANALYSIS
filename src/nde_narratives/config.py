from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from .constants import PARTICIPANT_CODE_HEADER, PLACEHOLDER_PREFIX, PROJECT_ROOT


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
    sampled_private_workbook: Path
    human_annotation_workbook: Path
    llm_predictions_path: Path


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
    return PathsConfig(
        path=resolved,
        survey_csv=_resolve_path(paths["survey_csv"], base_dir),
        annotation_output_dir=_resolve_path(paths["annotation_output_dir"], base_dir),
        llm_batch_dir=_resolve_path(paths["llm_batch_dir"], base_dir),
        evaluation_output_dir=_resolve_path(paths["evaluation_output_dir"], base_dir),
        sampled_private_workbook=_resolve_path(paths["sampled_private_workbook"], base_dir),
        human_annotation_workbook=_resolve_path(paths["human_annotation_workbook"], base_dir),
        llm_predictions_path=_resolve_path(paths["llm_predictions_path"], base_dir),
    )
