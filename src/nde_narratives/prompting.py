from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PathsConfig, StudyConfig
from .constants import PROMPT_INPUT_TOKEN, PROJECT_ROOT, SAMPLED_PRIVATE_SHEET
from .io_utils import read_tabular_file
from .sampling import assign_participant_codes, filter_source_data


def load_prompt_template(section: str, project_root: Path = PROJECT_ROOT) -> str:
    path = project_root / "prompts" / f"{section}_prompt.md"
    return path.read_text(encoding="utf-8")


def load_response_schema(section: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = project_root / "schemas" / f"{section}_output.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def render_prompt(section: str, input_text: str, project_root: Path = PROJECT_ROOT) -> str:
    template = load_prompt_template(section, project_root=project_root)
    return template.replace(PROMPT_INPUT_TOKEN, input_text)


def load_batch_source(
    study: StudyConfig,
    paths: PathsConfig,
    source: str,
    input_path: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if source == "sampled-private":
        workbook = Path(input_path or paths.sampled_private_workbook)
        if not workbook.exists():
            raise FileNotFoundError(f"Sampled private workbook not found: {workbook}")
        df = pd.read_excel(workbook, sheet_name=SAMPLED_PRIVATE_SHEET)
    elif source == "survey":
        survey_path = Path(input_path or paths.survey_csv)
        if not survey_path.exists():
            raise FileNotFoundError(f"Survey source not found: {survey_path}")
        raw = read_tabular_file(survey_path)
        filtered = filter_source_data(raw, study)
        filtered = filtered.sort_values(study.id_column).reset_index(drop=True)
        df = assign_participant_codes(filtered, study)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if "participant_code" not in df.columns:
        raise ValueError("Batch source must include participant_code.")

    if limit is not None:
        df = df.head(limit).copy()
    return df


def build_llm_batch_records(sampled_df: pd.DataFrame, study: StudyConfig) -> dict[str, list[dict[str, Any]]]:
    batches: dict[str, list[dict[str, Any]]] = {section: [] for section in study.section_order}
    for _, row in sampled_df.iterrows():
        participant_code = row["participant_code"]
        for section_name in study.section_order:
            section = study.sections[section_name]
            input_text = str(row[section.source_column])
            batches[section_name].append(
                {
                    "participant_code": participant_code,
                    "section": section_name,
                    "input_text": input_text,
                    "prompt": render_prompt(section_name, input_text),
                    "response_schema": load_response_schema(section_name),
                }
            )
    return batches


def write_llm_batches(
    study: StudyConfig,
    paths: PathsConfig,
    source: str,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, str]:
    sampled_df = load_batch_source(study=study, paths=paths, source=source, input_path=input_path, limit=limit)
    batches = build_llm_batch_records(sampled_df, study)

    batch_dir = Path(output_dir or paths.llm_batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for section_name, records in batches.items():
        batch_path = batch_dir / f"{section_name}_batch.jsonl"
        with batch_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        written[section_name] = str(batch_path)
    return written
