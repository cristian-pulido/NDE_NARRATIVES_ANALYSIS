from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from nde_narratives.constants import PARTICIPANT_CODE_HEADER

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli


def test_validate_config_passes_with_synthetic_fixture(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    result = run_cli("validate-config", "--study-config", str(study_config), "--paths-config", str(paths_config))

    assert result.returncode == 0, result.stderr
    assert "Configuration valid." in result.stdout


def test_build_annotation_sample_creates_workbooks(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))

    assert result.returncode == 0, result.stderr
    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"
    column_map_workbook = tmp_path / "annotation_outputs" / "nde_annotation_column_mapping_private.xlsx"

    assert annotation_workbook.exists()
    assert mapping_workbook.exists()
    assert column_map_workbook.exists()

    workbook = load_workbook(annotation_workbook)
    assert set(workbook.sheetnames) == {"annotation", "instructions"}
    worksheet = workbook["annotation"]
    headers = [cell.value for cell in worksheet[1]]
    assert PARTICIPANT_CODE_HEADER in headers
    assert "Context Narrative" in headers
    assert len(worksheet.data_validations.dataValidation) == 2

    sampled_private = pd.read_excel(mapping_workbook, sheet_name="sampled_private")
    assert len(sampled_private) == 3
    assert "participant_code" in sampled_private.columns


def test_build_annotation_sample_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    first_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert first_result.returncode == 0, first_result.stderr

    second_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))

    assert second_result.returncode == 1
    assert "Refusing to overwrite existing annotation artifacts" in second_result.stderr
    assert "nde build-annotation-sample --force" in second_result.stderr
    assert "completed human annotation workbook" in second_result.stderr



def test_build_annotation_sample_force_allows_intentional_overwrite(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    first_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert first_result.returncode == 0, first_result.stderr

    force_result = run_cli(
        "build-annotation-sample",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--force",
    )

    assert force_result.returncode == 0, force_result.stderr


def test_build_llm_batch_creates_three_section_files(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    result = run_cli("build-llm-batch", "--study-config", str(study_config), "--paths-config", str(paths_config))

    assert result.returncode == 0, result.stderr

    for section in ("context", "experience", "aftereffects"):
        batch_path = tmp_path / "llm_batches" / f"{section}_batch.jsonl"
        assert batch_path.exists()
        records = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(records) == 3
        assert {record["section"] for record in records} == {section}
        assert all(record["response_schema"]["title"].startswith("nde_") for record in records)
