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


def test_build_llm_batch_writes_experiment_directory_and_manifest(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    result = run_cli(
        "build-llm-batch",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--experiment-id",
        "exp-alpha",
        "--prompt-variant",
        "baseline",
        "--run-id",
        "run-01",
    )

    assert result.returncode == 0, result.stderr

    batch_dir = tmp_path / "llm_batches" / "exp-alpha__run-01"
    manifest_path = batch_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "exp-alpha"
    assert manifest["prompt_variant"] == "baseline"
    assert manifest["run_id"] == "run-01"

    for section in ("context", "experience", "aftereffects"):
        batch_path = batch_dir / f"{section}_batch.jsonl"
        assert batch_path.exists()
        records = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(records) == 3
        assert {record["section"] for record in records} == {section}
        assert all(record["experiment"]["artifact_id"] == "exp-alpha__run-01" for record in records)


def test_validate_config_accepts_minimal_data_dir_paths_config(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    data_dir = tmp_path / "nde_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    content = f"""[paths]
data_dir = "{data_dir.as_posix()}"
survey_csv = "{survey_csv.as_posix()}"
"""
    paths_config = tmp_path / "paths.local.toml"
    paths_config.write_text(content, encoding="utf-8")

    result = run_cli("validate-config", "--study-config", str(study_config), "--paths-config", str(paths_config))

    assert result.returncode == 0, result.stderr
    assert (data_dir / "annotation_outputs").exists()
    assert (data_dir / "human_annotations").exists()
    assert (data_dir / "llm_outputs").exists()
