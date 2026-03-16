from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from nde_narratives.config import load_study_config
from nde_narratives.constants import ANNOTATION_SHEET

from tests.cli_helpers import (
    FIXTURES,
    make_paths_config,
    populate_human_annotation_workbook,
    run_cli,
    write_llm_predictions_fixture,
)


def test_evaluate_reuses_existing_vader_scores_and_reports_vader_metrics(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"
    prediction_path = tmp_path / "llm_outputs" / "nde_predictions.jsonl"
    vader_dir = tmp_path / "evaluation_outputs" / "vader_sentiment"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)
    write_llm_predictions_fixture(mapping_workbook, study_config, prediction_path)

    vader_result = run_cli(
        "sentiment-sensitivity",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--output-dir",
        str(vader_dir),
    )
    assert vader_result.returncode == 0, vader_result.stderr

    eval_result = run_cli(
        "evaluate",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--vader-scores",
        str(vader_dir / "vader_sentiment_scores.csv"),
    )
    assert eval_result.returncode == 0, eval_result.stderr

    metrics = pd.read_csv(tmp_path / "evaluation_outputs" / "evaluation_metrics.csv")
    report_path = tmp_path / "evaluation_outputs" / "alignment_report.md"
    help_others = metrics[(metrics["comparison"] == "human_vs_llm") & (metrics["field"] == "m9_help_others")].iloc[0]
    assert set(metrics["comparison"]).issuperset({"human_vs_vader", "llm_vs_vader"})
    assert len(metrics[metrics["comparison"] == "human_vs_vader"]) == 3
    assert help_others["accuracy"] == 2 / 3
    assert round(help_others["cohen_kappa"], 4) == 0.4
    assert round(help_others["macro_f1"], 4) == 0.6667
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Objective" in report_text
    assert "## Methodology" in report_text
    assert "## Global Results" in report_text
    assert "## Interpretation" in report_text
    assert "LLM-derived comparisons were available" in report_text
    assert "Context Tone" in report_text
    assert "Bright light" in report_text
    assert "Greater willingness to forgive" in report_text
    assert "context_tone" not in report_text
    assert (tmp_path / "evaluation_outputs" / "figures" / "alignment" / "comparison_summary.png").exists()

    workbook = load_workbook(annotation_workbook)
    worksheet = workbook[ANNOTATION_SHEET]
    headers = {cell.value: index for index, cell in enumerate(worksheet[1], start=1)}
    worksheet.cell(row=2, column=headers["Context Tone"], value="invalid")
    workbook.save(annotation_workbook)

    invalid_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert invalid_result.returncode == 1
    assert "invalid labels" in invalid_result.stderr


def test_evaluate_runs_without_default_llm_predictions_and_reports_available_comparisons(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 0, eval_result.stderr

    metrics = pd.read_csv(tmp_path / "evaluation_outputs" / "evaluation_metrics.csv")
    report_path = tmp_path / "evaluation_outputs" / "alignment_report.md"
    study = load_study_config(study_config)
    assert set(metrics["comparison"]) == {"human_vs_questionnaire", "human_vs_vader"}
    assert len(metrics[metrics["comparison"] == "human_vs_vader"]) == len(study.tone_columns())
    assert len(metrics[metrics["comparison"] == "human_vs_questionnaire"]) == len(study.binary_columns())
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "human vs questionnaire" in report_text.lower()
    assert "human vs vader" in report_text.lower()
    assert "Context Tone" in report_text
    assert "Out-of-body sensation" in report_text
    assert "LLM-derived comparisons were not available" in report_text
    assert "m8_out_of_body" not in report_text
    assert (tmp_path / "evaluation_outputs" / "figures" / "alignment" / "tone_alignment.png").exists()


def test_evaluate_generates_sample_vader_scores_when_global_file_is_missing(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 0, eval_result.stderr

    sample_scores_path = tmp_path / "evaluation_outputs" / "vader_sentiment_sample" / "vader_sentiment_scores.csv"
    report_path = tmp_path / "evaluation_outputs" / "alignment_report.md"
    assert sample_scores_path.exists()
    assert report_path.exists()
    metrics = pd.read_csv(tmp_path / "evaluation_outputs" / "evaluation_metrics.csv")
    assert len(metrics[metrics["comparison"] == "human_vs_vader"]) == 3


def test_evaluate_fails_for_missing_explicit_vader_scores_path(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"
    prediction_path = tmp_path / "llm_outputs" / "nde_predictions.jsonl"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)
    write_llm_predictions_fixture(mapping_workbook, study_config, prediction_path)

    missing_path = tmp_path / "evaluation_outputs" / "does_not_exist.csv"
    eval_result = run_cli(
        "evaluate",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--vader-scores",
        str(missing_path),
    )
    assert eval_result.returncode == 1
    assert "VADER scores file not found" in eval_result.stderr
