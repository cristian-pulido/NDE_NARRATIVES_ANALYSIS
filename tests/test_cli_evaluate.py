from __future__ import annotations

import json
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


def _annotation_headers(worksheet) -> dict[str, int]:
    return {cell.value: index for index, cell in enumerate(worksheet[1], start=1)}


def _clear_annotation_row(annotation_workbook: Path, row_index: int) -> None:
    workbook = load_workbook(annotation_workbook)
    worksheet = workbook[ANNOTATION_SHEET]
    headers = _annotation_headers(worksheet)
    for column_index in headers.values():
        if column_index == headers["Participant Code"]:
            continue
        worksheet.cell(row=row_index, column=column_index, value="")
    workbook.save(annotation_workbook)


def _set_annotation_value(annotation_workbook: Path, row_index: int, header: str, value: str) -> None:
    workbook = load_workbook(annotation_workbook)
    worksheet = workbook[ANNOTATION_SHEET]
    headers = _annotation_headers(worksheet)
    worksheet.cell(row=row_index, column=headers[header], value=value)
    workbook.save(annotation_workbook)


def _delete_annotation_row(annotation_workbook: Path, row_index: int) -> None:
    workbook = load_workbook(annotation_workbook)
    worksheet = workbook[ANNOTATION_SHEET]
    worksheet.delete_rows(row_index)
    workbook.save(annotation_workbook)


def _append_extra_llm_prediction(prediction_path: Path, participant_code: str, study_config: Path) -> None:
    study = load_study_config(study_config)
    payload = {"participant_code": participant_code}
    for section_name in study.section_order:
        section = study.sections[section_name]
        payload[section.tone_internal_column] = study.tone_labels[0]
        for column in section.binary_labels:
            payload[column] = study.binary_labels[0]

    with prediction_path.open("a", encoding="utf-8") as handle:
        for _ in study.section_order:
            handle.write(json.dumps(payload))
            handle.write("\n")


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
    summary = json.loads((tmp_path / "evaluation_outputs" / "evaluation_summary.json").read_text(encoding="utf-8"))
    help_others = metrics[(metrics["comparison"] == "human_vs_llm") & (metrics["field"] == "m9_help_others")].iloc[0]
    assert set(metrics["comparison"]).issuperset({"human_vs_vader", "llm_vs_vader"})
    assert len(metrics[metrics["comparison"] == "human_vs_vader"]) == 3
    assert help_others["accuracy"] == 2 / 3
    assert round(help_others["cohen_kappa"], 4) == 0.4
    assert round(help_others["macro_f1"], 4) == 0.6667
    assert summary["coverage"] == {
        "n_sampled_total": 3,
        "n_human_rows_total": 3,
        "n_human_evaluable": 3,
        "n_skipped_unannotated": 0,
    }
    assert "human_vs_vader" in summary["comparisons"]
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Objective" in report_text
    assert "## Methodology" in report_text
    assert "## Evaluation Coverage" in report_text
    assert "## Global Results" in report_text
    assert "## Interpretation" in report_text
    assert "LLM-derived comparisons were available" in report_text
    assert "fully blank rows are skipped" in report_text
    assert "Context Tone" in report_text
    assert "Bright light" in report_text
    assert "Greater willingness to forgive" in report_text
    assert "context_tone" not in report_text
    assert (tmp_path / "evaluation_outputs" / "figures" / "alignment" / "comparison_summary.png").exists()

    _set_annotation_value(annotation_workbook, 2, "Context Tone", "invalid")

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



def test_evaluate_skips_fully_blank_human_rows(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)
    _clear_annotation_row(annotation_workbook, 2)

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 0, eval_result.stderr

    metrics = pd.read_csv(tmp_path / "evaluation_outputs" / "evaluation_metrics.csv")
    summary = json.loads((tmp_path / "evaluation_outputs" / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert set(metrics["n"]) == {2}
    assert summary["coverage"] == {
        "n_sampled_total": 3,
        "n_human_rows_total": 3,
        "n_human_evaluable": 2,
        "n_skipped_unannotated": 1,
    }



def test_evaluate_ignores_invalid_questionnaire_values_for_skipped_human_rows(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)
    _clear_annotation_row(annotation_workbook, 2)

    study = load_study_config(study_config)
    workbook = load_workbook(mapping_workbook)
    worksheet = workbook["sampled_private"]
    headers = _annotation_headers(worksheet)
    questionnaire_column = study.questionnaire["m8"]["columns"]["m8_out_of_body"]
    worksheet.cell(row=2, column=headers[questionnaire_column], value="INVALID")
    workbook.save(mapping_workbook)

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 0, eval_result.stderr

    metrics = pd.read_csv(tmp_path / "evaluation_outputs" / "evaluation_metrics.csv")
    assert set(metrics["n"]) == {2}
def test_evaluate_fails_for_partially_completed_human_row(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)
    _set_annotation_value(annotation_workbook, 2, "Context Tone", "")

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 1
    assert "partially completed rows" in eval_result.stderr



def test_evaluate_accepts_removed_human_row_and_tracks_coverage(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)
    _delete_annotation_row(annotation_workbook, 2)

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 0, eval_result.stderr

    metrics = pd.read_csv(tmp_path / "evaluation_outputs" / "evaluation_metrics.csv")
    summary = json.loads((tmp_path / "evaluation_outputs" / "evaluation_summary.json").read_text(encoding="utf-8"))
    report_text = (tmp_path / "evaluation_outputs" / "alignment_report.md").read_text(encoding="utf-8")
    assert set(metrics["n"]) == {2}
    assert summary["coverage"] == {
        "n_sampled_total": 3,
        "n_human_rows_total": 2,
        "n_human_evaluable": 2,
        "n_skipped_unannotated": 0,
    }
    assert "Sampled rows removed from the workbook before evaluation: 1" in report_text



def test_evaluate_ignores_extra_llm_predictions_outside_human_subset(tmp_path: Path) -> None:
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
    _delete_annotation_row(annotation_workbook, 2)
    _append_extra_llm_prediction(prediction_path, "ANN_9999_EXTRA", study_config)

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 0, eval_result.stderr

    metrics = pd.read_csv(tmp_path / "evaluation_outputs" / "evaluation_metrics.csv")
    assert set(metrics[metrics["comparison"] == "human_vs_llm"]["n"]) == {2}



def test_evaluate_fails_when_vader_missing_human_evaluable_participant(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    annotation_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"
    vader_dir = tmp_path / "evaluation_outputs" / "vader_sentiment"

    populate_human_annotation_workbook(annotation_workbook, mapping_workbook, study_config)

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

    vader_scores = pd.read_csv(vader_dir / "vader_sentiment_scores.csv")
    response_id_to_drop = pd.read_excel(mapping_workbook, sheet_name="sampled_private").loc[0, "response_id"]
    vader_scores = vader_scores[vader_scores["response_id"] != response_id_to_drop].copy()
    vader_scores.to_csv(vader_dir / "vader_sentiment_scores.csv", index=False)

    eval_result = run_cli(
        "evaluate",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--vader-scores",
        str(vader_dir / "vader_sentiment_scores.csv"),
    )
    assert eval_result.returncode == 1
    assert "Participant mismatch between human annotations and VADER predictions" in eval_result.stderr



