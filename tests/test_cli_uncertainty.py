from __future__ import annotations

import pandas as pd

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli


def _write_minimal_evaluation_metrics(path) -> None:
    rows = [
        {
            "comparison": "questionnaire_vs_llm:exp_alpha__run-01",
            "field": "experience_tone",
            "n": 20,
            "accuracy": 0.60,
            "cohen_kappa": 0.22,
            "macro_f1": 0.48,
        },
        {
            "comparison": "questionnaire_vs_llm:exp_alpha__run-01",
            "field": "outside_of_body_experience",
            "n": 20,
            "accuracy": 0.70,
            "cohen_kappa": 0.34,
            "macro_f1": 0.62,
        },
        {
            "comparison": "questionnaire_vs_llm:exp_alpha__run-01",
            "field": "fear_of_death",
            "n": 20,
            "accuracy": 0.58,
            "cohen_kappa": 0.14,
            "macro_f1": 0.46,
        },
        {
            "comparison": "questionnaire_vs_llm:exp_beta__run-01",
            "field": "experience_tone",
            "n": 20,
            "accuracy": 0.68,
            "cohen_kappa": 0.31,
            "macro_f1": 0.54,
        },
        {
            "comparison": "questionnaire_vs_llm:exp_beta__run-01",
            "field": "outside_of_body_experience",
            "n": 20,
            "accuracy": 0.74,
            "cohen_kappa": 0.41,
            "macro_f1": 0.68,
        },
        {
            "comparison": "questionnaire_vs_llm:exp_beta__run-01",
            "field": "fear_of_death",
            "n": 20,
            "accuracy": 0.63,
            "cohen_kappa": 0.21,
            "macro_f1": 0.52,
        },
        {
            "comparison": "human_reference_vs_llm:exp_alpha__run-01",
            "field": "experience_tone",
            "n": 12,
            "accuracy": 0.32,
            "cohen_kappa": 0.05,
            "macro_f1": 0.21,
        },
        {
            "comparison": "human_reference_vs_llm:exp_alpha__run-01",
            "field": "outside_of_body_experience",
            "n": 12,
            "accuracy": 0.44,
            "cohen_kappa": 0.10,
            "macro_f1": 0.29,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_evaluate_uncertainty_uses_default_paths_and_writes_outputs(tmp_path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    evaluation_dir = tmp_path / "evaluation_outputs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_evaluation_metrics(evaluation_dir / "evaluation_metrics.csv")

    result = run_cli(
        "evaluate-uncertainty",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--bootstrap-samples",
        "400",
        "--random-seed",
        "7",
    )
    assert result.returncode == 0, result.stderr

    uncertainty_dir = evaluation_dir / "uncertainty"
    assert (uncertainty_dir / "uncertainty_report.md").exists()
    assert (uncertainty_dir / "uncertainty_scope.csv").exists()
    assert (uncertainty_dir / "uncertainty_comparison.csv").exists()
    assert (uncertainty_dir / "uncertainty_scope_family.csv").exists()
    assert (uncertainty_dir / "uncertainty_comparison_family.csv").exists()
    assert (uncertainty_dir / "figures" / "uncertainty_scope_ci.png").exists()
    assert (uncertainty_dir / "figures" / "questionnaire_llm_family_ci.png").exists()

    scope_df = pd.read_csv(uncertainty_dir / "uncertainty_scope.csv")
    assert "questionnaire_vs_llm" in set(scope_df["scope"])
    assert {
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "cohen_kappa_ci_low",
        "cohen_kappa_ci_high",
    }.issubset(scope_df.columns)


def test_evaluate_uncertainty_fails_when_input_directory_is_missing(tmp_path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    missing_dir = tmp_path / "does_not_exist"
    result = run_cli(
        "evaluate-uncertainty",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--input-dir",
        str(missing_dir),
    )

    assert result.returncode == 1
    assert "Evaluation input directory not found" in result.stderr
