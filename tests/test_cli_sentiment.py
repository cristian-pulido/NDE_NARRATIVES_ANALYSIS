from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli


def test_sentiment_sensitivity_preserves_schema_with_zero_rows(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)
    output_dir = tmp_path / "sentiment_outputs_zero"

    result = run_cli(
        "sentiment-sensitivity",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--output-dir",
        str(output_dir),
        "--limit",
        "0",
    )

    assert result.returncode == 0, result.stderr
    scores_path = output_dir / "vader_sentiment_scores.csv"
    scores = pd.read_csv(scores_path)
    assert len(scores) == 0
    assert list(scores.columns) == [
        "response_id",
        "participant_code",
        "section",
        "source_column",
        "neg",
        "neu",
        "pos",
        "compound",
        "vader_label",
    ]


def test_sentiment_sensitivity_generates_scores_figures_and_report(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)
    output_dir = tmp_path / "sentiment_outputs"

    result = run_cli(
        "sentiment-sensitivity",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    scores_path = output_dir / "vader_sentiment_scores.csv"
    report_path = output_dir / "vader_report.md"
    assert scores_path.exists()
    assert report_path.exists()
    for section in ("context", "experience", "aftereffects"):
        assert (output_dir / "figures" / f"{section}_distribution.png").exists()

    scores = pd.read_csv(scores_path)
    assert len(scores) == 9
    assert set(scores["section"]) == {"context", "experience", "aftereffects"}
    assert set(scores["vader_label"]).issubset({"positive", "negative", "mixed"})
    assert "text" not in scores.columns

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Methodology" in report_text
    assert "## Results" in report_text

    all_records_result = run_cli(
        "sentiment-sensitivity",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--output-dir",
        str(tmp_path / "sentiment_outputs_all"),
        "--all-records",
    )
    assert all_records_result.returncode == 0, all_records_result.stderr
    all_scores = pd.read_csv(tmp_path / "sentiment_outputs_all" / "vader_sentiment_scores.csv")
    assert len(all_scores) == 12

    include_text_result = run_cli(
        "sentiment-sensitivity",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--output-dir",
        str(tmp_path / "sentiment_outputs_text"),
        "--include-text",
    )
    assert include_text_result.returncode == 0, include_text_result.stderr
    include_text_scores = pd.read_csv(tmp_path / "sentiment_outputs_text" / "vader_sentiment_scores.csv")
    assert "text" in include_text_scores.columns
