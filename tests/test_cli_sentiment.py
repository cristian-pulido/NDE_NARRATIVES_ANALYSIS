from __future__ import annotations

from pathlib import Path
import importlib.util

import pandas as pd
import pytest

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli

_HAS_VADER = importlib.util.find_spec("vaderSentiment") is not None


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


def test_sentiment_sensitivity_prefers_cleaned_dataset_when_available(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)
    output_dir = tmp_path / "sentiment_outputs_cleaned"

    cleaned_dir = tmp_path / "preprocessing_outputs"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned = pd.DataFrame(
        [
            {
                "response_id": 101,
                "participant_code": "ANN_CLEANED",
                "nde_context": "UNIQUE CLEANED CONTEXT TEXT",
                "nde_description": "UNIQUE CLEANED EXPERIENCE TEXT",
                "nde_aftereffects": "UNIQUE CLEANED AFTEREFFECTS TEXT",
                "n_valid_sections": 3,
                "n_valid_sections_cleaned": 3,
            }
        ]
    )
    cleaned.to_csv(cleaned_dir / "cleaned_dataset.csv", index=False)

    result = run_cli(
        "sentiment-sensitivity",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--output-dir",
        str(output_dir),
        "--include-text",
    )

    assert result.returncode == 0, result.stderr
    scores = pd.read_csv(output_dir / "vader_sentiment_scores.csv")
    assert len(scores) == 3
    assert "UNIQUE CLEANED CONTEXT TEXT" in set(scores["text"])
    assert "UNIQUE CLEANED EXPERIENCE TEXT" in set(scores["text"])
    assert "UNIQUE CLEANED AFTEREFFECTS TEXT" in set(scores["text"])


@pytest.mark.skipif(not _HAS_VADER, reason="vaderSentiment is not installed")
def test_sentiment_sensitivity_prefers_cleaned_dataset_over_translated(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)
    output_dir = tmp_path / "sentiment_outputs_translated"

    pre_dir = tmp_path / "preprocessing_outputs"
    pre_dir.mkdir(parents=True, exist_ok=True)

    cleaned = pd.DataFrame(
        [
            {
                "response_id": 1001,
                "participant_code": "ANN_CLEANED",
                "nde_context": "ONLY CLEANED CONTEXT",
                "nde_description": "ONLY CLEANED EXPERIENCE",
                "nde_aftereffects": "ONLY CLEANED AFTEREFFECTS",
                "n_valid_sections": 3,
                "n_valid_sections_cleaned": 3,
            }
        ]
    )
    cleaned.to_csv(pre_dir / "cleaned_dataset.csv", index=False)

    translated = cleaned.copy()
    translated.loc[:, "response_id"] = 1002
    translated.loc[:, "nde_context"] = "ONLY TRANSLATED CONTEXT"
    translated.loc[:, "nde_description"] = "ONLY TRANSLATED EXPERIENCE"
    translated.loc[:, "nde_aftereffects"] = "ONLY TRANSLATED AFTEREFFECTS"
    translated.to_csv(pre_dir / "translated_dataset.csv", index=False)

    result = run_cli(
        "sentiment-sensitivity",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--output-dir",
        str(output_dir),
        "--include-text",
    )

    assert result.returncode == 0, result.stderr
    scores = pd.read_csv(output_dir / "vader_sentiment_scores.csv")
    assert "ONLY CLEANED CONTEXT" in set(scores["text"])
    assert "ONLY CLEANED EXPERIENCE" in set(scores["text"])
    assert "ONLY CLEANED AFTEREFFECTS" in set(scores["text"])
