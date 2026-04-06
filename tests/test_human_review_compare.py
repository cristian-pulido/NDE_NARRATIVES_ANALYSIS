from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nde_narratives.human_review_compare import parse_human_md

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli


def _write_human_md(path: Path) -> None:
    path.write_text(
        """
response_id: 1
Context Narrative: Context one
Context Tone (4): Neutral
Context Tone (3): Negative
Experience Narrative: Experience one
Experience Tone (4): Positive
Experience Tone (3): Mixed
Out-of-body sensation: Yes
Bright light: No
Altered time perception: Yes
Encounter with a presence: No
Aftereffects Narrative: After one
Aftereffects Tone (4): Positive
Aftereffects Tone (3): Positive
Stronger moral principles: Yes

response_id: 2
Context Narrative: Context two
Context Tone (4): Negative
Context Tone (3): Negative
Experience Narrative: Experience two
Experience Tone (4): Mixed
Experience Tone (3): Mixed
Out-of-body sensation: No
Bright light: Yes
Altered time perception: No
Encounter with a presence: Yes
Aftereffects Narrative: After two
Aftereffects Tone (4): Neutral
Aftereffects Tone (3): Mixed
Stronger moral principles: No
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_cleaned_dataset(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "response_id": 1,
                "nde_context": "Context one",
                "nde_description": "Experience one",
                "nde_aftereffects": "After one",
                "n_valid_sections_cleaned": 3,
            },
            {
                "response_id": 2,
                "nde_context": "Context two",
                "nde_description": "Experience two",
                "nde_aftereffects": "After two",
                "n_valid_sections_cleaned": 2,
            },
        ]
    ).to_csv(path, index=False)


def _write_questionnaire(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "response_id": 1,
                "valence": "Positive",
                "q_m8_out_of_body": "Extremely",
                "q_m8_bright_light": "Not at all",
                "q_m8_time_distortion": "Extremely",
                "q_m8_presence": "Not at all",
                "q_m9_moral_rules": "Increased",
            },
            {
                "response_id": 2,
                "valence": "Negative",
                "q_m8_out_of_body": "Not at all",
                "q_m8_bright_light": "Extremely",
                "q_m8_time_distortion": "Not at all",
                "q_m8_presence": "Extremely",
                "q_m9_moral_rules": "Not changed",
            },
        ]
    ).to_csv(path, index=False)


def _write_llm_artifact(
    root: Path, name: str, run_id: str, tones: tuple[str, str, str]
) -> None:
    artifact_dir = root / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": name,
                "model_variant": name,
                "run_id": run_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        {
            "response_id": 1,
            "context": {"tone": tones[0]},
            "experience": {
                "tone": tones[1],
                "m8_out_of_body": "yes",
                "m8_bright_light": "no",
                "m8_time_distortion": "yes",
                "m8_presence": "no",
            },
            "aftereffects": {"tone": tones[2], "m9_moral_rules": "yes"},
        },
        {
            "response_id": 2,
            "context": {"tone": "negative"},
            "experience": {
                "tone": "mixed",
                "m8_out_of_body": "no",
                "m8_bright_light": "yes",
                "m8_time_distortion": "no",
                "m8_presence": "yes",
            },
            "aftereffects": {"tone": "neutral", "m9_moral_rules": "no"},
        },
    ]
    with (artifact_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_parse_human_md_normalizes_fields(tmp_path: Path) -> None:
    human_md = tmp_path / "Human.md"
    _write_human_md(human_md)
    df = parse_human_md(human_md)
    assert len(df) == 2
    assert set(df["context_tone_4"].tolist()) == {"neutral", "negative"}
    assert set(df["out_of_body"].tolist()) == {"yes", "no"}


def test_compare_human_review_cli_outputs_and_excludes_ra1(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    human_md = tmp_path / "Human.md"
    cleaned_csv = tmp_path / "cleaned_dataset.csv"
    questionnaire_csv = tmp_path / "NDE_traslated.csv"
    llm_dir = tmp_path / "llm_outputs"
    out_dir = tmp_path / "human_compare"

    _write_human_md(human_md)
    _write_cleaned_dataset(cleaned_csv)
    _write_questionnaire(questionnaire_csv)
    _write_llm_artifact(
        llm_dir, "model_a__01", "01", ("neutral", "positive", "positive")
    )
    _write_llm_artifact(
        llm_dir, "model_a__RA1", "RA1", ("negative", "negative", "negative")
    )

    result = run_cli(
        "compare-human-review",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--human-md",
        str(human_md),
        "--cleaned-dataset",
        str(cleaned_csv),
        "--questionnaire-csv",
        str(questionnaire_csv),
        "--llm-results-dir",
        str(llm_dir),
        "--output-dir",
        str(out_dir),
        "--export-figures-pdf",
    )
    assert result.returncode == 0, result.stderr

    metrics = pd.read_csv(out_dir / "human_review_metrics.csv")
    assert "model_a__01" in set(metrics["model"].astype(str))
    assert "model_a__RA1" not in set(metrics["model"].astype(str))
    questionnaire_tone_rows = metrics[
        (metrics["comparison"] == "human_vs_questionnaire")
        & (metrics["family"] == "tone")
    ]
    assert len(questionnaire_tone_rows) == 1
    assert questionnaire_tone_rows.iloc[0]["field"] == "experience_tone"

    assert (out_dir / "human_review_alignment_report.md").exists()
    assert (out_dir / "human_review_summary.json").exists()
    report_text = (out_dir / "human_review_alignment_report.md").read_text(
        encoding="utf-8"
    )
    assert (
        "![Human-family alignment combined](figures/human_family_alignment_combined.png)"
        in report_text
    )
    assert (out_dir / "human_review_unit_alignment_long.csv").exists()
    assert (out_dir / "human_review_unit_alignment_metrics.csv").exists()
    assert (out_dir / "figures" / "human_family_alignment_combined.png").exists()
