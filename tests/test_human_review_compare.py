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
Heightened awareness: Yes
Altered time perception: Yes
Encounter with a presence: No
Peace or wellbeing: Yes
Aftereffects Narrative: After one
Aftereffects Tone (4): Positive
Aftereffects Tone (3): Positive
Fear of death: Yes
Inner meaning in my life: Yes
Compassion toward others: Yes
Spiritual feelings: Yes
Desire to help others: Yes
Personal vulnerability: Yes
Interest in material goods: No
Interest in religion: Yes
Understanding myself: Yes
Social justice issues: Yes

response_id: 2
Context Narrative: Context two
Context Tone (4): Negative
Context Tone (3): Negative
Experience Narrative: Experience two
Experience Tone (4): Mixed
Experience Tone (3): Mixed
Out-of-body sensation: No
Bright light: Yes
Heightened awareness: No
Altered time perception: No
Encounter with a presence: Yes
Peace or wellbeing: No
Aftereffects Narrative: After two
Aftereffects Tone (4): Neutral
Aftereffects Tone (3): Mixed
Fear of death: No
Inner meaning in my life: Yes
Compassion toward others: Yes
Spiritual feelings: No
Desire to help others: Yes
Personal vulnerability: Yes
Interest in material goods: No
Interest in religion: No
Understanding myself: Yes
Social justice issues: Yes
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
                "outside_of_body_experience": "Extremely",
                "feeling_bright_light": "Not at all - absence",
                "feeling_awareness": "Medium",
                "presence_encounter": "Not at all - absence",
                "saw_relived_past_events": "Slightly",
                "time_perception_altered": "Extremely",
                "border_point_of_no_return": "Medium",
                "non_existence_feeling": "Slightly",
                "feeling_peace_wellbeing": "Extremely",
                "saw_entered_gateway": "Slightly",
                "fear_of_death": "Decreased",
                "inner_meaning_in_my_life": "Strongly increased",
                "compassion_toward_others": "Increased",
                "spiritual_feelings": "Increased",
                "desire_to_help_others": "Strongly increased",
                "personal_vulnerability": "Increased",
                "interest_in_material_goods": "Decreased",
                "interest_in_religion": "Increased",
                "understanding_myself": "Strongly increased",
                "social_justice_issues": "Increased",
            },
            {
                "response_id": 2,
                "valence": "Negative",
                "outside_of_body_experience": "Not at all - absence",
                "feeling_bright_light": "Extremely",
                "feeling_awareness": "Slightly",
                "presence_encounter": "Extremely",
                "saw_relived_past_events": "Not at all - absence",
                "time_perception_altered": "Not at all - absence",
                "border_point_of_no_return": "Slightly",
                "non_existence_feeling": "Not at all - absence",
                "feeling_peace_wellbeing": "Slightly",
                "saw_entered_gateway": "Not at all - absence",
                "fear_of_death": "Not changed",
                "inner_meaning_in_my_life": "Increased",
                "compassion_toward_others": "Increased",
                "spiritual_feelings": "Not changed",
                "desire_to_help_others": "Increased",
                "personal_vulnerability": "Increased",
                "interest_in_material_goods": "Not changed",
                "interest_in_religion": "Not changed",
                "understanding_myself": "Increased",
                "social_justice_issues": "Missing",
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
                "outside_of_body_experience": "yes",
                "feeling_bright_light": "no",
                "feeling_awareness": "yes",
                "presence_encounter": "no",
                "saw_relived_past_events": "no",
                "time_perception_altered": "yes",
                "border_point_of_no_return": "yes",
                "non_existence_feeling": "no",
                "feeling_peace_wellbeing": "yes",
                "saw_entered_gateway": "no",
            },
            "aftereffects": {
                "tone": tones[2],
                "fear_of_death": "yes",
                "inner_meaning_in_my_life": "yes",
                "compassion_toward_others": "yes",
                "spiritual_feelings": "yes",
                "desire_to_help_others": "yes",
                "personal_vulnerability": "yes",
                "interest_in_material_goods": "no",
                "interest_in_religion": "yes",
                "understanding_myself": "yes",
                "social_justice_issues": "yes",
            },
        },
        {
            "response_id": 2,
            "context": {"tone": "negative"},
            "experience": {
                "tone": "mixed",
                "outside_of_body_experience": "no",
                "feeling_bright_light": "yes",
                "feeling_awareness": "no",
                "presence_encounter": "yes",
                "saw_relived_past_events": "no",
                "time_perception_altered": "no",
                "border_point_of_no_return": "no",
                "non_existence_feeling": "no",
                "feeling_peace_wellbeing": "no",
                "saw_entered_gateway": "no",
            },
            "aftereffects": {
                "tone": "neutral",
                "fear_of_death": "no",
                "inner_meaning_in_my_life": "yes",
                "compassion_toward_others": "yes",
                "spiritual_feelings": "no",
                "desire_to_help_others": "yes",
                "personal_vulnerability": "yes",
                "interest_in_material_goods": "no",
                "interest_in_religion": "no",
                "understanding_myself": "yes",
                "social_justice_issues": "yes",
            },
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
