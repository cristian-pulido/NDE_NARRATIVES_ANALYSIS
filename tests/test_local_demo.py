from __future__ import annotations

from nde_narratives.local_demo import _alignment_markdown, _labels_table_rows


def _sample_predictions() -> dict[str, dict[str, dict[str, object]]]:
    return {
        "context": {
            "context": {
                "tone": "neutral",
                "death_context_nature": "objective_medical_context",
                "evidence_segments": ["clinical context evidence"],
            }
        },
        "experience": {
            "experience": {
                "tone": "positive",
                "evidence_segments": ["bright light and peace"],
                "m8_bright_light": "yes",
                "m8_peace": "yes",
            }
        },
        "aftereffects": {
            "aftereffects": {
                "tone": "positive",
                "evidence_segments": ["less fear of death"],
                "m9_help_others": "yes",
            }
        },
    }


def test_labels_table_rows_are_present_independent_of_valence() -> None:
    rows = _labels_table_rows(_sample_predictions())
    assert rows
    assert any(row[1] == "Tone" for row in rows)
    assert any(row[1] == "Supporting Evidence" for row in rows)
    assert any(row[1] == "Bright light perception" for row in rows)


def test_alignment_message_skips_when_no_valence() -> None:
    message = _alignment_markdown(_sample_predictions(), "")
    assert "no alignment check" in message.lower()
