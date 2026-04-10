from __future__ import annotations

import json
from pathlib import Path

from nde_narratives.config import load_llm_config, load_paths_config, load_study_config
from nde_narratives.interactive import (
    analyze_single_narrative,
    analyze_three_sections,
    parse_ollama_tags_payload,
)
from nde_narratives.llm.types import LLMProviderResponse

from tests.cli_helpers import FIXTURES, make_paths_config


class SequencedProvider:
    def __init__(self) -> None:
        self.sections: list[str] = []

    def generate_structured(self, request) -> LLMProviderResponse:
        self.sections.append(str(request.section))
        if request.section == "interactive_preprocess_resegment":
            payload = {
                "context": "I had surgery complications.",
                "experience": "I saw bright light and felt peace.",
                "aftereffects": "I no longer fear death.",
            }
            return LLMProviderResponse(
                provider="fake", model=request.model, raw_text=json.dumps(payload)
            )

        if request.section == "context":
            payload = {
                "context": {
                    "tone": "neutral",
                    "death_context_nature": "objective_medical_context",
                    "evidence_segments": ["I had surgery complications."],
                }
            }
            return LLMProviderResponse(
                provider="fake", model=request.model, raw_text=json.dumps(payload)
            )

        if request.section == "experience":
            payload = {
                "experience": {
                    "tone": "positive",
                    "evidence_segments": ["I saw bright light and felt peace."],
                    "m8_out_of_body": "no",
                    "m8_bright_light": "yes",
                    "m8_peace": "yes",
                    "m8_time_distortion": "no",
                    "m8_presence": "no",
                }
            }
            return LLMProviderResponse(
                provider="fake", model=request.model, raw_text=json.dumps(payload)
            )

        payload = {
            "aftereffects": {
                "tone": "positive",
                "evidence_segments": ["I no longer fear death."],
                "m9_moral_rules": "yes",
                "m9_long_term_thinking": "yes",
                "m9_consider_others": "yes",
                "m9_help_others": "yes",
                "m9_forgiveness": "yes",
            }
        }
        return LLMProviderResponse(
            provider="fake", model=request.model, raw_text=json.dumps(payload)
        )


def _llm_block() -> str:
    return """
[llm]
provider = "ollama"
base_url = "http://localhost:11434"
timeout_seconds = 30
max_attempts = 2
temperature = 0.0
source = "survey"
all_records = false

[preprocessing]
provider = "ollama"
base_url = "http://localhost:11434"
timeout_seconds = 30
max_attempts = 2
temperature = 0.0
model = "mock-model"

[[llm.experiments]]
experiment_id = "exp-alpha"
enabled = true
model = "mock-model"
run_id = "run-01"
temperature = 0.0
"""


def test_parse_ollama_tags_payload_handles_common_shapes() -> None:
    payload = {
        "models": [
            {"name": "qwen3.5:9b"},
            {"name": "llama3.1:8b"},
            {"model": "qwen3.5:9b"},
            {"size": 123},
        ]
    }
    models = parse_ollama_tags_payload(payload)
    assert models == ["llama3.1:8b", "qwen3.5:9b"]


def test_analyze_three_sections_returns_expected_structure(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(
        tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_llm_block()
    )
    paths = load_paths_config(paths_config)
    llm_config = load_llm_config(paths_config)

    provider = SequencedProvider()
    result = analyze_three_sections(
        study=study,
        paths=paths,
        llm_config=llm_config,
        model="mock-model",
        context_text="I had surgery complications.",
        experience_text="I saw bright light and felt peace.",
        aftereffects_text="I no longer fear death.",
        provider_factory=lambda _runtime: provider,
    )

    assert result["mode"] == "three_sections"
    assert set(result["predictions"].keys()) == {
        "context",
        "experience",
        "aftereffects",
    }
    assert provider.sections == ["context", "experience", "aftereffects"]


def test_analyze_single_narrative_runs_resegmentation_then_sections(
    tmp_path: Path,
) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(
        tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_llm_block()
    )
    paths = load_paths_config(paths_config)
    llm_config = load_llm_config(paths_config)

    provider = SequencedProvider()
    result = analyze_single_narrative(
        study=study,
        paths=paths,
        llm_config=llm_config,
        model="mock-model",
        single_narrative_text="Surgery complication, bright light, and then lasting life changes.",
        provider_factory=lambda _runtime: provider,
    )

    assert result["mode"] == "single_narrative"
    assert provider.sections[0] == "interactive_preprocess_resegment"
    assert provider.sections[1:] == ["context", "experience", "aftereffects"]
    assert (
        result["resegmentation"]["parsed"]["experience"]
        == "I saw bright light and felt peace."
    )
