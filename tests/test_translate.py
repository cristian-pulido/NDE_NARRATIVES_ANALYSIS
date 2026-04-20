from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nde_narratives.config import (
    load_paths_config,
    load_study_config,
    load_translate_config,
)
from nde_narratives.llm.types import LLMProviderResponse
from nde_narratives.preprocessing_translate import run_translate_pipeline

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli


class FakeTranslateProvider:
    def generate_structured(self, request) -> LLMProviderResponse:
        section = str(request.section).replace("preprocess_translate_", "")
        payload = {"translation": f"translated {section}", "source_language": "es"}
        return LLMProviderResponse(
            provider="fake", model=request.model, raw_text=json.dumps(payload)
        )


def _translate_block() -> str:
    return """
[translate]
provider = "ollama"
base_url = "http://localhost:11434"
timeout_seconds = 30
max_attempts = 2
temperature = 0.0
model = "mock-translate-model"
prompt_version = "v1"
dynamic_context_enabled = true
num_ctx_min = 4096
num_ctx_max = 16384
chars_per_token = 4.0
"""


def test_run_translate_pipeline_writes_translated_outputs_and_detected_language(
    tmp_path: Path,
) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(
        tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_translate_block()
    )
    paths = load_paths_config(paths_config)
    translate = load_translate_config(paths_config)

    result = run_translate_pipeline(
        study=study,
        paths=paths,
        translate=translate,
        limit=1,
        provider_factory=lambda _: FakeTranslateProvider(),
    )

    assert Path(result["translated_dataset_csv"]).exists()
    assert Path(result["translated_dataset_xlsx"]).exists()
    assert Path(result["participant_results_file"]).exists()
    assert Path(result["raw_responses_file"]).exists()
    assert Path(result["errors_file"]).exists()

    translated = pd.read_csv(result["translated_dataset_csv"])
    assert len(translated) == 1
    assert translated.iloc[0]["detected_language_row"] == "es"
    assert (
        translated.iloc[0][study.sections["context"].source_column]
        == "translated context"
    )
    assert (
        translated.iloc[0][study.sections["experience"].source_column]
        == "translated experience"
    )
    assert (
        translated.iloc[0][study.sections["aftereffects"].source_column]
        == "translated aftereffects"
    )


def test_translate_command_executes_with_translate_block(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_translate_block())

    result = run_cli(
        "translate",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--limit",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["records"] == 1
    assert Path(payload["translated_dataset_csv"]).exists()


def test_translate_tolerates_literal_newlines_in_model_json_string(
    tmp_path: Path,
) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(
        tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_translate_block()
    )
    paths = load_paths_config(paths_config)
    translate = load_translate_config(paths_config)

    class NewlineLiteralProvider:
        def generate_structured(self, request) -> LLMProviderResponse:
            raw = '{\n  "translation": "line 1\nline 2",\n  "source_language": "nl"\n}'
            return LLMProviderResponse(
                provider="fake", model=request.model, raw_text=raw
            )

    result = run_translate_pipeline(
        study=study,
        paths=paths,
        translate=translate,
        limit=1,
        provider_factory=lambda _: NewlineLiteralProvider(),
    )

    translated = pd.read_csv(result["translated_dataset_csv"])
    value = str(translated.iloc[0][study.sections["context"].source_column])
    assert "line 1" in value
    assert "line 2" in value
