from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nde_narratives.config import load_paths_config, load_preprocessing_config, load_study_config
from nde_narratives.llm.types import LLMProviderResponse
from nde_narratives.preprocessing import run_preprocessing_pipeline

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli


class FakePreprocessingProvider:
    def __init__(self, invalid_participant: str | None = None) -> None:
        self.invalid_participant = invalid_participant
        self.validation_calls: list[str] = []
        self.resegment_calls: list[str] = []
        self.validation_num_ctx: list[int | None] = []
        self.resegment_num_ctx: list[int | None] = []
        self.validation_call_counts: dict[str, int] = {}

    def generate_structured(self, request) -> LLMProviderResponse:
        if request.section == "preprocess_validate":
            self.validation_calls.append(request.participant_code)
            self.validation_num_ctx.append(request.metadata.get("num_ctx"))
            count = self.validation_call_counts.get(str(request.participant_code), 0) + 1
            self.validation_call_counts[str(request.participant_code)] = count
            if request.participant_code == self.invalid_participant and count == 1:
                payload = {
                    "context_assessment": "invalid",
                    "experience_assessment": "valid",
                    "aftereffects_assessment": "invalid",
                    "needs_resegmentation": "yes",
                }
            else:
                payload = {
                    "context_assessment": "valid",
                    "experience_assessment": "valid",
                    "aftereffects_assessment": "valid",
                    "needs_resegmentation": "no",
                }
            return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))

        if request.section == "preprocess_validate_post_resegment":
            payload = {
                "context_assessment": "valid",
                "experience_assessment": "valid",
                "aftereffects_assessment": "valid",
                "needs_resegmentation": "no",
            }
            return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))

        self.resegment_calls.append(request.participant_code)
        self.resegment_num_ctx.append(request.metadata.get("num_ctx"))
        payload = {
            "context": "clean context",
            "experience": "clean experience",
            "aftereffects": "clean aftereffects",
        }
        return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))


def _preprocessing_block() -> str:
    return '''
[preprocessing]
provider = "ollama"
base_url = "http://localhost:11434"
timeout_seconds = 30
max_attempts = 2
temperature = 0.0
model = "mock-preprocess-model"
prompt_version = "v1"
dynamic_context_enabled = true
num_ctx_min = 4096
num_ctx_max = 16384
chars_per_token = 4.0
'''


def test_preprocess_help_mentions_cleaned_dataset_and_validation_sample() -> None:
    result = run_cli("preprocess", "--help")

    assert result.returncode == 0, result.stderr
    assert "cleaned dataset" in result.stdout.lower()
    assert "validation sample" in result.stdout.lower()
    assert "nde preprocess" in result.stdout


def test_run_preprocessing_pipeline_writes_cleaned_outputs_and_validation_sample(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    source_preview = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=3,
        provider_factory=lambda _: FakePreprocessingProvider(),
        generate_validation_sample=True,
        validation_n_total=2,
        force_validation_sample=True,
    )

    assert source_preview["summary"]["records"] == 3
    assert Path(source_preview["cleaned_dataset_csv"]).exists()
    assert Path(source_preview["cleaned_dataset_xlsx"]).exists()
    assert Path(source_preview["participant_results_file"]).exists()
    assert Path(source_preview["validation_sample_workbook"]).exists()
    assert Path(source_preview["validation_mapping_workbook"]).exists()
    cleaned_df = pd.read_csv(source_preview["cleaned_dataset_csv"])
    assert "n_valid_sections_original" in cleaned_df.columns
    assert "n_valid_sections_cleaned" in cleaned_df.columns
    assert study.stratify_column in cleaned_df.columns
    quality_column = str(study.dataset.get("quality_label_column"))
    if quality_column:
        assert quality_column in cleaned_df.columns
    assert "n_rows_with_3_valid_sections_original" in source_preview["summary"]
    assert "n_rows_with_3_valid_sections_cleaned" in source_preview["summary"]


def test_run_preprocessing_pipeline_resegments_invalid_rows(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    preview_provider = FakePreprocessingProvider()
    preview = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        provider_factory=lambda _: preview_provider,
    )
    cleaned = Path(preview["cleaned_dataset_csv"]).read_text(encoding="utf-8")
    participant_code = cleaned.splitlines()[1].split(",")[1]

    provider = FakePreprocessingProvider(invalid_participant=participant_code)
    result = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        output_dir=tmp_path / "alt_preprocessing",
        provider_factory=lambda _: provider,
    )

    records = [json.loads(line) for line in Path(result["participant_results_file"]).read_text(encoding="utf-8-sig").splitlines() if line]
    assert records[0]["preprocessing_status"] == "fully_resegmented"
    assert records[0]["context_clean"] == "clean context"
    assert records[0]["n_valid_sections_original"] == 1
    assert records[0]["n_valid_sections_cleaned"] == 3
    assert provider.resegment_calls == [participant_code]


def test_preprocessing_dynamic_num_ctx_is_attached_to_requests_and_audited(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    raw = pd.read_csv(FIXTURES / "survey_fixture.csv")
    target_id = int(raw.iloc[0][study.id_column])
    repeated = " ".join(["narrative"] * 800)
    raw.loc[raw[study.id_column] == target_id, study.sections["context"].source_column] = repeated
    raw.loc[raw[study.id_column] == target_id, study.sections["experience"].source_column] = repeated
    raw.loc[raw[study.id_column] == target_id, study.sections["aftereffects"].source_column] = repeated
    custom_survey = tmp_path / "survey_long_prompt.csv"
    raw.to_csv(custom_survey, index=False)

    paths_config = make_paths_config(tmp_path, custom_survey, llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    provider = FakePreprocessingProvider(invalid_participant=None)
    result = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        provider_factory=lambda _: provider,
    )

    assert provider.validation_num_ctx
    assert provider.validation_num_ctx[0] is not None
    assert preprocessing.num_ctx_min <= int(provider.validation_num_ctx[0]) <= preprocessing.num_ctx_max

    raw_responses = [
        json.loads(line)
        for line in Path(result["raw_responses_file"]).read_text(encoding="utf-8-sig").splitlines()
        if line
    ]
    assert raw_responses
    assert "request_metadata" in raw_responses[0]
    assert "num_ctx" in raw_responses[0]["request_metadata"]


def test_run_preprocessing_pipeline_from_scratch_discards_previous_ledger_state(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    first = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        provider_factory=lambda _: FakePreprocessingProvider(),
    )
    first_records = [json.loads(line) for line in Path(first["participant_results_file"]).read_text(encoding="utf-8-sig").splitlines() if line]
    first_code = first_records[0]["participant_code"]

    second = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        from_scratch=True,
        output_dir=paths.preprocessing_output_dir,
        provider_factory=lambda _: FakePreprocessingProvider(invalid_participant=first_code),
    )
    second_records = [json.loads(line) for line in Path(second["participant_results_file"]).read_text(encoding="utf-8-sig").splitlines() if line]

    assert len(second_records) == 1
    assert second_records[0]["preprocessing_status"] == "fully_resegmented"


def test_preprocess_source_frame_keeps_rows_with_partial_narratives_for_rescue(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    raw = pd.read_csv(FIXTURES / "survey_fixture.csv")
    target_id = int(raw.iloc[0][study.id_column])
    raw.loc[raw[study.id_column] == target_id, study.sections["experience"].source_column] = ""
    raw.loc[raw[study.id_column] == target_id, study.sections["aftereffects"].source_column] = ""
    custom_survey = tmp_path / "survey_partial.csv"
    raw.to_csv(custom_survey, index=False)

    paths_config = make_paths_config(tmp_path, custom_survey, llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    result = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        provider_factory=lambda _: FakePreprocessingProvider(),
    )

    cleaned_df = pd.read_csv(result["cleaned_dataset_csv"])
    assert target_id in set(cleaned_df[study.id_column])


def test_preprocessing_treats_nan_section_values_as_empty_strings(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    raw = pd.read_csv(FIXTURES / "survey_fixture.csv")
    target_id = int(raw.iloc[0][study.id_column])
    raw.loc[raw[study.id_column] == target_id, study.sections["experience"].source_column] = pd.NA
    raw.loc[raw[study.id_column] == target_id, study.sections["aftereffects"].source_column] = pd.NA
    custom_survey = tmp_path / "survey_with_nan_sections.csv"
    raw.to_csv(custom_survey, index=False)

    paths_config = make_paths_config(tmp_path, custom_survey, llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    result = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        provider_factory=lambda _: FakePreprocessingProvider(),
    )

    cleaned_df = pd.read_csv(result["cleaned_dataset_csv"])
    row = cleaned_df.iloc[0]

    assert pd.isna(row[study.sections["experience"].source_column])
    assert pd.isna(row[study.sections["aftereffects"].source_column])
    assert row["n_valid_sections_cleaned"] == 1
    assert pd.isna(row["original_experience"])
    assert pd.isna(row["original_aftereffects"])


def test_preprocess_prefers_translated_dataset_when_present_and_no_input_path(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    translated_path = paths.preprocessing_output_dir / "translated_dataset.csv"
    translated_df = pd.read_csv(FIXTURES / "survey_fixture.csv")
    translated_df.loc[translated_df.index[0], study.id_column] = 999
    translated_df.to_csv(translated_path, index=False)

    result = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        provider_factory=lambda _: FakePreprocessingProvider(),
    )

    assert result["summary"]["input_path"].endswith("translated_dataset.csv")
    cleaned_df = pd.read_csv(result["cleaned_dataset_csv"])
    assert 999 in set(cleaned_df[study.id_column])


def test_resegmentation_fails_when_post_validation_still_invalid(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    class AlwaysInvalidPostValidationProvider(FakePreprocessingProvider):
        def generate_structured(self, request):
            if request.section == "preprocess_validate":
                payload = {
                    "context_assessment": "invalid",
                    "experience_assessment": "valid",
                    "aftereffects_assessment": "invalid",
                    "needs_resegmentation": "yes",
                }
                return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))
            if request.section == "preprocess_validate_post_resegment":
                payload = {
                    "context_assessment": "invalid",
                    "experience_assessment": "valid",
                    "aftereffects_assessment": "invalid",
                    "needs_resegmentation": "yes",
                }
                return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))
            return super().generate_structured(request)

    result = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        provider_factory=lambda _: AlwaysInvalidPostValidationProvider(),
    )

    records = [json.loads(line) for line in Path(result["participant_results_file"]).read_text(encoding="utf-8-sig").splitlines() if line]
    assert records[0]["preprocessing_status"] == "post_validation_failed"
    assert records[0]["status"] == "failed"
    cleaned_df = pd.read_csv(result["cleaned_dataset_csv"])
    assert "post_validation_assessments" in cleaned_df.columns
    assert "post_validation_needs_resegmentation" in cleaned_df.columns


def test_resegmentation_allows_warning_when_two_sections_valid_after_post_validation(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    paths_config = make_paths_config(tmp_path, FIXTURES / "survey_fixture.csv", llm_block=_preprocessing_block())
    paths = load_paths_config(paths_config)
    preprocessing = load_preprocessing_config(paths_config)

    class WarningPostValidationProvider(FakePreprocessingProvider):
        def generate_structured(self, request):
            if request.section == "preprocess_validate":
                payload = {
                    "context_assessment": "invalid",
                    "experience_assessment": "valid",
                    "aftereffects_assessment": "invalid",
                    "needs_resegmentation": "yes",
                }
                return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))
            if request.section == "preprocess_validate_post_resegment":
                payload = {
                    "context_assessment": "valid",
                    "experience_assessment": "invalid",
                    "aftereffects_assessment": "valid",
                    "needs_resegmentation": "yes",
                }
                return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))
            return super().generate_structured(request)

    result = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        limit=1,
        provider_factory=lambda _: WarningPostValidationProvider(),
    )

    records = [json.loads(line) for line in Path(result["participant_results_file"]).read_text(encoding="utf-8-sig").splitlines() if line]
    assert records[0]["status"] == "success"
    assert records[0]["preprocessing_status"] == "post_validation_warning"
    assert records[0]["n_valid_sections_cleaned"] == 2
