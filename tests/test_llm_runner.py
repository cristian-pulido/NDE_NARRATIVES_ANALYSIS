from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from nde_narratives.config import load_llm_config, load_paths_config, load_study_config
from nde_narratives.llm.ollama import OllamaProvider
from nde_narratives.llm.types import LLMProviderResponse
from nde_narratives.llm_runner import run_llm_experiments
from nde_narratives.preprocessing import run_preprocessing_pipeline
from nde_narratives.prompting import load_batch_source
from nde_narratives.sampling import assign_participant_codes, create_annotation_frames

from tests.cli_helpers import (
    FIXTURES,
    copy_completed_annotation_to_human_dir,
    make_paths_config,
    populate_human_annotation_workbook,
    run_cli,
    write_human_manifest,
)


class FakeProvider:
    def __init__(self, predictions_by_key: dict[tuple[str, str], dict[str, Any]], invalid_once: set[tuple[str, str]] | None = None) -> None:
        self.predictions_by_key = predictions_by_key
        self.invalid_once = invalid_once or set()
        self.calls: dict[tuple[str, str], int] = defaultdict(int)
        self.temperatures: list[float | None] = []

    def generate_structured(self, request) -> LLMProviderResponse:
        key = (request.participant_code, request.section)
        self.calls[key] += 1
        self.temperatures.append(request.temperature)
        if key in self.invalid_once and self.calls[key] == 1:
            return LLMProviderResponse(provider="fake", model=request.model, raw_text="not valid json")
        return LLMProviderResponse(
            provider="fake",
            model=request.model,
            raw_text=json.dumps(self.predictions_by_key[key]),
        )


class InterruptingProvider(FakeProvider):
    def __init__(self, predictions_by_key: dict[tuple[str, str], dict[str, Any]], *, interrupt_on_call: int) -> None:
        super().__init__(predictions_by_key)
        self.interrupt_on_call = interrupt_on_call
        self.total_calls = 0

    def generate_structured(self, request) -> LLMProviderResponse:
        self.total_calls += 1
        if self.total_calls == self.interrupt_on_call:
            raise KeyboardInterrupt("simulated interruption")
        return super().generate_structured(request)


class FakePreprocessingProvider:
    def __init__(self, invalid_participant: str | None = None) -> None:
        self.invalid_participant = invalid_participant

    def generate_structured(self, request) -> LLMProviderResponse:
        if request.section == "preprocess_validate":
            payload = {
                "context_assessment": "invalid" if request.participant_code == self.invalid_participant else "valid",
                "experience_assessment": "valid",
                "aftereffects_assessment": "invalid" if request.participant_code == self.invalid_participant else "valid",
                "needs_resegmentation": "yes" if request.participant_code == self.invalid_participant else "no",
            }
            return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))
        payload = (
            {
                "context": "clean context",
                "experience": "clean experience",
                "aftereffects": "",
            }
            if request.participant_code == self.invalid_participant
            else {
                "context": "clean context",
                "experience": "clean experience",
                "aftereffects": "clean aftereffects",
            }
        )
        return LLMProviderResponse(provider="fake", model=request.model, raw_text=json.dumps(payload))


def _llm_block(*, temperature: float = 0.2) -> str:
    return f'''
[llm]
provider = "ollama"
base_url = "http://localhost:11434"
timeout_seconds = 30
max_attempts = 2
temperature = {temperature}
source = "survey"
all_records = false

[[llm.experiments]]
experiment_id = "exp-alpha"
enabled = true
model = "mock-model"
prompt_variant = "baseline"
run_id = "run-01"
model_variant = "mock-model"
temperature = 0.0
'''


def _prediction_payloads(study, source_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    fixture = pd.read_csv(FIXTURES / "llm_predictions_fixture.csv").set_index("response_id")
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in source_df.iterrows():
        participant_code = row["participant_code"]
        prediction_row = fixture.loc[row[study.id_column]]
        for section_name in study.section_order:
            section = study.sections[section_name]
            section_payload: dict[str, Any] = {
                "tone": str(prediction_row[section.tone_internal_column]),
                "evidence_segments": [f"fixture evidence for {section_name}"],
            }
            if section_name == "context":
                section_payload["death_context_nature"] = "subjective_threat_only"
            for column in section.binary_labels:
                section_payload[column] = str(prediction_row[column])
            payload = {section_name: section_payload}
            payloads[(participant_code, section_name)] = payload
    return payloads


def _prepare_human_artifact(source_workbook: Path, human_root: Path, annotator_id: str) -> Path:
    target_dir = human_root / annotator_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = copy_completed_annotation_to_human_dir(source_workbook, target_dir / f"{annotator_id}.xlsx")
    write_human_manifest(target_dir, annotator_id)
    return target


def test_load_llm_config_and_runtime_precedence(tmp_path: Path) -> None:
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_llm_block())

    llm_config = load_llm_config(paths_config)

    assert llm_config.runtime.provider == "ollama"
    assert llm_config.runtime.source == "survey"
    assert llm_config.runtime.temperature == 0.2
    assert len(llm_config.experiments) == 1
    assert llm_config.experiments[0].temperature == 0.0
    assert llm_config.experiments[0].run_id == "run-01"


def test_participant_codes_are_stable_across_sample_and_survey_modes(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    survey_csv = FIXTURES / "survey_fixture.csv"
    raw = pd.read_csv(survey_csv)
    _, _, _, sampled_private_df, _ = create_annotation_frames(raw, study)
    sampled_codes = sampled_private_df.set_index(study.id_column)["participant_code"].to_dict()

    paths_config = make_paths_config(tmp_path, survey_csv)
    paths = load_paths_config(paths_config)
    filtered_df = load_batch_source(study, paths, source="survey", all_records=False)
    all_records_df = load_batch_source(study, paths, source="survey", all_records=True)

    filtered_codes = filtered_df.set_index(study.id_column)["participant_code"].to_dict()
    all_codes = all_records_df.set_index(study.id_column)["participant_code"].to_dict()

    assert sampled_private_df["participant_code"].is_unique
    assert filtered_df["participant_code"].is_unique
    assert all_records_df["participant_code"].is_unique

    for response_id, participant_code in sampled_codes.items():
        assert filtered_codes[response_id] == participant_code
        assert all_codes[response_id] == participant_code


def test_assign_participant_codes_rejects_duplicate_response_ids(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    source_df = pd.read_csv(FIXTURES / "survey_fixture.csv")
    duplicated = pd.concat([source_df.iloc[[0]], source_df.iloc[[0]]], ignore_index=True)

    try:
        assign_participant_codes(duplicated, study)
    except ValueError as exc:
        assert f"duplicate {study.id_column} values" in str(exc)
    else:
        raise AssertionError("Expected duplicate response ids to be rejected.")


def test_survey_batch_source_applies_require_all_texts_by_default(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    raw = pd.read_csv(FIXTURES / "survey_fixture.csv")
    raw.loc[raw[study.id_column] == 101, study.sections["aftereffects"].source_column] = ""
    custom_survey = tmp_path / "survey_with_blank.csv"
    raw.to_csv(custom_survey, index=False)

    paths_config = make_paths_config(tmp_path, custom_survey)
    paths = load_paths_config(paths_config)

    filtered_df = load_batch_source(study, paths, source="survey", all_records=False)
    all_records_df = load_batch_source(study, paths, source="survey", all_records=True)

    assert 101 not in set(filtered_df[study.id_column])
    assert 101 in set(all_records_df[study.id_column])


def test_run_llm_resumes_failures_and_emits_noop_when_complete(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_llm_block())
    paths = load_paths_config(paths_config)
    llm_config = load_llm_config(paths_config)
    source_df = load_batch_source(study, paths, source="survey", all_records=False)
    payloads = _prediction_payloads(study, source_df)
    failed_key = next((key for key in payloads if int(source_df.set_index("participant_code").loc[key[0], study.id_column]) == 102 and key[1] == "aftereffects"))
    provider = FakeProvider(payloads, invalid_once={failed_key})

    first = run_llm_experiments(
        study=study,
        paths=paths,
        llm_config=llm_config,
        experiment_ids=["exp-alpha"],
        provider_factory=lambda runtime: provider,
    )

    artifact = first["experiments"][0]
    artifact_dir = Path(artifact["manifest_file"]).parent
    assert artifact["no_op"] is False
    assert Path(artifact["section_results_file"]).exists()
    assert Path(artifact["predictions_file"]).exists()
    assert Path(artifact["raw_responses_file"]).exists()
    assert Path(artifact["errors_file"]).exists()

    predictions_after_first = [json.loads(line) for line in Path(artifact["predictions_file"]).read_text(encoding="utf-8").splitlines() if line]
    assert len(predictions_after_first) == 2
    section_results_first = [json.loads(line) for line in (artifact_dir / "section_results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    failed_entries = [record for record in section_results_first if record["participant_code"] == failed_key[0] and record["section"] == failed_key[1]]
    assert failed_entries[0]["status"] == "failed"
    assert failed_entries[0]["attempts"] == 1

    second = run_llm_experiments(
        study=study,
        paths=paths,
        llm_config=llm_config,
        experiment_ids=["exp-alpha"],
        provider_factory=lambda runtime: provider,
    )
    artifact_second = second["experiments"][0]
    predictions_after_second = [json.loads(line) for line in Path(artifact_second["predictions_file"]).read_text(encoding="utf-8").splitlines() if line]
    assert artifact_second["no_op"] is False
    assert len(predictions_after_second) == 3
    assert provider.calls[failed_key] == 2
    assert all(temperature == 0.0 for temperature in provider.temperatures)

    third = run_llm_experiments(
        study=study,
        paths=paths,
        llm_config=llm_config,
        experiment_ids=["exp-alpha"],
        provider_factory=lambda runtime: provider,
    )
    artifact_third = third["experiments"][0]
    assert artifact_third["no_op"] is True
    assert "No pending LLM calls" in artifact_third["message"]


def test_run_llm_persists_progress_during_interrupted_run(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_llm_block())
    paths = load_paths_config(paths_config)
    llm_config = load_llm_config(paths_config)
    source_df = load_batch_source(study, paths, source="survey", all_records=False)
    payloads = _prediction_payloads(study, source_df)
    total_tasks = len(source_df) * len(study.section_order)

    crashing_provider = InterruptingProvider(payloads, interrupt_on_call=2)
    try:
        run_llm_experiments(
            study=study,
            paths=paths,
            llm_config=llm_config,
            experiment_ids=["exp-alpha"],
            provider_factory=lambda runtime: crashing_provider,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Expected the simulated interruption to abort the run.")

    artifact_dir = paths.llm_results_dir / f"{llm_config.experiments[0].experiment_id}__{llm_config.experiments[0].run_id}"
    section_results = [json.loads(line) for line in (artifact_dir / "section_results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    raw_responses = [json.loads(line) for line in (artifact_dir / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines() if line]
    partial_summary = json.loads((artifact_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert len(section_results) == 1
    assert section_results[0]["status"] == "success"
    assert len(raw_responses) == 1
    assert partial_summary["execution"]["n_calls_attempted"] == 1
    assert "Processed 1 LLM calls so far" in partial_summary["execution"]["message"]

    provider = FakeProvider(payloads)
    resumed = run_llm_experiments(
        study=study,
        paths=paths,
        llm_config=llm_config,
        experiment_ids=["exp-alpha"],
        provider_factory=lambda runtime: provider,
    )

    artifact = resumed["experiments"][0]
    final_summary = artifact["summary"]
    assert artifact["no_op"] is False
    assert final_summary["execution"]["n_calls_attempted"] == total_tasks - 1
    assert sum(provider.calls.values()) == total_tasks - 1


def test_evaluate_ignores_internal_runner_artifacts(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    study = load_study_config(study_config)
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_llm_block())
    paths = load_paths_config(paths_config)
    llm_config = load_llm_config(paths_config)

    build_result = run_cli("build-annotation-sample", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert build_result.returncode == 0, build_result.stderr

    source_workbook = tmp_path / "annotation_outputs" / "nde_annotation_sample.xlsx"
    mapping_workbook = tmp_path / "annotation_outputs" / "nde_annotation_mapping_private.xlsx"
    human_root = tmp_path / "human_annotations"
    populate_human_annotation_workbook(source_workbook, mapping_workbook, study_config)
    _prepare_human_artifact(source_workbook, human_root, "ann_a")
    _prepare_human_artifact(source_workbook, human_root, "ann_b")

    source_df = load_batch_source(study, paths, source="survey", all_records=False)
    payloads = _prediction_payloads(study, source_df)
    provider = FakeProvider(payloads)
    run_llm_experiments(
        study=study,
        paths=paths,
        llm_config=llm_config,
        experiment_ids=["exp-alpha"],
        provider_factory=lambda runtime: provider,
    )

    eval_result = run_cli("evaluate", "--study-config", str(study_config), "--paths-config", str(paths_config))
    assert eval_result.returncode == 0, eval_result.stderr

    summary = json.loads((tmp_path / "evaluation_outputs" / "evaluation_summary.json").read_text(encoding="utf-8"))
    llm_manifest = json.loads((tmp_path / "evaluation_outputs" / "llm_artifacts_manifest.json").read_text(encoding="utf-8"))

    assert summary["coverage"]["n_valid_llm_artifacts"] == 1
    assert summary["coverage"]["n_rejected_llm_artifacts"] == 0
    assert len(llm_manifest["accepted"]) == 1
    assert len(llm_manifest["rejected"]) == 0
    assert llm_manifest["accepted"][0]["artifact_id"] == "exp_alpha__run-01"


def test_load_batch_source_prefers_preprocessed_dataset_and_can_filter_by_cleaned_sections(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_llm_block())
    paths = load_paths_config(paths_config)

    preview_df = load_batch_source(study, paths, source="survey", all_records=False)
    invalid_participant = str(preview_df.iloc[0]["participant_code"])

    class PreprocessConfigStub:
        model = "mock-preprocess-model"
        provider = "ollama"
        base_url = "http://localhost:11434"
        timeout_seconds = 30
        max_attempts = 2
        temperature = 0.0
        prompt_version = "v1"

        def to_dict(self):
            return {}

    run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=PreprocessConfigStub(),
        provider_factory=lambda _: FakePreprocessingProvider(invalid_participant=invalid_participant),
    )

    preferred_df = load_batch_source(study, paths, source="survey", all_records=False)
    assert "n_valid_sections_cleaned" in preferred_df.columns
    assert invalid_participant not in set(preferred_df["participant_code"])
    assert set(preferred_df["participant_code"]) == set(preview_df["participant_code"]) - {invalid_participant}

    filtered_df = load_batch_source(study, paths, source="survey", all_records=False, min_valid_sections=3)
    assert invalid_participant not in set(filtered_df["participant_code"])


def test_load_batch_source_uses_post_preprocessing_validity_and_to_drop_filters(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_llm_block())
    paths = load_paths_config(paths_config)

    class PreprocessConfigStub:
        model = "mock-preprocess-model"
        provider = "ollama"
        base_url = "http://localhost:11434"
        timeout_seconds = 30
        max_attempts = 2
        temperature = 0.0
        prompt_version = "v1"

        def to_dict(self):
            return {}

    run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=PreprocessConfigStub(),
        all_records=True,
        provider_factory=lambda _: FakePreprocessingProvider(),
    )

    cleaned_path = paths.preprocessing_output_dir / "cleaned_dataset.csv"
    cleaned_df = pd.read_csv(cleaned_path)
    cleaned_df.loc[cleaned_df[study.id_column] == 104, "TO_DROP"] = False
    cleaned_df.loc[cleaned_df[study.id_column] == 104, "n_valid_sections_cleaned"] = 3
    cleaned_df.loc[cleaned_df[study.id_column] == 104, "n_valid_sections"] = 3
    cleaned_df.loc[cleaned_df[study.id_column] == 104, study.sections["aftereffects"].source_column] = "clean aftereffects"
    cleaned_df.to_csv(cleaned_path, index=False)

    filtered_df = load_batch_source(study, paths, source="survey", all_records=False)
    all_records_df = load_batch_source(study, paths, source="survey", all_records=True)

    assert 104 in set(filtered_df[study.id_column])
    assert 104 in set(all_records_df[study.id_column])


def test_load_batch_source_excludes_preprocessed_rows_marked_to_drop(tmp_path: Path) -> None:
    study = load_study_config(FIXTURES / "study_test.toml")
    survey_csv = FIXTURES / "survey_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv, llm_block=_llm_block())
    paths = load_paths_config(paths_config)

    class PreprocessConfigStub:
        model = "mock-preprocess-model"
        provider = "ollama"
        base_url = "http://localhost:11434"
        timeout_seconds = 30
        max_attempts = 2
        temperature = 0.0
        prompt_version = "v1"

        def to_dict(self):
            return {}

    run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=PreprocessConfigStub(),
        all_records=True,
        provider_factory=lambda _: FakePreprocessingProvider(),
    )

    cleaned_path = paths.preprocessing_output_dir / "cleaned_dataset.csv"
    cleaned_df = pd.read_csv(cleaned_path)
    cleaned_df.loc[cleaned_df[study.id_column] == 101, "TO_DROP"] = True
    cleaned_df.to_csv(cleaned_path, index=False)

    filtered_df = load_batch_source(study, paths, source="survey", all_records=False)
    all_records_df = load_batch_source(study, paths, source="survey", all_records=True)

    assert 101 not in set(filtered_df[study.id_column])
    assert 101 in set(all_records_df[study.id_column])



def test_ollama_provider_accepts_structured_output_in_thinking() -> None:
    payload = {
        "thinking": '{\n  "context": {\n    "tone": "neutral",\n    "death_context_nature": "no_death_context",\n    "evidence_segments": ["The report describes events in sequence."]\n  }\n}',
        "response": "",
    }

    raw_text, source_field = OllamaProvider._extract_raw_text(payload)

    assert source_field == "thinking"
    assert '"tone": "neutral"' in raw_text
