# Output Contract

The repository normalizes human annotations, LLM outputs, questionnaire-derived labels, and VADER tone outputs to the current shared schema.

## Translation Artifact Contract

`nde translate` writes optional translation artifacts under `preprocessing_output_dir`.

Typical files:

- `manifest_translate.json`
- `participant_results_translate.jsonl`
- `raw_responses_translate.jsonl`
- `errors_translate.jsonl`
- `run_summary_translate.json`
- `translated_dataset.csv`
- `translated_dataset.xlsx`

The translated dataset preserves the original section source column names (same shape expected by downstream extraction), but values are normalized to English and augmented with audit fields:

- `detected_language_context`
- `detected_language_experience`
- `detected_language_aftereffects`
- `detected_language_row`

## Preprocessing Artifact Contract

[`nde preprocess`](../src/nde_narratives/cli.py:202) writes a dedicated preprocessing artifact directory under `preprocessing_output_dir`.

Typical files:

- `manifest.json`
- `participant_results.jsonl`
- `raw_responses.jsonl`
- `errors.jsonl`
- `run_summary.json`
- `cleaned_dataset.csv`
- `cleaned_dataset.xlsx`
- optional `preprocessing_validation_sample.xlsx`
- optional `preprocessing_validation_mapping_private.xlsx`

The preprocessing stage is resumable:

- rows with `success` are not rerun
- rows with `failed` are retried until `max_attempts`
- rows with `exhausted` are left untouched unless `--retry-exhausted` is passed
- [`nde preprocess --from-scratch`](../src/nde_narratives/cli.py:247) clears the existing preprocessing artifacts in the target output directory before rebuilding

By default, preprocessing accepts rows with at least one meaningful narrative section, even if the original source does not have all three sections populated. The stricter `3 valid sections` requirement is intended for downstream analysis, not for admission into preprocessing.

The cleaned dataset includes:

- study `id_column`
- `participant_code`
- cleaned text written back to the three configured section source columns
- `preprocessing_status`
- `n_valid_sections` as the post-cleaning count for backward compatibility
- `n_valid_sections_original` as the count judged valid before resegmentation
- `n_valid_sections_cleaned` as the count of non-empty sections after cleaning
- `preprocessing_run_status`
- `changed_sections`

The preprocessing run summary includes:

- `n_rows_with_3_valid_sections_original`
- `n_rows_with_3_valid_sections_cleaned`

This makes it explicit how many rows already had 3 usable sections before preprocessing and how many rows ended with 3 usable sections after preprocessing.

The preprocessing prompts are intentionally separate from downstream analysis prompts:

- [`prompts/preprocessing/`](../prompts/preprocessing/)
- [`prompts/analysis/`](../prompts/analysis/)

Prompt repository policy:

- default downstream prompts must exist only in [`prompts/analysis/`](../prompts/analysis/)
- duplicate default prompt files under `prompts/*.md` at repository root are not supported
- variant prompts are loaded from `prompt_variants_dir/<variant>/` when configured

## Shared Fields

- `participant_code`
- `context_tone`
- `experience_tone`
- `aftereffects_tone`
- `outside_of_body_experience`
- `feeling_bright_light`
- `feeling_awareness`
- `presence_encounter`
- `saw_relived_past_events`
- `time_perception_altered`
- `border_point_of_no_return`
- `non_existence_feeling`
- `feeling_peace_wellbeing`
- `saw_entered_gateway`
- `fear_of_death`
- `inner_meaning_in_my_life`
- `compassion_toward_others`
- `spiritual_feelings`
- `desire_to_help_others`
- `personal_vulnerability`
- `interest_in_material_goods`
- `interest_in_religion`
- `understanding_myself`
- `social_justice_issues`

## Allowed Values

- Tone fields: `positive`, `negative`, `mixed`, `neutral`
- Binary fields: `yes`, `no`

## Questionnaire Binarization Contract

Questionnaire-derived binary labels are normalized to `yes` / `no` / `NA` before evaluation.

### NDE-C mapping

- uses the configured `yes_values` and `no_values` per item from the study questionnaire config
- matching is case-insensitive and whitespace-normalized
- blank values and `Missing` are mapped to `NA`
- configured `na_values` are also mapped to `NA` (case-insensitive)

### LCI-R mapping

- explicit `increased` or `decreased` responses map to `yes` (change present)
- explicit `not changed` responses map to `no` (no change)
- `Missing` or blank responses map to `NA`
- `NA` cells are excluded from that item's metric denominator

## Human Annotation Artifact Contract

`nde evaluate` can now read either:

- a single workbook passed explicitly with `--human-annotation-workbook`, or
- a folder of candidate workbooks discovered under `human_annotations_dir`

Each candidate workbook is validated independently.

For any accepted workbook:

- rows with all required label fields blank are treated as not evaluated and are skipped
- rows with all required label fields completed are included
- rows with only some required label fields completed are invalid and cause that workbook to be rejected
- `participant_code` must be present for every remaining row

Rejected workbooks are reported in `human_artifacts_manifest.json` and do not abort the whole evaluation run.

## Human Artifact Identity

Human artifact identity is resolved in this order:

1. sibling `<filename>.manifest.json` / `<filename>.manifest.toml`
2. folder-level `manifest.json` / `manifest.toml`
3. workbook filename stem

For human artifacts, manifests may define:

- `annotator_id`

Folder-level manifests apply to every artifact in that folder.

## Majority Human Reference Contract

Evaluation no longer uses a single human workbook as the default reference.

Instead:

- valid human workbooks are consolidated by `annotator_id`
- a field-level majority vote is computed for each `participant_code`
- ties remain unresolved for that field/participant pair
- unresolved cells are excluded only from the affected field metrics

`nde evaluate` writes:

- `human_reference_majority.csv`
- `adjudication_summary.csv`
- `human_agreement_pairwise.csv`
- `human_agreement_summary.csv`

## LLM Batch Format

Each JSONL record produced by `nde build-llm-batch` contains:

- `participant_code`
- `section`
- `input_text`
- `prompt`
- `response_schema`
- `experiment`

`experiment` includes:

- `experiment_id`
- `artifact_id`
- optional `prompt_variant`
- optional `run_id`
- optional `model_variant`

Each batch directory also writes a `manifest.json` with the same metadata plus the resolved prompt root and section batch paths.

## LLM Prediction Format for Evaluation

`nde evaluate` accepts either:

- a single prediction artifact passed explicitly with `--llm-predictions`, or
- a folder of candidate artifacts discovered under `llm_results_dir`

Supported artifact formats remain:

- `.jsonl`
- `.csv`
- `.xlsx`
- `.xls`

For JSONL, records must follow the nested section contract:

```json
{
  "participant_code": "ANN_XXXX",
  "prediction": {
    "context": {
      "tone": "neutral",
      "death_context_nature": "subjective_threat_only",
      "evidence_segments": ["short verbatim span"]
    },
    "experience": {
      "tone": "positive",
      "evidence_segments": ["short verbatim span"],
      "outside_of_body_experience": "no",
      "feeling_bright_light": "yes",
      "feeling_awareness": "yes",
      "presence_encounter": "no",
      "saw_relived_past_events": "no",
      "time_perception_altered": "no",
      "border_point_of_no_return": "no",
      "non_existence_feeling": "no",
      "feeling_peace_wellbeing": "yes",
      "saw_entered_gateway": "no"
    },
    "aftereffects": {
      "tone": "mixed",
      "evidence_segments": ["short verbatim span"],
      "fear_of_death": "yes",
      "inner_meaning_in_my_life": "yes",
      "compassion_toward_others": "yes",
      "spiritual_feelings": "yes",
      "desire_to_help_others": "yes",
      "personal_vulnerability": "yes",
      "interest_in_material_goods": "no",
      "interest_in_religion": "yes",
      "understanding_myself": "yes",
      "social_justice_issues": "no"
    }
  }
}
```

### Context-only extraction field (LLM output)

The context section schema now also requires:

- `death_context_nature`: `no_death_context` | `subjective_threat_only` | `objective_medical_context`

This field is intentionally extraction-only for the current phase and is not part of the shared evaluation label matrix. Evaluation tolerates this additional field in LLM outputs and still aligns human vs LLM metrics on the existing shared columns.

Human annotation workbooks may also include extra columns; evaluation ignores non-required columns and validates/aligned only on the required study-defined labels.

Each candidate artifact is validated independently. Invalid artifacts are reported in `llm_artifacts_manifest.json` and skipped.

## LLM Artifact Identity

LLM artifact identity is resolved in this order:

1. sibling `<filename>.manifest.json` / `<filename>.manifest.toml`
2. folder-level `manifest.json` / `manifest.toml`
3. parent folder name / filename stem fallback

LLM manifests may define:

- `experiment_id`
- optional `prompt_variant`
- optional `run_id`
- optional `model_variant`

The effective comparison key written into metrics is `artifact_id`, which is `experiment_id` or `experiment_id__run_id` when a run id is present.

## VADER Sensitivity Output Format

`nde sentiment-sensitivity` produces a reusable tabular file with one row per source record and narrative section. Required fields are:

- study `id_column` from `config/study.toml` (for example `response_id`)
- `participant_code` when available in the input source; blank otherwise
- `section`
- `source_column`
- `neg`
- `neu`
- `pos`
- `compound`
- `vader_label`

Source selection behavior for `nde sentiment-sensitivity`:

- if `--input-path` is provided, that file is used
- otherwise, if [`preprocessing_outputs/cleaned_dataset.csv`](../src/nde_narratives/preprocessing.py:23) exists, VADER uses the cleaned dataset
- otherwise, if `preprocessing_outputs/translated_dataset.csv` exists, VADER uses the translated dataset
- otherwise, VADER falls back to the configured survey source

Source priority for all `survey`-mode downstream consumers without explicit `--input-path`:

1. `cleaned_dataset.csv`
2. `translated_dataset.csv`
3. configured `survey_csv`

If none exists, execution fails with explicit `FileNotFoundError` including attempted paths.

Optional field:

- `text` only when the command is run with `--include-text`

`vader_label` is mapped only for tone comparison using the standard VADER thresholds:

- `compound >= 0.05` -> `positive`
- `compound <= -0.05` -> `negative`
- otherwise -> `mixed`

VADER outputs apply only to section tone fields and are not used for binary questionnaire-derived labels.

## Evaluation Alignment Rules

In addition to the standard alignment tables and figures, `nde evaluate` now emits a questionnaire-focused contradiction analysis package for Experience Tone:

- `questionnaire_contradictions.csv`
- `questionnaire_contradiction_examples.csv`
- contradiction figures under `evaluation_outputs/figures/alignment/`:
  - `questionnaire_contradiction_overview.png`
  - `questionnaire_contradiction_unigram_wordcloud.png`
  - `questionnaire_contradiction_bigrams.png`
  - `questionnaire_contradiction_trigrams.png`

### Contradiction definition

- strict polarity contradiction only: questionnaire `positive` vs automated `negative`, or questionnaire `negative` vs automated `positive`
- scope limited to `experience_tone`

### Source selection and evidence policy

- VADER is included only as a quantitative baseline in contradiction counts/rates
- qualitative contradiction evidence uses only selected LLM outputs
- selected LLMs are the top 3 `questionnaire_vs_llm:*` comparisons ranked by `macro_f1` on `experience_tone`

### Evidence segments policy

- for nested JSONL outputs, section-level `evidence_segments` are retained for qualitative analysis
- tabular prediction files may not include usable evidence segments for qualitative contradiction examples

`nde evaluate` aligns comparison sources to the participant codes present in the majority human reference.

- questionnaire-derived labels are filtered to that participant subset
- VADER tone labels are filtered to that participant subset
- each accepted LLM artifact is evaluated separately against that same subset
- internal LLM runner files such as `section_results.jsonl`, `raw_responses.jsonl`, and `errors.jsonl` are ignored during artifact discovery
- extra rows in automated sources are ignored naturally by field-level overlap
- unresolved human majority cells reduce field-level `n` instead of aborting evaluation

Primary evaluation uses variable-wise sample size (`n`) per item/comparison pair after per-field missingness and unresolved-majority exclusions. Complete-case evaluation is reported separately as a sensitivity analysis.

## Evaluation Report Outputs

`nde evaluate` writes:

- `evaluation_metrics.csv`
- `evaluation_metrics_complete_case.csv`
- `evaluation_summary.json`
- `human_reference_majority.csv`
- `adjudication_summary.csv`
- `human_agreement_pairwise.csv`
- `human_agreement_summary.csv`
- `human_artifacts_manifest.json`
- `llm_artifacts_manifest.json`
- `questionnaire_contradictions.csv`
- `questionnaire_contradiction_examples.csv`
- `alignment_report.md`
- `alignment_report_questionnaire.md`
- `alignment_metrics_long.csv`
- `alignment_family_metrics.csv`
- `figures/alignment/*.png`
- `figures/alignment/*.pdf` when `nde evaluate --export-figures-pdf` is used
- `experiments/<artifact_id>/evaluation_metrics.csv`
- `experiments/<artifact_id>/evaluation_summary.json`

`evaluation_summary.json` now contains these top-level sections:

- `coverage`: counts for sampled rows, valid/rejected human artifacts, reference participants, and valid/rejected LLM artifacts
- `adjudication`: majority-reference coverage and unresolved counts
- `comparisons`: mean metric summaries by comparison name
- `human_artifacts`: accepted and rejected human artifact registry
- `llm_artifacts`: accepted and rejected LLM artifact registry
- `questionnaire_contradictions`: contradiction-focused quantitative and qualitative analysis payload for questionnaire vs automated Experience Tone
