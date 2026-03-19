# Output Contract

The repository normalizes human annotations, LLM outputs, questionnaire-derived labels, and VADER tone outputs to comparable field names where appropriate.

## Shared Fields

- `participant_code`
- `context_tone`
- `experience_tone`
- `aftereffects_tone`
- `m8_out_of_body`
- `m8_bright_light`
- `m8_peace`
- `m8_time_distortion`
- `m8_presence`
- `m9_moral_rules`
- `m9_long_term_thinking`
- `m9_consider_others`
- `m9_help_others`
- `m9_forgiveness`

## Allowed Values

- Tone fields: `positive`, `negative`, `mixed`
- Binary fields: `yes`, `no`

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

For JSONL, the record may either expose normalized fields directly or wrap them under a `prediction` object. In both cases, keys must follow the shared field names above.

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

Optional field:

- `text` only when the command is run with `--include-text`

`vader_label` is mapped only for tone comparison using the standard VADER thresholds:

- `compound >= 0.05` -> `positive`
- `compound <= -0.05` -> `negative`
- otherwise -> `mixed`

VADER outputs apply only to section tone fields and are not used for binary questionnaire-derived labels.

## Evaluation Alignment Rules

`nde evaluate` aligns comparison sources to the participant codes present in the majority human reference.

- questionnaire-derived labels are filtered to that participant subset
- VADER tone labels are filtered to that participant subset
- each accepted LLM artifact is evaluated separately against that same subset
- internal LLM runner files such as `section_results.jsonl`, `raw_responses.jsonl`, and `errors.jsonl` are ignored during artifact discovery
- extra rows in automated sources are ignored naturally by field-level overlap
- unresolved human majority cells reduce field-level `n` instead of aborting evaluation

## Evaluation Report Outputs

`nde evaluate` writes:

- `evaluation_metrics.csv`
- `evaluation_summary.json`
- `human_reference_majority.csv`
- `adjudication_summary.csv`
- `human_agreement_pairwise.csv`
- `human_agreement_summary.csv`
- `human_artifacts_manifest.json`
- `llm_artifacts_manifest.json`
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
