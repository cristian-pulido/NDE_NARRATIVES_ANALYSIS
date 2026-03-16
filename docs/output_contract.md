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

## Human Annotation Workbook Behavior

For `nde evaluate`, the human workbook is interpreted row by row using the shared fields above:

- rows with all required label fields blank are treated as not evaluated and are skipped
- rows with all required label fields completed are included in evaluation
- rows with only some required label fields completed are invalid and cause evaluation to fail
- `participant_code` must still be present for every remaining row in the workbook

This allows experts to leave a case unevaluated by clearing the row or by removing the row from the completed workbook, without introducing a special label value.

## LLM Batch Format

Each JSONL record produced by `nde build-llm-batch` contains:

- `participant_code`
- `section`
- `input_text`
- `prompt`
- `response_schema`

## LLM Prediction Format for Evaluation

`nde evaluate` accepts:

- JSONL with one record per `participant_code` and section, or
- CSV/XLSX with one wide row per `participant_code`

For JSONL, the record may either expose normalized fields directly or wrap them under a `prediction` object. In both cases, keys must follow the shared field names above.

During evaluation, LLM rows are filtered to the human-evaluable participant subset. Extra LLM rows outside that subset are ignored, but missing LLM rows for human-evaluable participants still cause evaluation to fail when LLM comparisons are requested.

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

`nde evaluate` aligns all comparison sources to the participant codes that are fully annotated in the human workbook.

- Questionnaire-derived labels and VADER tone labels are filtered to that human-evaluable subset.
- Optional LLM predictions are filtered to that same subset.
- Extra rows in automated sources are ignored.
- Missing rows in automated sources for human-evaluable participants still trigger a participant mismatch error.

## Evaluation Report Outputs

`nde evaluate` continues to write `evaluation_metrics.csv` and `evaluation_summary.json`, and now also produces:

- `alignment_report.md`
- `alignment_metrics_long.csv`
- `figures/alignment/*.png`

The alignment report is built from the available comparisons in each run. At minimum it supports `human_vs_questionnaire` and `human_vs_vader`, and it incorporates LLM comparisons automatically when LLM predictions are present.

`evaluation_summary.json` now contains two top-level sections:

- `coverage`: counts for sampled rows, rows still present in the human workbook, fully evaluable human rows, and skipped blank rows
- `comparisons`: mean metric summaries by comparison name
