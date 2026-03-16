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


## Evaluation Report Outputs

`nde evaluate` continues to write `evaluation_metrics.csv` and `evaluation_summary.json`, and now also produces:

- `alignment_report.md`
- `alignment_metrics_long.csv`
- `figures/alignment/*.png`

The alignment report is built from the available comparisons in each run. At minimum it supports `human_vs_questionnaire` and `human_vs_vader`, and it incorporates LLM comparisons automatically when LLM predictions are present.
