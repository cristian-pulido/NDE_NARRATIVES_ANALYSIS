# Output Contract

The repository normalizes human annotations, LLM outputs, and questionnaire-derived labels to the same field names.

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