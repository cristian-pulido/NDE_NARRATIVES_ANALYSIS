# Summary

This PR adds a dedicated preprocessing workflow for narrative cleaning before downstream analysis.

## What changed

- Added a new CLI entrypoint: `nde preprocess`.
- Implemented resumable preprocessing with validation + conditional resegmentation.
- Separated preprocessing prompts from downstream analysis prompts.
- Added preprocessing schemas and output artifacts.
- Added automatic downstream consumption of the cleaned dataset when present.
- Added downstream filtering based on post-preprocessing validity and `TO_DROP != True`.
- Added support for restarting preprocessing from scratch with `--from-scratch`.
- Added summary metrics reporting how many rows had 3 valid sections before vs after preprocessing.

## Main files changed

- `src/nde_narratives/cli.py`
- `src/nde_narratives/config.py`
- `src/nde_narratives/preprocessing.py`
- `src/nde_narratives/prompting.py`
- `src/nde_narratives/llm_runner.py`
- `prompts/preprocessing/validate_sections_prompt.md`
- `prompts/preprocessing/resegment_narrative_prompt.md`
- `prompts/analysis/context_prompt.md`
- `prompts/analysis/experience_prompt.md`
- `prompts/analysis/aftereffects_prompt.md`
- `schemas/preprocess_validation_output.schema.json`
- `schemas/preprocess_resegmentation_output.schema.json`
- `README.md`
- `docs/llm_workflow.md`
- `docs/output_contract.md`
- `tests/test_preprocessing.py`
- `tests/test_llm_runner.py`
- `tests/test_cli_config_and_build.py`
- `tests/cli_helpers.py`

## Behavior

### Preprocessing

- Accepts rows with at least one meaningful narrative section so partially populated originals can still be rescued.
- Validates whether the three original sections are correctly aligned.
- If not, merges and resegments while trimming irrelevant material.
- Writes a cleaned dataset plus resumable ledgers and summaries.

### Downstream analysis

- If `preprocessing_outputs/cleaned_dataset.csv` exists, downstream survey-based analysis uses it automatically.
- Filtered downstream runs use post-preprocessing validity.
- Rows with fewer than 3 cleaned valid sections are excluded by default in filtered downstream loading.
- Rows with `TO_DROP = True` are excluded from filtered downstream loading.
- `--min-valid-sections` remains available for explicit threshold control.

## Validation

Executed successfully:

- `PYTHONPATH=src:. pytest -q tests/test_preprocessing.py`
- `PYTHONPATH=src:. pytest tests/test_llm_runner.py -q -k 'prefers_preprocessed_dataset or uses_post_preprocessing_validity_and_to_drop_filters or excludes_preprocessed_rows_marked_to_drop'`

## Notes

- Human validation / agreement for preprocessing is intentionally left as an external review step.
- The decision to continue with downstream analysis remains human-led after reviewing preprocessing outputs.
