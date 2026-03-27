# PR: Add prompt-variant filtering and custom output directory support in evaluate CLI

## Summary
This PR extends the `nde evaluate` workflow with:
- `--prompt-variant` filtering for LLM artifact discovery.
- `--output-dir` override support for evaluation outputs.
- Backward-compatible prompt-variant inference from manifest metadata (`prompt_variant` with `prompt_root` fallback).
- Test coverage for prompt-variant filtering plus custom output directory behavior.
- README documentation updates for the new evaluation options.

## What Changed
- **CLI updates** (`src/nde_narratives/cli.py`)
  - Added `--prompt-variant` argument to evaluation filters.
  - Wired `prompt_variants` into `evaluate_outputs(...)`.
- **Evaluation pipeline updates** (`src/nde_narratives/evaluation.py`)
  - Added `_infer_prompt_variant_from_metadata(...)`.
  - Updated experiment metadata resolution to use inferred prompt variant.
  - Added `prompt_variants` filter in `discover_llm_prediction_artifacts(...)`.
  - Threaded `prompt_variants` through `evaluate_outputs(...)`.
- **Tests** (`tests/test_cli_evaluate.py`)
  - Added end-to-end CLI test validating:
    - filtering by `--prompt-variant sentence_majority_v1`
    - writing outputs to `--output-dir`
    - expected accepted artifact and output files.
- **Docs** (`README.md`)
  - Added explanation for `--prompt-variant` and `--output-dir` in evaluation options.

## Validation
- Code review completed for code, documentation, and integration flow.
- Targeted test execution was attempted; environment lacked one runtime dependency (`vaderSentiment`), so full runtime validation could not be completed locally in this session.

## Risk / Compatibility
- Low risk.
- Changes are additive and preserve previous behavior when new flags are not provided.
- Prompt variant matching uses normalized identifiers for robust filtering.

## Reviewer Checklist
- [ ] Confirm CLI help text for `--prompt-variant` and `--output-dir` is clear.
- [ ] Verify prompt-variant fallback behavior for legacy manifests with `prompt_root`.
- [ ] Run full test suite in environment with project dependencies installed.

## Suggested test command
```bash
PYTHONPATH=src:. pytest -q tests/test_cli_evaluate.py
```
