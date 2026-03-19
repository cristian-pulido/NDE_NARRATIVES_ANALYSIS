# Summary

This PR refines the evaluation reporting workflow to better support article-facing interpretation of alignment results.

## What changed

- Reorganized reporting into clearer analytical layers: general results, family-level results, and item-level detail.
- Split reporting into a main human-centered report and a separate questionnaire-centered report.
- Added higher-quality article-oriented figures, including per-scope figure sets for human and questionnaire analyses.
- Added configurable figure export options in [`nde evaluate`](src/nde_narratives/cli.py:444), including high-DPI raster output and optional PDF export.
- Added family-level aggregation outputs in [`alignment_family_metrics.csv`](docs/output_contract.md:172).
- Added binary positive-class diagnostics in [`compute_comparison_metrics()`](src/nde_narratives/evaluation.py:451), including:
  - `precision_yes`
  - `recall_yes`
  - `f1_yes`
  - prevalence and prevalence-gap summaries
- Updated questionnaire reporting to better support interpretation of the question: whether lower alignment reflects model limitations or systematic differences in narrative recoverability.
- Updated tests and output contract documentation.

# Motivation

The previous reporting layout was useful for technical inspection but less effective for article writing. In particular, it did not clearly separate:

- overall alignment patterns
- family-level differences between `Tone`, `M8`, and `M9`
- fine-grained item detail

This PR makes the report structure more interpretable and adds diagnostics that help distinguish between:

- lower alignment due to model weakness
- lower alignment due to weaker recovery of positive cases
- lower alignment due to systematic differences in how constructs are expressed in text

# Key files changed

- [`src/nde_narratives/evaluation_report.py`](src/nde_narratives/evaluation_report.py:1)
- [`src/nde_narratives/evaluation.py`](src/nde_narratives/evaluation.py:1)
- [`src/nde_narratives/cli.py`](src/nde_narratives/cli.py:1)
- [`docs/output_contract.md`](docs/output_contract.md:1)
- [`tests/test_cli_evaluate.py`](tests/test_cli_evaluate.py:1)

# Validation

- Syntax checked with `python3 -m py_compile` for the updated Python modules.
- Full pytest execution was not completed in this environment because the external dependency `vaderSentiment` is unavailable.

# Notes

- [`TAREA.md`](TAREA.md) was intentionally not included in this commit.
- The questionnaire-based interpretation note created outside this repository was also not included in this commit.
