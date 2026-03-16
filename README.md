# NDE Narratives Analysis

Structured coding workflow for Near-Death Experience (NDE) narratives with human annotation, LLM-ready batches, VADER-based sensitivity analysis, and evaluation against questionnaire-derived labels.

## Repository Principles

- Real participant data is never committed to the repository.
- The CLI reads and writes external paths defined in `config/paths.local.toml`.
- The repository versions code, prompt templates, schemas, docs, and synthetic fixtures only.

## Layout

- `config/`: Study configuration and local path templates.
- `docs/`: Research proposal, annotation guidelines, and output contracts.
- `prompts/`: Section-specific LLM prompt templates.
- `schemas/`: Versioned JSON schemas for normalized LLM outputs.
- `src/`: Python package and CLI.
- `tests/fixtures/`: Synthetic survey, annotation, and prediction fixtures.

## Setup on Windows PowerShell

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e .[dev]
    Copy-Item config\paths.example.toml config\paths.local.toml

Then edit `config/paths.local.toml` to point to your external data and output folders, and replace the questionnaire column placeholders in `config/study.toml`.

## CLI

Validate that the study config, path config, and source CSV are aligned:

    nde validate-config

Build the annotation workbook sample outside the repository:

    nde build-annotation-sample

If the target annotation artifacts already exist, the command stops instead of overwriting them. Use `--force` only when you intentionally want to replace the generated files:

    nde build-annotation-sample --force

Build three JSONL batches, one per narrative section:

    nde build-llm-batch --source sampled-private

Run a first-layer VADER sensitivity analysis over the configured narrative text columns:

    nde sentiment-sensitivity

Use `--all-records` to bypass the study-level row filters, repeat `--quality-value` to override the configured quality subset, and add `--include-text` only when you want the raw text echoed into the score file.

Run evaluation against human annotations, questionnaire-derived labels, and VADER tone labels. If LLM predictions are available, they are included automatically as extra comparisons:

    nde evaluate

Run the test suite:

    python -m pytest

## Notes

- `nde build-annotation-sample` writes three external files: the generated annotator workbook, the private mapping workbook, and the private column map workbook.
- The generated annotator workbook is the base file for coding. The completed human-annotation workbook used by `nde evaluate` is a separate artifact configured through `paths.human_annotation_workbook`.
- `nde build-annotation-sample` protects existing annotation artifacts by default and requires `--force` for intentional overwrites.
- `nde build-llm-batch` expects either the private sampled workbook or the external survey CSV as input.
- `nde sentiment-sensitivity` writes a reusable `vader_sentiment_scores.csv`, per-section PNG figures, a Markdown report, and a JSON summary under the chosen output directory. Raw text is excluded by default and included only with `--include-text`.
- `nde evaluate` always requires completed human annotations. LLM predictions are optional unless you pass an explicit `--llm-predictions` path, in which case the file must exist and follow the contract documented in `docs/output_contract.md`.
- `nde evaluate` will reuse a previously generated VADER score file when available, or generate sample-level VADER scores automatically for the annotated subset.
- `nde evaluate` now also writes an alignment-focused Markdown report, comparison figures, and an auxiliary long-format metrics table under the evaluation output directory.
