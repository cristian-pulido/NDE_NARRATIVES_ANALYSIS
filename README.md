# NDE Narratives Analysis

Structured coding workflow for Near-Death Experience (NDE) narratives with human annotation, LLM-ready batches, and evaluation against questionnaire-derived labels.

## Repository Principles

- Real participant data is never committed to the repository.
- The CLI reads and writes external paths defined in `config/paths.local.toml`.
- The repository versions code, prompt templates, schemas, docs, and synthetic fixtures only.

## Layout

- `config/`: Study configuration and local path templates.
- `docs/`: Research proposal and annotation guidelines.
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

Run evaluation against human annotations, LLM predictions, and questionnaire-derived labels:

    nde evaluate

Run the test suite:

    python -m pytest

## Notes

- `nde build-annotation-sample` writes three external files: the generated annotator workbook, the private mapping workbook, and the private column map workbook.
- The generated annotator workbook is the base file for coding. The completed human-annotation workbook used by `nde evaluate` is a separate artifact configured through `paths.human_annotation_workbook`.
- `nde build-annotation-sample` protects existing annotation artifacts by default and requires `--force` for intentional overwrites.
- `nde build-llm-batch` expects either the private sampled workbook or the external survey CSV as input.
- `nde evaluate` expects completed human annotations and normalized LLM predictions that use the versioned field contract documented in `docs/output_contract.md`.