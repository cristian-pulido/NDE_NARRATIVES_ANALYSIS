# NDE Narratives Analysis

Structured coding workflow for Near-Death Experience (NDE) narratives with multi-annotator human annotation, experiment-scoped LLM batches, VADER-based sensitivity analysis, and evaluation against a majority-vote human reference.

## Repository Principles

- Real participant data is never committed to the repository.
- The CLI reads and writes external paths defined in `config/paths.local.toml`.
- The repository versions code, prompt templates, schemas, docs, and synthetic fixtures only.
- Human annotators and LLM experiments are treated as traceable external artifacts, not anonymous overwriteable files.

## Layout

- `config/`: Study configuration and local path templates.
- `docs/`: Research proposal, annotation guidelines, and output contracts.
- `prompts/`: Default section-specific LLM prompt templates.
- `schemas/`: Versioned JSON schemas for normalized LLM outputs.
- `src/`: Python package and CLI.
- `tests/fixtures/`: Synthetic survey, annotation, and prediction fixtures.

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
Copy-Item config\paths.example.toml config\paths.local.toml
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp config/paths.example.toml config/paths.local.toml
```

Then edit `config/paths.local.toml` and replace the questionnaire column placeholders in `config/study.toml`.

After the virtual environment is active, the examples in this repository use `nde ...`. If the `nde` command is not available in your shell, use `python -m nde_narratives.cli ...` on Windows or `python3 -m nde_narratives.cli ...` on Linux/macOS.

## Minimal Config

In most cases you only need:

```toml
[paths]
data_dir = "D:/data/nde"
survey_csv = "D:/data/nde/results-survey.csv"
```

If you omit the rest, the repository resolves these defaults relative to `data_dir`:

- `annotation_output_dir = data_dir/annotation_outputs`
- `human_annotations_dir = data_dir/human_annotations`
- `llm_batch_dir = data_dir/llm_batches`
- `llm_results_dir = data_dir/llm_outputs`
- `evaluation_output_dir = data_dir/evaluation_outputs`
- `prompt_variants_dir = data_dir/prompt_variants`
- `sampled_private_workbook = data_dir/annotation_outputs/nde_annotation_mapping_private.xlsx`
- `human_annotation_workbook = data_dir/human_annotations/nde_annotation_sample_completed.xlsx`
- `llm_predictions_path = data_dir/llm_outputs/nde_predictions.jsonl`

`human_annotation_workbook` and `llm_predictions_path` are now optional in the TOML. They remain as compatibility defaults and as useful single-file overrides.

Local LLM runtime settings and experiment registrations now live in the same file under `[llm]` and `[[llm.experiments]]`.

For the practical LLM setup, smoke-test flow, prompt variants layout, and full run instructions, see [LLM Workflow](docs/llm_workflow.md).

## User Options

You now have two valid ways to work.

### Option 1: Folder-based workflow

Recommended for real studies with multiple annotators or multiple LLM experiments.

- Store completed human workbooks under `human_annotations_dir`.
- Store LLM prediction artifacts under `llm_results_dir`.
- Run `nde evaluate` and let the CLI discover everything automatically.

### Option 2: Single-file workflow

Useful for quick runs or backward compatibility.

- Keep a single completed human workbook and/or a single predictions file.
- Either place them in the default derived paths, or pass them directly by CLI.

Examples:

    nde evaluate --human-annotation-workbook D:/data/nde/human_annotations/ana.xlsx
    nde evaluate --llm-predictions D:/data/nde/llm_outputs/exp1.jsonl

If you pass a file explicitly by CLI, that file overrides folder discovery for that source.

## Artifact Conventions

### Human artifacts

The evaluator accepts `.xlsx` / `.xls` workbooks.

Simplest case: no manifest.

```text
D:/data/nde/human_annotations/ana.xlsx
D:/data/nde/human_annotations/luis.xlsx
```

In that case, the annotator ids are inferred from the filenames: `ana`, `luis`.

With separate folders and one manifest per annotator:

```text
D:/data/nde/human_annotations/ana/ana.xlsx
D:/data/nde/human_annotations/ana/manifest.json
D:/data/nde/human_annotations/luis/luis.xlsx
D:/data/nde/human_annotations/luis/manifest.json
```

Example manifest:

```json
{
  "annotator_id": "ana"
}
```

With multiple files in the same folder, use per-file manifests if you want explicit ids:

```text
D:/data/nde/human_annotations/ana.xlsx
D:/data/nde/human_annotations/ana.manifest.json
D:/data/nde/human_annotations/luis.xlsx
D:/data/nde/human_annotations/luis.manifest.json
```

Important rule:

- A folder-level `manifest.json` applies to every artifact in that folder.
- So if you keep many annotators in the same folder, do not use one shared folder-level manifest unless they all intentionally share the same identity.
- For multiple annotators, either use separate folders or per-file manifests.

### LLM artifacts

The evaluator accepts `.jsonl`, `.csv`, `.xlsx`, and `.xls`.

Without manifest:

```text
D:/data/nde/llm_outputs/gpt4o_run01.jsonl
D:/data/nde/llm_outputs/gpt41_run01.jsonl
```

With experiment folder and manifest:

```text
D:/data/nde/llm_outputs/gpt4o_run01/predictions.jsonl
D:/data/nde/llm_outputs/gpt4o_run01/manifest.json
```

Example manifest:

```json
{
  "experiment_id": "gpt4o_baseline",
  "prompt_variant": "baseline",
  "run_id": "run_01",
  "model_variant": "gpt-4o"
}
```

Just like human artifacts:

- a folder-level manifest applies to all artifacts inside that folder
- if you store many prediction files in one folder, use per-file manifests instead

## CLI

Validate that the study config, path config, and source CSV are aligned:

    nde validate-config

Build the annotation workbook sample outside the repository:

    nde build-annotation-sample

If the target annotation artifacts already exist, the command stops instead of overwriting them. Use `--force` only when you intentionally want to replace the generated files:

    nde build-annotation-sample --force

Build three JSONL batches, one per narrative section, inside an experiment-specific output folder:

    nde build-llm-batch --source survey --experiment-id exp_alpha --prompt-variant baseline --run-id run_01

Useful batch options:

- `--experiment-id`: stable experiment identifier.
- `--prompt-variant`: prompt folder name under `prompt_variants_dir`.
- `--run-id`: run identifier appended to the experiment artifact id.
- `--model-variant`: optional metadata only.
- `--prompt-root`: explicit prompt folder override.
- `--all-records`: bypass study-level row filters and batch the full survey source.

Run configured LLM experiments directly, resume missing or failed rows, and write normalized artifacts under `llm_results_dir`:

    nde run-llm --experiment-id qwen25_baseline

Use `--all-experiments` to execute every enabled `[[llm.experiments]]` entry in `paths.local.toml`. The command preserves successful rows, retries pending or failed rows up to `max_attempts`, and returns a no-op message when the artifact is already complete for that configuration.

Run a first-layer VADER sensitivity analysis over the configured narrative text columns:

    nde sentiment-sensitivity

Use `--all-records` to bypass the study-level row filters, repeat `--quality-value` to override the configured quality subset, and add `--include-text` only when you want the raw text echoed into the score file.

Run evaluation against the majority-vote human reference, questionnaire-derived labels, VADER tone labels, and any accepted LLM experiments:

    nde evaluate

Useful evaluation options:

- `--human-annotations-dir`: evaluate all discovered human workbooks in a folder.
- `--llm-results-dir`: evaluate all discovered LLM prediction artifacts in a folder.
- `--annotator-id`: restrict evaluation to selected annotators.
- `--experiment-id`: restrict evaluation to selected experiments.
- `--human-annotation-workbook` and `--llm-predictions`: explicit single-file overrides.

## Typical Workflow

### Multiple annotators and multiple experiments

1. Configure `data_dir` and `survey_csv`.
2. Run:

       nde build-annotation-sample

3. Distribute the generated workbook template to annotators.
4. Save completed workbooks under `human_annotations_dir`.
5. Save each LLM result artifact under `llm_results_dir`.
6. Optionally add manifests for stable ids and metadata.
7. Run:

       nde evaluate

### One human file and one LLM file

1. Keep your files anywhere you want.
2. Run:

       nde evaluate --human-annotation-workbook D:/path/to/completed.xlsx --llm-predictions D:/path/to/predictions.jsonl

## Outputs

`nde evaluate` now writes, at minimum:

- `evaluation_metrics.csv`
- `evaluation_summary.json`
- `human_reference_majority.csv`
- `adjudication_summary.csv`
- `human_agreement_pairwise.csv`
- `human_agreement_summary.csv`
- `human_artifacts_manifest.json`
- `llm_artifacts_manifest.json`
- `alignment_report.md`
- comparison figures and `alignment_metrics_long.csv`

Per-experiment outputs are written under:

- `evaluation_outputs/experiments/<artifact_id>/`

## Notes

- `nde build-annotation-sample` still writes the generated annotator workbook, private mapping workbook, and private column map workbook.
- The generated annotator workbook is a template for coders; completed workbooks should be stored under `paths.human_annotations_dir` or passed explicitly by CLI.
- `nde build-llm-batch` writes experiment-specific batch folders and a `manifest.json` describing the batch.
- `nde run-llm` now persists section-level progress during execution, so interrupted runs can resume from the latest saved state instead of waiting for a single final write.
- `nde evaluate` no longer assumes a single completed human workbook or a single LLM predictions file.
- `nde evaluate` will reuse a previously generated VADER score file when available, or generate sample-level VADER scores automatically for the majority-reference participant subset.
- LLM execution is now configured locally through `[llm]` and `[[llm.experiments]]` in `paths.local.toml`, with Ollama as the first backend.

