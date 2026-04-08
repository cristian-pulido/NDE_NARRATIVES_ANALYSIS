# NDE Narratives Analysis

Structured coding workflow for Near-Death Experience (NDE) narratives with multi-annotator human annotation, experiment-scoped LLM batches, VADER-based sensitivity analysis, and evaluation against a majority-vote human reference.

## Workflow Diagram

The following diagram summarizes how configuration, prompts, schemas, human annotation, LLM execution, VADER scoring, and evaluation artifacts connect across the project workflow.

![NDE narratives analysis workflow](Pipeline.svg)

## Repository Principles

- Real participant data is never committed to the repository.
- The CLI reads and writes external paths defined in `config/paths.local.toml`.
- The repository versions code, prompt templates, schemas, docs, and synthetic fixtures only.
- Human annotators and LLM experiments are treated as traceable external artifacts, not anonymous overwriteable files.

## Layout

- `config/`: Study configuration and local path templates.
- `docs/`: Research proposal, annotation guidelines, and output contracts.
- `prompts/`: Prompt templates split into preprocessing and downstream analysis.
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

Local runtime settings are split by purpose in [`config/paths.local.toml`](config/paths.local.toml):

- `[translate]` defines the optional one-time translation stage used by `nde translate`
- `[preprocessing]` defines the one canonical segmentation/validation cleaning model used by [`nde preprocess`](src/nde_narratives/cli.py:202)
- `[llm]` and `[[llm.experiments]]` define downstream analysis experiments used by [`nde run-llm`](src/nde_narratives/cli.py:382)

For the practical LLM setup, smoke-test flow, prompt variants layout, and full run instructions, see [LLM Workflow](docs/llm_workflow.md).

For the optional external baseline validation pipeline (Amazon download, normalization, benchmark run, and benchmark report), see [Benchmark Workflow](docs/benchmark_workflow.md).

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

Optional translation stage (detect source language per record and translate the three narrative columns to English):

    nde translate

Run the one-time segmentation/validation preprocessing stage that validates section structure, resegments invalid rows when needed, resumes failures, and writes a cleaned dataset copy under `preprocessing_output_dir`:

    nde preprocess

Useful preprocessing options:

- `--all-records`: bypass study-level row filters.
- `--retry-exhausted`: retry rows previously marked exhausted.
- `--from-scratch`: borrar el ledger previo y recomenzar la corrida desde cero en la carpeta de salida elegida.
- `--generate-validation-sample`: write a human-review workbook after preprocessing.
- `--validation-n-total`: size of the optional validation sample.

Useful preprocessing config knobs in `paths.local.toml` under `[preprocessing]`:

- `timeout_seconds`: HTTP timeout per request.
- `dynamic_context_enabled`: when true, preprocessing estimates prompt size and sends a dynamic `num_ctx` to Ollama.
- `num_ctx_min` / `num_ctx_max`: lower/upper bounds for dynamic `num_ctx` bucketing.
- `chars_per_token`: coarse tokenizer ratio used for context estimation.

By default, preprocessing is intentionally more inclusive than downstream analysis: it accepts rows with at least one meaningful narrative section so partially populated originals can still be rescued before the final `3-section` filter is applied.

When no explicit `--input-path` is passed for `survey` source, downstream commands now resolve source with this priority:

1. `preprocessing_output_dir/cleaned_dataset.csv`
2. `preprocessing_output_dir/translated_dataset.csv`
3. configured `survey_csv`

If none of these sources exists, the workflow fails with an explicit `FileNotFoundError` listing attempted paths.

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

If you want to keep only fully usable cleaned narratives, add:

    nde run-llm --experiment-id qwen25_baseline --min-valid-sections 3

Use `--all-experiments` to execute every enabled `[[llm.experiments]]` entry in `paths.local.toml`. The command preserves successful rows, retries pending or failed rows up to `max_attempts`, and returns a no-op message when the artifact is already complete for that configuration.

The expected prompt separation is now:

- [`prompts/preprocessing/`](prompts/preprocessing/) for the one-time cleaning stage
- [`prompts/analysis/`](prompts/analysis/) for downstream experiment-driven extraction and coding

Prompt location policy:

- Repository default analysis prompts must live only under [`prompts/analysis/`](prompts/analysis/).
- Do not create duplicate defaults under `prompts/*.md` at repository root.
- Optional experiment variants should live under `prompt_variants_dir/<variant>/` and include the same three files (`context_prompt.md`, `experience_prompt.md`, `aftereffects_prompt.md`).

Run a first-layer VADER sensitivity analysis over the configured narrative text columns:

    nde sentiment-sensitivity

Use `--all-records` to bypass the study-level row filters, repeat `--quality-value` to override the configured quality subset, and add `--include-text` only when you want the raw text echoed into the score file.

Run evaluation against the majority-vote human reference, questionnaire-derived labels, VADER tone labels, and any accepted LLM experiments:

    nde evaluate

Run the alternate human-review protocol from `Human.md` against cleaned segmentation, questionnaire labels, and default-prompt LLM outputs (excluding RA1 artifacts):

    nde compare-human-review --human-md /path/Human.md --cleaned-dataset /path/cleaned_dataset.csv --questionnaire-csv /path/NDE_traslated.csv --llm-results-dir /path/llm_outputs_translate_v0_segment_run --output-dir /path/human_review_report --export-figures-pdf

Note: for unit-classification segmentation metrics, use a cleaned dataset in the same language as the questionnaire text (for example `preprocessing_outputs_from_translated_v0/cleaned_dataset.csv` when using `NDE_traslated.csv`).

Useful evaluation options:

- `--human-annotations-dir`: evaluate all discovered human workbooks in a folder.
- `--llm-results-dir`: evaluate all discovered LLM prediction artifacts in a folder.
- `--annotator-id`: restrict evaluation to selected annotators.
- `--experiment-id`: restrict evaluation to selected experiments.
- `--prompt-variant`: restrict evaluation to artifacts whose manifest prompt variant matches one or more selected values (with prompt-root fallback for legacy manifests).
- `--output-dir`: write metrics, manifests, figures, and reports to a custom destination instead of the default evaluation output folder.
- `--human-annotation-workbook` and `--llm-predictions`: explicit single-file overrides.

Compare multiple evaluation output folders (for example, to quantify preprocessing effects such as translation or re-segmentation) and generate a manuscript-style appendix report with unified tables and figures:

    nde compare-evaluation-outputs --condition standard=./Data/Alignment/Results/evaluation_outputs --condition translate_run=./Data/Alignment/Results/evaluation_outputs_translate_run --output-dir ./Data/Alignment/Results/comparison_outputs

Input modes for `nde compare-evaluation-outputs`:

- Repeatable `--condition NAME=PATH` (recommended for quick runs).
- `--config path/to/compare_config.toml` for reusable multi-condition setups.

Important precedence rule:

- If `--config` is provided, it takes precedence and inline `--condition` entries are ignored.

Useful comparison options:

- `--baseline NAME`: choose the reference condition used for delta columns and narratives.
- `--focus-scope questionnaire_vs_llm|human_reference_vs_llm|all`: restrict analysis to selected comparison scopes.
- `--metric macro_f1|accuracy|cohen_kappa`: select the primary ranking metric used in summary panels.
- `--title "Custom comparison title"`: override report title.
- `--output-dir PATH`: write comparison tables, figures, and markdown report to a custom destination.

Compute bootstrap uncertainty intervals for evaluation outputs (with emphasis on questionnaire vs LLM metrics), writing a dedicated `uncertainty/` package with CSV tables, figures, and Markdown report:

    nde evaluate-uncertainty

Useful uncertainty options:

- `--input-dir PATH`: source evaluation outputs directory (must contain `evaluation_metrics.csv`).
- `--output-dir PATH`: destination directory for uncertainty artifacts; default is `<input-dir>/uncertainty`.
- `--bootstrap-samples N`: bootstrap replicates (default `5000`).
- `--confidence-level P`: interval confidence level in `(0, 1)` (default `0.95`).
- `--random-seed N`: deterministic seed for reproducible intervals.
- `--export-figures-pdf`: also export PDF copies of uncertainty figures.

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

### `run-llm` Valid-Section Filtering Behavior

When `nde run-llm` loads a survey input that includes either `n_valid_sections_cleaned` or `n_valid_sections`, it now applies a valid-section threshold by default, even when `--input-path` is provided explicitly.

- Default behavior: keep rows with at least `3` valid sections.
- Override threshold: pass `--min-valid-sections N`.
- Disable valid-section filtering: pass `--min-valid-sections 0`.
- `--all-records` still bypasses study-level row filters; if you pass `--all-records` together with `--min-valid-sections N`, the valid-section threshold is applied intentionally.

This ensures explicit `--input-path` runs and auto-detected preprocessed runs behave consistently.
