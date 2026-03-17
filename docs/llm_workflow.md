# LLM Workflow

This guide explains the practical LLM workflow for a user who just installed the repository and wants to run a small smoke test before launching a full experiment.

## 1. Install and Prepare

From the repository root, create and activate the virtual environment.

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

Then edit `config/paths.local.toml`:

- set `[paths].data_dir`
- set `[paths].survey_csv`
- confirm or override the derived output folders if needed
- add the `[llm]` and `[[llm.experiments]]` blocks described below

If your study config still contains questionnaire placeholder columns, replace them in `config/study.toml` before running the CLI.

After the environment is active, the commands below use `nde ...`. If your shell does not expose that entrypoint, use `python -m nde_narratives.cli ...` on Windows or `python3 -m nde_narratives.cli ...` on Linux/macOS.

## 2. Configure Ollama and Experiments

The runtime configuration and experiment registry live in `config/paths.local.toml`.

Example:

```toml
[paths]
data_dir = "D:/data/nde"
survey_csv = "D:/data/nde/results-survey.csv"

[llm]
provider = "ollama"
base_url = "http://localhost:11434"
timeout_seconds = 120
max_attempts = 2
temperature = 0.0
source = "survey"
all_records = false

[[llm.experiments]]
experiment_id = "smoke_qwen08"
enabled = true
model = "qwen3.5:0.8b"
run_id = "smoke01"
model_variant = "qwen3.5:0.8b"
temperature = 0.0

[[llm.experiments]]
experiment_id = "baseline_qwen35"
enabled = true
model = "qwen3.5:latest"
run_id = "run01"
model_variant = "qwen3.5:latest"
temperature = 0.0
```

Important notes:

- `source = "survey"` means the runner uses the survey CSV, not the human annotation sample.
- If you want to run only the generated validation sample, set `source = "sampled-private"` instead. That makes `run-llm` use `sampled_private_workbook`, which is the same sample used by the human validation workflow.
- `all_records = false` means it applies the study-level row filters by default.
- Each experiment should have its own `experiment_id` or `run_id` so artifacts stay traceable.
- Use one experiment for smoke tests and a different one for the full run.

## 3. Validate the Setup

Run:

```bash
nde validate-config
```

This checks the study config, the source CSV, the path config, and the LLM configuration block.

## 4. Run a Small Smoke Test

Before running the full dataset, execute a tiny subset:

```bash
nde run-llm --experiment-id smoke_qwen08 --limit 2
```

This is the recommended first test because it:

- verifies the Ollama connection
- verifies the prompts and schemas
- writes a real artifact you can inspect quickly
- avoids waiting for the full dataset

If you prefer to smoke-test only the validation sample instead of the filtered survey, first set this in `config/paths.local.toml`:

```toml
[llm]
source = "sampled-private"
```

Then run the same command with `--limit`.

## 5. Understand the Output

Each experiment is written under:

```text
<llm_results_dir>/<artifact_id>/
```

Typical files:

- `manifest.json`: effective configuration for the run
- `section_results.jsonl`: one row per `participant_code + section`
- `predictions.jsonl`: only participants with all three sections completed successfully
- `raw_responses.jsonl`: raw model outputs for debugging
- `errors.jsonl`: structured failures
- `run_summary.json`: counts and status summary

The runner is resumable:

- it does not rerun `success`
- it retries `pending` and `failed` until `max_attempts`
- it leaves `exhausted` alone unless you pass `--retry-exhausted`

If everything is already done for that exact artifact, rerunning the same command returns a no-op summary.

These files are updated during execution, not only at the end of the run. That means a long experiment can be resumed from the latest persisted section-level state after an interruption.

For live progress, inspect `section_results.jsonl` or `run_summary.json`. `predictions.jsonl` only changes when a participant has all three sections completed successfully, so it is normal for that file to stay empty for a while at the beginning of a run.

For smoke tests, `run_summary.json` is the fastest file to inspect:

- `n_complete_predictions = 0` means no participant finished all three sections successfully yet
- `status_counts.failed > 0` means you should inspect `errors.jsonl`
- `raw_responses.jsonl` may still be empty if the provider failed before the runner accepted any model text as a valid candidate response

## 6. Run the Full Experiment

Once the smoke test looks good:

```bash
nde run-llm --experiment-id baseline_qwen35
```

Or run every enabled experiment:

```bash
nde run-llm --all-experiments
```

If you need to force another pass on exhausted rows:

```bash
nde run-llm --experiment-id baseline_qwen35 --retry-exhausted
```

## 7. Evaluate the Results

After human annotations and LLM outputs are available:

```bash
nde evaluate --experiment-id baseline_qwen35
```

Evaluation still compares automated outputs only against the adjudicated human subset.

## 8. Prompt Variants

By default, prompts are loaded from the repository `prompts/` folder.

If you define `prompt_variant = "baseline_v2"` in an experiment, the CLI first looks for:

```text
<prompt_variants_dir>/baseline_v2/
```

That variant folder must contain:

- `context_prompt.md`
- `experience_prompt.md`
- `aftereffects_prompt.md`

If the variant folder does not exist, the CLI falls back to the repository default prompts.

With the default path layout, `prompt_variants_dir` resolves to:

```text
<data_dir>/prompt_variants/
```

Example:

```text
D:/data/nde/prompt_variants/baseline_v2/context_prompt.md
D:/data/nde/prompt_variants/baseline_v2/experience_prompt.md
D:/data/nde/prompt_variants/baseline_v2/aftereffects_prompt.md
```

## 9. Optional Low-Level Debugging

If you want to inspect the rendered prompts and per-section batch records without calling the model:

```bash
nde build-llm-batch --source survey --experiment-id debug_batch --run-id run01 --limit 2
```

This is useful for prompt inspection, but for normal use `run-llm` is the primary entrypoint.

## 10. Troubleshooting

### Smoke test ran but every section failed

If `run_summary.json` looks like this:

- `n_section_tasks > 0`
- `n_complete_predictions = 0`
- `status_counts.success = 0`
- `status_counts.failed = all tasks`

open `errors.jsonl` first.

### Error: `Ollama response did not include a non-empty 'response' field.`

This can happen with some reasoning-capable Ollama models when `format` is supplied. In that case, Ollama may leave `response` blank and place the structured JSON in `thinking`.

The runner now accepts that fallback, but if you still see this pattern:

- rerun the same artifact once after updating the code
- if the artifact already exists, rerunning is safe because the runner resumes only failed rows
- if you want a clean smoke test, create a new `run_id`

Example:

```toml
[[llm.experiments]]
experiment_id = "smoke_qwen08"
run_id = "smoke02"
```

then run:

```bash
nde run-llm --experiment-id smoke_qwen08 --limit 2
```

### Different source scopes for smoke test and full run

Avoid reusing the same artifact id across:

- `source = "survey"`
- `source = "sampled-private"`
- different `--limit` values

If you change source scope, use a different `run_id` or `experiment_id`.
