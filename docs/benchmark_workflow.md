# Benchmark Workflow (Optional)

This optional workflow establishes an external sentiment baseline for the current tone pipeline.

Initial target:

- Amazon Reviews (mapped to 3 classes)
- Label mapping: `1-2 -> negative`, `3 -> neutral`, `4-5 -> positive`

## 1) Configure benchmark blocks

Add benchmark settings in `config/paths.local.toml`:

```toml
[benchmark.runtime]
provider = "ollama"
base_url = "http://localhost:11434"
timeout_seconds = 120
max_attempts = 2
temperature = 0.0

[benchmark.dataset]
dataset_name = "SetFit/amazon_reviews_multi_en"
dataset_config = ""
split = "train"
text_column = "text"
label_column = "label"
max_rows = 2000
random_state = 20

[[benchmark.datasets]]
dataset_name = "amazon_reviews_multi"
dataset_config = "en"
split = "train"
text_column = "review_body"
label_column = "stars"
max_rows = 2000
random_state = 20

[[benchmark.datasets]]
dataset_name = "imdb"
dataset_config = ""
split = "test"
text_column = "text"
label_column = "label"
max_rows = 2000
random_state = 20

[[benchmark.experiments]]
experiment_id = "amazon_qwen25_baseline"
enabled = true
model = "qwen2.5:7b"
prompt_variant = "baseline"
run_id = "run-01"
model_variant = "qwen2.5:7b"
temperature = 0.0
```

If `[[benchmark.experiments]]` is omitted, benchmark execution reuses `[[llm.experiments]]` automatically and keeps unique artifact ids.

If `[[benchmark.datasets]]` is present, `nde benchmark-all` runs every dataset in that list and generates one comparative report automatically.

Benchmark artifacts are stored under optional benchmark path keys in `[paths]`:

- `benchmark_raw_dir`
- `benchmark_processed_dir`
- `benchmark_runs_dir`
- `benchmark_reports_dir`
- `benchmark_prompt_variants_dir`

## 2) Download and normalize benchmark data

```bash
nde benchmark-download
```

This writes:

- raw JSONL snapshot
- normalized CSV with `record_id`, `text`, `gold_label`
- dataset manifest JSON

## 3) Run baseline models (VADER + LLM experiments)

```bash
nde benchmark-run
```

You may override input or scope:

```bash
nde benchmark-run --dataset-path /path/to/amazon_reviews_multi_normalized.csv --max-rows 1500 --prompt-variant baseline_v2
```

Resume behavior:

- `benchmark-run` is resumable by default.
- Re-running in the same output directory reuses existing prediction files and processes only missing rows.
- To force a full reset of the run artifact:

```bash
nde benchmark-run --from-scratch
```

Run outputs include:

- prediction files
- metrics table (Accuracy, Macro F1, Cohen Kappa)
- confusion table and per-label table
- run summary manifest

## 4) Generate benchmark report

```bash
nde benchmark-report
```

Optional benchmark vs NDE comparison table in the report:

```bash
nde benchmark-report --nde-metrics /path/to/evaluation_metrics.csv
```

The generated report follows this fixed section order:

1. Source
2. Methodology
3. Prompts
4. Metrics
5. Interpretation
6. Limitations

## 5) End-to-end command

```bash
nde benchmark-all
```

This runs download + normalize + model execution + report generation in one command.

To force full reset of the run artifact in this end-to-end flow:

```bash
nde benchmark-all --from-scratch
```

## Prompt customization

- Default prompt lives at `prompts/benchmark/sentiment_prompt.md`.
- Variant prompts can be stored in `<benchmark_prompt_variants_dir>/<variant>/sentiment_prompt.md` and selected with `--prompt-variant`.
