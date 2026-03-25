from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

from .config import (
    default_paths_config_path,
    default_study_config_path,
    load_benchmark_config,
    load_llm_config,
    load_paths_config,
    load_preprocessing_config,
    load_study_config,
)
from .constants import PROJECT_ROOT
from .excel import write_annotation_outputs
from .io_utils import read_tabular_file
from .prompting import PREPROCESSED_DATASET_FILENAME, write_llm_batches
from .sampling import create_annotation_frames


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Za-z0-9<>\- '&/]+:$")


def _supports_color(stream: object) -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR") or os.getenv("CLICOLOR_FORCE"):
        return True
    is_tty = getattr(stream, "isatty", None)
    if callable(is_tty) and is_tty():
        return os.getenv("TERM", "").lower() != "dumb"
    return False


def _style(text: str, *codes: str) -> str:
    if not text:
        return text
    return "".join(codes) + text + ANSI_RESET


def _colorize_help(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("usage:"):
            lines.append(_style("usage:", ANSI_BOLD, ANSI_CYAN) + line[len("usage:"):])
            continue
        if stripped == "Use 'nde <command> --help' for command-specific help.":
            lines.append(_style(line, ANSI_DIM))
            continue
        if stripped == "<command>":
            lines.append(line.replace("<command>", _style("<command>", ANSI_YELLOW)))
            continue
        if line.startswith("  nde "):
            lines.append(_style(line, ANSI_GREEN))
            continue
        if SECTION_HEADER_RE.match(stripped):
            prefix = line[: len(line) - len(line.lstrip())]
            lines.append(prefix + _style(stripped, ANSI_BOLD, ANSI_CYAN))
            continue
        if line.startswith("    ") and not line.startswith("     "):
            content = line[4:]
            parts = re.split(r"\s{2,}", content, maxsplit=1)
            command = parts[0]
            if re.fullmatch(r"[A-Za-z][\w-]+", command):
                rest = f"  {parts[1]}" if len(parts) == 2 else ""
                lines.append(f"    {_style(command, ANSI_CYAN)}{rest}")
                continue
        if line.startswith("  -"):
            indent = line[: len(line) - len(line.lstrip())]
            content = line[len(indent):]
            parts = re.split(r"\s{2,}", content, maxsplit=1)
            option_text = parts[0]
            rest = f"  {parts[1]}" if len(parts) == 2 else ""
            lines.append(f"{indent}{_style(option_text, ANSI_YELLOW)}{rest}")
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


class NDEArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None) -> None:  # type: ignore[override]
        target = file if file is not None else sys.stdout
        text = self.format_help()
        if _supports_color(target):
            text = _colorize_help(text)
        self._print_message(text, target)


class NDEHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_help_position", 32)
        super().__init__(*args, **kwargs)

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if action.default in {None, False, argparse.SUPPRESS}:
            return help_text
        display_default = getattr(action, "display_default", action.default)
        if display_default in {None, False, argparse.SUPPRESS}:
            return help_text
        if action.option_strings and "%(default)" not in help_text:
            return f"{help_text} (default: {display_default})"
        return help_text


def _examples_block(*lines: str) -> str:
    return "Examples:\n" + "\n".join(f"  {line}" for line in lines)


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Config")
    study_action = group.add_argument(
        "--study-config",
        metavar="PATH",
        default=str(default_study_config_path()),
        help="Path to the study TOML that defines source columns, labels, and section layout.",
    )
    study_action.display_default = _display_path(default_study_config_path())
    paths_action = group.add_argument(
        "--paths-config",
        metavar="PATH",
        default=str(default_paths_config_path()),
        help="Path to the local TOML that defines data locations and LLM runtime settings.",
    )
    paths_action.display_default = _display_path(default_paths_config_path())


def build_parser() -> argparse.ArgumentParser:
    parser = NDEArgumentParser(
        prog="nde",
        formatter_class=NDEHelpFormatter,
        description=dedent(
            """\
            Run the NDE narratives workflow from configuration through annotation prep,
            LLM execution, VADER sensitivity, and evaluation.
            """
        ),
        epilog=dedent(
            """\
            Common entry points:
              nde validate-config
              nde preprocess
              nde build-annotation-sample
              nde run-llm --experiment-id smoke_qwen08 --limit 2
              nde run-llm --all-experiments
              nde evaluate

            Use 'nde <command> --help' for command-specific help.
            """
        ),
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="<command>",
        dest="command",
        required=True,
        parser_class=NDEArgumentParser,
    )

    validate = subparsers.add_parser(
        "validate-config",
        formatter_class=NDEHelpFormatter,
        help="Validate study, path, and LLM config against the source data.",
        description=dedent(
            """\
            Check that the study config, local paths config, source survey file, and
            configured LLM runtime can be resolved together before running the workflow.
            """
        ),
        epilog=_examples_block(
            "nde validate-config",
            "nde validate-config --paths-config config/paths.local.toml",
        ),
    )
    _add_config_arguments(validate)
    validate.set_defaults(handler=cmd_validate_config)

    preprocess = subparsers.add_parser(
        "preprocess",
        formatter_class=NDEHelpFormatter,
        help="Run the narrative preprocessing pipeline and write a cleaned dataset copy.",
        description=dedent(
            """\
            Validate narrative section structure with the configured preprocessing model,
            re-segment invalid cases, resume prior failures, and write a cleaned dataset
            copy without modifying the original source file.
            """
        ),
        epilog=_examples_block(
            "nde preprocess",
            "nde preprocess --limit 10",
            "nde preprocess --retry-exhausted",
            "nde preprocess --generate-validation-sample --validation-n-total 20",
        ),
    )
    _add_config_arguments(preprocess)
    source_group = preprocess.add_argument_group("Source And Scope")
    source_group.add_argument(
        "--input-path",
        metavar="PATH",
        default=None,
        help="Explicit tabular input file path that overrides the configured survey source.",
    )
    source_group.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Limit the number of source rows for debugging or smoke tests.",
    )
    source_group.add_argument(
        "--all-records",
        action="store_true",
        help="Bypass study-level row filters and process every source row.",
    )
    execution_group = preprocess.add_argument_group("Execution Controls")
    execution_group.add_argument(
        "--retry-exhausted",
        action="store_true",
        help="Retry rows already marked as exhausted instead of leaving them untouched.",
    )
    execution_group.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete any existing preprocessing ledger state in the target output directory and rebuild the run from zero.",
    )
    execution_group.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Write preprocessing artifacts to this directory instead of preprocessing_output_dir.",
    )
    validation_group = preprocess.add_argument_group("Optional Validation Sample")
    validation_group.add_argument(
        "--generate-validation-sample",
        action="store_true",
        help="Generate a human-review sample from the cleaned preprocessing outputs.",
    )
    validation_group.add_argument(
        "--validation-n-total",
        metavar="N",
        type=int,
        default=None,
        help="Override the total number of rows to include in the validation sample.",
    )
    validation_group.add_argument(
        "--validation-random-state",
        metavar="N",
        type=int,
        default=None,
        help="Random seed for reproducible validation sample selection.",
    )
    validation_group.add_argument(
        "--force-validation-sample",
        action="store_true",
        help="Overwrite any existing preprocessing validation sample files.",
    )
    preprocess.set_defaults(handler=cmd_preprocess)

    build_annotation = subparsers.add_parser(
        "build-annotation-sample",
        formatter_class=NDEHelpFormatter,
        help="Generate the human annotation sample workbooks outside the repository.",
        description=dedent(
            """\
            Create the coder-facing annotation workbook plus the private mapping workbooks
            used to align the sampled records with participant ids and questionnaire data.
            """
        ),
        epilog=_examples_block(
            "nde build-annotation-sample",
            "nde build-annotation-sample --n-total 50 --random-state 42",
            "nde build-annotation-sample --force",
        ),
    )
    _add_config_arguments(build_annotation)
    sampling_group = build_annotation.add_argument_group("Sampling")
    sampling_group.add_argument(
        "--n-total",
        metavar="N",
        type=int,
        default=None,
        help="Override the total number of records to sample for annotation.",
    )
    sampling_group.add_argument(
        "--random-state",
        metavar="N",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    output_group = build_annotation.add_argument_group("Output")
    output_group.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing annotation output files if they already exist.",
    )
    build_annotation.set_defaults(handler=cmd_build_annotation_sample)

    build_batch = subparsers.add_parser(
        "build-llm-batch",
        formatter_class=NDEHelpFormatter,
        help="Render prompts and JSONL batch files without calling the model.",
        description=dedent(
            """\
            Build section-specific JSONL batches for inspection, debugging, or external
            execution. This command prepares prompts and schemas but does not invoke an LLM.
            """
        ),
        epilog=_examples_block(
            "nde build-llm-batch --experiment-id exp_alpha --run-id run_01",
            "nde build-llm-batch --source sampled-private --experiment-id sample_debug --limit 5",
            "nde build-llm-batch --prompt-variant baseline_v2 --output-dir /tmp/nde_batches",
        ),
    )
    _add_config_arguments(build_batch)
    source_group = build_batch.add_argument_group("Source And Scope")
    source_group.add_argument(
        "--source",
        choices=["survey", "sampled-private"],
        default="survey",
        help="Input source to batch: filtered survey rows or the sampled private workbook.",
    )
    source_group.add_argument(
        "--input-path",
        metavar="PATH",
        default=None,
        help="Explicit input file path that overrides the configured source location.",
    )
    source_group.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Limit the number of source rows for a quick smoke batch.",
    )
    source_group.add_argument(
        "--all-records",
        action="store_true",
        help="Bypass study-level row filters and batch every available record from the source.",
    )
    experiment_group = build_batch.add_argument_group("Experiment Metadata")
    experiment_group.add_argument(
        "--experiment-id",
        metavar="ID",
        default=None,
        help="Stable experiment identifier stored in the batch manifest and records.",
    )
    experiment_group.add_argument(
        "--prompt-variant",
        metavar="ID",
        default=None,
        help="Prompt variant folder name under prompt_variants_dir.",
    )
    experiment_group.add_argument(
        "--run-id",
        metavar="ID",
        default=None,
        help="Run identifier appended to the artifact id.",
    )
    experiment_group.add_argument(
        "--model-variant",
        metavar="ID",
        default=None,
        help="Optional model label stored as metadata only.",
    )
    output_group = build_batch.add_argument_group("Prompt And Output")
    output_group.add_argument(
        "--prompt-root",
        metavar="PATH",
        default=None,
        help="Explicit prompt directory to use instead of the configured default or prompt variant lookup.",
    )
    output_group.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory where the experiment batch folder should be written.",
    )
    build_batch.set_defaults(handler=cmd_build_llm_batch)

    run_llm = subparsers.add_parser(
        "run-llm",
        formatter_class=NDEHelpFormatter,
        help="Execute configured LLM experiments and resume existing artifacts.",
        description=dedent(
            """\
            Run one or more configured LLM experiments against the selected source data.
            Existing artifacts are resumed in place: successful rows are preserved, failed
            rows are retried up to max_attempts, and completed runs return a no-op summary.
            """
        ),
        epilog=_examples_block(
            "nde run-llm --experiment-id smoke_qwen08 --limit 2",
            "nde run-llm --experiment-id baseline_qwen35",
            "nde run-llm --all-experiments",
            "nde run-llm --experiment-id baseline_qwen35 --retry-exhausted",
        ),
    )
    _add_config_arguments(run_llm)
    selection_group = run_llm.add_argument_group("Experiment Selection")
    selection_group.add_argument(
        "--experiment-id",
        metavar="ID",
        action="append",
        default=None,
        help="Run one configured experiment_id. Repeat the flag to run several selected experiments.",
    )
    selection_group.add_argument(
        "--all-experiments",
        action="store_true",
        help="Run every enabled [[llm.experiments]] entry from the current paths config.",
    )
    source_group = run_llm.add_argument_group("Source And Scope Overrides")
    source_group.add_argument(
        "--input-path",
        metavar="PATH",
        default=None,
        help="Explicit input file path that overrides the configured survey or sampled-private source.",
    )
    source_group.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Limit the number of source rows for smoke tests or debugging.",
    )
    source_group.add_argument(
        "--all-records",
        action="store_true",
        help="Bypass study-level row filters and process the full configured source.",
    )
    source_group.add_argument(
        "--min-valid-sections",
        metavar="N",
        type=int,
        default=None,
        help="When a preprocessed dataset is auto-detected, keep only rows with at least N cleaned valid sections.",
    )
    execution_group = run_llm.add_argument_group("Execution Controls")
    execution_group.add_argument(
        "--retry-exhausted",
        action="store_true",
        help="Retry rows already marked as exhausted instead of leaving them untouched.",
    )
    execution_group.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Write experiment artifact folders to this directory instead of llm_results_dir.",
    )
    run_llm.set_defaults(handler=cmd_run_llm)

    sentiment = subparsers.add_parser(
        "sentiment-sensitivity",
        formatter_class=NDEHelpFormatter,
        help="Run VADER sentiment scoring across the configured narrative sections.",
        description=dedent(
            """\
            Generate section-level VADER scores and labels from the configured narrative
            text columns. Useful as a lightweight baseline and for evaluation inputs.
            """
        ),
        epilog=_examples_block(
            "nde sentiment-sensitivity",
            "nde sentiment-sensitivity --limit 10 --include-text",
            "nde sentiment-sensitivity --all-records --output-dir /tmp/vader_run",
        ),
    )
    _add_config_arguments(sentiment)
    source_group = sentiment.add_argument_group("Source And Scope")
    source_group.add_argument(
        "--input-path",
        metavar="PATH",
        default=None,
        help="Explicit tabular input file path that overrides the configured survey source.",
    )
    source_group.add_argument(
        "--all-records",
        action="store_true",
        help="Bypass study-level row filters and score every source row.",
    )
    source_group.add_argument(
        "--quality-value",
        dest="quality_values",
        metavar="VALUE",
        action="append",
        default=None,
        help="Repeat to override the configured quality subset filter with one or more accepted values.",
    )
    source_group.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Limit the number of scored source rows for debugging.",
    )
    output_group = sentiment.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory where the VADER score files, figures, and report should be written.",
    )
    output_group.add_argument(
        "--include-text",
        action="store_true",
        help="Include the raw narrative text in the output score file for inspection.",
    )
    sentiment.set_defaults(handler=cmd_sentiment_sensitivity)

    benchmark_download = subparsers.add_parser(
        "benchmark-download",
        formatter_class=NDEHelpFormatter,
        help="Download and normalize the Amazon benchmark dataset for baseline evaluation.",
        description=dedent(
            """\
            Download an external benchmark dataset and write both raw and normalized artifacts
            so the sentiment baseline can be reproduced and rerun without redownloading.
            """
        ),
        epilog=_examples_block(
            "nde benchmark-download",
            "nde benchmark-download --max-rows 1000",
            "nde benchmark-download --processed-dir /tmp/benchmark/processed",
        ),
    )
    _add_config_arguments(benchmark_download)
    benchmark_download.add_argument("--max-rows", metavar="N", type=int, default=None, help="Override benchmark.dataset.max_rows.")
    benchmark_download.add_argument(
        "--raw-dir",
        metavar="PATH",
        default=None,
        help="Directory where raw benchmark download artifacts should be written.",
    )
    benchmark_download.add_argument(
        "--processed-dir",
        metavar="PATH",
        default=None,
        help="Directory where normalized benchmark CSV and manifest should be written.",
    )
    benchmark_download.set_defaults(handler=cmd_benchmark_download)

    benchmark_run = subparsers.add_parser(
        "benchmark-run",
        formatter_class=NDEHelpFormatter,
        help="Run VADER and configured LLM experiments on the normalized benchmark dataset.",
        description=dedent(
            """\
            Execute the benchmark sentiment baseline using VADER plus configured benchmark
            LLM experiments, then write metrics and run manifests.
            """
        ),
        epilog=_examples_block(
            "nde benchmark-run",
            "nde benchmark-run --dataset-path /data/nde/benchmark/processed/amazon_reviews_multi_normalized.csv",
            "nde benchmark-run --prompt-variant baseline_v2",
        ),
    )
    _add_config_arguments(benchmark_run)
    benchmark_run.add_argument(
        "--dataset-path",
        metavar="PATH",
        default=None,
        help="Existing normalized benchmark dataset CSV. If omitted, benchmark-run triggers download+normalize first.",
    )
    benchmark_run.add_argument("--max-rows", metavar="N", type=int, default=None, help="Override benchmark.dataset.max_rows.")
    benchmark_run.add_argument(
        "--prompt-variant",
        metavar="ID",
        default=None,
        help="Optional prompt variant under benchmark_prompt_variants_dir.",
    )
    benchmark_run.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory where benchmark run artifacts should be written.",
    )
    benchmark_run.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete any existing resumable benchmark artifact in the target output dir and rerun from zero.",
    )
    benchmark_run.set_defaults(handler=cmd_benchmark_run)

    benchmark_report = subparsers.add_parser(
        "benchmark-report",
        formatter_class=NDEHelpFormatter,
        help="Generate a benchmark Markdown report from a benchmark run summary.",
        description=dedent(
            """\
            Build the benchmark report with required section order: source, methodology,
            prompts, metrics, interpretation, limitations.
            """
        ),
        epilog=_examples_block(
            "nde benchmark-report",
            "nde benchmark-report --run-summary /data/nde/benchmark/runs/amazon_baseline__20260121T010203Z/run_summary.json",
        ),
    )
    _add_config_arguments(benchmark_report)
    benchmark_report.add_argument(
        "--run-summary",
        metavar="PATH",
        default=None,
        help="Path to benchmark run_summary.json. If omitted, CLI uses the latest run under benchmark_runs_dir.",
    )
    benchmark_report.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory where benchmark_report.md should be written.",
    )
    benchmark_report.add_argument(
        "--nde-metrics",
        metavar="PATH",
        default=None,
        help="Optional path to NDE evaluation_metrics.csv to include a Benchmark vs NDE comparison table.",
    )
    benchmark_report.add_argument(
        "--compare-run-summary",
        metavar="PATH",
        action="append",
        default=None,
        help="Optional additional run_summary.json paths to include multi-dataset benchmark comparison in one report.",
    )
    benchmark_report.set_defaults(handler=cmd_benchmark_report)

    benchmark_all = subparsers.add_parser(
        "benchmark-all",
        formatter_class=NDEHelpFormatter,
        help="Run benchmark-download, benchmark-run, and benchmark-report in one command.",
        description=dedent(
            """\
            End-to-end benchmark baseline flow: download + normalize, run VADER and LLM
            experiments, then generate the Markdown report.
            """
        ),
        epilog=_examples_block(
            "nde benchmark-all",
            "nde benchmark-all --max-rows 1500 --prompt-variant baseline_v2",
        ),
    )
    _add_config_arguments(benchmark_all)
    benchmark_all.add_argument("--max-rows", metavar="N", type=int, default=None, help="Override benchmark.dataset.max_rows.")
    benchmark_all.add_argument(
        "--prompt-variant",
        metavar="ID",
        default=None,
        help="Optional prompt variant under benchmark_prompt_variants_dir.",
    )
    benchmark_all.add_argument(
        "--raw-dir",
        metavar="PATH",
        default=None,
        help="Directory where raw benchmark download artifacts should be written.",
    )
    benchmark_all.add_argument(
        "--processed-dir",
        metavar="PATH",
        default=None,
        help="Directory where normalized benchmark CSV and manifest should be written.",
    )
    benchmark_all.add_argument(
        "--run-output-dir",
        metavar="PATH",
        default=None,
        help="Directory where benchmark run artifacts should be written.",
    )
    benchmark_all.add_argument(
        "--report-output-dir",
        metavar="PATH",
        default=None,
        help="Directory where benchmark_report.md should be written.",
    )
    benchmark_all.add_argument(
        "--nde-metrics",
        metavar="PATH",
        default=None,
        help="Optional path to NDE evaluation_metrics.csv to include a Benchmark vs NDE comparison table.",
    )
    benchmark_all.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete any existing resumable benchmark artifact in the target run output dir and rerun from zero.",
    )
    benchmark_all.set_defaults(handler=cmd_benchmark_all)

    evaluate = subparsers.add_parser(
        "evaluate",
        formatter_class=NDEHelpFormatter,
        help="Evaluate humans, questionnaire labels, VADER, and discovered LLM artifacts together.",
        description=dedent(
            """\
            Build the majority human reference and compare it against questionnaire-derived
            labels, VADER tone labels, and any accepted LLM artifacts discovered on disk.

            If the default VADER score file is missing, evaluation generates the sample-level
            VADER output automatically. LLM artifacts are discovered from llm_results_dir or
            an explicit override, even if those experiment folders are no longer present in
            the current paths config.
            """
        ),
        epilog=_examples_block(
            "nde evaluate",
            "nde evaluate --experiment-id smoke_qwen08",
            "nde evaluate --human-annotation-workbook data/human/ana.xlsx --llm-predictions data/llm/predictions.jsonl",
            "nde evaluate --llm-results-dir /data/nde/llm_outputs --output-dir /data/nde/evaluation_outputs",
        ),
    )
    _add_config_arguments(evaluate)
    human_group = evaluate.add_argument_group("Human Annotation Inputs")
    human_group.add_argument(
        "--human-annotation-workbook",
        metavar="PATH",
        default=None,
        help="Evaluate one explicit completed human workbook instead of discovering all workbooks in human_annotations_dir.",
    )
    human_group.add_argument(
        "--human-annotations-dir",
        metavar="PATH",
        default=None,
        help="Directory to scan for completed human annotation workbooks and optional manifests.",
    )
    llm_group = evaluate.add_argument_group("LLM Artifact Discovery")
    llm_group.add_argument(
        "--llm-predictions",
        metavar="PATH",
        default=None,
        help="Evaluate one explicit LLM predictions artifact instead of discovering artifacts in llm_results_dir.",
    )
    llm_group.add_argument(
        "--llm-results-dir",
        metavar="PATH",
        default=None,
        help="Directory to scan for LLM result artifacts; existing folders are evaluated even if the experiment is no longer in current config.",
    )
    filter_group = evaluate.add_argument_group("Filtering")
    filter_group.add_argument(
        "--annotator-id",
        metavar="ID",
        action="append",
        default=None,
        help="Restrict evaluation to one annotator id. Repeat the flag to include several annotators.",
    )
    filter_group.add_argument(
        "--experiment-id",
        metavar="ID",
        action="append",
        default=None,
        help="Restrict evaluation to selected experiment ids or artifact ids discovered on disk.",
    )
    auxiliary_group = evaluate.add_argument_group("Auxiliary Inputs And Output")
    auxiliary_group.add_argument(
        "--sampled-private-workbook",
        metavar="PATH",
        default=None,
        help="Override the sampled private workbook used for questionnaire alignment and VADER participant matching.",
    )
    auxiliary_group.add_argument(
        "--vader-scores",
        metavar="PATH",
        default=None,
        help="Optional path to an existing VADER scores file. If omitted and the default file is missing, evaluate generates it automatically.",
    )
    auxiliary_group.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory where evaluation tables, manifests, figures, and reports should be written.",
    )
    auxiliary_group.add_argument(
        "--figure-dpi",
        metavar="N",
        type=int,
        default=300,
        help="Resolution for saved evaluation figures in raster formats. Use higher values for manuscript-quality exports.",
    )
    auxiliary_group.add_argument(
        "--export-figures-pdf",
        action="store_true",
        help="Also save evaluation figures as PDF alongside the default PNG files for article-ready vector export.",
    )
    evaluate.set_defaults(handler=cmd_evaluate)

    return parser


def _ensure_output_locations(paths_config) -> list[str]:
    created: list[str] = []
    for path in (
        paths_config.annotation_output_dir,
        paths_config.llm_batch_dir,
        paths_config.evaluation_output_dir,
        paths_config.preprocessing_output_dir,
        paths_config.human_annotations_dir,
        paths_config.llm_results_dir,
        paths_config.sampled_private_workbook.parent,
        paths_config.human_annotation_workbook.parent,
        paths_config.llm_predictions_path.parent,
        paths_config.benchmark_raw_dir,
        paths_config.benchmark_processed_dir,
        paths_config.benchmark_runs_dir,
        paths_config.benchmark_reports_dir,
        paths_config.benchmark_prompt_variants_dir,
    ):
        if path is None:
            continue
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path))
    return created


def cmd_validate_config(args: argparse.Namespace) -> int:
    study = load_study_config(args.study_config)
    paths = load_paths_config(args.paths_config)
    llm_config = load_llm_config(args.paths_config)

    placeholders = study.placeholder_questionnaire_columns()
    if placeholders:
        raise ValueError(f"Replace questionnaire column placeholders before validation: {placeholders}")

    if not paths.survey_csv.exists():
        raise FileNotFoundError(f"Survey source not found: {paths.survey_csv}")

    survey_df = read_tabular_file(paths.survey_csv, nrows=5)
    missing_columns = [column for column in study.required_source_columns() if column not in survey_df.columns]
    if missing_columns:
        raise ValueError(f"Survey source is missing required columns: {missing_columns}")

    created_dirs = _ensure_output_locations(paths)

    summary = {
        "study_config": str(Path(args.study_config).resolve()),
        "paths_config": str(Path(args.paths_config).resolve()),
        "survey_source": str(paths.survey_csv),
        "required_columns_checked": len(study.required_source_columns()),
        "created_output_directories": created_dirs,
        "llm_runtime": llm_config.runtime.to_dict(),
        "llm_experiments": [experiment.to_dict() for experiment in llm_config.experiments],
    }
    print("Configuration valid.")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_build_annotation_sample(args: argparse.Namespace) -> int:
    study = load_study_config(args.study_config)
    paths = load_paths_config(args.paths_config)
    preprocessed_path = paths.preprocessing_output_dir / PREPROCESSED_DATASET_FILENAME
    source_path = paths.survey_csv
    source_df = read_tabular_file(source_path)

    if preprocessed_path.exists():
        preprocessed_df = read_tabular_file(preprocessed_path)
        required_columns = [study.id_column, study.stratify_column, *study.text_columns().values()]
        if all(column in preprocessed_df.columns for column in required_columns):
            source_path = preprocessed_path
            source_df = preprocessed_df

    annotation_df, mapping_df, column_map_df, sampled_private_df, summary = create_annotation_frames(
        source_df=source_df,
        study=study,
        n_total=args.n_total,
        random_state=args.random_state,
    )
    written = write_annotation_outputs(
        annotation_df=annotation_df,
        mapping_df=mapping_df,
        column_map_df=column_map_df,
        sampled_private_df=sampled_private_df,
        study=study,
        paths=paths,
        force=args.force,
    )
    print(json.dumps({**summary, "source_path": str(source_path), **written}, indent=2))
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .preprocessing import run_preprocessing_pipeline

    study = load_study_config(args.study_config)
    paths = load_paths_config(args.paths_config)
    preprocessing = load_preprocessing_config(args.paths_config)
    summary = run_preprocessing_pipeline(
        study=study,
        paths=paths,
        preprocessing=preprocessing,
        input_path=Path(args.input_path).resolve() if args.input_path else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        limit=args.limit,
        all_records=bool(args.all_records),
        retry_exhausted=bool(args.retry_exhausted),
        from_scratch=bool(args.from_scratch),
        generate_validation_sample=bool(args.generate_validation_sample),
        validation_n_total=args.validation_n_total,
        validation_random_state=args.validation_random_state,
        force_validation_sample=bool(args.force_validation_sample),
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_build_llm_batch(args: argparse.Namespace) -> int:
    study = load_study_config(args.study_config)
    paths = load_paths_config(args.paths_config)
    written = write_llm_batches(
        study=study,
        paths=paths,
        source=args.source,
        input_path=Path(args.input_path).resolve() if args.input_path else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        limit=args.limit,
        all_records=args.all_records,
        experiment_id=args.experiment_id,
        prompt_variant=args.prompt_variant,
        run_id=args.run_id,
        model_variant=args.model_variant,
        prompt_root=Path(args.prompt_root).resolve() if args.prompt_root else None,
    )
    print(json.dumps(written, indent=2))
    return 0


def cmd_run_llm(args: argparse.Namespace) -> int:
    from .llm_runner import run_llm_experiments

    study = load_study_config(args.study_config)
    paths = load_paths_config(args.paths_config)
    llm_config = load_llm_config(args.paths_config)
    summary = run_llm_experiments(
        study=study,
        paths=paths,
        llm_config=llm_config,
        experiment_ids=list(args.experiment_id) if args.experiment_id else None,
        all_experiments=bool(args.all_experiments),
        input_path=Path(args.input_path).resolve() if args.input_path else None,
        limit=args.limit,
        all_records=True if args.all_records else None,
        min_valid_sections=args.min_valid_sections,
        retry_exhausted=bool(args.retry_exhausted),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_sentiment_sensitivity(args: argparse.Namespace) -> int:
    from .vader_analysis import run_vader_sensitivity

    study = load_study_config(args.study_config)
    paths = load_paths_config(args.paths_config)
    scores_df, summary, written = run_vader_sensitivity(
        study,
        paths,
        input_path=Path(args.input_path).resolve() if args.input_path else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        all_records=args.all_records,
        quality_values=list(args.quality_values) if args.quality_values else None,
        limit=args.limit,
        include_text=args.include_text,
    )
    print(json.dumps({"rows": len(scores_df), "summary": summary, **written}, indent=2))
    return 0


def _latest_benchmark_run_summary(paths_config) -> Path:
    runs_dir = Path(paths_config.benchmark_runs_dir or (paths_config.evaluation_output_dir / "benchmark_runs"))
    if not runs_dir.exists():
        raise FileNotFoundError(f"Benchmark runs directory not found: {runs_dir}")
    candidates = [candidate / "run_summary.json" for candidate in runs_dir.iterdir() if candidate.is_dir()]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        raise FileNotFoundError(f"No benchmark run_summary.json found under: {runs_dir}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def cmd_benchmark_download(args: argparse.Namespace) -> int:
    from .benchmark import download_and_prepare_benchmark_dataset

    paths = load_paths_config(args.paths_config)
    benchmark = load_benchmark_config(args.paths_config)
    benchmark = replace(benchmark, dataset=benchmark.datasets[0])
    dataset_df, written, summary = download_and_prepare_benchmark_dataset(
        paths=paths,
        benchmark=benchmark,
        max_rows=args.max_rows,
        output_raw_dir=Path(args.raw_dir).resolve() if args.raw_dir else None,
        output_processed_dir=Path(args.processed_dir).resolve() if args.processed_dir else None,
    )
    print(
        json.dumps(
            {
                "rows": int(len(dataset_df)),
                "raw_file": str(written.raw_file),
                "processed_file": str(written.processed_file),
                "manifest_file": str(written.manifest_file),
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0


def cmd_benchmark_run(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark_pipeline

    paths = load_paths_config(args.paths_config)
    benchmark = load_benchmark_config(args.paths_config)
    summary = run_benchmark_pipeline(
        paths=paths,
        benchmark=benchmark,
        dataset_path=Path(args.dataset_path).resolve() if args.dataset_path else None,
        run_output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        artifact_prefix=None,
        prompt_variant=args.prompt_variant,
        max_rows=args.max_rows,
        resume=not bool(args.from_scratch),
        from_scratch=bool(args.from_scratch),
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_benchmark_report(args: argparse.Namespace) -> int:
    from .benchmark import write_benchmark_report

    paths = load_paths_config(args.paths_config)
    run_summary = Path(args.run_summary).resolve() if args.run_summary else _latest_benchmark_run_summary(paths)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(paths.benchmark_reports_dir)
    report_path = write_benchmark_report(
        run_summary,
        output_dir=output_dir,
        nde_metrics_path=Path(args.nde_metrics).resolve() if args.nde_metrics else None,
        comparison_run_summaries=[Path(path).resolve() for path in args.compare_run_summary]
        if args.compare_run_summary
        else None,
    )
    figure_path = output_dir / "figures" / "benchmark_macro_f1_vs_kappa.png"
    payload = {"run_summary": str(run_summary), "report_file": str(report_path)}
    if figure_path.exists():
        payload["figure_file"] = str(figure_path)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_benchmark_all(args: argparse.Namespace) -> int:
    from .benchmark import download_and_prepare_benchmark_dataset, run_benchmark_pipeline, write_benchmark_report

    paths = load_paths_config(args.paths_config)
    benchmark = load_benchmark_config(args.paths_config)
    configured_datasets = benchmark.datasets if benchmark.datasets else [benchmark.dataset]
    run_root_base = (
        Path(args.run_output_dir).resolve()
        if args.run_output_dir
        else Path(paths.benchmark_runs_dir or (paths.evaluation_output_dir / "benchmark_runs")).resolve()
    )

    per_dataset_rows: list[dict[str, object]] = []
    summary_paths: list[Path] = []

    for dataset_cfg in configured_datasets:
        dataset_benchmark = replace(benchmark, dataset=dataset_cfg)
        dataset_slug = re.sub(r"[^a-z0-9]+", "_", dataset_cfg.dataset_name.lower()).strip("_") or "dataset"

        dataset_df, written, download_summary = download_and_prepare_benchmark_dataset(
            paths=paths,
            benchmark=dataset_benchmark,
            max_rows=args.max_rows,
            output_raw_dir=Path(args.raw_dir).resolve() if args.raw_dir else None,
            output_processed_dir=Path(args.processed_dir).resolve() if args.processed_dir else None,
        )
        run_summary = run_benchmark_pipeline(
            paths=paths,
            benchmark=dataset_benchmark,
            dataset_path=written.processed_file,
            run_output_dir=run_root_base / dataset_slug,
            artifact_prefix=f"{dataset_slug}_baseline",
            prompt_variant=args.prompt_variant,
            max_rows=args.max_rows,
            resume=not bool(args.from_scratch),
            from_scratch=bool(args.from_scratch),
        )
        summary_path = Path(run_summary["summary_file"])  # type: ignore[index]
        summary_paths.append(summary_path)
        per_dataset_rows.append(
            {
                "dataset_name": dataset_cfg.dataset_name,
                "rows": int(len(dataset_df)),
                "download": {
                    "raw_file": str(written.raw_file),
                    "processed_file": str(written.processed_file),
                    "manifest_file": str(written.manifest_file),
                    "summary": download_summary,
                },
                "run": run_summary,
            }
        )

    if not summary_paths:
        raise ValueError("No benchmark datasets configured. Define [benchmark.dataset] or [[benchmark.datasets]].")

    report_path = write_benchmark_report(
        summary_paths[0],
        output_dir=Path(args.report_output_dir).resolve() if args.report_output_dir else Path(paths.benchmark_reports_dir),
        nde_metrics_path=Path(args.nde_metrics).resolve() if args.nde_metrics else None,
        comparison_run_summaries=summary_paths[1:] if len(summary_paths) > 1 else None,
    )
    report_output_dir = Path(args.report_output_dir).resolve() if args.report_output_dir else Path(paths.benchmark_reports_dir)
    figure_path = report_output_dir / "figures" / "benchmark_macro_f1_vs_kappa.png"
    response_payload = {
        "datasets": per_dataset_rows,
        "report_file": str(report_path),
    }
    if figure_path.exists():
        response_payload["figure_file"] = str(figure_path)
    print(
        json.dumps(
            response_payload,
            indent=2,
        )
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_outputs

    study = load_study_config(args.study_config)
    paths = load_paths_config(args.paths_config)
    metrics_df, summary, written = evaluate_outputs(
        study=study,
        paths=paths,
        human_annotation_workbook=Path(args.human_annotation_workbook).resolve() if args.human_annotation_workbook else None,
        human_annotations_dir=Path(args.human_annotations_dir).resolve() if args.human_annotations_dir else None,
        llm_predictions_path=Path(args.llm_predictions).resolve() if args.llm_predictions else None,
        llm_results_dir=Path(args.llm_results_dir).resolve() if args.llm_results_dir else None,
        annotator_ids=list(args.annotator_id) if args.annotator_id else None,
        experiment_ids=list(args.experiment_id) if args.experiment_id else None,
        sampled_private_workbook=Path(args.sampled_private_workbook).resolve() if args.sampled_private_workbook else None,
        vader_scores_path=Path(args.vader_scores).resolve() if args.vader_scores else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        figure_dpi=int(args.figure_dpi),
        export_figures_pdf=bool(args.export_figures_pdf),
    )
    print(json.dumps({"rows": len(metrics_df), "summary": summary, **written}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
