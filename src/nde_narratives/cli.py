from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from textwrap import dedent

from .config import (
    default_paths_config_path,
    default_study_config_path,
    load_llm_config,
    load_paths_config,
    load_study_config,
)
from .constants import PROJECT_ROOT
from .evaluation import evaluate_outputs
from .excel import write_annotation_outputs
from .io_utils import read_tabular_file
from .llm_runner import run_llm_experiments
from .prompting import write_llm_batches
from .sampling import create_annotation_frames
from .vader_analysis import run_vader_sensitivity


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
    evaluate.set_defaults(handler=cmd_evaluate)

    return parser


def _ensure_output_locations(paths_config) -> list[str]:
    created: list[str] = []
    for path in (
        paths_config.annotation_output_dir,
        paths_config.llm_batch_dir,
        paths_config.evaluation_output_dir,
        paths_config.human_annotations_dir,
        paths_config.llm_results_dir,
        paths_config.sampled_private_workbook.parent,
        paths_config.human_annotation_workbook.parent,
        paths_config.llm_predictions_path.parent,
    ):
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
    source_df = read_tabular_file(paths.survey_csv)

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
    print(json.dumps({**summary, **written}, indent=2))
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
        retry_exhausted=bool(args.retry_exhausted),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_sentiment_sensitivity(args: argparse.Namespace) -> int:
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


def cmd_evaluate(args: argparse.Namespace) -> int:
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
