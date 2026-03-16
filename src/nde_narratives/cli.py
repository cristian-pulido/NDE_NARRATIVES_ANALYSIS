from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import default_paths_config_path, default_study_config_path, load_paths_config, load_study_config
from .evaluation import evaluate_outputs
from .excel import write_annotation_outputs
from .io_utils import read_tabular_file
from .prompting import write_llm_batches
from .sampling import create_annotation_frames
from .vader_analysis import run_vader_sensitivity


def _config_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--study-config", default=str(default_study_config_path()))
    parent.add_argument("--paths-config", default=str(default_paths_config_path()))
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nde", description="NDE narratives research CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", parents=[_config_parent()], help="Validate study and path configs.")
    validate.set_defaults(handler=cmd_validate_config)

    build_annotation = subparsers.add_parser(
        "build-annotation-sample",
        parents=[_config_parent()],
        help="Generate the external annotation workbook sample.",
    )
    build_annotation.add_argument("--n-total", type=int, default=None)
    build_annotation.add_argument("--random-state", type=int, default=None)
    build_annotation.set_defaults(handler=cmd_build_annotation_sample)

    build_batch = subparsers.add_parser(
        "build-llm-batch",
        parents=[_config_parent()],
        help="Generate JSONL batches for the three narrative sections.",
    )
    build_batch.add_argument("--source", choices=["sampled-private", "survey"], default="sampled-private")
    build_batch.add_argument("--input-path", default=None)
    build_batch.add_argument("--output-dir", default=None)
    build_batch.add_argument("--limit", type=int, default=None)
    build_batch.set_defaults(handler=cmd_build_llm_batch)

    sentiment = subparsers.add_parser(
        "sentiment-sensitivity",
        parents=[_config_parent()],
        help="Run VADER sentiment sensitivity analysis across narrative text columns.",
    )
    sentiment.add_argument("--input-path", default=None)
    sentiment.add_argument("--output-dir", default=None)
    sentiment.add_argument("--all-records", action="store_true")
    sentiment.add_argument("--quality-value", dest="quality_values", action="append", default=None)
    sentiment.add_argument("--limit", type=int, default=None)
    sentiment.add_argument("--include-text", action="store_true")
    sentiment.set_defaults(handler=cmd_sentiment_sensitivity)

    evaluate = subparsers.add_parser(
        "evaluate",
        parents=[_config_parent()],
        help="Compare human annotations, LLM predictions, questionnaire-derived labels, and VADER tone labels.",
    )
    evaluate.add_argument("--human-annotation-workbook", default=None)
    evaluate.add_argument("--llm-predictions", default=None)
    evaluate.add_argument("--sampled-private-workbook", default=None)
    evaluate.add_argument("--vader-scores", default=None)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.set_defaults(handler=cmd_evaluate)

    return parser


def _ensure_output_locations(paths_config) -> list[str]:
    created: list[str] = []
    for path in (
        paths_config.annotation_output_dir,
        paths_config.llm_batch_dir,
        paths_config.evaluation_output_dir,
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
    )
    print(json.dumps(written, indent=2))
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
        llm_predictions_path=Path(args.llm_predictions).resolve() if args.llm_predictions else None,
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
