from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nde_narratives.benchmark import compute_metrics

from tests.cli_helpers import FIXTURES, make_paths_config, run_cli


def _load_json(stdout: str) -> dict:
    return json.loads(stdout)


def test_benchmark_run_with_local_dataset_generates_metrics_artifacts(
    tmp_path: Path,
) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    benchmark_csv = FIXTURES / "benchmark_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    result = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(tmp_path / "benchmark_runs"),
    )

    assert result.returncode == 0, result.stderr
    payload = _load_json(result.stdout)

    summary_path = Path(payload["summary_file"])
    metrics_path = Path(payload["metrics_file"])
    confusion_path = Path(payload["confusion_file"])
    per_label_path = Path(payload["per_label_file"])

    assert summary_path.exists()
    assert metrics_path.exists()
    assert confusion_path.exists()
    assert per_label_path.exists()

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["source"]) == {"vader"}
    assert {"accuracy", "macro_f1", "cohen_kappa"}.issubset(set(metrics.columns))


def test_benchmark_report_sections_follow_required_order(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    benchmark_csv = FIXTURES / "benchmark_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    run_result = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(tmp_path / "benchmark_runs"),
    )
    assert run_result.returncode == 0, run_result.stderr
    run_payload = _load_json(run_result.stdout)

    report_result = run_cli(
        "benchmark-report",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--run-summary",
        str(run_payload["summary_file"]),
        "--output-dir",
        str(tmp_path / "benchmark_reports"),
    )
    assert report_result.returncode == 0, report_result.stderr
    report_payload = _load_json(report_result.stdout)
    report_path = Path(report_payload["report_file"])
    assert report_path.exists()
    assert "figure_file" in report_payload
    figure_path = Path(report_payload["figure_file"])
    assert figure_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    section_order = [
        "## Source",
        "## Methodology",
        "## Prompts",
        "## Metrics",
        "## Visual Summary",
        "## Interpretation",
        "## Limitations",
    ]
    positions = [report_text.index(section) for section in section_order]
    assert positions == sorted(positions)
    assert (
        "![Benchmark macro_f1 vs cohen_kappa](figures/benchmark_macro_f1_vs_kappa.png)"
        in report_text
    )
    assert (
        "Marker color encodes dataset, marker shape encodes model identity (aligned with the principal figure), and compact mini bars"
        in report_text
    )
    assert "- Dataset: benchmark_fixture" in report_text


def test_benchmark_report_can_include_benchmark_vs_nde_table(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    benchmark_csv = FIXTURES / "benchmark_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    nde_metrics = pd.DataFrame(
        [
            {
                "comparison": "human_reference_vs_vader",
                "field": "experience_tone",
                "macro_f1": 0.41,
            },
            {
                "comparison": "human_reference_vs_llm:exp_a",
                "field": "experience_tone",
                "macro_f1": 0.56,
            },
        ]
    )
    nde_metrics_path = tmp_path / "evaluation_metrics.csv"
    nde_metrics.to_csv(nde_metrics_path, index=False)

    run_result = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(tmp_path / "benchmark_runs"),
    )
    assert run_result.returncode == 0, run_result.stderr
    run_payload = _load_json(run_result.stdout)

    report_result = run_cli(
        "benchmark-report",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--run-summary",
        str(run_payload["summary_file"]),
        "--output-dir",
        str(tmp_path / "benchmark_reports"),
        "--nde-metrics",
        str(nde_metrics_path),
    )
    assert report_result.returncode == 0, report_result.stderr
    report_payload = _load_json(report_result.stdout)
    report_text = Path(report_payload["report_file"]).read_text(encoding="utf-8")
    assert "### Benchmark vs NDE" in report_text
    assert "figure_file" in report_payload
    assert Path(report_payload["figure_file"]).exists()


def test_benchmark_report_supports_comparison_run_summaries(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    benchmark_csv = FIXTURES / "benchmark_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)

    amazon_run = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(tmp_path / "benchmark_runs_amazon"),
    )
    assert amazon_run.returncode == 0, amazon_run.stderr
    amazon_payload = _load_json(amazon_run.stdout)

    imdb_run = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(tmp_path / "benchmark_runs_imdb"),
    )
    assert imdb_run.returncode == 0, imdb_run.stderr
    imdb_payload = _load_json(imdb_run.stdout)

    imdb_summary_path = Path(imdb_payload["summary_file"])
    imdb_summary = json.loads(imdb_summary_path.read_text(encoding="utf-8"))
    imdb_summary["dataset"]["dataset_name"] = "imdb"
    imdb_summary_path.write_text(json.dumps(imdb_summary, indent=2), encoding="utf-8")

    report_result = run_cli(
        "benchmark-report",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--run-summary",
        str(amazon_payload["summary_file"]),
        "--compare-run-summary",
        str(imdb_payload["summary_file"]),
        "--output-dir",
        str(tmp_path / "benchmark_reports"),
    )
    assert report_result.returncode == 0, report_result.stderr
    report_payload = _load_json(report_result.stdout)
    report_text = Path(report_payload["report_file"]).read_text(encoding="utf-8")

    assert "Compared datasets in report" in report_text
    assert "imdb" in report_text
    assert "Overall evidence level for this benchmark" in report_text


def test_compute_metrics_returns_expected_keys() -> None:
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "negative", "positive", "neutral"]
    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) >= {
        "n",
        "accuracy",
        "macro_f1",
        "cohen_kappa",
        "per_label",
        "confusion",
    }
    assert metrics["n"] == 4
    assert len(metrics["confusion"]) == 9


def test_benchmark_run_resumes_and_from_scratch_resets_artifact(tmp_path: Path) -> None:
    study_config = FIXTURES / "study_test.toml"
    survey_csv = FIXTURES / "survey_fixture.csv"
    benchmark_csv = FIXTURES / "benchmark_fixture.csv"
    paths_config = make_paths_config(tmp_path, survey_csv)
    run_root = tmp_path / "benchmark_runs"

    first = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(run_root),
    )
    assert first.returncode == 0, first.stderr
    first_payload = _load_json(first.stdout)
    artifact_dir = Path(first_payload["artifact_dir"])
    marker = artifact_dir / "resume_marker.txt"
    marker.write_text("keep-me", encoding="utf-8")

    second = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(run_root),
    )
    assert second.returncode == 0, second.stderr
    second_payload = _load_json(second.stdout)
    assert Path(second_payload["artifact_dir"]) == artifact_dir
    assert marker.exists()

    reset = run_cli(
        "benchmark-run",
        "--study-config",
        str(study_config),
        "--paths-config",
        str(paths_config),
        "--dataset-path",
        str(benchmark_csv),
        "--output-dir",
        str(run_root),
        "--from-scratch",
    )
    assert reset.returncode == 0, reset.stderr
    assert not marker.exists()
