from __future__ import annotations

import pytest

from pathlib import Path

from nde_narratives.config import LLMRuntimeConfig, PreprocessingConfig, load_llm_config
from nde_narratives.llm.factory import build_llm_provider


def test_load_llm_config_parses_bedrock_runtime_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "paths.local.toml"
    config_path.write_text(
        """[paths]
data_dir = "/tmp/nde-data"
survey_csv = "/tmp/nde-data/survey.csv"

[llm]
provider = "bedrock"
aws_region = "us-east-1"
aws_profile = "default"
timeout_seconds = 45
max_attempts = 3
temperature = 0.1
max_tokens = 700
top_p = 0.95
top_k = 250
stop_sequences = ["</END>"]
source = "survey"
all_records = false

[[llm.experiments]]
experiment_id = "exp-bedrock"
enabled = true
model = "anthropic.claude-3-haiku-20240307-v1:0"
run_id = "run-01"
""",
        encoding="utf-8",
    )

    llm_config = load_llm_config(config_path)

    assert llm_config.runtime.provider == "bedrock"
    assert llm_config.runtime.aws_region == "us-east-1"
    assert llm_config.runtime.aws_profile == "default"
    assert llm_config.runtime.max_tokens == 700
    assert llm_config.runtime.top_p == 0.95
    assert llm_config.runtime.top_k == 250
    assert llm_config.runtime.stop_sequences == ["</END>"]


def test_build_llm_provider_routes_to_bedrock(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeBedrockProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "nde_narratives.llm.factory.BedrockProvider", FakeBedrockProvider
    )

    runtime = LLMRuntimeConfig(
        provider="bedrock",
        aws_region="us-east-1",
        aws_profile="default",
        timeout_seconds=60,
        temperature=0.0,
        max_tokens=512,
        top_p=0.9,
        top_k=100,
        stop_sequences=["END"],
    )

    provider = build_llm_provider(runtime)
    assert isinstance(provider, FakeBedrockProvider)
    assert captured["region_name"] == "us-east-1"
    assert captured["aws_profile"] == "default"
    assert captured["max_tokens"] == 512


def test_build_llm_provider_routes_preprocessing_bedrock(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeBedrockProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "nde_narratives.llm.factory.BedrockProvider", FakeBedrockProvider
    )

    runtime = PreprocessingConfig(
        path=Path("/tmp/paths.local.toml"),
        provider="bedrock",
        aws_region="us-west-2",
        aws_profile="dev",
        timeout_seconds=75,
        temperature=0.15,
        max_tokens=1024,
        top_p=0.9,
        top_k=50,
        stop_sequences=["</stop>"],
    )
    provider = build_llm_provider(runtime)

    assert isinstance(provider, FakeBedrockProvider)
    assert captured["region_name"] == "us-west-2"
    assert captured["aws_profile"] == "dev"
    assert captured["timeout_seconds"] == 75
    assert captured["temperature"] == 0.15
    assert captured["max_tokens"] == 1024
    assert captured["top_p"] == 0.9
    assert captured["top_k"] == 50
    assert captured["stop_sequences"] == ["</stop>"]


def test_load_llm_config_rejects_string_stop_sequences(tmp_path: Path) -> None:
    config_path = tmp_path / "paths.local.toml"
    config_path.write_text(
        """[paths]
data_dir = "/tmp/nde-data"
survey_csv = "/tmp/nde-data/survey.csv"

[llm]
provider = "bedrock"
stop_sequences = "</END>"

[[llm.experiments]]
experiment_id = "exp-bedrock"
model = "anthropic.claude-3-haiku-20240307-v1:0"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="llm.stop_sequences must be a TOML array"):
        load_llm_config(config_path)
