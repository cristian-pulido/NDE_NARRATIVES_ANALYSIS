from nde_narratives.evaluation_report import _presentation_model_name


def test_presentation_model_name_handles_qwen36_run_suffix() -> None:
    assert _presentation_model_name("qwen36_27__01") == "Qwen 3.6 27B"


def test_presentation_model_name_handles_qwen35() -> None:
    assert _presentation_model_name("qwen35_35__01") == "Qwen 3.5 35B"


def test_presentation_model_name_handles_known_aliases() -> None:
    assert _presentation_model_name("llama31_8") == "Llama 3.1 8B"
    assert _presentation_model_name("claude35_sonnet") == "Claude 3.5 Sonnet"


def test_presentation_model_name_keeps_reasonable_fallback() -> None:
    assert _presentation_model_name("custom_model_x") == "Custom Model X"
