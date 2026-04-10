from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import (
    load_llm_config,
    load_paths_config,
    load_preprocessing_config,
    load_study_config,
)
from .constants import PROJECT_ROOT
from .interactive import (
    analyze_single_narrative,
    analyze_three_sections,
    build_evidence_summary_markdown,
    configured_model_fallbacks,
    list_ollama_models,
)


SECTION_LABELS = {
    "context": "Before the NDE",
    "experience": "Core NDE Experience",
    "aftereffects": "Aftereffects",
}

CONTEXT_NATURE_LABELS = {
    "no_death_context": "No clear death-related context",
    "subjective_threat_only": "Subjective threat only",
    "objective_medical_context": "Objective medical context",
}

LABEL_DISPLAY_NAMES = {
    "m8_out_of_body": "Out-of-body sensation",
    "m8_bright_light": "Bright light perception",
    "m8_peace": "Sense of peace",
    "m8_time_distortion": "Time distortion",
    "m8_presence": "Perceived presence",
    "m9_moral_rules": "Change in moral orientation",
    "m9_long_term_thinking": "Long-term thinking",
    "m9_consider_others": "Considering others",
    "m9_help_others": "Helping others",
    "m9_forgiveness": "Forgiveness",
}

VALUE_DISPLAY_NAMES = {
    "yes": "Yes",
    "no": "No",
    "positive": "Positive",
    "negative": "Negative",
    "mixed": "Mixed",
    "neutral": "Neutral",
}

DISCLAIMER_MD = """
## Important Disclaimer

- This tool is for research and educational use only; it is not a medical or psychological diagnostic system.
- Do not submit personally identifying or highly sensitive information.
- Model outputs may be inaccurate, incomplete, or biased and require human review.
""".strip()


INTRO_MD = """
## Why This Problem Matters

Near-death experiences can be represented in two different but complementary ways:

- **Narratives** (rich, temporal, contextual first-person text)
- **Questionnaire-style labels** (standardized, structured categories)

These representations do not always align perfectly. A narrative may contain local fear, confusion, or ambiguity while the overall questionnaire appraisal is later positive or meaningful.

This interface is designed to make that representational mismatch visible through transparent stages instead of a black-box prediction.
""".strip()


RESEARCH_QUESTIONS_MD = """
## Core Questions From the Study

1. How much do narrative-derived signals align with questionnaire-style labels?
2. Is narrative tone enough to recover self-reported valence?
3. Are concrete experiential features easier to recover than reflective aftereffects?
4. Is disagreement mainly model error, or a representational mismatch between formats?
""".strip()


WHAT_WAS_DONE_MD = """
## What This Demo Reproduces

- Section-aware analysis of **context**, **core experience**, and **aftereffects**.
- LLM-based extraction of tone, contextual framing, and structured features.
- Evidence-grounded outputs so you can inspect *why* a signal was detected.
- Optional comparison between your provided valence and model-derived overall tone.
""".strip()


DEMO_PURPOSE_MD = """
## Demo Purpose

This is a practical demonstration of how LLMs extract structured information from narratives.

Use it to compare:
- what the model extracts from your text, and
- what you would extract manually from the same narrative.
""".strip()


KEY_ELEMENTS_MD = """
## Key Elements You Can Inspect

- **Segmentation quality:** how input is split into context, experience, and aftereffects.
- **Tone recovery:** what emotional signal appears in each section and globally.
- **Structured signal extraction:** what perceptual features and reflective aftereffects are detected and where evidence appears.
- **Alignment layer:** how optional valence compares against overall experience-weighted tone.
- **Interpretation boundaries:** what appears recoverable and what remains ambiguous.
""".strip()


USER_GUIDE_MD = """
## How To Use This Demo

1. Choose an input path:
   - **Guided mode**: paste each section manually.
   - **Complex mode**: paste one full narrative and let the app re-segment it.
2. (Optional) add a valence label if you want an explicit alignment check.
3. Click **Run Analysis**.
4. Review each stage and tab, then compare model output with your own reading.
""".strip()


STAGES_MD = """
## Pipeline Stages

1. **Input**: provide either one narrative (complex mode) or three sections (guided mode).
2. **Segmentation**: inspect the three sections actually used for inference.
3. **Module Analysis**: review tone, structured features, and optional valence alignment.
4. **Interpretation**: understand recoverable signals, uncertainty, and limits versus questionnaire-style labels.
""".strip()


VIDEO_SUMMARY_MD = """
## Results Overview Video

This short video summarizes key findings and the narrative-versus-questionnaire perspective.
""".strip()


def _display_value(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        return ""
    return VALUE_DISPLAY_NAMES.get(normalized, str(value).strip())


def _display_section(section_name: str) -> str:
    return SECTION_LABELS.get(section_name, section_name.replace("_", " ").title())


def _display_label(label_key: str) -> str:
    return LABEL_DISPLAY_NAMES.get(label_key, label_key.replace("_", " ").title())


def _extract_section_payload(
    predictions: dict[str, dict[str, Any]], section_name: str
) -> dict[str, Any]:
    payload = predictions.get(section_name, {})
    section_payload = payload.get(section_name, {}) if isinstance(payload, dict) else {}
    if isinstance(section_payload, dict):
        return section_payload
    return {}


def _tone_by_section(predictions: dict[str, dict[str, Any]]) -> dict[str, str]:
    tones: dict[str, str] = {}
    for section_name in ("context", "experience", "aftereffects"):
        section_payload = _extract_section_payload(predictions, section_name)
        tones[section_name] = str(section_payload.get("tone", "")).strip().lower()
    return tones


def _overall_experience_tone(predictions: dict[str, dict[str, Any]]) -> str:
    tones = _tone_by_section(predictions)
    weighted_tones: list[str] = []

    if tones.get("experience"):
        weighted_tones.extend([tones["experience"], tones["experience"]])
    if tones.get("context"):
        weighted_tones.append(tones["context"])
    if tones.get("aftereffects"):
        weighted_tones.append(tones["aftereffects"])

    if not weighted_tones:
        return "unknown"

    counts = Counter(weighted_tones)
    top_tone, top_count = counts.most_common(1)[0]
    if list(counts.values()).count(top_count) > 1:
        return "mixed"
    return top_tone


def _global_tone_markdown(predictions: dict[str, dict[str, Any]]) -> str:
    tones = _tone_by_section(predictions)
    overall = _overall_experience_tone(predictions)
    return (
        "### Tone Snapshot\n"
        f"- Before the NDE: {_display_value(tones.get('context', 'unknown')) or 'Unknown'}\n"
        f"- Core NDE Experience: {_display_value(tones.get('experience', 'unknown')) or 'Unknown'}\n"
        f"- Aftereffects: {_display_value(tones.get('aftereffects', 'unknown')) or 'Unknown'}\n"
        f"- Overall experience-weighted tone: {_display_value(overall) or 'Unknown'}"
    )


def _alignment_markdown(predictions: dict[str, dict[str, Any]], valence: str) -> str:
    normalized_valence = str(valence).strip().lower()
    if not normalized_valence:
        return "No optional valence was provided, so no alignment check was executed."

    overall_tone = _overall_experience_tone(predictions)
    evidence_lines: list[str] = []
    for section_name in ("context", "experience", "aftereffects"):
        section_payload = _extract_section_payload(predictions, section_name)
        evidence = section_payload.get("evidence_segments", [])
        if isinstance(evidence, list):
            for segment in evidence:
                evidence_lines.append(f"- {_display_section(section_name)}: {segment}")

    evidence_block = (
        "\n".join(evidence_lines)
        if evidence_lines
        else "- No evidence segments were returned."
    )

    if overall_tone == normalized_valence:
        return (
            f"The provided valence matches the overall experience tone ({_display_value(overall_tone)}).\n\n"
            f"Evidence used:\n{evidence_block}"
        )

    return (
        f"The provided valence does not match the overall experience tone "
        f"(valence={_display_value(normalized_valence)}, tone={_display_value(overall_tone)}).\n\n"
        f"Evidence used:\n{evidence_block}"
    )


def _interpretation_markdown(predictions: dict[str, dict[str, Any]]) -> str:
    strong_sections: list[str] = []
    uncertain_sections: list[str] = []

    for section_name in ("context", "experience", "aftereffects"):
        payload = _extract_section_payload(predictions, section_name)
        evidence = payload.get("evidence_segments", [])
        tone = str(payload.get("tone", "")).strip().lower()
        if isinstance(evidence, list) and evidence and tone and tone != "neutral":
            strong_sections.append(_display_section(section_name))
        else:
            uncertain_sections.append(_display_section(section_name))

    strong_text = ", ".join(strong_sections) if strong_sections else "none"
    uncertain_text = ", ".join(uncertain_sections) if uncertain_sections else "none"

    return (
        "### Interpretation\n"
        f"- What seems recoverable from narrative text: {strong_text}.\n"
        f"- What remains ambiguous or weakly expressed: {uncertain_text}.\n"
        "- This output is an interpretation layer and is not equivalent to questionnaire ground truth.\n"
        "- Narrative and questionnaire representations can align partially while still diverging in important details."
    )


def _labels_table_rows(predictions: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for section_name in ("context", "experience", "aftereffects"):
        section_payload = _extract_section_payload(predictions, section_name)
        tone_value = str(section_payload.get("tone", "")).strip()
        if tone_value:
            rows.append(
                [
                    _display_section(section_name),
                    "Tone",
                    _display_value(tone_value),
                ]
            )

        if section_name == "context":
            raw_context_nature = str(
                section_payload.get("death_context_nature", "")
            ).strip()
            if raw_context_nature:
                rows.append(
                    [
                        _display_section(section_name),
                        "Context Type",
                        CONTEXT_NATURE_LABELS.get(
                            raw_context_nature, raw_context_nature
                        ),
                    ]
                )

        evidence = section_payload.get("evidence_segments", [])
        if isinstance(evidence, list) and evidence:
            rows.append(
                [
                    _display_section(section_name),
                    "Supporting Evidence",
                    " | ".join(str(item) for item in evidence),
                ]
            )

        for key, value in section_payload.items():
            if key in {"tone", "evidence_segments", "death_context_nature"}:
                continue
            rows.append(
                [
                    _display_section(section_name),
                    _display_label(str(key)),
                    _display_value(str(value)),
                ]
            )

    if rows:
        return rows
    return [
        [
            "No extracted data",
            "Status",
            "The model response did not include extractable fields.",
        ]
    ]


def _section_table_rows(predictions: dict[str, dict[str, Any]]) -> list[list[str]]:
    output: list[list[str]] = []
    for section_name in ("context", "experience", "aftereffects"):
        section_payload = _extract_section_payload(predictions, section_name)
        evidence = section_payload.get("evidence_segments", [])
        evidence_text = (
            " | ".join(str(item) for item in evidence)
            if isinstance(evidence, list)
            else ""
        )
        context_nature = ""
        if section_name == "context":
            raw_context_nature = str(
                section_payload.get("death_context_nature", "")
            ).strip()
            context_nature = CONTEXT_NATURE_LABELS.get(
                raw_context_nature, raw_context_nature
            )
        output.append(
            [
                _display_section(section_name),
                _display_value(str(section_payload.get("tone", ""))),
                context_nature,
                evidence_text,
            ]
        )
    return output


def _segmentation_table_rows(segmentation: dict[str, Any]) -> list[list[str]]:
    return [
        [_display_section("context"), str(segmentation.get("context", ""))],
        [_display_section("experience"), str(segmentation.get("experience", ""))],
        [_display_section("aftereffects"), str(segmentation.get("aftereffects", ""))],
    ]


def launch_local_demo(
    *,
    study_config_path: str,
    paths_config_path: str,
    host: str,
    port: int,
    share: bool,
) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Install UI dependencies with: pip install -e .[ui]"
        ) from exc

    study = load_study_config(study_config_path)
    paths = load_paths_config(paths_config_path)
    llm_config = load_llm_config(paths_config_path)
    preprocessing = load_preprocessing_config(paths_config_path)

    fallback_models = configured_model_fallbacks(llm_config, preprocessing)
    default_base_url = str(llm_config.runtime.base_url)
    video_path = Path(PROJECT_ROOT) / "Stories_vs_Surveys.mp4"

    def _discover_models(base_url: str) -> tuple[list[str], str]:
        try:
            discovered = list_ollama_models(
                base_url,
                timeout_seconds=max(5, int(llm_config.runtime.timeout_seconds)),
            )
            if discovered:
                return discovered, f"Discovered {len(discovered)} model(s) from Ollama."
            if fallback_models:
                return (
                    fallback_models,
                    "No models returned by /api/tags. Showing fallback models from local config.",
                )
            return [], "No Ollama models found and no fallback models configured."
        except Exception as exc:  # noqa: BLE001
            if fallback_models:
                return (
                    fallback_models,
                    f"Could not fetch Ollama models ({exc}). Showing fallback models from local config.",
                )
            return [], f"Could not fetch Ollama models: {exc}"

    initial_models, initial_model_message = _discover_models(default_base_url)
    initial_model = initial_models[0] if initial_models else None

    def refresh_models(base_url: str):
        models, message = _discover_models(base_url or default_base_url)
        selected = models[0] if models else None
        return gr.Dropdown(choices=models, value=selected), message

    def toggle_mode(mode: str):
        three_visible = mode == "Guided Mode: Three Sections"
        single_visible = mode == "Complex Mode: Single Narrative"
        return (
            gr.Textbox(visible=three_visible),
            gr.Textbox(visible=three_visible),
            gr.Textbox(visible=three_visible),
            gr.Textbox(visible=single_visible),
        )

    def run_analysis(
        mode: str,
        base_url: str,
        model: str,
        temperature: float,
        prompt_variant: str,
        context_text: str,
        experience_text: str,
        aftereffects_text: str,
        single_narrative_text: str,
        optional_valence: str,
    ):
        if not model or not str(model).strip():
            raise ValueError("Select an Ollama model before running analysis.")

        effective_variant = str(prompt_variant).strip() or None
        effective_base_url = str(base_url).strip() or default_base_url

        if mode == "Guided Mode: Three Sections":
            result = analyze_three_sections(
                study=study,
                paths=paths,
                llm_config=llm_config,
                model=model,
                context_text=context_text,
                experience_text=experience_text,
                aftereffects_text=aftereffects_text,
                prompt_variant=effective_variant,
                base_url=effective_base_url,
                temperature=float(temperature),
            )
            segmentation_note = (
                "Stage 2 segmentation source: user-provided sections (guided mode)."
            )
        else:
            result = analyze_single_narrative(
                study=study,
                paths=paths,
                llm_config=llm_config,
                model=model,
                single_narrative_text=single_narrative_text,
                prompt_variant=effective_variant,
                base_url=effective_base_url,
                temperature=float(temperature),
            )
            segmentation_note = "Stage 2 segmentation source: model-assisted resegmentation (complex mode)."

        predictions = dict(result.get("predictions", {}))
        evidence_md = build_evidence_summary_markdown(predictions)
        segmentation = dict(result.get("segmentation", {}))
        section_rows = _section_table_rows(predictions)
        label_rows = _labels_table_rows(predictions)
        segmentation_rows = _segmentation_table_rows(segmentation)
        alignment_md = _alignment_markdown(predictions, optional_valence)
        global_tone_md = _global_tone_markdown(predictions)
        interpretation_md = _interpretation_markdown(predictions)
        status = (
            f"Analysis completed with model {result.get('model')} "
            f"using endpoint {result.get('base_url')}."
        )
        return (
            status,
            segmentation_note,
            segmentation_rows,
            global_tone_md,
            section_rows,
            label_rows,
            alignment_md,
            evidence_md,
            interpretation_md,
        )

    with gr.Blocks(
        title="NDE Local Interactive Demo",
        theme=gr.themes.Soft(
            primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate"
        ),
        css="""
footer, .footer, #footer {display: none !important;}
.gradio-container {
  color: #0f172a;
  background:
    radial-gradient(1200px 400px at 10% -10%, #d9f5ec 0%, transparent 55%),
    radial-gradient(900px 380px at 100% 0%, #dcecff 0%, transparent 50%),
    linear-gradient(180deg, #f8fbff 0%, #f3f7fb 100%);
}
#overview-video {
  max-width: 680px;
  margin-left: auto;
  margin-right: auto;
}
#overview-video video {
  max-height: 300px !important;
}
#overview-video .wrap,
#overview-video .container,
#overview-video .block {
  max-width: 680px;
  margin-left: auto;
  margin-right: auto;
}
.gradio-container .prose,
.gradio-container .prose * {
  color: #0f172a !important;
}
.gradio-container button[role="tab"],
.gradio-container .tab-nav button,
.gradio-container [role="tablist"] button {
  background: #e5e7eb !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
}
.gradio-container button[role="tab"][aria-selected="true"],
.gradio-container .tab-nav button.selected,
.gradio-container [role="tablist"] button[aria-selected="true"] {
  background: #1e3a8a !important;
  color: #ffffff !important;
  border-color: #1e40af !important;
}
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .secondary-text,
.gradio-container .svelte-1ipelgc {
  color: #1f2937 !important;
}
#component-0,
.gradio-container .block {
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 12px;
}
""",
    ) as demo:
        gr.Markdown("# NDE Local Interactive Demo")
        gr.Markdown(INTRO_MD)
        gr.Markdown(RESEARCH_QUESTIONS_MD)
        gr.Markdown(WHAT_WAS_DONE_MD)
        gr.Markdown(DEMO_PURPOSE_MD)
        gr.Markdown(KEY_ELEMENTS_MD)
        gr.Markdown(STAGES_MD)
        gr.Markdown(USER_GUIDE_MD)

        gr.Markdown(VIDEO_SUMMARY_MD)
        if video_path.exists():
            gr.Video(
                value=str(video_path),
                label="Stories vs Surveys Overview",
                elem_id="overview-video",
                height=300,
            )
        else:
            gr.Markdown(
                "Video not found at repository root (`Stories_vs_Surveys.mp4`)."
            )

        gr.Markdown(DISCLAIMER_MD)

        gr.Markdown("## Stage 1 — Input")
        with gr.Row():
            base_url_input = gr.Textbox(label="Ollama Base URL", value=default_base_url)
            model_dropdown = gr.Dropdown(
                choices=initial_models,
                value=initial_model,
                label="Model",
                allow_custom_value=True,
            )
            refresh_button = gr.Button("Refresh Models")

        model_status = gr.Markdown(initial_model_message)

        with gr.Row():
            mode_selector = gr.Radio(
                choices=[
                    "Guided Mode: Three Sections",
                    "Complex Mode: Single Narrative",
                ],
                value="Guided Mode: Three Sections",
                label="Input Path",
                info="Guided mode lets you control section boundaries; complex mode runs model-assisted resegmentation before analysis.",
            )
            prompt_variant_input = gr.Textbox(
                label="Prompt Variant (optional)",
                value="",
                placeholder="Use analysis default when empty",
            )
            temperature_input = gr.Number(
                label="Temperature",
                value=float(llm_config.runtime.temperature),
                precision=3,
            )
            optional_valence_input = gr.Dropdown(
                choices=["", "positive", "negative", "mixed", "neutral"],
                value="",
                label="Optional Valence",
                info="Used only in Stage 3 alignment module.",
            )

        context_input = gr.Textbox(
            label="Before the NDE (Context)",
            lines=6,
            placeholder="Describe circumstances before/during the life-threatening event.",
            info="Use this for medical context, perceived threat, and immediate setting.",
        )
        experience_input = gr.Textbox(
            label="Core NDE Experience",
            lines=6,
            placeholder="Describe the central experience (perceptions, emotions, phenomena).",
            info="Use this for the main experiential content.",
        )
        aftereffects_input = gr.Textbox(
            label="Aftereffects",
            lines=6,
            placeholder="Describe lasting changes after the event.",
            info="Use this for moral, relational, or long-term perspective changes.",
        )
        single_input = gr.Textbox(
            label="Single Narrative",
            lines=12,
            visible=False,
            placeholder="Paste the full narrative. The app will segment it before analysis.",
            info="Best for quick trials; review Stage 2 segmentation before interpreting results.",
        )

        run_button = gr.Button("Run Analysis", variant="primary")
        status_output = gr.Markdown()

        gr.Markdown("## Stage 2 — Segmentation")
        segmentation_note_output = gr.Markdown()
        segmentation_table_output = gr.Dataframe(
            headers=["Narrative Part", "Text Used for Analysis"],
            datatype=["str", "str"],
            label="Segmented Narrative",
            interactive=False,
        )

        gr.Markdown("## Stage 3 — Module Analysis")
        with gr.Tabs():
            with gr.TabItem("Tone Estimation"):
                global_tone_output = gr.Markdown()
                section_table_output = gr.Dataframe(
                    headers=[
                        "Narrative Part",
                        "Detected Tone",
                        "Context Type",
                        "Supporting Evidence",
                    ],
                    datatype=["str", "str", "str", "str"],
                    label="Section-Level Summary",
                    interactive=False,
                )

            with gr.TabItem("Structured Features"):
                labels_table_output = gr.Dataframe(
                    headers=["Narrative Part", "Dimension", "Detected Value"],
                    datatype=["str", "str", "str"],
                    label="Extracted Dimensions",
                    interactive=False,
                )
                evidence_output = gr.Markdown(label="Evidence Details")

            with gr.TabItem("Alignment with Questionnaire-style Layer"):
                alignment_output = gr.Markdown(label="Valence Alignment")

        gr.Markdown("## Stage 4 — Interpretation")
        interpretation_output = gr.Markdown()

        refresh_button.click(
            fn=refresh_models,
            inputs=[base_url_input],
            outputs=[model_dropdown, model_status],
        )
        mode_selector.change(
            fn=toggle_mode,
            inputs=[mode_selector],
            outputs=[context_input, experience_input, aftereffects_input, single_input],
        )
        run_button.click(
            fn=run_analysis,
            inputs=[
                mode_selector,
                base_url_input,
                model_dropdown,
                temperature_input,
                prompt_variant_input,
                context_input,
                experience_input,
                aftereffects_input,
                single_input,
                optional_valence_input,
            ],
            outputs=[
                status_output,
                segmentation_note_output,
                segmentation_table_output,
                global_tone_output,
                section_table_output,
                labels_table_output,
                alignment_output,
                evidence_output,
                interpretation_output,
            ],
        )

    demo.launch(
        server_name=host, server_port=int(port), share=bool(share), show_error=True
    )
