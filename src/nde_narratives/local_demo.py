from __future__ import annotations

from collections import Counter
from html import escape
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
    "outside_of_body_experience": "Out-of-body sensation",
    "feeling_bright_light": "Bright light perception",
    "feeling_awareness": "Heightened awareness",
    "presence_encounter": "Encounter with a presence",
    "saw_relived_past_events": "Relived past events",
    "time_perception_altered": "Altered time perception",
    "border_point_of_no_return": "Point of no return",
    "non_existence_feeling": "Feeling of non-existence",
    "feeling_peace_wellbeing": "Peace and wellbeing",
    "saw_entered_gateway": "Entered a gateway",
    "fear_of_death": "Fear of death",
    "inner_meaning_in_my_life": "Inner meaning in life",
    "compassion_toward_others": "Compassion toward others",
    "spiritual_feelings": "Spiritual feelings",
    "desire_to_help_others": "Desire to help others",
    "personal_vulnerability": "Personal vulnerability",
    "interest_in_material_goods": "Interest in material goods",
    "interest_in_religion": "Interest in religion",
    "understanding_myself": "Understanding myself",
    "social_justice_issues": "Social justice issues",
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
### Research Use Only
- Not a diagnostic tool.
- Avoid personal identifiers.
- Outputs require human judgment.
""".strip()


HERO_SUBTITLE_MD = """
Narratives and questionnaires don't describe the same experience.
""".strip()


HERO_SUPPORT_MD = """
This demo lets you explore that mismatch in action.
""".strip()


AUTHOR_MD = """
**Autor original:** [Cristian Pulido](https://cristian-pulido.github.io/)
""".strip()


CORE_INSIGHT_MD = """
### What you are about to see
- Narratives capture local moments (fear, confusion, ambiguity).
- Questionnaires summarize global meaning (often positive or transformative).
- These differences are not errors.
- They reflect different representations of the same experience.
""".strip()


TRY_IT_MD = """
### Try it yourself
- Write or paste a narrative.
- Run the analysis.
- Compare what the model extracts, what you would extract, and what a questionnaire would capture.
""".strip()


SIMPLIFIED_FLOW_MD = """
### Quick flow
1. Add your text.
2. Run analysis.
3. Compare interpretations.
""".strip()


LEARN_MORE_MD = """
### Learn more
- The pipeline segments text into context, core experience, and aftereffects.
- It extracts tone, structured features, and supporting snippets.
- Alignment checks compare narrative signals with a questionnaire-style valence.
""".strip()


TOOLTIP_MD = """
ℹ️ Use **Guided Input** if you already have three sections.
Use **Full Narrative** if you want automatic segmentation.
""".strip()


STAGES_MD = """
## Pipeline Stages

1. **Input**: provide either one narrative (complex mode) or three sections (guided mode).
2. **Segmentation**: inspect the three sections actually used for inference.
3. **Module Analysis**: review tone, structured features, and optional valence alignment.
4. **Interpretation**: understand recoverable signals, uncertainty, and limits versus questionnaire-style labels.
""".strip()


VIDEO_SUMMARY_MD = """
See how this works (2 min video)
""".strip()


ARTICLE_URL = "https://cristian-pulido.github.io/representational-mismatch-nde/"


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


def _post_result_insight_markdown(
    predictions: dict[str, dict[str, Any]], optional_valence: str
) -> str:
    tones = _tone_by_section(predictions)
    experience_tone = tones.get("experience", "") or "unknown"
    distinct_tones = {tone for tone in tones.values() if tone}
    has_local_variation = len(distinct_tones) > 1

    normalized_valence = str(optional_valence).strip().lower()
    has_valence = bool(normalized_valence)
    valence_mismatch = has_valence and normalized_valence != experience_tone

    if valence_mismatch and has_local_variation:
        return (
            "### Study Insight (This Run)\n"
            "Notice how local section tone is mixed, and it does not align with the selected questionnaire-style valence. "
            "This mismatch is the core finding of the study."
        )

    if valence_mismatch:
        return (
            "### Study Insight (This Run)\n"
            "The selected questionnaire-style valence does not match the core narrative tone. "
            "Different representations of the same experience can diverge."
        )

    if has_local_variation:
        return (
            "### Study Insight (This Run)\n"
            "Notice how tone and features shift across sections. "
            "Local narrative moments can differ from the global meaning a questionnaire captures."
        )

    return (
        "### Study Insight (This Run)\n"
        "This run looks more aligned across sections and overall meaning. "
        "The key point remains: alignment is possible, but not guaranteed."
    )


def _stepper_html(active_step: int) -> str:
    steps = [
        (1, "Input", "✍️"),
        (2, "Segmentation", "✂️"),
        (3, "Extraction", "🧩"),
        (4, "Interpretation", "🧠"),
    ]
    parts: list[str] = ["<div class='pipeline-stepper'>"]
    for index, label, icon in steps:
        state_class = "step-pending"
        if active_step > index:
            state_class = "step-complete"
        elif active_step == index:
            state_class = "step-active"
        parts.append(
            "<div class='pipeline-step {state}'>"
            "<span class='step-icon'>{icon}</span>"
            "<span class='step-label'>{label}</span>"
            "</div>".format(state=state_class, icon=icon, label=escape(label))
        )
        if index < len(steps):
            parts.append("<div class='pipeline-connector'></div>")
    parts.append("</div>")
    return "".join(parts)


def _tone_badge_html(value: str) -> str:
    normalized = str(value).strip().lower() or "unknown"
    css_class = {
        "positive": "badge-positive",
        "negative": "badge-negative",
        "mixed": "badge-mixed",
        "neutral": "badge-mixed",
        "unknown": "badge-unknown",
    }.get(normalized, "badge-unknown")
    return f"<span class='tone-badge {css_class}'>{escape(_display_value(normalized) or 'Unknown')}</span>"


def _tone_cards_html(predictions: dict[str, dict[str, Any]]) -> str:
    tones = _tone_by_section(predictions)
    cards = ["<div class='tone-grid'>"]
    for section_name in ("context", "experience", "aftereffects"):
        section_payload = _extract_section_payload(predictions, section_name)
        evidence = section_payload.get("evidence_segments", [])
        evidence_items: list[str] = []
        if isinstance(evidence, list):
            for segment in evidence[:3]:
                text = str(segment).strip()
                if text:
                    evidence_items.append(
                        f"<li class='evidence-item'>{escape(text)}</li>"
                    )
        evidence_html = (
            f"<ul class='evidence-list'>{''.join(evidence_items)}</ul>"
            if evidence_items
            else "<div class='inline-note'>No explicit evidence returned for this section.</div>"
        )
        cards.append(
            "<div class='result-card'>"
            f"<div class='result-title'>{escape(_display_section(section_name))}</div>"
            f"{_tone_badge_html(tones.get(section_name, 'unknown'))}"
            "<div class='evidence-title'>Why the model says this</div>"
            f"{evidence_html}"
            "</div>"
        )
    cards.append("</div>")
    return "".join(cards)


def _structured_features_html(predictions: dict[str, dict[str, Any]]) -> str:
    sections_html: list[str] = ["<div class='feature-sections'>"]
    has_features = False
    for section_name in ("context", "experience", "aftereffects"):
        payload = _extract_section_payload(predictions, section_name)
        chips: list[str] = []
        if section_name == "context":
            raw_context_nature = str(payload.get("death_context_nature", "")).strip()
            if raw_context_nature:
                chips.append(
                    "<span class='feature-chip'>Context: {value}</span>".format(
                        value=escape(
                            CONTEXT_NATURE_LABELS.get(
                                raw_context_nature, raw_context_nature
                            )
                        )
                    )
                )

        for key, value in payload.items():
            if key in {"tone", "evidence_segments", "death_context_nature"}:
                continue
            has_features = True
            raw_value = str(value).strip().lower()
            display_value = escape(_display_value(str(value)) or str(value))
            if raw_value == "yes":
                display_value = (
                    f"<span class='feature-value-yes'>{display_value}</span>"
                )
            chips.append(
                "<span class='feature-chip'><strong>{label}:</strong> {value}</span>".format(
                    label=escape(_display_label(str(key))),
                    value=display_value,
                )
            )

        if not chips:
            chips.append(
                "<span class='feature-chip feature-empty'>No strong features detected.</span>"
            )

        sections_html.append(
            "<div class='result-card'>"
            f"<div class='result-title'>{escape(_display_section(section_name))}</div>"
            f"<div class='chip-wrap'>{''.join(chips)}</div>"
            "</div>"
        )
    sections_html.append("</div>")

    if not has_features:
        sections_html.append(
            "<div class='inline-note'>Only tone and evidence were detected for this narrative.</div>"
        )
    return "".join(sections_html)


def _segmentation_html(segmentation: dict[str, Any], note: str) -> str:
    cards = ["<div class='segment-grid'>"]
    for section_name in ("context", "experience", "aftereffects"):
        text = str(segmentation.get(section_name, "")).strip() or "No text provided."
        cards.append(
            "<div class='result-card'>"
            f"<div class='result-title'>{escape(_display_section(section_name))}</div>"
            f"<div class='segment-text'>{escape(text)}</div>"
            "</div>"
        )
    cards.append("</div>")
    cards.append(f"<div class='inline-note'>{escape(note)}</div>")
    return "".join(cards)


def _alignment_html(predictions: dict[str, dict[str, Any]], valence: str) -> str:
    normalized_valence = str(valence).strip().lower()
    experience_payload = _extract_section_payload(predictions, "experience")
    experience_tone = (
        str(experience_payload.get("tone", "")).strip().lower() or "unknown"
    )

    if not normalized_valence:
        return (
            "<div class='result-card'>"
            "<div class='result-title'>Your Valence vs Core NDE Experience</div>"
            "<div class='inline-note'>No user valence selected. Add one in Input to compare.</div>"
            "</div>"
        )

    is_match = experience_tone == normalized_valence
    match_text = "Aligned" if is_match else "Mismatch"
    match_class = "badge-positive" if is_match else "badge-negative"
    return (
        "<div class='alignment-grid'>"
        "<div class='result-card'><div class='result-title'>Your Valence</div>"
        f"{_tone_badge_html(normalized_valence)}</div>"
        "<div class='result-card'><div class='result-title'>Core NDE Experience (LLM)</div>"
        f"{_tone_badge_html(experience_tone)}</div>"
        "</div>"
        f"<div class='alignment-status'><span class='tone-badge {match_class}'>{match_text}</span></div>"
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
    video_path = Path(PROJECT_ROOT) / "Stories_vs_Surveys.mov"

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

    def _compute_results(
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
        segmentation = dict(result.get("segmentation", {}))
        segmentation_html = _segmentation_html(segmentation, segmentation_note)
        tone_html = _tone_cards_html(predictions)
        features_html = _structured_features_html(predictions)
        alignment_html = _alignment_html(predictions, optional_valence)
        post_result_insight_md = _post_result_insight_markdown(
            predictions, optional_valence
        )
        interpretation_md = _interpretation_markdown(predictions)
        status = (
            f"Analysis completed with model {result.get('model')} "
            f"using endpoint {result.get('base_url')}."
        )
        return (
            status,
            _stepper_html(4),
            segmentation_html,
            tone_html,
            features_html,
            alignment_html,
            post_result_insight_md,
            interpretation_md,
        )

    with gr.Blocks(
        title="NDE Local Interactive Demo",
        theme=gr.themes.Soft(
            primary_hue="orange", secondary_hue="blue", neutral_hue="slate"
        ),
        css="""
footer, .footer, #footer {display: none !important;}
.gradio-container {
  --body-background-fill: #071227;
  --block-background-fill: rgba(11, 28, 58, 0.75);
  --block-border-color: rgba(146, 168, 201, 0.18);
  --input-background-fill: #0d2347;
  --input-border-color: rgba(168, 188, 220, 0.3);
  color-scheme: dark;
  color: #edf2ff;
  background:
    radial-gradient(1000px 420px at 12% -6%, rgba(255, 153, 61, 0.2) 0%, transparent 58%),
    radial-gradient(880px 360px at 90% -12%, rgba(110, 145, 255, 0.25) 0%, transparent 52%),
    linear-gradient(180deg, #061024 0%, #081a36 45%, #071227 100%);
}
.hero-card, .stage-card, .note-card {
  background: rgba(10, 27, 56, 0.78);
  border: 1px solid rgba(146, 168, 201, 0.2);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 16px 40px rgba(3, 10, 24, 0.35);
  backdrop-filter: blur(6px);
}
.hero-title {
  font-size: 2.2rem;
  margin: 0;
  line-height: 1.1;
  color: #ffffff;
}
.hero-subtitle {
  margin: 0.4rem 0 0.8rem;
  color: #b7c9ec;
  font-size: 1.02rem;
}
.hero-support {
  margin: 0.15rem 0 0.9rem;
  color: #d4e3ff;
  font-size: 0.98rem;
}
.cta-btn {
  display: inline-block;
  margin-top: 0.4rem;
  padding: 0.55rem 1rem;
  border-radius: 999px;
  background: linear-gradient(90deg, #ff8c36 0%, #ff6f2a 100%);
  color: #fff !important;
  font-weight: 700;
  text-decoration: none;
  transition: transform 0.16s ease, box-shadow 0.16s ease;
}
.cta-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(255, 111, 42, 0.35);
}
.cta-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.cta-btn-secondary {
  background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
}
.cta-btn-secondary:hover {
  box-shadow: 0 8px 20px rgba(34, 197, 94, 0.35);
}
.pipeline-stepper {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.pipeline-step {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(136, 158, 190, 0.38);
  font-size: 0.86rem;
}
.step-active {
  background: rgba(255, 140, 54, 0.22);
  border-color: rgba(255, 140, 54, 0.55);
}
.step-complete {
  background: rgba(55, 211, 153, 0.18);
  border-color: rgba(55, 211, 153, 0.45);
}
.step-pending {
  opacity: 0.72;
}
.pipeline-connector {
  height: 1px;
  width: 26px;
  background: rgba(140, 163, 197, 0.45);
}
.tone-grid, .segment-grid, .alignment-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
}
.feature-sections {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}
.result-card {
  border-radius: 14px;
  border: 1px solid rgba(147, 170, 206, 0.24);
  background: rgba(9, 22, 46, 0.84);
  padding: 12px;
}
.result-card-accent {
  border-color: rgba(255, 140, 54, 0.55);
  box-shadow: inset 0 0 0 1px rgba(255, 140, 54, 0.26);
}
.result-title {
  font-size: 0.9rem;
  color: #d4e3ff;
  margin-bottom: 8px;
}
.evidence-title {
  margin-top: 10px;
  margin-bottom: 4px;
  font-size: 0.78rem;
  color: #b8caea;
}
.evidence-list {
  margin: 0;
  padding-left: 18px;
}
.evidence-item {
  margin-bottom: 4px;
  color: #dbe7ff;
  line-height: 1.35;
}
.tone-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.84rem;
  border: 1px solid transparent;
  font-weight: 700;
}
.badge-positive { background: rgba(34, 197, 94, 0.2); color: #bbf7d0; border-color: rgba(34,197,94,0.45); }
.badge-negative { background: rgba(239, 68, 68, 0.2); color: #fecaca; border-color: rgba(239,68,68,0.45); }
.badge-mixed { background: rgba(250, 204, 21, 0.22); color: #fef3c7; border-color: rgba(250,204,21,0.45); }
.badge-unknown { background: rgba(148, 163, 184, 0.22); color: #e2e8f0; border-color: rgba(148,163,184,0.45); }
.chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.feature-chip {
  border-radius: 10px;
  padding: 4px 8px;
  font-size: 0.8rem;
  background: rgba(91, 127, 186, 0.18);
  border: 1px solid rgba(139, 165, 208, 0.4);
  color: #deebff;
}
.feature-value-yes {
  color: #86efac;
  font-weight: 700;
}
.feature-empty {
  opacity: 0.8;
}
.segment-text {
  color: #d9e8ff;
  max-height: 160px;
  overflow-y: auto;
  white-space: pre-wrap;
  line-height: 1.4;
}
.inline-note {
  margin-top: 8px;
  font-size: 0.85rem;
  color: #b8caea;
}
.alignment-status {
  margin-top: 10px;
}
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container label,
.gradio-container .block-label,
.gradio-container .block-title,
.gradio-container .block-info,
.gradio-container .component-description,
.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container [role="tab"] {
  color: #edf2ff !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: #a4bcdf !important;
  opacity: 1 !important;
}
.gradio-container button.primary {
  background: linear-gradient(90deg, #ff8c36 0%, #ff6f2a 100%) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 700 !important;
}
#overview-video video {
  max-height: 320px !important;
}
.loading-card {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 30px;
  color: #b7c9ec;
  gap: 10px;
}
.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: #ff8c36;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
""",
    ) as demo:
        # Loading state helpers and click wrappers
        def _loading_placeholder_html() -> str:
            return (
                "<div class='result-card loading-card'>"
                "<div class='spinner'></div>"
                "<span>Loading results…</span>"
                "</div>"
            )

        def _loading_placeholder_markdown() -> str:
            return "⌛ Analyzing interpretation…"

        def run_guided_click(
            base_url,
            model,
            temperature,
            prompt_variant,
            context,
            experience,
            aftereffects,
            single,
            valence,
        ):
            loading_stepper = _stepper_html(1)
            loading_card = _loading_placeholder_html()
            loading_insight = "⌛ Generating study insight…"
            loading_interp = _loading_placeholder_markdown()
            yield [
                gr.update(interactive=False),
                "⏳ Processing...",
                loading_stepper,
                loading_card,
                loading_card,
                loading_card,
                loading_card,
                loading_insight,
                loading_interp,
            ]
            try:
                results = _compute_results(
                    "Guided Mode: Three Sections",
                    base_url,
                    model,
                    temperature,
                    prompt_variant,
                    context,
                    experience,
                    aftereffects,
                    single,
                    valence,
                )
                yield [gr.update(interactive=True)] + list(results)
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                yield [
                    gr.update(interactive=True),
                    error_msg,
                    loading_stepper,
                    loading_card,
                    loading_card,
                    loading_card,
                    loading_card,
                    loading_insight,
                    loading_interp,
                ]

        def run_full_click(
            base_url,
            model,
            temperature,
            prompt_variant,
            context,
            experience,
            aftereffects,
            single,
            valence,
        ):
            loading_stepper = _stepper_html(1)
            loading_card = _loading_placeholder_html()
            loading_insight = "⌛ Generating study insight…"
            loading_interp = _loading_placeholder_markdown()
            yield [
                gr.update(interactive=False),
                "⏳ Processing...",
                loading_stepper,
                loading_card,
                loading_card,
                loading_card,
                loading_card,
                loading_insight,
                loading_interp,
            ]
            try:
                results = _compute_results(
                    "Complex Mode: Single Narrative",
                    base_url,
                    model,
                    temperature,
                    prompt_variant,
                    context,
                    experience,
                    aftereffects,
                    single,
                    valence,
                )
                yield [gr.update(interactive=True)] + list(results)
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                yield [
                    gr.update(interactive=True),
                    error_msg,
                    loading_stepper,
                    loading_card,
                    loading_card,
                    loading_card,
                    loading_card,
                    loading_insight,
                    loading_interp,
                ]

        with gr.Group(elem_classes="hero-card"):
            gr.HTML("<h1 class='hero-title'>From Stories to Structure</h1>")
            gr.HTML(f"<p class='hero-subtitle'>{escape(HERO_SUBTITLE_MD)}</p>")
            gr.HTML(f"<p class='hero-support'>{escape(HERO_SUPPORT_MD)}</p>")
            gr.HTML(
                "<div class='cta-row'>"
                "<a class='cta-btn' href='#input-workspace'>Try the Demo ↓</a>"
                f"<a class='cta-btn cta-btn-secondary' href='{ARTICLE_URL}' target='_blank' rel='noopener noreferrer'>Article Page ↗</a>"
                "</div>"
            )
            gr.Markdown(AUTHOR_MD)

        with gr.Group(elem_classes="note-card"):
            gr.Markdown(CORE_INSIGHT_MD)

        with gr.Accordion("Learn more", open=False):
            gr.Markdown(LEARN_MORE_MD)
            gr.Markdown(TOOLTIP_MD)

        with gr.Group(elem_classes="note-card"):
            gr.Markdown(TRY_IT_MD)
            gr.Markdown(SIMPLIFIED_FLOW_MD)

        with gr.Group(elem_classes="note-card"):
            gr.Markdown(f"### {VIDEO_SUMMARY_MD}")
            if video_path.exists():
                gr.Video(
                    value=str(video_path),
                    label="Stories vs Surveys Overview",
                    elem_id="overview-video",
                    height=300,
                )
            else:
                gr.Markdown(
                    "Video not found at repository root (`Stories_vs_Surveys.mov`)."
                )

        with gr.Group(elem_classes="note-card"):
            gr.Markdown(DISCLAIMER_MD)

        with gr.Group(elem_classes="stage-card", elem_id="input-workspace"):
            gr.Markdown("## Input")
            stepper_output = gr.HTML(_stepper_html(1))

            optional_valence_input = gr.Dropdown(
                choices=["", "positive", "negative", "mixed", "neutral"],
                value="",
                label="Optional Valence",
            )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    base_url_input = gr.Textbox(
                        label="Ollama Base URL", value=default_base_url
                    )
                    model_dropdown = gr.Dropdown(
                        choices=initial_models,
                        value=initial_model,
                        label="Model",
                        allow_custom_value=True,
                    )
                    refresh_button = gr.Button("Refresh Models")
                with gr.Row():
                    temperature_input = gr.Number(
                        label="Temperature",
                        value=float(llm_config.runtime.temperature),
                        precision=3,
                    )
                    prompt_variant_input = gr.Textbox(
                        label="Prompt Variant",
                        value="",
                        placeholder="Default prompt variant",
                    )
                model_status = gr.Markdown(initial_model_message)

            with gr.Tabs():
                with gr.Tab("Guided Input"):
                    context_input = gr.Textbox(
                        label="Before (Context)",
                        lines=5,
                        placeholder="Example: I was in surgery after a severe accident...",
                    )
                    experience_input = gr.Textbox(
                        label="Experience",
                        lines=5,
                        placeholder="Example: I saw a bright light and felt calm...",
                    )
                    aftereffects_input = gr.Textbox(
                        label="After (Effects)",
                        lines=5,
                        placeholder="Example: Since then I feel less fear of death...",
                    )
                    run_guided_button = gr.Button("Run Analysis", variant="primary")

                with gr.Tab("Full Narrative"):
                    single_input = gr.Textbox(
                        label="Full Narrative",
                        lines=13,
                        placeholder="Paste the complete story. The app will segment it automatically.",
                    )
                    run_full_button = gr.Button("Run Analysis", variant="primary")

            status_output = gr.Markdown()

        with gr.Group(elem_classes="stage-card"):
            gr.Markdown("## Results")
            with gr.Tabs():
                with gr.Tab("Segmentation"):
                    segmentation_output = gr.HTML()
                with gr.Tab("Tone"):
                    global_tone_output = gr.HTML()
                with gr.Tab("Structured Features"):
                    features_output = gr.HTML()
                with gr.Tab("Alignment"):
                    alignment_output = gr.HTML()

        with gr.Group(elem_classes="note-card"):
            post_result_insight_output = gr.Markdown(
                "### Study Insight (This Run)\nRun an analysis to see how narrative and questionnaire-style interpretations align or diverge."
            )

        with gr.Group(elem_classes="stage-card"):
            gr.Markdown("## Interpretation")
            interpretation_output = gr.Markdown()

        guided_mode_state = gr.State("Guided Mode: Three Sections")
        full_mode_state = gr.State("Complex Mode: Single Narrative")

        refresh_button.click(
            fn=refresh_models,
            inputs=[base_url_input],
            outputs=[model_dropdown, model_status],
        )

        run_guided_button.click(
            fn=run_guided_click,
            inputs=[
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
                run_guided_button,
                status_output,
                stepper_output,
                segmentation_output,
                global_tone_output,
                features_output,
                alignment_output,
                post_result_insight_output,
                interpretation_output,
            ],
        )

        run_full_button.click(
            fn=run_full_click,
            inputs=[
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
                run_full_button,
                status_output,
                stepper_output,
                segmentation_output,
                global_tone_output,
                features_output,
                alignment_output,
                post_result_insight_output,
                interpretation_output,
            ],
        )

    demo.launch(
        server_name=host, server_port=int(port), share=bool(share), show_error=True
    )
