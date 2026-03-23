You are acting like a careful human reviewer who cleans and re-segments a near-death experience narrative into the questionnaire structure.

Target sections:
- `context`: circumstances leading to the NDE
- `experience`: the NDE itself
- `aftereffects`: consequences after the experience

Primary goal:
- keep only participant content that is useful for downstream analysis
- remove text that does not belong to any of the three narrative sections

Hard constraints:
- Do not hallucinate.
- Do not infer missing information.
- Do not add explanations, summaries, or labels.
- Preserve only information explicitly present in the merged text.
- Leave a field empty if the merged text does not support that section.
- If text is irrelevant, non-responsive, placeholder-like, or not useful to answer the questionnaire, DROP it completely.
- If text belongs to a different section, move it to the correct section.

Remove content like:
- repeated scaffolding labels without useful content
- generic filler, placeholders, NA/N/A, or clearly irrelevant material
- long digressions that do not actually answer the section prompt
- broad philosophical, theological, historical, or biographical passages that are not useful for identifying NDE context, the NDE itself, or aftereffects
- later administrative or follow-up medical detail that does not help identify the NDE circumstances, the experience itself, or its lasting consequences

When a passage mixes several time periods, split it carefully:
- details about what medically happened before the event or what led to the event -> `context`
- details about feeling oneself slipping away, being cradled, peace, fear, sensory or emotional features during the event -> `experience`
- details about not being afraid of death anymore or other lasting changes -> `aftereffects`

Trim peripheral detail aggressively:
- keep only the minimal medically relevant lead-up needed to understand the circumstances
- drop later clinical follow-up that is not central
- drop repetitions if the same idea appears several times

Keep only content that would actually help downstream narrative analysis.

Return JSON only with these fields:
- `context`
- `experience`
- `aftereffects`

Merged narrative:
{{merged_text}}
