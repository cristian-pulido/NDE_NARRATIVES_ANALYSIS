You are acting like a careful human reviewer of near-death experience narratives.

Your job in this step is ONLY to judge whether each section contains useful participant content that belongs in that section.
Do not rewrite the text in this step.

Original questionnaire intent:
- `context` = events or circumstances that led to the NDE
- `experience` = the NDE itself: sensory, emotional, cognitive, symbolic, memorable, or encountered elements during the experience
- `aftereffects` = changes after the experience: life impact, worldview, relationships, purpose, long-term consequences

Use calibrated review and these labels:
- `valid` = the section is mostly useful and mostly belongs in that section (minor overlap is acceptable)
- `invalid` = off-section or irrelevant material dominates enough to make the section unreliable as-is
- `empty` = there is no meaningful participant content for that section after ignoring irrelevant material

Treat the section as `empty` when it is mostly or entirely made of things like:
- generic statements about the writing rather than the experience itself
- vague or non-responsive text that does not answer the section prompt
- summaries or framing text instead of participant content
- repeated labels such as "Narrative:", "Context:", "Aftereffect:" without substantive content
- generic refusals, placeholders, missing values, `NA`, `N/A`, blank text
- clearly irrelevant content that should not survive cleaning

Treat the section as `invalid` when:
- substantial material belongs to another section and dominates the section
- useful content is present but heavily mixed with irrelevant/off-section content so reliability is low
- long reflective/theological/philosophical/historical/biographical material dominates and is not anchored to the section intent
- labels/summaries/explanatory framing dominate the section content
- temporal layers are mixed in a way that materially harms section reliability

Do NOT mark `invalid` only because:
- there is minor spillover from adjacent sections
- grammar is poor, wording is repetitive, or style is messy
- there is brief reflective interpretation that is still anchored to the participant's NDE narrative

When judging mixed passages:
- material about what happened before the NDE or what medically led to it belongs in `context`
- material about sensations, slipping away, peace, fearlessness, lights, beings, out-of-body states, or what was experienced in the moment belongs in `experience`
- material about not fearing death anymore, worldview changes, relationship changes, behavior changes, or lasting consequences belongs in `aftereffects`
- later administrative detail, medical follow-up logistics, or intervention planning should usually be ignored unless directly relevant to section intent

Important constraints:
- Ignore spelling, grammar, and formatting problems.
- Do not infer missing information.
- Be strict with clearly irrelevant content, but calibrated with minor overlap.
- If a section would become empty after removing irrelevant/off-section material, mark it `empty`.
- Mark `needs_resegmentation` as `yes` if at least one section is `invalid` or if useful content appears clearly misplaced across sections.

Return JSON only with these fields:
- `context_assessment`
- `experience_assessment`
- `aftereffects_assessment`
- `needs_resegmentation`

Context section text:
{{context_text}}

Experience section text:
{{experience_text}}

Aftereffects section text:
{{aftereffects_text}}
