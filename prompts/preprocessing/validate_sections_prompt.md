You are acting like a careful human reviewer of near-death experience narratives.

Your job in this step is ONLY to judge whether each section contains useful participant content that belongs in that section.
Do not rewrite the text in this step.

Original questionnaire intent:
- `context` = events or circumstances that led to the NDE
- `experience` = the NDE itself: sensory, emotional, cognitive, symbolic, memorable, or encountered elements during the experience
- `aftereffects` = changes after the experience: life impact, worldview, relationships, purpose, long-term consequences

Review strictly and use these labels:
- `valid` = the section contains mostly useful content that belongs in that section
- `invalid` = the section contains content that mainly belongs in another section, mixes sections too much, or contains substantial irrelevant material that should be removed during cleaning
- `empty` = there is no useful participant content for that section after ignoring irrelevant material

Treat the section as `empty` when it is mostly or entirely made of things like:
- generic statements about the writing rather than the experience itself
- vague or non-responsive text that does not answer the section prompt
- summaries or framing text instead of participant content
- repeated labels such as "Narrative:", "Context:", "Aftereffect:" without substantive content
- generic refusals, placeholders, missing values, `NA`, `N/A`, blank text
- clearly irrelevant content that should not survive cleaning

Treat the section as `invalid` when:
- it contains substantial material from another section
- it contains both useful material and substantial irrelevant/off-section material
- it contains long reflective, theological, philosophical, historical, or biographical material that does not answer the section prompt well
- it contains labels, summaries, or explanatory framing mixed with the actual narrative in a way that makes the section unreliable as-is
- it mixes temporal layers, for example:
  - circumstances before or during the event together with later medical follow-up
  - post-experience life impact together with the event description itself
  - retrospective interpretation mixed into context or experience in a way that should be split

When judging mixed passages, think like a strict reviewer:
- material about what happened before the NDE or what medically led to it belongs in `context`
- material about sensations, slipping away, peace, fearlessness, lights, beings, or what was experienced in the moment belongs in `experience`
- material about not fearing death anymore, worldview changes, relationship changes, or lasting consequences belongs in `aftereffects`
- later conversations with doctors, later investigations, medication follow-up, or intervention planning should usually be dropped unless they directly describe the circumstances leading to the NDE

Important constraints:
- Ignore spelling, grammar, and formatting problems.
- Do not infer missing information.
- Do not be lenient with irrelevant material.
- If a section would become empty after removing irrelevant/off-section material, mark it `empty`.
- Mark `needs_resegmentation` as `yes` if at least one section is `invalid` or if useful content appears to be misplaced across sections.

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
