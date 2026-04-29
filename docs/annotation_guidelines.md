# Annotation Guidelines

## Purpose

The annotation task transforms open-ended NDE narratives into a structured format that can be compared with:

- human judgments
- LLM predictions
- questionnaire-based measures from NDE-C and LCI-R

The task does not determine whether the event was medically verified or diagnostically classifiable as an NDE.

## Annotation Units

Each participant is coded independently for:

- `context`
- `experience`
- `aftereffects`

Information must not be transferred across sections.

## General Rules

- Code only what is explicitly expressed.
- Use the overall meaning of the section, not isolated words.
- If tone is ambiguous, use `mixed`.
- If the section is mainly factual/descriptive and emotionally uncharged, use `neutral`.
- If an element is implied but not explicit, code `no`.
- If an element is denied, code `no`.
- Prioritize the participant's framing of the event.
- For aftereffects items, the objective is to detect whether the text explicitly mentions a change, not to infer change direction when direction is unstated.

## Evaluation Workflow Note

- If a case should not be evaluated, leave the full row unlabeled or remove the row from the completed workbook.
- Do not leave a row partially completed. `nde evaluate` treats partially filled rows as invalid and will stop with an error.
- Do not edit `Participant Code`.

## Tone Labels

- `positive`: peace, relief, clarity, insight, beneficial meaning, favorable transformation
- `negative`: fear, distress, confusion, suffering, threat, adverse interpretation
- `mixed`: clear positive and negative elements, ambivalence, or unresolved tone
- `neutral`: mainly descriptive or factual language with little or no emotional valence

## Experience Elements

Code these only from the `experience` narrative:

- `outside_of_body_experience`: leaving the body, floating above it, viewing the body from outside
- `feeling_bright_light`: bright, radiant, or luminous light
- `feeling_awareness`: unusual heightened awareness or clarity
- `presence_encounter`: an encountered being, entity, or felt presence
- `saw_relived_past_events`: reliving or replaying prior life events
- `time_perception_altered`: time slowing, speeding up, stopping, or losing meaning
- `border_point_of_no_return`: explicit border/limit threshold before return
- `non_existence_feeling`: explicit dissolution or non-existence feeling
- `feeling_peace_wellbeing`: intense peace, calm, serenity, or wellbeing
- `saw_entered_gateway`: entering or approaching a gateway/tunnel/door threshold

## Aftereffects Elements

Code these only from the `aftereffects` narrative:

For these LCI-R aligned items, mark `yes` only when the narrative explicitly states a change (for example, increased, decreased, less, more, stronger, weaker). Do not estimate a direction if the narrative does not clearly express one.

- `fear_of_death`: explicit change in fear of dying/death compared with before
- `inner_meaning_in_my_life`: explicit change in perceived meaning or purpose in life
- `compassion_toward_others`: explicit change in empathy, compassion, or concern for others
- `spiritual_feelings`: explicit change in spirituality, sacred connection, or spiritual feeling
- `desire_to_help_others`: explicit change in helping/prosocial motivation toward others
- `personal_vulnerability`: explicit change in felt vulnerability, fragility, or openness to harm
- `interest_in_material_goods`: explicit change in attachment to possessions/material success
- `interest_in_religion`: explicit change in religious interest, practice, or affiliation salience
- `understanding_myself`: explicit change in self-knowledge, self-understanding, or personal insight
- `social_justice_issues`: explicit change in concern for fairness, inequality, or justice issues

## Allowed Outputs

- Tone columns: `positive`, `negative`, `mixed`, `neutral`
- Element columns: `yes`, `no`

## LLM Evidence Spans

For each section-level tone assignment, LLM outputs now include 1 to 3 short verbatim evidence spans:

- spans must be copied directly from the section text
- spans should justify the assigned tone label
- spans should not be paraphrases or summaries
