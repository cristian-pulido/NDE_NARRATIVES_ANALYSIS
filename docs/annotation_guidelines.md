# Annotation Guidelines

## Purpose

The annotation task transforms open-ended NDE narratives into a structured format that can be compared with:

- human judgments
- LLM predictions
- questionnaire-based measures from M8 and M9

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
- If an element is implied but not explicit, code `no`.
- If an element is denied, code `no`.
- Prioritize the participant's framing of the event.

## Evaluation Workflow Note

- If a case should not be evaluated, leave the full row unlabeled or remove the row from the completed workbook.
- Do not leave a row partially completed. `nde evaluate` treats partially filled rows as invalid and will stop with an error.
- Do not edit `Participant Code`.

## Tone Labels

- `positive`: peace, relief, clarity, insight, beneficial meaning, favorable transformation
- `negative`: fear, distress, confusion, suffering, threat, adverse interpretation
- `mixed`: clear positive and negative elements, ambivalence, or unresolved tone

## Experience Elements

Code these only from the `experience` narrative:

- `m8_out_of_body`: leaving the body, floating above it, viewing the body from outside
- `m8_bright_light`: bright, radiant, or luminous light
- `m8_peace`: intense peace, calm, serenity, or wellbeing
- `m8_time_distortion`: time slowing, speeding up, stopping, or losing meaning
- `m8_presence`: an encountered being, entity, or felt presence

## Aftereffects Elements

Code these only from the `aftereffects` narrative:

- `m9_moral_rules`: stronger commitment to moral principles
- `m9_long_term_thinking`: greater attention to long-term consequences
- `m9_consider_others`: stronger consideration of others' feelings or perspectives
- `m9_help_others`: stronger responsibility or willingness to help others
- `m9_forgiveness`: greater willingness to forgive

## Allowed Outputs

- Tone columns: `positive`, `negative`, `mixed`
- Element columns: `yes`, `no`
