You are given a participant's aftereffects narrative from a near-death experience questionnaire.

Classify:
- overall tone as one of `positive`, `negative`, `mixed`
- whether the text explicitly mentions each long-term change as `yes` or `no`

Return JSON only with these fields:
- `aftereffects_tone`
- `m9_moral_rules`
- `m9_long_term_thinking`
- `m9_consider_others`
- `m9_help_others`
- `m9_forgiveness`

Only mark `yes` when the feature is explicitly present in the text.

Text:
<INPUT_TEXT>
