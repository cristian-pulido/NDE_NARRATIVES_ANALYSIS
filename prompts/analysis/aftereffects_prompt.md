You are given a participant's aftereffects narrative from a near-death experience questionnaire.

Classify:
- overall tone as one of `positive`, `negative`, `mixed`, `neutral`
- whether the text explicitly mentions each long-term change as `yes` or `no`

Definitions:
- `mixed`: both positive and negative emotional elements are clearly present
- `neutral`: factual or descriptive language with little or no emotional valence

Return JSON only with this structure:
{
  "aftereffects": {
    "tone": "positive | negative | mixed | neutral",
    "evidence_segments": ["short verbatim span 1"],
    "m9_moral_rules": "yes | no",
    "m9_long_term_thinking": "yes | no",
    "m9_consider_others": "yes | no",
    "m9_help_others": "yes | no",
    "m9_forgiveness": "yes | no"
  }
}

Only mark `yes` when the feature is explicitly present in the text.
Use 1 to 3 short verbatim evidence spans copied directly from the input text.

Evidence constraints:
- Evidence must be literal substrings from the participant text.
- Do not output placeholders or meta text such as "<INPUT_TEXT>", "[[INPUT_TEXT]]", "Text:", or "No text provided".
- Do not invent evidence; only quote what is actually present in the text.

Text:
[[INPUT_TEXT]]
