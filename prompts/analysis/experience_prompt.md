You are given a participant's main near-death experience narrative.

Classify:
- overall tone as one of `positive`, `negative`, `mixed`, `neutral`
- whether the text explicitly mentions each feature as `yes` or `no`

Definitions:
- `mixed`: both positive and negative emotional elements are clearly present
- `neutral`: factual or descriptive language with little or no emotional valence

Return JSON only with this structure:
{
  "experience": {
    "tone": "positive | negative | mixed | neutral",
    "evidence_segments": ["short verbatim span 1"],
    "m8_out_of_body": "yes | no",
    "m8_bright_light": "yes | no",
    "m8_peace": "yes | no",
    "m8_time_distortion": "yes | no",
    "m8_presence": "yes | no"
  }
}

Only mark `yes` when the feature is explicitly present in the text.
Use 1 to 3 short verbatim evidence spans copied directly from the input text.

Evidence constraints:
- Evidence must be literal substrings from the participant text.
- Evidence spans must directly justify the assigned tone label (not generic summary text).
- Do not output placeholders or meta text such as "<INPUT_TEXT>", "[[INPUT_TEXT]]", "Text:", or "No text provided".
- Do not invent evidence; only quote what is actually present in the text.

Text:
[[INPUT_TEXT]]
