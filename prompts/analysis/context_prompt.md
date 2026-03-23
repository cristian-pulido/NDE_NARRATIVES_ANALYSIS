You are given a participant's context narrative from a near-death experience questionnaire.

Classify the tone of the text as exactly one of:
- positive
- negative
- mixed
- neutral

Definitions:
- `mixed`: both positive and negative emotional elements are clearly present
- `neutral`: factual or descriptive language with little or no emotional valence

Return JSON only with this structure:
{
  "context": {
    "tone": "positive | negative | mixed | neutral",
    "evidence_segments": ["short verbatim span 1"]
  }
}

Use 1 to 3 short verbatim evidence spans copied directly from the input text.

Evidence constraints:
- Evidence must be literal substrings from the participant text.
- Do not output placeholders or meta text such as "<INPUT_TEXT>", "[[INPUT_TEXT]]", "Text:", or "No text provided".
- Do not invent evidence; only quote what is actually present in the text.

Text:
[[INPUT_TEXT]]
