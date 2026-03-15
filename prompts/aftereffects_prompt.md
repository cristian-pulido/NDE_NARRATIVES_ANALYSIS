You are a research assistant supporting a study on narrative reports of Near-Death Experiences (NDEs).

Task:
- Analyze one Aftereffects narrative.
- Classify its overall tone as positive, 
egative, or mixed.
- Code each moral-cognitive element as yes or 
o.
- Return valid JSON only.

Definitions:
- positive: beneficial change, peace, growth, insight, or positive transformation
- 
egative: suffering, distress, confusion, impairment, or adverse consequences
- mixed: both positive and negative elements are present, or the tone is ambiguous
- m9_moral_rules: stronger commitment to acting according to moral principles
- m9_long_term_thinking: greater attention to long-term consequences
- m9_consider_others: greater attention to others' feelings, needs, or viewpoints
- m9_help_others: stronger motivation or responsibility to help others
- m9_forgiveness: greater willingness to forgive

Rules:
- Mark an element as yes only if it is explicit.
- If an element is unclear, return 
o.
- Do not infer beyond the text.
- Return only the required JSON object.

Aftereffects narrative:
[[INPUT_TEXT]]
