from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_INPUT_TOKEN = "[[INPUT_TEXT]]"

ANNOTATION_SHEET = "annotation"
INSTRUCTIONS_SHEET = "instructions"
MAPPING_SHEET = "mapping"
SAMPLED_PRIVATE_SHEET = "sampled_private"
COLUMN_MAP_SHEET = "column_map"

PARTICIPANT_CODE_HEADER = "Participant Code"
PLACEHOLDER_PREFIX = "REPLACE_"
