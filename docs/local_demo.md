# Local Demo Guide (Ollama)

This guide explains how to run the local interactive web app for quick narrative trials without launching full experiment batches.

## Purpose

The local demo is designed for rapid exploration and sanity checks:

- test prompts and models with custom text
- inspect structured outputs in tables
- compare optional user-provided valence against detected overall experience tone

It is not a replacement for the full experiment + evaluation workflow.

## Prerequisites

1. Repository installed with UI extras:

   ```bash
   pip install -e .[ui]
   ```

2. Ollama running locally (or reachable from the host running the app):

   - default expected endpoint: `http://localhost:11434`
   - at least one pulled model available in Ollama

3. `config/paths.local.toml` configured for your environment.

## Launch

Run from repository root:

```bash
nde local-demo
```

Optional host/port override:

```bash
nde local-demo --host 0.0.0.0 --port 7860
```

## Input Modes

### Three Sections (Required)

Provide all three narrative parts explicitly:

- Before the NDE (Context)
- Core NDE Experience
- Aftereffects

### Single Narrative (Auto-segment)

Provide one narrative text. The app segments it into the three sections before running structured analysis.

## Outputs

The demo uses human-readable tables by default:

1. **Section-Level Summary**
   - narrative part
   - detected tone
   - context type (for context section)
   - supporting evidence

2. **Extracted Dimensions**
   - extracted dimensions and values in readable labels
   - includes tone/context/evidence and section-specific coded dimensions

3. **Segmented Narrative**
   - the final text slices used for model analysis

4. **Valence Alignment**
   - optional check only when valence is provided by the user
   - compares provided valence with model-detected overall experience tone
   - returns evidence snippets used for the comparison

## Optional Valence Behavior

- If valence is blank: the app skips alignment and reports that no alignment check was executed.
- If valence is provided: the app reports match/mismatch against overall experience tone and shows evidence.
- Valence input affects only the alignment message, not extraction tables.

## Remote Server Usage (SSH)

If the app is running on a remote server, use SSH tunnel forwarding to open it from your local browser:

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server>
```

Then open:

`http://127.0.0.1:7860`

## About `--share`

`--share` creates a temporary public Gradio URL, but this is usually not useful when your backend model is local Ollama (`localhost:11434`) on the server host. Public users can open the page but may not be able to execute model inference unless backend connectivity is properly exposed.

## Troubleshooting

- **No models in dropdown**: verify Ollama is running and check `Ollama Base URL`.
- **Model call errors**: confirm selected model is pulled in Ollama.
- **Remote access issue**: verify SSH tunnel or host/port binding settings.
- **Empty dimensions table**: check whether the model returned only minimal fields; section summary and evidence should still populate.
