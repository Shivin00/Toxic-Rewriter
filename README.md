# Toxic Rewriter

Toxic Rewriter is an advanced Streamlit NLP project for rewriting toxic, passive-aggressive, or emotionally charged text into safer communication while preserving intent, assertiveness, and context.

## What It Does

- Detects toxicity before and after rewriting
- Scores multiple toxicity dimensions instead of using a single label
- Rewrites text into different politeness and communication modes
- Uses context, persona, emotional target, and firmness controls
- Preserves user intent such as complaint, disagreement, or boundary-setting
- Explains exactly what was changed and why
- Flags hidden toxicity and residual abuse risk
- Tracks session-local feedback to simulate an adaptive tone profile

## Implemented Features

- Multi-dimension toxicity classification:
  hate speech, harassment, threats, profanity, discrimination, sarcasm, gaslighting, and microaggressions
- Hidden toxicity detection:
  passive aggression, manipulative politeness, and condescension
- Multiple politeness modes:
  formal professional, casual friendly, empathetic supportive, assertive non-toxic, therapeutic, diplomatic corporate, and neutral factual
- Context-aware rewriting:
  conversation context, persona-sensitive rewriting, and emotional rewrite direction
- Intent preservation engine:
  inferred goal, target entity, and firmness slider
- Rewrite explainability:
  highlighted replacements, toxicity map, and a detected -> interpreted -> rewritten -> safety flow
- Safety layer:
  residual threat/manipulation checks on the output
- Communication scoring:
  toxicity, politeness, assertiveness, empathy, clarity, professionalism, and meaning retention
- Advanced editor behavior:
  live preview, side-by-side comparison, batch mode, and CSV export
- Privacy-first demo behavior:
  no persistent storage of feedback in the current prototype

## Project Structure

```text
Toxicity-Rewriter/
|-- app.py
|-- main.py
|-- requirements.txt
|-- README.md
|-- .streamlit/
|   `-- config.toml
`-- toxic_rewriter/
    |-- __init__.py
    |-- nlp.py
    `-- scoring.py
```

## Core Techniques

- `unitary/toxic-bert` for toxicity classification when available
- `google/flan-t5-base` for text rewriting when available
- Heuristic fallback layers for reliable Streamlit deployment
- Lexicon and pattern-based toxicity dimension scoring
- Intent inference and safety heuristics
- Semantic similarity scoring for meaning retention

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

## Demo Flow

1. Open `Rewrite Studio`.
2. Paste a harsh or passive-aggressive message.
3. Add conversation context and choose a persona.
4. Select a politeness mode and emotional direction.
5. Adjust the firmness slider.
6. Review the toxicity map, explainability flow, safety layer, and rewritten output.
7. Use `Batch Lab` for a multi-example experiment.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app from your repository.
4. Set the main file path to `app.py`.
5. Deploy.

If hosted model loading fails, the app still works through the built-in heuristic fallback.

## Notes on Scope

This version implements a strong prototype of many advanced features using heuristics and model-assisted rewriting when available. The following ideas are partially represented as product direction rather than full production systems:

- Adaptive learning with persistent personalization
- Multi-language rewriting
- Real-time API integrations
- Encrypted enterprise deployment
- Conflict mediation for full conversations

## Strong Next Steps

- add multilingual support with locale-specific tone rules
- expose a lightweight API for Slack, Discord, or CRM integration
- store user feedback in an opt-in local profile
- fine-tune a dedicated politeness-transfer model
- add full conversation mediation and dual-rewrite mode
