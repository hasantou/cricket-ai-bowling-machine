# app/

The MVP demo — a Streamlit web app implementing the human-in-the-loop
design from `docs/product/MVP_RD_Plan_Software_First.docx`.

## Run it

```
pip install -r requirements.txt
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Log a handful of deliveries against the
same style with good outcomes (e.g. "middled") and watch it get marked
mastered, then see the recommended next style appear.

## What's real vs. placeholder

- **Real, tested**: the mastery-scoring and style-recommendation logic in
  `adaptation-engine/` — see its test suite (`pytest adaptation-engine/tests/`).
- **Placeholder, on purpose**: delivery outcomes are typed in by a human
  here instead of detected by computer vision — that's the deliberate
  Phase 1 scope (`docs/product/MVP_RD_Plan_Software_First.docx`, Section
  2.1). Wiring this up to `cv-pipeline/` is later work.
- **Not built**: any link to a physical bowling machine.
