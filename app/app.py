"""
MVP demo app — the human-in-the-loop web interface described in
docs/product/MVP_RD_Plan_Software_First.docx.

What this is: a coach/operator logs each delivery (the style bowled and
how the batter handled it). The adaptation engine scores mastery per
style live, and the moment a style is mastered, the app recommends a
genuinely different next style — the coach then manually dials that into
the real bowling machine. No embedded hardware, no CV model wired up yet
(see cv-pipeline/README.md for why) — this proves the decision logic
works and is usable, which is exactly what the MVP is scoped to test.

Run with:  streamlit run app/app.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adaptation-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cv-pipeline"))

import streamlit as st
from scoring import MasteryScorer, OUTCOME_QUALITY, MASTERY_THRESHOLD, MIN_SAMPLE
from recommender import recommend_next_styles, explain_recommendation
from styles import STYLE_LIBRARY, STYLE_BY_KEY
from video_pipeline import estimate_outcome_from_video

st.set_page_config(page_title="AI-Adaptive Bowling Machine — MVP", page_icon="🏏", layout="wide")

if "scorer" not in st.session_state:
    st.session_state.scorer = MasteryScorer()
if "log" not in st.session_state:
    st.session_state.log = []  # list of (style_key, outcome) for display
if "recent_styles" not in st.session_state:
    st.session_state.recent_styles = []

scorer: MasteryScorer = st.session_state.scorer

st.title("🏏 AI-Adaptive Bowling Machine — MVP")
st.caption(
    "Human-in-the-loop prototype. Log what happened on each ball; the engine tracks "
    "mastery per style and recommends a genuinely different one the moment a pattern "
    "is solved — you then set that on the machine yourself."
)

left, right = st.columns([1, 1.4])

with left:
    st.subheader("Log a delivery")
    style_options = {s.label: s.key for s in STYLE_LIBRARY}
    chosen_label = st.selectbox("Style bowled", list(style_options.keys()))
    chosen_key = style_options[chosen_label]

    outcome = st.radio(
        "Outcome", list(OUTCOME_QUALITY.keys()),
        horizontal=True, index=0,
    )

    st.markdown("**Timing & footwork**")
    input_mode = st.radio(
        "How should timing/footwork be judged?",
        ["Enter manually", "Estimate from a video clip"],
        horizontal=True,
    )

    on_time, footwork_correct = True, True
    vision_estimate = None
    ready_to_log = True

    if input_mode == "Enter manually":
        on_time = st.checkbox("Batter was on time", value=True)
        footwork_correct = st.checkbox("Footwork was correct", value=True)
    else:
        st.caption(
            "Uploads a clip of this one delivery through cv-pipeline's pretrained "
            "pose model. Estimates timing/footwork only — not the outcome above, which "
            "still needs your judgement (see cv-pipeline/README.md for what this can "
            "and can't do yet)."
        )
        clip = st.file_uploader("Delivery clip", type=["mp4", "mov", "avi", "mkv"])
        ready_to_log = False
        if clip is not None:
            suffix = os.path.splitext(clip.name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(clip.read())
                tmp_path = tmp.name
            try:
                vision_estimate = estimate_outcome_from_video(tmp_path)
                on_time = vision_estimate.on_time
                footwork_correct = vision_estimate.footwork_correct
                ready_to_log = True
                st.success(
                    f"Estimated from video — on time: {'yes' if on_time else 'no'}, "
                    f"footwork correct: {'yes' if footwork_correct else 'no'}"
                )
                with st.expander("Raw features"):
                    st.json(vision_estimate.features.__dict__)
            except FileNotFoundError:
                st.error(
                    "Pose model not downloaded yet. Run "
                    "`python cv-pipeline/pose_estimation.py` once, then retry."
                )
            except ValueError as e:
                st.error(str(e))
            finally:
                os.unlink(tmp_path)

    if st.button("Log delivery", type="primary", disabled=not ready_to_log):
        record = scorer.record_delivery(chosen_key, outcome, on_time, footwork_correct)
        st.session_state.log.append((chosen_key, outcome))
        st.session_state.recent_styles.append(chosen_key)
        st.session_state.recent_styles = st.session_state.recent_styles[-2:]
        st.rerun()

    st.divider()
    st.subheader("Session log")
    if not st.session_state.log:
        st.write("No deliveries logged yet.")
    else:
        for i, (key, outcome_) in enumerate(reversed(st.session_state.log[-10:]), 1):
            st.write(f"{len(st.session_state.log) - i + 1}. {STYLE_BY_KEY[key].label} → **{outcome_}**")

with right:
    st.subheader("Live mastery scores")
    records = scorer.all_records()
    if not records:
        st.info("Log a few deliveries to see mastery scores build up here.")
    else:
        for key, record in sorted(records.items(), key=lambda kv: -kv[1].score):
            style = STYLE_BY_KEY[key]
            pct = min(record.score, 1.0)
            label = f"{style.label} — {record.deliveries_seen} seen"
            if record.mastered:
                st.success(f"✅ MASTERED — {label} (score {record.score:.2f})")
            else:
                st.write(label)
            st.progress(pct)

    st.divider()
    st.subheader("Recommendation")
    mastered = scorer.mastered_styles()
    if not mastered:
        st.write("No style mastered yet — keep logging deliveries.")
    else:
        # Most recently mastered style drives the recommendation.
        just_mastered = mastered[-1]
        recs = recommend_next_styles(
            just_mastered, top_n=3, exclude=st.session_state.recent_styles
        )
        st.warning(explain_recommendation(just_mastered, recs[0]))
        st.markdown("**Top alternatives, ranked by contrast:**")
        for r in recs:
            st.write(f"- {r.label}")

st.divider()
with st.expander("What's real here vs. what's a placeholder"):
    st.markdown(
        "- **Real, tested logic**: mastery scoring (`adaptation-engine/scoring.py`) and "
        "style recommendation (`adaptation-engine/recommender.py`) — both have passing "
        "unit tests in `adaptation-engine/tests/`.\n"
        "- **Real, working, but unvalidated**: uploading a delivery clip runs a genuine "
        "pretrained pose model (`cv-pipeline/pose_estimation.py`) and feature extraction "
        "(`cv-pipeline/feature_extraction.py`) to estimate timing/footwork automatically. "
        "The estimate itself is a documented heuristic (`cv-pipeline/outcome_bridge.py`), "
        "not a trained classifier, and has never been checked against a real coach's "
        "judgement — see `cv-pipeline/README.md`.\n"
        "- **Placeholder for this MVP, by design**: shot outcome (middled/edged/missed/...) "
        "is always entered by a human — ball tracking against the bat isn't built.\n"
        "- **Not built yet**: any connection to an actual bowling machine. This app only "
        "recommends; a person still sets the real machine."
    )
