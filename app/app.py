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

import csv
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adaptation-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cv-pipeline"))

import streamlit as st
from scoring import MasteryScorer, OUTCOME_QUALITY, MASTERY_THRESHOLD, MIN_SAMPLE
from neural_scorer import NeuralMasteryScorer
from recommender import recommend_next_styles, explain_recommendation
from styles import STYLE_LIBRARY, STYLE_BY_KEY
from video_pipeline import estimate_outcomes_from_video

st.set_page_config(page_title="AI-Adaptive Bowling Machine — MVP", page_icon="🏏", layout="wide")

SCORER_ENGINES = ["Rule-based (EMA threshold)", "Neural (trained MLP)"]


def _new_scorer(engine: str):
    return NeuralMasteryScorer() if engine == "Neural (trained MLP)" else MasteryScorer()


@st.cache_data(show_spinner="Running pose estimation on the clip...")
def _analyse_clip(video_bytes: bytes, suffix: str):
    """Cached on the uploaded file's bytes so re-running the app (e.g. the
    user picking a different delivery from the dropdown below) doesn't
    re-run pose estimation on the whole clip every time — that's real
    compute, not free, especially on a multi-minute session clip."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        return estimate_outcomes_from_video(tmp_path)
    finally:
        os.unlink(tmp_path)


if "scorer_engine" not in st.session_state:
    st.session_state.scorer_engine = SCORER_ENGINES[0]
if "scorer" not in st.session_state:
    st.session_state.scorer = _new_scorer(st.session_state.scorer_engine)
if "log" not in st.session_state:
    st.session_state.log = []  # list of (style_key, outcome) for display
if "recent_styles" not in st.session_state:
    st.session_state.recent_styles = []

with st.sidebar:
    st.header("Session settings")
    chosen_engine = st.radio(
        "Scoring engine", SCORER_ENGINES,
        index=SCORER_ENGINES.index(st.session_state.scorer_engine),
    )
    st.caption(
        "Neural scorer is a real trained model (`adaptation-engine/neural_scorer.py`, "
        "held-out AUC ~0.96) — but trained entirely on simulated batters, never a real "
        "player (see `adaptation-engine/simulator.py`). Switching engines starts a "
        "fresh session rather than comparing mid-session."
    )
    if chosen_engine != st.session_state.scorer_engine:
        try:
            new_scorer = _new_scorer(chosen_engine)
        except FileNotFoundError:
            st.error(
                "Neural scorer model not found. Run "
                "`python adaptation-engine/neural_scorer.py` once to train and save it, "
                "then retry."
            )
        else:
            st.session_state.scorer = new_scorer
            st.session_state.scorer_engine = chosen_engine
            st.session_state.log = []
            st.session_state.recent_styles = []
            st.rerun()

    if st.button("Reset session"):
        st.session_state.scorer = _new_scorer(st.session_state.scorer_engine)
        st.session_state.log = []
        st.session_state.recent_styles = []
        st.rerun()

    if st.session_state.log:
        st.divider()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ball_number", "style_key", "style_label", "outcome"])
        for i, (key, outcome_) in enumerate(st.session_state.log, 1):
            writer.writerow([i, key, STYLE_BY_KEY[key].label, outcome_])
        st.download_button(
            "Download session log (CSV)", buf.getvalue(),
            file_name="session_log.csv", mime="text/csv",
        )

scorer = st.session_state.scorer

st.title("🏏 AI-Adaptive Bowling Machine — MVP")
st.caption(
    "Human-in-the-loop prototype. Log what happened on each ball; the engine tracks "
    "mastery per style and recommends a genuinely different one the moment a pattern "
    "is solved — you then set that on the machine yourself. "
    f"Scoring engine: **{st.session_state.scorer_engine}** (change in the sidebar)."
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
            "Uploads a clip through cv-pipeline's pretrained pose model — a whole nets "
            "session works, not just one pre-trimmed ball; delivery_segmentation.py finds "
            "every delivery in it automatically. Estimates timing/footwork only — not the "
            "outcome above, which still needs your judgement (see cv-pipeline/README.md "
            "for what this can and can't do yet)."
        )
        clip = st.file_uploader("Delivery clip (one ball or a whole session)", type=["mp4", "mov", "avi", "mkv"])
        ready_to_log = False
        if clip is not None:
            suffix = os.path.splitext(clip.name)[1]
            try:
                vision_estimates = _analyse_clip(clip.getvalue(), suffix)
            except FileNotFoundError:
                st.error(
                    "Pose model not downloaded yet. Run "
                    "`python cv-pipeline/pose_estimation.py` once, then retry."
                )
                vision_estimates = None
            except ValueError as e:
                st.error(str(e))
                vision_estimates = None

            if vision_estimates:
                n = len(vision_estimates)
                st.success(f"Detected {n} deliver{'y' if n == 1 else 'ies'} in this clip.")
                if n > 1:
                    idx = st.selectbox(
                        "Which detected delivery does this log entry apply to?",
                        options=list(range(n)),
                        format_func=lambda i: (
                            f"Delivery {i + 1} — on time: "
                            f"{'yes' if vision_estimates[i].on_time else 'no'}, footwork: "
                            f"{'correct' if vision_estimates[i].footwork_correct else 'incorrect'}"
                        ),
                    )
                else:
                    idx = 0
                vision_estimate = vision_estimates[idx]
                on_time = vision_estimate.on_time
                footwork_correct = vision_estimate.footwork_correct
                ready_to_log = True
                with st.expander("Raw features for this delivery"):
                    st.json(vision_estimate.features.__dict__)

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
        "- **Real, tested logic**: mastery scoring (`adaptation-engine/scoring.py`), "
        "the neural alternative (`adaptation-engine/neural_scorer.py`, switchable in "
        "the sidebar), and style recommendation (`adaptation-engine/recommender.py`) — "
        "all have passing unit tests in `adaptation-engine/tests/`.\n"
        "- **Real, but never validated on a real player**: the neural scorer is trained "
        "entirely on `adaptation-engine/simulator.py`'s virtual batters — held-out AUC "
        "~0.96 there, but that's a simulated ground truth, not a real coach's judgement.\n"
        "- **Real, tested against actual footage, still unvalidated**: uploading a clip "
        "runs a genuine pretrained pose model (`cv-pipeline/pose_estimation.py`) and "
        "automatic delivery detection (`cv-pipeline/delivery_segmentation.py`) — a whole "
        "nets session works, not just one pre-trimmed ball; pick which detected delivery "
        "above this log entry applies to. Run against 5 real WhatsApp clips: 26 "
        "deliveries detected across them, and both footwork and timing "
        "(`on_time`, via `footwork_lead_seconds`) discriminate meaningfully instead of "
        "one heuristic being structurally stuck on a single answer (an earlier bug — see "
        "`cv-pipeline/README.md`). That confirms the pipeline measures *something* real, "
        "not that the something is *correct* — nobody has watched the source footage to "
        "confirm the 26 detections are all genuine swings, and no coach's independent "
        "verdict has checked the on_time/footwork numbers yet.\n"
        "- **Placeholder for this MVP, by design**: shot outcome (middled/edged/missed/...) "
        "is always entered by a human — ball tracking against the bat isn't built.\n"
        "- **Not built yet**: any connection to an actual bowling machine. This app only "
        "recommends; a person still sets the real machine."
    )
