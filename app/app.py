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
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adaptation-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cv-pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trajectory-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "machine-control"))

import streamlit as st
from scoring import MasteryScorer, OUTCOME_QUALITY, MASTERY_THRESHOLD, MIN_SAMPLE
from neural_scorer import NeuralMasteryScorer
from recommender import recommend_next_styles, explain_recommendation
from styles import STYLE_LIBRARY, STYLE_BY_KEY
from video_pipeline import estimate_outcomes_from_video
from cricket_trajectory import (
    BallProperties, Environment, PlayerProfile, expected_success, OUTCOME_SCORES,
    Scorecard, classify_delivery_legality, run_simulation,
)
from cricket_trajectory.adaptive import suggest_next_delivery, delivery_difficulty_rating
from cricket_trajectory.machine import WheelMachine
from machine_control.session_store import save_profile, load_profile, save_scorecard, load_scorecard
from machine_control.controller import SimulatedMachineController
from machine_control.safety import SafeMachineController, SafetyLimits, SafetyViolation

st.set_page_config(page_title="AI-Adaptive Bowling Machine — MVP", page_icon="🏏", layout="wide")

# Vintage cricket-pavilion styling on top of .streamlit/config.toml's base
# theme — a serif display face for headings (evoking an old club
# noticeboard) and a typewriter face specifically for the scorecard text
# (st.text output below), since a scorecard is exactly the kind of thing
# that would once have been typed on a real typewriter. Google Fonts is on
# the artifact/app CDN allowlist; this is cosmetic only, nothing here
# depends on it loading.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Courier+Prime&display=swap');

h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif !important; }
.stApp { background-color: #f4ecd8; }
div[data-testid="stMetricValue"] { font-family: 'Playfair Display', Georgia, serif !important; }
pre, code, div[data-testid="stText"] {
    font-family: 'Courier Prime', 'Courier New', monospace !important;
}
hr { border-top: 1px solid #b8a888 !important; }
</style>
""", unsafe_allow_html=True)

ENGINE_FAMILIES = ["Style library (adaptation-engine)", "Physics-based (trajectory-engine)"]
SCORER_ENGINES = ["Rule-based (EMA threshold)", "Neural (trained MLP)"]


def _new_scorer(engine: str):
    return NeuralMasteryScorer() if engine == "Neural (trained MLP)" else MasteryScorer()


def _style_session_to_json(scorer_engine, log) -> str:
    """The style-library engine has no dedicated persistence module (its
    scorer state lives inside MasteryScorer/NeuralMasteryScorer's private
    records) — reconstructing by replaying `log` through record_delivery()
    respects that encapsulation instead of reaching into private
    attributes, and only needs the log itself to be saved."""
    return json.dumps({"scorer_engine": scorer_engine, "log": log}, indent=2)


def _style_session_from_json(data: bytes):
    parsed = json.loads(data)
    scorer_engine = parsed["scorer_engine"]
    scorer = _new_scorer(scorer_engine)
    log = [tuple(entry) for entry in parsed["log"]]
    for key, outcome, on_time, footwork_correct in log:
        scorer.record_delivery(key, outcome, on_time, footwork_correct)
    recent_styles = [key for key, *_ in log[-2:]]
    return scorer, scorer_engine, log, recent_styles


def _traj_session_to_json(profile, card) -> str:
    """Reuses machine-control's actual save_profile/save_scorecard —
    round-tripped through real temp files (same pattern already used for
    video uploads in this file) rather than reimplementing the same
    serialization inline — and bundled into one file for a single
    download instead of two."""
    with tempfile.TemporaryDirectory() as d:
        p_path, c_path = os.path.join(d, "profile.json"), os.path.join(d, "card.json")
        save_profile(profile, p_path)
        save_scorecard(card, c_path)
        with open(p_path) as f:
            profile_data = json.load(f)
        with open(c_path) as f:
            card_data = json.load(f)
    return json.dumps({"profile": profile_data, "card": card_data}, indent=2)


def _traj_session_from_json(data: bytes):
    combined = json.loads(data)
    with tempfile.TemporaryDirectory() as d:
        p_path, c_path = os.path.join(d, "profile.json"), os.path.join(d, "card.json")
        with open(p_path, "w") as f:
            json.dump(combined["profile"], f)
        with open(c_path, "w") as f:
            json.dump(combined["card"], f)
        return load_profile(p_path), load_scorecard(c_path)


def _margin_for_target(target_success: float) -> float:
    """Inverts the Elo expected-score curve used by expected_success() so a
    chosen target win rate (e.g. 0.5) turns into the challenge_margin
    suggest_next_delivery() actually searches around — clamped away from
    the 0/1 edges where the inverse blows up."""
    target_success = min(max(target_success, 0.05), 0.95)
    return 400.0 * math.log10((1.0 - target_success) / target_success)


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


if "engine_family" not in st.session_state:
    st.session_state.engine_family = ENGINE_FAMILIES[0]
if "scorer_engine" not in st.session_state:
    st.session_state.scorer_engine = SCORER_ENGINES[0]
if "scorer" not in st.session_state:
    st.session_state.scorer = _new_scorer(st.session_state.scorer_engine)
if "log" not in st.session_state:
    st.session_state.log = []  # list of (style_key, outcome) for display
if "recent_styles" not in st.session_state:
    st.session_state.recent_styles = []

if "traj_ball" not in st.session_state:
    st.session_state.traj_ball = BallProperties()
if "traj_env" not in st.session_state:
    st.session_state.traj_env = Environment()
if "traj_profile" not in st.session_state:
    st.session_state.traj_profile = PlayerProfile(name="Player 1", rating=1000.0)
if "traj_card" not in st.session_state:
    st.session_state.traj_card = Scorecard(batter_name=st.session_state.traj_profile.name)
if "traj_target_success" not in st.session_state:
    st.session_state.traj_target_success = 0.5
if "traj_next_delivery" not in st.session_state:
    st.session_state.traj_next_delivery = suggest_next_delivery(
        st.session_state.traj_profile, st.session_state.traj_ball,
        challenge_margin=_margin_for_target(st.session_state.traj_target_success),
    )
if "traj_controller" not in st.session_state:
    # No real machine exists to connect to yet - SimulatedMachineController
    # stands in for one, wrapped in the same SafeMachineController a real
    # driver would be. Swapping the simulated controller for a real one
    # later is the only change needed anywhere in this app; every button
    # below already calls the real machine-control interface, not a mock
    # built for the UI.
    st.session_state.traj_controller = SafeMachineController(
        SimulatedMachineController(cycle_time_s=1.5), SafetyLimits(max_wheel_rpm=6000.0)
    )

with st.sidebar:
    st.header("Session settings")
    chosen_family = st.radio(
        "Engine", ENGINE_FAMILIES,
        index=ENGINE_FAMILIES.index(st.session_state.engine_family),
    )
    st.caption(
        "Two independent, uncalibrated-against-each-other approaches, kept as separate "
        "engines to compare rather than merged — see `trajectory-engine/README.md` and "
        "the comparison in project notes for why."
    )
    if chosen_family != st.session_state.engine_family:
        st.session_state.engine_family = chosen_family
        st.rerun()

    st.divider()

    if st.session_state.engine_family == ENGINE_FAMILIES[0]:
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
            writer.writerow(["ball_number", "style_key", "style_label", "outcome", "on_time", "footwork_correct"])
            for i, (key, outcome_, on_time_, footwork_) in enumerate(st.session_state.log, 1):
                writer.writerow([i, key, STYLE_BY_KEY[key].label, outcome_, on_time_, footwork_])
            st.download_button(
                "Download session log (CSV)", buf.getvalue(),
                file_name="session_log.csv", mime="text/csv",
            )

        st.divider()
        st.caption(
            "Save/load restores your scorer + full delivery log — download it before "
            "closing the tab, since nothing here is stored anywhere else (Streamlit Cloud "
            "resets on every redeploy)."
        )
        st.download_button(
            "Save session (JSON)",
            _style_session_to_json(st.session_state.scorer_engine, st.session_state.log),
            file_name="style_session.json", mime="application/json",
            disabled=not st.session_state.log,
        )
        uploaded = st.file_uploader("Load session (JSON)", type=["json"], key="style_session_upload")
        if uploaded is not None and st.button("Apply loaded session"):
            try:
                scorer, scorer_engine, log, recent_styles = _style_session_from_json(uploaded.getvalue())
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                st.error(f"Couldn't load that session file: {e}")
            else:
                st.session_state.scorer = scorer
                st.session_state.scorer_engine = scorer_engine
                st.session_state.log = log
                st.session_state.recent_styles = recent_styles
                st.rerun()
    else:
        st.caption(
            "Elo-style player rating vs. a delivery-difficulty rating computed directly "
            "from real aerodynamics (`trajectory-engine/cricket_trajectory/adaptive.py`) "
            "— continuously varied deliveries, not a fixed style list. Wide/no-ball is "
            "ruled automatically from the simulated flight (`scorecard.py`) — only a fair "
            "ball needs your judgement call. Untrained (no simulated-data caveat), but its "
            "Elo tuning is unvalidated against a real, continuously-improving player — see "
            "project notes for the measured ~100-point rating lag against a simulated "
            "improving batter."
        )

        target_pct = st.slider(
            "Target success rate", min_value=30, max_value=70,
            value=int(round(st.session_state.traj_target_success * 100)), step=5,
            format="%d%%",
        )
        st.caption(
            "What share of deliveries the engine aims for you to handle well. Lower = "
            "harder deliveries picked; higher = easier ones — it's a target the engine "
            "aims at, not a guarantee for any single ball, so expect it to land a few "
            "points either side."
        )
        new_target = target_pct / 100.0
        if abs(new_target - st.session_state.traj_target_success) > 1e-9:
            st.session_state.traj_target_success = new_target
            st.session_state.traj_next_delivery = suggest_next_delivery(
                st.session_state.traj_profile, st.session_state.traj_ball,
                challenge_margin=_margin_for_target(new_target),
            )
            st.rerun()

        if st.button("Reset session", key="reset_traj"):
            st.session_state.traj_profile = PlayerProfile(name="Player 1", rating=1000.0)
            st.session_state.traj_card = Scorecard(batter_name=st.session_state.traj_profile.name)
            st.session_state.traj_next_delivery = suggest_next_delivery(
                st.session_state.traj_profile, st.session_state.traj_ball,
                challenge_margin=_margin_for_target(st.session_state.traj_target_success),
            )
            st.session_state.traj_controller = SafeMachineController(
                SimulatedMachineController(cycle_time_s=1.5), SafetyLimits(max_wheel_rpm=6000.0)
            )
            st.rerun()

        if st.session_state.traj_card.balls:
            st.divider()
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["over", "ball_in_over", "delivery_label", "delivery_difficulty",
                              "legality", "outcome", "runs", "is_wicket", "dismissal",
                              "rating_before", "rating_after"])
            for r in st.session_state.traj_card.balls:
                writer.writerow([
                    r.over, r.ball_in_over, r.delivery_label, f"{r.delivery_difficulty:.0f}",
                    r.legality or "", r.outcome or "", r.runs, r.is_wicket, r.dismissal or "",
                    f"{r.player_rating_before:.0f}" if r.player_rating_before is not None else "",
                    f"{r.player_rating_after:.0f}" if r.player_rating_after is not None else "",
                ])
            st.download_button(
                "Download session log (CSV)", buf.getvalue(),
                file_name="trajectory_session_log.csv", mime="text/csv",
            )

        st.divider()
        st.caption(
            "Save/load restores your player rating + full scorecard "
            "(`machine-control/session_store.py`) — download it before closing the "
            "tab, since nothing here is stored anywhere else (Streamlit Cloud resets "
            "on every redeploy)."
        )
        st.download_button(
            "Save session (JSON)",
            _traj_session_to_json(st.session_state.traj_profile, st.session_state.traj_card),
            file_name="trajectory_session.json", mime="application/json",
            disabled=not st.session_state.traj_card.balls,
        )
        traj_uploaded = st.file_uploader("Load session (JSON)", type=["json"], key="traj_session_upload")
        if traj_uploaded is not None and st.button("Apply loaded session", key="apply_traj_session"):
            try:
                profile, card = _traj_session_from_json(traj_uploaded.getvalue())
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                st.error(f"Couldn't load that session file: {e}")
            else:
                st.session_state.traj_profile = profile
                st.session_state.traj_card = card
                st.session_state.traj_next_delivery = suggest_next_delivery(
                    profile, st.session_state.traj_ball,
                    challenge_margin=_margin_for_target(st.session_state.traj_target_success),
                )
                st.rerun()

scorer = st.session_state.scorer

st.title("🏏 AI-Adaptive Bowling Machine — MVP")

if st.session_state.engine_family == ENGINE_FAMILIES[0]:
    st.caption(
        "Human-in-the-loop prototype. Log what happened on each ball; the engine tracks "
        "mastery per style and recommends a genuinely different one the moment a pattern "
        "is solved — you then set that on the machine yourself. "
        f"Scoring engine: **{st.session_state.scorer_engine}** (change in the sidebar)."
    )
else:
    st.caption(
        "Physics-based engine. Every delivery is a continuously varied set of real "
        "parameters (speed, seam angle, spin), not picked from a fixed list — the "
        f"engine targets keeping your expected success near "
        f"**{st.session_state.traj_target_success*100:.0f}%** (the 'challenge point', "
        "adjustable in the sidebar), rather than waiting for a fixed style to be "
        "'mastered'. Swing is modelled on "
        "the seam-turbulence mechanism from Mehta, R.D. (1985), *Aerodynamics of "
        "Sports Balls*, Annual Review of Fluid Mechanics, 17:151-189 — not a guessed "
        "curve (see `trajectory-engine/cricket_trajectory/aerodynamics.py`)."
    )

if st.session_state.engine_family == ENGINE_FAMILIES[0]:
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Log a delivery")
        style_options = {s.label: s.key for s in STYLE_LIBRARY}
        chosen_label = st.selectbox("Style bowled", list(style_options.keys()))
        chosen_key = style_options[chosen_label]

        st.markdown("**Timing & footwork**")
        input_mode = st.radio(
            "How should timing/footwork be judged?",
            ["Enter manually", "Estimate from a video clip"],
            horizontal=True,
        )

        if input_mode == "Enter manually":
            outcome = st.radio(
                "Outcome", list(OUTCOME_QUALITY.keys()),
                horizontal=True, index=0,
            )
            on_time = st.checkbox("Batter was on time", value=True)
            footwork_correct = st.checkbox("Footwork was correct", value=True)

            if st.button("Log delivery", type="primary"):
                scorer.record_delivery(chosen_key, outcome, on_time, footwork_correct)
                st.session_state.log.append((chosen_key, outcome, on_time, footwork_correct))
                st.session_state.recent_styles.append(chosen_key)
                st.session_state.recent_styles = st.session_state.recent_styles[-2:]
                st.rerun()
        else:
            st.caption(
                "Uploads a clip through cv-pipeline's pretrained pose model — a whole nets "
                "session works, not just one pre-trimmed ball; delivery_segmentation.py finds "
                "every delivery in it automatically and estimates timing/footwork for each one. "
                "Outcome still needs your judgement per ball (see cv-pipeline/README.md for "
                "what this can and can't do yet) — set it below for each detected delivery, "
                "then log them all in one go."
            )
            clip = st.file_uploader("Delivery clip (one ball or a whole session)", type=["mp4", "mov", "avi", "mkv"])
            vision_estimates = None
            if clip is not None:
                suffix = os.path.splitext(clip.name)[1]
                try:
                    vision_estimates = _analyse_clip(clip.getvalue(), suffix)
                except FileNotFoundError:
                    st.error(
                        "Pose model not downloaded yet. Run "
                        "`python cv-pipeline/pose_estimation.py` once, then retry."
                    )
                except ValueError as e:
                    st.error(str(e))

            row_outcomes = []
            if vision_estimates:
                n = len(vision_estimates)
                st.success(
                    f"Detected {n} deliver{'y' if n == 1 else 'ies'} in this clip — "
                    f"all will be logged as **{chosen_label}**. Set each one's outcome:"
                )
                outcome_keys = list(OUTCOME_QUALITY.keys())
                for i, est in enumerate(vision_estimates):
                    label_col, outcome_col = st.columns([2, 1])
                    label_col.write(
                        f"**Delivery {i + 1}** — on time: {'yes' if est.on_time else 'no'}, "
                        f"footwork: {'correct' if est.footwork_correct else 'incorrect'}"
                    )
                    row_outcome = outcome_col.selectbox(
                        f"Outcome for delivery {i + 1}", outcome_keys,
                        key=f"video_outcome_{i}", label_visibility="collapsed",
                    )
                    row_outcomes.append(row_outcome)
                with st.expander("Raw features for all detected deliveries"):
                    st.json([est.features.__dict__ for est in vision_estimates])

            if st.button(
                f"Log {len(row_outcomes)} deliver{'y' if len(row_outcomes) == 1 else 'ies'}"
                if row_outcomes else "Log deliveries",
                type="primary", disabled=not row_outcomes,
            ):
                for est, row_outcome in zip(vision_estimates, row_outcomes):
                    scorer.record_delivery(chosen_key, row_outcome, est.on_time, est.footwork_correct)
                    st.session_state.log.append((chosen_key, row_outcome, est.on_time, est.footwork_correct))
                    st.session_state.recent_styles.append(chosen_key)
                st.session_state.recent_styles = st.session_state.recent_styles[-2:]
                st.rerun()

        st.divider()
        st.subheader("Session log")
        if not st.session_state.log:
            st.write("No deliveries logged yet.")
        else:
            for i, (key, outcome_, *_rest) in enumerate(reversed(st.session_state.log[-10:]), 1):
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

else:
    left, right = st.columns([1, 1.4])
    profile = st.session_state.traj_profile
    ball = st.session_state.traj_ball
    env = st.session_state.traj_env
    card = st.session_state.traj_card
    next_ball = st.session_state.traj_next_delivery
    controller = st.session_state.traj_controller
    if "traj_delivery_sent" not in st.session_state:
        st.session_state.traj_delivery_sent = False

    # Simulate the flight up front — legality (wide/no-ball) is judged from
    # this trajectory alone, before any outcome is even asked for.
    sim_result = run_simulation(ball, env, next_ball)
    legality = classify_delivery_legality(sim_result)

    with left:
        st.subheader("Next delivery to bowl")
        d_rating = delivery_difficulty_rating(next_ball, ball)
        pre_expected = expected_success(profile.rating, d_rating)
        st.info(f"**{next_ball.label}**")
        st.caption(
            f"Difficulty rating {d_rating:.0f} vs. your rating {profile.rating:.0f} → "
            f"expected success **{pre_expected*100:.0f}%** "
            f"(engine targets ~{st.session_state.traj_target_success*100:.0f}%, "
            "adjustable in the sidebar)"
        )
        spin_mag = sum(w * w for w in next_ball.spin_rad_s) ** 0.5
        machine = WheelMachine()
        rpm1, rpm2 = machine.wheel_rpms_for_delivery(next_ball.speed_mps, spin_mag, ball)

        st.markdown("**Machine connection**")
        st.caption(
            "No real machine is wired in yet — this runs against "
            "`SimulatedMachineController`, wrapped in the same `SafeMachineController` "
            "hard-limit/kill-switch layer a real driver would be "
            "(`machine-control/machine_control/`). Swapping the simulated controller for "
            "a real one is the only change needed anywhere in this app; every button "
            "here already calls the real interface, not something built just for this UI."
        )
        status_col1, status_col2 = st.columns(2)
        status_col1.metric("Enabled", "yes" if controller.enabled else "NO — stopped")
        status_col2.metric("Ready for next command", "yes" if controller.is_ready() else "not yet")

        if not controller.enabled:
            st.error("Machine is emergency-stopped. Confirm it's safe before resuming.")
            if st.button("Reset (confirm safe to resume)", type="primary"):
                controller.reset()
                st.rerun()
        elif not st.session_state.traj_delivery_sent:
            if st.button(f"Send to machine — wheel 1 ≈ {rpm1:.0f} rpm, wheel 2 ≈ {rpm2:.0f} rpm", type="primary"):
                try:
                    controller.set_delivery(rpm1, rpm2)
                except SafetyViolation as e:
                    st.error(f"Machine refused this command: {e}")
                except RuntimeError as e:
                    st.warning(f"Machine not ready yet: {e}")
                else:
                    st.session_state.traj_delivery_sent = True
                    st.rerun()
            if st.button("Emergency stop", key="estop_before_send"):
                controller.emergency_stop()
                st.rerun()
        else:
            st.success("Command sent — this delivery has been bowled (simulated).")
            if st.button("Emergency stop", key="estop_after_send"):
                controller.emergency_stop()
                st.session_state.traj_delivery_sent = False
                st.rerun()

            if legality is not None:
                st.warning(
                    f"Simulated flight rules this a **{legality.upper()}** — computed directly "
                    "from the trajectory (line/height at the batting crease), no umpire input "
                    "needed for this part. Scores as +1 extra; the machine re-bowls."
                )
                if st.button("Confirm — re-bowl", type="primary"):
                    card.record_ball(profile, ball, next_ball, sim_result)
                    st.session_state.traj_next_delivery = suggest_next_delivery(
                        profile, ball,
                        challenge_margin=_margin_for_target(st.session_state.traj_target_success),
                    )
                    st.session_state.traj_delivery_sent = False
                    st.rerun()
            else:
                outcome = st.radio(
                    "Outcome", list(OUTCOME_SCORES.keys()),
                    horizontal=True, index=0,
                )
                if st.button("Log delivery", type="primary"):
                    card.record_ball(profile, ball, next_ball, sim_result, outcome=outcome)
                    st.session_state.traj_next_delivery = suggest_next_delivery(
                        profile, ball,
                        challenge_margin=_margin_for_target(st.session_state.traj_target_success),
                    )
                    st.session_state.traj_delivery_sent = False
                    st.rerun()

        inner_history = getattr(controller._inner, "history", None)
        if inner_history:
            with st.expander(f"Machine command log ({len(inner_history)} sent)"):
                for cmd in reversed(inner_history[-10:]):
                    tag = "EMERGENCY STOP" if cmd.stopped else f"{cmd.wheel1_rpm:.0f} / {cmd.wheel2_rpm:.0f} rpm"
                    st.write(f"- {tag}")

        st.divider()
        st.subheader("Scorecard")
        if not card.balls:
            st.write("No deliveries logged yet.")
        else:
            st.text(card.render_text())

    with right:
        st.subheader("Player rating")
        st.metric(profile.name, f"{profile.rating:.0f}", help=profile.skill_tier())
        st.caption(f"Skill tier: **{profile.skill_tier()}**")

        scored_balls = [r for r in card.balls if r.player_rating_after is not None]
        if scored_balls:
            st.divider()
            st.subheader("Rating over the session")
            ratings = [scored_balls[0].player_rating_before] + [r.player_rating_after for r in scored_balls]
            st.line_chart(ratings)

            st.divider()
            st.subheader("Expected success per ball")
            st.caption(
                f"Engine aims to keep this near {st.session_state.traj_target_success*100:.0f}% "
                "(the 'challenge point', adjustable in the sidebar) — not too easy, not too hard."
            )
            expected_per_ball = [
                expected_success(r.player_rating_before, r.delivery_difficulty) for r in scored_balls
            ]
            st.line_chart(expected_per_ball)
        else:
            st.info("Log a few deliveries to see the rating and challenge tracking build up here.")

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
        "is always entered by a human in the style-library engine — ball tracking against "
        "the bat isn't built.\n"
        "- **Physics-based engine (`trajectory-engine/`)**: the trajectory simulation, "
        "twin-wheel machine mapping, and Elo-style adaptive rating are real, verified by "
        "hand (see project notes) and by running the package's own demos — not yet by a "
        "test suite of its own. Wide/no-ball rulings are computed automatically from the "
        "simulated flight (`scorecard.py`) with no human input; a fair ball's outcome "
        "(runs/wicket) still needs a human judgement call, same honesty gap as the "
        "style-library engine. The Elo tuning itself is uncalibrated — see the sidebar "
        "caption for the measured rating lag against a continuously improving player.\n"
        "- **Not built yet, in either engine**: live actuation of a real bowling machine. "
        "The physics engine computes the correct wheel-RPM numbers to set "
        "(`cricket_trajectory/machine.py`); nothing here sends them to actual hardware — "
        "a person still reads the number and sets the real machine."
    )
