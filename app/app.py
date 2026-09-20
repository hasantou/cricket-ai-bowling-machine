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
import random
import time
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
from video_pipeline import analyse_video, _expected_frame_count
from footage_check import assess_footage
from pose_estimation import PoseEstimator
from outcome_bridge import VisionOutcomeEstimator
from live_video_source import LiveVideoSource
from live_delivery_detector import LiveDeliveryDetector
from cricket_trajectory import (
    BallProperties, Environment, PlayerProfile, expected_success, OUTCOME_SCORES,
    Scorecard, classify_delivery_legality, run_simulation, score_outcome, build_delivery_report,
    analyse_shot,
)
from cricket_trajectory.adaptive import suggest_next_delivery, delivery_difficulty_rating
from cricket_trajectory.machine import WheelMachine
from machine_control.session_store import save_profile, load_profile, save_scorecard, load_scorecard
from machine_control.controller import SimulatedMachineController
from machine_control.safety import SafeMachineController, SafetyLimits, SafetyViolation
from machine_control.serial_controller import SerialCommunicationError, SerialMachineController
from machine_control.release_sensor import SimulatedReleaseSensor, SpeedCalibrator
from machine_control.impact_sensor import SimulatedVelocitySensor, VelocitySensorOutcomeObserver
from cricket_trajectory.net_outcome import classify_exit_velocity, NET_OUTCOMES
from cricket_trajectory.shot_vocabulary import by_detectability as _shot_detectability
try:
    from serial.tools import list_ports as _serial_list_ports
except ImportError:
    _serial_list_ports = None

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

:root {
    --pavilion-maroon: #7a2e2e;
    --pavilion-maroon-dark: #5c2222;
    --pavilion-ink: #3b2f22;
    --pavilion-parchment: #f4ecd8;
    --pavilion-parchment-dark: #e8dcc0;
    --pavilion-brass: #a9822f;
}

.stApp { background-color: var(--pavilion-parchment); }

h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--pavilion-ink) !important;
    letter-spacing: 0.01em;
}

/* Masthead: a scorebook-style double rule under the main title, instead
   of Streamlit's default plain heading. */
h1:first-of-type {
    border-bottom: 3px double var(--pavilion-maroon);
    padding-bottom: 0.4em;
    margin-bottom: 0.6em !important;
}

div[data-testid="stMetricValue"] { font-family: 'Playfair Display', Georgia, serif !important; }
div[data-testid="stMetricLabel"] {
    font-family: 'Playfair Display', Georgia, serif !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.78em !important;
    color: var(--pavilion-brass) !important;
}
/* Metric "cards": a bordered ledger-card look instead of bare numbers
   floating on the page. */
div[data-testid="stMetric"] {
    background-color: var(--pavilion-parchment-dark);
    border: 1px solid var(--pavilion-brass);
    border-top: 3px solid var(--pavilion-maroon);
    border-radius: 2px;
    padding: 0.7em 1em;
}

pre, code, div[data-testid="stText"] {
    font-family: 'Courier Prime', 'Courier New', monospace !important;
}
/* The scorecard's render_text() output - an aged-ledger card. */
div[data-testid="stText"] {
    background-color: #fffdf6;
    border: 1px solid var(--pavilion-brass);
    border-radius: 2px;
    padding: 1em 1.2em;
}

hr { border-top: 1px solid var(--pavilion-brass) !important; }

/* Buttons: a pressed-brass look, not a default rounded-rectangle web
   button. Primary = filled maroon; secondary = outlined. */
.stButton > button, .stDownloadButton > button {
    font-family: 'Playfair Display', Georgia, serif !important;
    letter-spacing: 0.03em;
    border-radius: 3px;
    transition: transform 0.05s ease-in-out;
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background-color: var(--pavilion-maroon);
    border: 1px solid var(--pavilion-maroon-dark);
    box-shadow: 0 2px 0 var(--pavilion-maroon-dark);
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
    background-color: var(--pavilion-maroon-dark);
}
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {
    border: 1px solid var(--pavilion-maroon);
    color: var(--pavilion-maroon);
}

/* Alerts (info/success/warning/error) - one consistent aged-notice-board
   look rather than default red/green/blue/orange, which clashed with the
   parchment palette. */
div[data-testid="stAlert"] {
    background-color: var(--pavilion-parchment-dark) !important;
    border: 1px solid var(--pavilion-brass) !important;
    border-left: 4px solid var(--pavilion-maroon) !important;
    border-radius: 2px !important;
    color: var(--pavilion-ink) !important;
}
div[data-testid="stAlert"] p { color: var(--pavilion-ink) !important; }

/* Sidebar: a shade darker, like a noticeboard set back from the main hall. */
section[data-testid="stSidebar"] {
    background-color: var(--pavilion-parchment-dark);
    border-right: 1px solid var(--pavilion-brass);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
    text-transform: uppercase;
    font-size: 1.1em !important;
    letter-spacing: 0.06em;
}

/* Expanders styled as pull-out ledger tabs. */
div[data-testid="stExpander"] {
    border: 1px solid var(--pavilion-brass) !important;
    border-radius: 2px !important;
    background-color: rgba(255, 253, 246, 0.5);
}

/* Radio groups rendered horizontally read like a set of ballot choices -
   a little letter-spacing on the label makes them read as a heading
   rather than a stray line of text. */
div[data-testid="stWidgetLabel"] p {
    font-weight: 600;
    color: var(--pavilion-ink);
}
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


def _simulate_impact_velocity(expected_success_pct: float, rng: random.Random):
    """A toy, clearly-labelled stand-in for a real post-contact sensor
    reading — not a real batter model. Returns an exit velocity vector
    (vx, vy, vz) in m/s (ball.py's x=down-pitch, y=lateral, z=up frame),
    or None for "no contact". Biased so a delivery the batter was
    expected to handle comfortably (high expected_success) skews toward
    higher exit speed, and a delivery they were expected to struggle with
    skews toward a gentler, more defensive speed. Exists to demonstrate
    VelocitySensorOutcomeObserver's real classification logic end-to-end
    in this UI — the same code path a real stereo-camera exit-trajectory
    rig would feed — not to claim any predictive accuracy about real
    batting.
    """
    if rng.random() < (1.0 - expected_success_pct) * 0.25:
        return None  # simulated miss
    mean_speed = 3.0 + expected_success_pct * 22.0
    speed = max(0.0, rng.gauss(mean_speed, 4.0))
    skied_chance = (1.0 - expected_success_pct) * 0.2
    elevation_deg = rng.gauss(28.0, 6.0) if rng.random() < skied_chance else rng.gauss(2.0, 5.0)
    azimuth_deg = rng.uniform(-70.0, 70.0)
    theta = math.radians(elevation_deg)
    phi = math.radians(azimuth_deg)
    vx = speed * math.cos(theta) * math.cos(phi)
    vy = speed * math.cos(theta) * math.sin(phi)
    vz = speed * math.sin(theta)
    return vx, vy, vz


@st.cache_data(show_spinner="Running pose estimation on the clip...")
def _analyse_clip(video_bytes: bytes, suffix: str, analyse_bowler: bool = False):
    """Cached on the uploaded file's bytes (and the bowler option) so
    re-running the app (e.g. the user picking a different delivery from the
    dropdown below) doesn't re-run pose estimation on the whole clip every
    time — that's real compute, not free, especially on a multi-minute
    session clip. Returns a video_pipeline.VideoAnalysis: the batter's
    estimates, a footage-quality verdict, and (only if asked) the bowler's
    action per delivery."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        return analyse_video(tmp_path, analyse_bowler=analyse_bowler)
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner="Processing frame-by-frame, as if streaming from a camera on the machine...")
def _analyse_clip_live(video_bytes: bytes, suffix: str):
    """Runs the SAME uploaded clip through the live-camera-shaped pipeline
    (cv-pipeline/live_video_source.py + live_delivery_detector.py)
    instead of the batch call above — one frame at a time, exactly the
    code path validated against real footage in
    cv-pipeline/demo_live_delivery_detection.py (see that script and
    cv-pipeline/README.md for the three real bugs found and fixed doing
    that). There's no real camera on Streamlit Cloud to point this at;
    feeding it an uploaded file frame-by-frame is the same honest stand-in
    that validation script uses — a file and a live camera are identical
    to cv2.VideoCapture.read() in a loop.

    Returns (estimates, frames_processed, frames_with_a_person, elapsed_s, fps,
    footage_report). (Bowler-action analysis is batch-only: it needs a second,
    multi-person pass over the frames, which a streaming camera loop doesn't keep.)
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        source = LiveVideoSource(device=tmp_path)
        fps = source.fps
        detector = LiveDeliveryDetector(fps=fps, buffer_seconds=15.0)
        vision_estimator = VisionOutcomeEstimator()
        estimates = []
        frame_count = 0
        person_frame_count = 0
        all_landmarks = []
        t0 = time.perf_counter()
        with PoseEstimator() as estimator:
            for frame in source.frames():
                frame_count += 1
                landmarks = estimator.extract_landmarks_from_one_live_frame(frame, fps)
                if landmarks is not None:
                    person_frame_count += 1
                    all_landmarks.append(landmarks)
                for delivery in detector.add_frame(landmarks):
                    window_landmarks = detector.buffered_landmarks()[delivery.start:delivery.end]
                    estimates.append(vision_estimator.estimate(window_landmarks, fps=fps))
        elapsed = time.perf_counter() - t0
        source.close()
        footage = assess_footage(
            all_landmarks, frame_count, frames_expected=_expected_frame_count(tmp_path)
        )
        return estimates, frame_count, person_frame_count, elapsed, fps, footage
    finally:
        os.unlink(tmp_path)


def _show_footage_report(footage):
    """Plain-language verdict on whether this clip is good enough for
    pose-based analysis to mean anything (cv-pipeline/footage_check.py)."""
    text = (
        f"**Footage quality: {footage.verdict.upper()}** — person {footage.person_height_fraction * 100:.0f}% "
        f"of frame height, found in {footage.detection_rate * 100:.0f}% of frames, "
        f"pose confidence {footage.median_visibility:.2f}. " + " ".join(footage.reasons)
    )
    {"good": st.success, "marginal": st.warning, "poor": st.error}[footage.verdict](text)


def _bowler_action_rows(action):
    return [
        ("Arm", action.arm_side),
        ("Action", f"{action.arm_action_label} (~{action.arm_angle_deg:.0f}° from vertical, projected)"),
        ("Release height", f"{action.release_height_torsos:.2f} torso-lengths above the shoulder"),
        ("Run-up pace", f"{action.run_up_speed_torsos_per_s:.1f} torso-lengths/s"),
        ("Arm speed", f"{action.arm_speed_torsos_per_s:.0f} torso-lengths/s"),
        ("Release → batter's swing",
         f"{action.release_to_swing_s:.2f} s" if action.release_to_swing_s is not None else "n/a"),
    ]


shot_detectability = _shot_detectability()


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
if "traj_wheel_machine" not in st.session_state:
    # The software's current belief about this machine's RPM<->speed
    # mapping - persisted (not recreated each render) so calibration
    # corrections actually stick across reruns instead of resetting.
    st.session_state.traj_wheel_machine = WheelMachine()
if "traj_true_wheel_machine" not in st.session_state:
    # Stands in for real hardware's actual (never perfectly known)
    # efficiency - every real machine has some manufacturing/wear
    # variance from the textbook value the software starts out assuming.
    # Randomised once per session so the calibration demo isn't identical
    # every time, but fixed for the session so it converges toward
    # something real rather than a moving target.
    st.session_state.traj_true_wheel_machine = WheelMachine(
        speed_efficiency=random.Random().uniform(0.78, 0.88)
    )
if "traj_release_sensor" not in st.session_state:
    st.session_state.traj_release_sensor = SimulatedReleaseSensor(
        st.session_state.traj_true_wheel_machine, noise_std_mps=0.3
    )
if "traj_speed_calibrator" not in st.session_state:
    st.session_state.traj_speed_calibrator = SpeedCalibrator(
        st.session_state.traj_wheel_machine, min_samples=5
    )
if "traj_impact_sensor" not in st.session_state:
    st.session_state.traj_impact_sensor = SimulatedVelocitySensor()

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
st.markdown(
    "<p style='margin-top:-0.8em; color:#a9822f; font-family:\"Playfair Display\",Georgia,serif; "
    "letter-spacing:0.12em; text-transform:uppercase; font-size:0.85em;'>"
    "A Training-Ground Companion &middot; Est. From Real Aerodynamics, 1985"
    "</p>",
    unsafe_allow_html=True,
)

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
            processing_mode = st.radio(
                "Processing mode",
                ["Batch (whole clip at once)", "Live (frame-by-frame, as if streaming from the machine's camera)"],
                horizontal=True,
            )
            analyse_bowler = st.checkbox(
                "Also read the bowler's action (Batch mode; slower)",
                value=False,
                help="Runs a second, multi-person pose pass. Only works where the bowler is in shot, "
                     "large enough, and delivers with the arm overhead — otherwise it says so. "
                     "Never estimates ball speed, line, length, swing or spin: a phone clip can't show those.",
            )
            vision_estimates = None
            bowler_actions = None
            footage = None
            if clip is not None:
                suffix = os.path.splitext(clip.name)[1]
                try:
                    if processing_mode.startswith("Batch"):
                        analysis = _analyse_clip(clip.getvalue(), suffix, analyse_bowler)
                        vision_estimates = analysis.estimates
                        footage = analysis.footage
                        bowler_actions = analysis.bowler_actions if analyse_bowler else None
                    else:
                        vision_estimates, n_frames, n_person_frames, elapsed, live_fps, footage = _analyse_clip_live(
                            clip.getvalue(), suffix
                        )
                        st.caption(
                            f"Live pipeline: {n_frames} frames processed ({n_person_frames} with a person "
                            f"detected) in {elapsed:.1f}s — {n_frames/elapsed:.0f} fps sustained, at "
                            f"{live_fps:.0f}fps source. Uses `cv-pipeline/live_video_source.py` + "
                            "`live_delivery_detector.py`, the exact code validated against real footage "
                            "in `demo_live_delivery_detection.py` (see `cv-pipeline/README.md` for the "
                            "three real bugs found and fixed doing that)."
                        )
                        if not vision_estimates:
                            st.warning(
                                "No delivery confirmed yet from this clip in live mode — either no clear "
                                "swing was found, or (for a clip near the buffer's ~15s window) its "
                                "follow-through hadn't finished within the clip, which a real continuous "
                                "camera would never hit. Try Batch mode on the same clip to compare."
                            )
                except FileNotFoundError:
                    st.error(
                        "Pose model not downloaded yet. Run "
                        "`python cv-pipeline/pose_estimation.py` once, then retry."
                    )
                except ValueError as e:
                    st.error(str(e))

            if footage is not None:
                _show_footage_report(footage)

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
                    if bowler_actions is not None:
                        action = bowler_actions[i]
                        if action is None:
                            st.caption(
                                f"Delivery {i + 1} — bowler: no overhead bowling action found (the bowler "
                                "may be out of shot, too small, or deliver low/side-on). Nothing is guessed."
                            )
                        else:
                            with st.expander(f"Delivery {i + 1} — bowler's action (estimate from pose)"):
                                st.markdown(
                                    "| Detail | Estimate |\n|---|---|\n"
                                    + "\n".join(f"| **{a}** | {b} |" for a, b in _bowler_action_rows(action))
                                )
                                st.caption(" ".join(action.caveats))
                st.caption(
                    "From video the app can estimate the **batter's** footwork and timing and, where "
                    "visible, the **bowler's action**. Ball speed, line, length, swing and spin are not "
                    "measured from video — on the machine those come from the delivery report and the "
                    "release / impact sensors."
                )
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
        sent_speed = (
            st.session_state.get("traj_last_release_speed")
            if st.session_state.traj_delivery_sent else None
        )
        delivery_report = build_delivery_report(next_ball, sim_result, measured_release_speed_mps=sent_speed)
        with st.expander("How this ball is bowled — delivery report", expanded=True):
            report_table = "| Detail | Value |\n|---|---|\n" + "\n".join(
                f"| **{label}** | {value} |" for label, value in delivery_report.rows()
            )
            st.markdown(report_table)
            st.caption(
                "Built from what the machine was commanded to do plus the physics simulation of "
                "the flight (`trajectory-engine/cricket_trajectory/delivery_report.py`) — real data "
                "the machine already has, no video needed. Length and line are measured where the "
                "ball first pitches, using conventional coaching bands (adjustable, uncalibrated). "
                + ("The sensor-confirmed pace is the (simulated) release-point sensor's reading."
                   if sent_speed is not None else
                   "Send the delivery to see the release sensor's confirmed pace here.")
            )
        spin_mag = sum(w * w for w in next_ball.spin_rad_s) ** 0.5
        machine = st.session_state.traj_wheel_machine
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
        with st.expander("Connect a real machine (serial)"):
            st.caption(
                "Speaks the line protocol in `machine-control/PROTOCOL.md` — any "
                "microcontroller (Arduino, ESP32, a Pi Pico) running that protocol "
                "works here unchanged. No firmware exists yet, because no physical "
                "machine has been chosen yet (see `ROADMAP.md`) — this side of the "
                "wire is real and tested against a fake serial link "
                "(`machine-control/tests/test_serial_controller.py`); there's just "
                "nothing plugged into the other end. Also: this only finds ports on "
                "whatever machine is actually running the app — on Streamlit Cloud "
                "that's a remote server with no physical ports at all, so this only "
                "does anything when you run the app locally on a machine wired to "
                "real hardware."
            )
            detected = [p.device for p in _serial_list_ports.comports()] if _serial_list_ports else []
            port_choice = st.selectbox("Detected ports", detected + ["(enter manually)"]) if detected else "(enter manually)"
            manual_port = st.text_input(
                "Serial port", value="COM3",
                disabled=bool(detected) and port_choice != "(enter manually)",
            )
            chosen_port = manual_port if port_choice == "(enter manually)" else port_choice
            baud = st.number_input("Baud rate", value=115200, step=9600)

            conn_col1, conn_col2 = st.columns(2)
            if conn_col1.button("Connect"):
                try:
                    real = SerialMachineController(port=chosen_port, baudrate=int(baud))
                except SerialCommunicationError as e:
                    st.error(f"Couldn't connect: {e}")
                except Exception as e:
                    st.error(f"Couldn't open {chosen_port}: {e}")
                else:
                    st.session_state.traj_controller = SafeMachineController(
                        real, SafetyLimits(max_wheel_rpm=6000.0)
                    )
                    st.session_state.traj_delivery_sent = False
                    st.success(f"Opened {chosen_port} — status below reflects a real PING to it now.")
                    st.rerun()
            if conn_col2.button("Use simulated machine instead"):
                try:
                    if hasattr(controller._inner, "close"):
                        controller._inner.close()
                except Exception:
                    pass  # best-effort; switching backends should never get stuck on this
                st.session_state.traj_controller = SafeMachineController(
                    SimulatedMachineController(cycle_time_s=1.5), SafetyLimits(max_wheel_rpm=6000.0)
                )
                st.session_state.traj_delivery_sent = False
                st.rerun()

        status_col1, status_col2 = st.columns(2)
        status_col1.metric("Enabled", "yes" if controller.enabled else "NO — stopped")
        backend_name = type(controller._inner).__name__
        status_col2.metric("Ready for next command", "yes" if controller.is_ready() else "not yet")
        st.caption(f"Backend: **{backend_name}**" + (f" on `{controller._inner.port}`" if hasattr(controller._inner, "port") else ""))

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
                    # A real release-point sensor would fire automatically on
                    # every delivery - simulate that here too, feeding the
                    # calibrator so speed_efficiency keeps correcting itself.
                    st.session_state.traj_release_sensor.arm(rpm1, rpm2)
                    measured = st.session_state.traj_release_sensor.measure_speed_mps()
                    st.session_state.traj_last_release_speed = measured
                    st.session_state.traj_speed_calibrator.record(rpm1, rpm2, measured)
                    st.session_state.traj_speed_calibrator.apply_calibration()
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
                outcome_mode = st.radio(
                    "How was the outcome determined?",
                    ["Manual entry", "Simulated impact sensor"],
                    horizontal=True,
                )
                if outcome_mode == "Manual entry":
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
                else:
                    st.caption(
                        "`machine_control/impact_sensor.py`'s `VelocitySensorOutcomeObserver` — an "
                        "exit-velocity reading (speed, launch angle, direction) instead of visually "
                        "tracking the ball through the shot, which `cv-pipeline/motion_ball_detector.py` "
                        "found genuinely hard on real footage. This is exactly what a stereo-camera "
                        "post-shot trajectory rig would measure — see `trajectory-engine/cricket_trajectory/"
                        "net_outcome.py`'s `ExitVelocity`. The reading below is a **toy simulated batter "
                        "model**, not a real one — it exists to demonstrate the real classification code "
                        "path end to end."
                    )
                    if "traj_sensor_reading" not in st.session_state:
                        if st.button("Generate sensor reading (simulated)", type="primary"):
                            rng = random.Random()
                            reading = _simulate_impact_velocity(pre_expected, rng)
                            sensor = st.session_state.traj_impact_sensor
                            sensor.arm(reading)
                            observer = VelocitySensorOutcomeObserver(sensor)
                            resolved_outcome = observer.observe(next_ball, ball)
                            if reading is not None:
                                outcome_obj, exit_velocity = classify_exit_velocity(*reading)
                                label = outcome_obj.label
                            else:
                                exit_velocity = None
                                label = NET_OUTCOMES["no_contact"].label
                            st.session_state.traj_sensor_reading = {
                                "exit_velocity": exit_velocity, "label": label, "outcome": resolved_outcome,
                            }
                            st.rerun()
                    else:
                        r = st.session_state.traj_sensor_reading
                        ev = r["exit_velocity"]
                        scoring = score_outcome(r["outcome"])
                        scoring_text = (
                            f"OUT ({scoring.dismissal})" if scoring.is_wicket
                            else f"{scoring.runs} run{'s' if scoring.runs != 1 else ''}"
                        )
                        if ev is None:
                            st.warning(f"Sensor reading: no contact detected. → **{scoring_text}**")
                        else:
                            st.info(
                                f"Sensor reading: **{ev.speed_mps:.1f} m/s** ({ev.speed_mps*3.6:.0f} km/h), "
                                f"elevation **{ev.elevation_deg:.0f}°**, "
                                f"direction **{ev.direction_label()}** ({ev.azimuth_deg:+.0f}°)\n\n"
                                f"→ classified as **{r['label']}** → **{scoring_text}**"
                            )
                        shot = analyse_shot(ev, delivery_report.length_label)
                        with st.expander(f"What shot was it? — {shot.shot}", expanded=True):
                            st.markdown(
                                "| Detail | Value |\n|---|---|\n"
                                + "\n".join(f"| **{a}** | {b} |" for a, b in shot.rows())
                            )
                            st.caption(
                                shot.rationale + " This is an inference from the ball's direction, height, "
                                "pace and length — a rule table, not a coach's verdict. Right-handed batter "
                                "assumed. Here the sensor reading is simulated."
                            )
                        with st.expander("Which shots can this tell apart, and which can't it?"):
                            for reason, shots in shot_detectability.items():
                                st.markdown(
                                    f"**{'Can name from the sensor reading' if reason == 'sensor' else 'Cannot yet — ' + reason}:** "
                                    + ", ".join(shots)
                                )
                            st.caption(
                                "Shot list transcribed from the project's saved shot-name reference "
                                "(`shot_vocabulary.py`); a plain reference list, not a coaching standard."
                            )
                        if st.button("Log delivery", type="primary"):
                            card.record_ball(profile, ball, next_ball, sim_result, outcome=r["outcome"])
                            st.session_state.traj_next_delivery = suggest_next_delivery(
                                profile, ball,
                                challenge_margin=_margin_for_target(st.session_state.traj_target_success),
                            )
                            st.session_state.traj_delivery_sent = False
                            del st.session_state.traj_sensor_reading
                            st.rerun()
                        if st.button("Generate a different reading"):
                            del st.session_state.traj_sensor_reading
                            st.rerun()

        inner_history = getattr(controller._inner, "history", None)
        if inner_history:
            with st.expander(f"Machine command log ({len(inner_history)} sent)"):
                for cmd in reversed(inner_history[-10:]):
                    tag = "EMERGENCY STOP" if cmd.stopped else f"{cmd.wheel1_rpm:.0f} / {cmd.wheel2_rpm:.0f} rpm"
                    st.write(f"- {tag}")

        st.divider()
        st.markdown("**Release-speed calibration** (`machine_control/release_sensor.py`)")
        calibrator = st.session_state.traj_speed_calibrator
        n_samples = len(calibrator.measurements)
        st.caption(
            "Unlike a human bowler, this system commands its own delivery — it doesn't need to "
            "*track* the ball to check the release was right, only *confirm* it with one cheap "
            "sensor (a photogate) at the release point. Simulated here: this session's machine has "
            f"a hidden true speed efficiency of **{st.session_state.traj_true_wheel_machine.speed_efficiency:.3f}** "
            "(manufacturing/wear variance every real machine has) that the software doesn't start out "
            "knowing — watch its belief below converge toward it as deliveries are sent."
        )
        cal_col1, cal_col2 = st.columns(2)
        cal_col1.metric("Deliveries measured", n_samples)
        cal_col2.metric("Believed speed efficiency", f"{machine.speed_efficiency:.3f}")
        if n_samples < calibrator.min_samples:
            st.info(f"Needs {calibrator.min_samples - n_samples} more measured deliveries before it will calibrate.")
        else:
            error = abs(machine.speed_efficiency - st.session_state.traj_true_wheel_machine.speed_efficiency)
            st.success(f"Calibrated — within {error:.3f} of the true value.")

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
        "verdict has checked the on_time/footwork numbers yet. **Processing mode** lets "
        "you pick Batch (the whole clip processed at once) or Live — the exact "
        "frame-by-frame pipeline (`cv-pipeline/live_video_source.py` + "
        "`live_delivery_detector.py`) meant for a camera mounted on the machine, "
        "validated by playing real clips back as if live and matching the batch result "
        "exactly on every delivery a continuous camera would also complete (three real "
        "bugs found and fixed doing that — see `cv-pipeline/README.md`). No real camera "
        "exists yet; Live mode here is fed the same uploaded file, same honest stand-in "
        "the validation itself uses.\n"
        "- **Video: what it can and can't tell you**: a *footage-quality* verdict (person "
        "size, pose confidence, dropouts, undecodable frames) is shown for every clip. From "
        "video the app estimates the **batter's** footwork and timing, and — optionally, and "
        "only where the bowler is in shot, large, and delivers overhead — the **bowler's "
        "action** (arm side, arm angle, release height, run-up pace; in torso-lengths, "
        "not km/h). Ball speed, line, length, swing and spin are **not** measured from "
        "video. The bowler-action reading is tested on synthetic skeletons only: none of "
        "the real clips available so far shows a bowler close enough to validate it, and "
        "an earlier version mistook a batter's backlift for a delivery (fixed).\n"
        "- **Shot naming (physics engine)**: after a sensor reading, the app names the shot "
        "(cover drive, pull, forward defence, ...) from the ball's exit direction, height and "
        "pace plus the delivery's length (`trajectory-engine/shot_analysis.py`). It is an "
        "inference from a rule table — not a measurement, not a coach's verdict, right-handed "
        "batter only — and the sensor reading here is simulated. Front/back foot is inferred "
        "from the ball's length unless observed. Not yet checked against real shots.\n"
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
        "- **Machine connection (physics engine)**: every delivery goes through the real "
        "`machine-control` interface (`SafeMachineController` + either "
        "`SimulatedMachineController` or a real `SerialMachineController` — pick in the "
        "'Connect a real machine' panel) rather than a person reading a number and setting "
        "the real machine by hand. No physical machine has been chosen yet, so the serial "
        "option has nothing plugged into the other end — see `machine-control/PROTOCOL.md`.\n"
        "- **Sensing architecture (physics engine)**: two new pieces sidestep visual ball "
        "tracking (which `cv-pipeline/motion_ball_detector.py` found genuinely hard on real "
        "footage) by exploiting that this system commands its own deliveries. "
        "**Release-speed calibration** measures each delivery's actual exit speed with one "
        "simulated cheap sensor and corrects `machine.py`'s RPM-to-speed mapping over "
        "time — watch the belief converge toward the (simulated) true value below the "
        "command log. **Simulated impact sensor** (an alternative to manual outcome entry) "
        "classifies the shot from a simulated exit-velocity reading (speed, launch angle, "
        "direction) — exactly what a stereo-camera post-shot trajectory rig would measure — "
        "through the real, already-tested `net_outcome.py` `classify_exit_velocity()` logic. "
        "All three are real, tested software (`machine-control/machine_control/"
        "release_sensor.py` and `impact_sensor.py`) — run against simulated sensors, since "
        "no real ones exist yet; the impact-sensor reading itself comes from a "
        "clearly-labelled toy batter model, not a claim about real batting."
    )
