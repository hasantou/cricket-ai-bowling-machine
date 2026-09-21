"""
Everything known about ONE delivery, from every source, with each fact labelled by where it came from.

A delivery has several independent witnesses, and they are not equally trustworthy:

  commanded          what the machine was told to do (exact)
  physics prediction what the simulation says the flight did (a model, uncalibrated)
  release sensor     the ball's measured exit speed
  crease sensor      the ball's measured position as it passed the batter
  impact sensor      the ball's measured velocity off the bat
  video (pose)       the batter's and bowler's bodies, measured from a clip
  video (rule)       a judgement made from those measurements by a rule of thumb
  inferred           deduced from the above by a stated rule (e.g. Law 32 would-be-bowled)
  not measured       nothing could tell us

The story never fills a gap: a fact with no source is reported as "not measured", and anything from a
simulated sensor says "simulated". It then does three things a list of readings does not:

  * OBSERVATIONS: coaching cross-checks that only exist when sources are combined (footwork against the
    length bowled, footwork timing against the swing);
  * FLAGS: places where the sources disagree (the sensor felt bat, the camera saw no swing; the release
    sensor differs from the command; the camera and sensor clocks don't line up) — surfaced, not hidden;
  * a NARRATIVE that reads the whole thing back in plain sentences, hedged by source.

Observations use stated rules of thumb (named thresholds below); they are prompts for a coach, not verdicts.
This module takes plain objects so the physics package does not depend on the computer-vision package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .delivery_report import DeliveryReport
from .scorecard import score_outcome

# --- provenance labels ---------------------------------------------------------------------
COMMANDED = "commanded"
PHYSICS = "physics prediction"
RELEASE_SENSOR = "release sensor"
CREASE_SENSOR = "crease sensor"
IMPACT_SENSOR = "impact sensor"
VIDEO_POSE = "video (pose)"
VIDEO_RULE = "video (rule of thumb)"
INFERRED = "inferred"
NOT_MEASURED = "not measured"

# --- rules of thumb for the observations -------------------------------------------------------
SPEED_DISAGREEMENT_KMH = 3.0         # release sensor vs command beyond this is worth a flag
LATE_FOOTWORK_S = 0.15               # front-foot stride began less than this before the swing peak
EARLY_FOOTWORK_S = 0.90              # ...or more than this before it
SYNC_TOLERANCE_S = 0.15


@dataclass(frozen=True)
class StoryRow:
    label: str
    value: str
    source: str
    simulated: bool = False

    def source_text(self) -> str:
        return f"simulated {self.source}" if self.simulated else self.source


@dataclass(frozen=True)
class StorySection:
    title: str
    rows: List[StoryRow]


@dataclass
class ImpactInfo:
    """What the post-contact sensor reported (None fields = no contact)."""
    exit_speed_kmh: Optional[float] = None
    elevation_deg: Optional[float] = None
    azimuth_deg: Optional[float] = None
    shot: Optional[str] = None                 # shot_analysis name, "no shot / missed" for no contact
    shot_confidence: str = ""
    region: Optional[str] = None
    outcome: Optional[str] = None              # scorecard outcome key (e.g. "boundary", "missed", "beaten")
    outcome_note: Optional[str] = None
    simulated: bool = False


@dataclass
class VideoDeliveryInfo:
    """What the camera pipeline found for this delivery (any field may be None)."""
    swing_verdict: Optional[str] = None        # "swing" / "no swing" / "unclear"
    peak_hand_speed: Optional[float] = None    # torso-lengths/s, relative to the body
    hand_path_net_direction: Optional[str] = None
    footwork_class: Optional[str] = None       # "front-foot" / "back-foot" / "minimal" / "unclear (feet not visible)"
    front_stride: Optional[float] = None       # torso-lengths toward the bowler
    hip_shift: Optional[float] = None
    lead_time_s: Optional[float] = None
    foreshortened: bool = False
    shot: Optional[str] = None                 # shot_from_video name, when one was justified
    shot_verdict: Optional[str] = None         # "shot named" / "family only" / "no shot" / "cannot tell"
    shot_family: Optional[str] = None
    shot_reasons: List[str] = field(default_factory=list)
    bowler_arm: Optional[str] = None
    bowler_action: Optional[str] = None
    bowler_release_height: Optional[float] = None
    footage_verdict: Optional[str] = None
    swing_time_s: Optional[float] = None       # swing-peak time on the video clock (for alignment)
    notes: List[str] = field(default_factory=list)


@dataclass
class SyncInfo:
    matched: bool
    residual_s: Optional[float] = None
    offset_s: Optional[float] = None


@dataclass
class DeliveryStory:
    sections: List[StorySection]
    observations: List[str]
    flags: List[str]
    narrative: str
    provenance: Dict[str, int]

    def all_rows(self) -> List[StoryRow]:
        return [r for sec in self.sections for r in sec.rows]

    def markdown(self) -> str:
        out = []
        for sec in self.sections:
            out.append(f"**{sec.title}**\n\n| Detail | Value | Source |\n|---|---|---|")
            out += [f"| {r.label} | {r.value} | {r.source_text()} |" for r in sec.rows]
            out.append("")
        return "\n".join(out)


def _outcome_text(key: Optional[str]) -> Tuple[str, Optional[object]]:
    if key is None:
        return "not measured", None
    info = score_outcome(key)
    text = f"OUT — {info.dismissal}" if info.is_wicket else f"{info.runs} run{'s' if info.runs != 1 else ''}"
    return text, info


def build_story(
    report: Optional[DeliveryReport] = None,
    decision=None,                       # machine_control LegalityDecision (crease sensor / model), duck-typed
    impact: Optional[ImpactInfo] = None,
    video: Optional[VideoDeliveryInfo] = None,
    fused=None,                          # shot_fusion.FusedShot
    sync: Optional[SyncInfo] = None,
    simulated_sensors: bool = True,
) -> DeliveryStory:
    sections: List[StorySection] = []
    obs: List[str] = []
    flags: List[str] = []
    sim = simulated_sensors

    # ---------------- 1. how it was bowled ----------------
    if report is None:
        rows = [StoryRow("Machine data", "none - a video-only story: the ball's pace, length, line and movement are not "
                         "measured from video", NOT_MEASURED)]
    else:
        rows = [StoryRow("Pace (commanded)", f"{report.commanded_speed_kmh:.0f} km/h", COMMANDED)]
        if report.measured_speed_kmh is not None:
            rows.append(StoryRow("Pace (measured at release)", f"{report.measured_speed_kmh:.0f} km/h "
                                 f"({report.speed_error_kmh:+.1f} vs commanded)", RELEASE_SENSOR, sim))
            if abs(report.speed_error_kmh) > SPEED_DISAGREEMENT_KMH:
                flags.append(f"The release sensor read {report.measured_speed_kmh:.0f} km/h against {report.commanded_speed_kmh:.0f} "
                             f"commanded ({report.speed_error_kmh:+.1f}); the machine is not delivering what it is told.")
        else:
            rows.append(StoryRow("Pace (measured at release)", "no release-sensor reading", NOT_MEASURED))
        length_text = ("full toss - reaches the batter without bouncing" if report.length_label == "full toss"
                       else f"{report.length_label} - pitches {report.pitch_from_stumps_m:.1f} m from the stumps")
        rows += [
            StoryRow("Length", length_text, PHYSICS),
            StoryRow("Line", f"{report.line_label} ({report.pitch_y_m * 100:+.0f} cm from centre)", PHYSICS),
            StoryRow("Ball movement in the air", f"{report.swing_label} ({report.swing_cm:+.1f} cm)", PHYSICS),
            StoryRow("Spin", report.spin_description + (f" at {report.spin_rpm:.0f} rpm" if report.spin_rpm >= 50 else ""), COMMANDED),
            StoryRow("Seam", f"{report.seam_angle_deg:.0f} deg to the flight" + (" - reverse swing regime" if report.reverse_swing else ""), COMMANDED),
        ]
    if video is not None and video.bowler_action:
        rows.append(StoryRow("Bowler's action", f"{video.bowler_arm}, {video.bowler_action}"
                             + (f", release {video.bowler_release_height:.2f} torso-lengths above the shoulder"
                                if video.bowler_release_height is not None else ""), VIDEO_POSE))
    sections.append(StorySection("How it was bowled", rows))

    # ---------------- 2. where it went at the batter (legality) ----------------
    rows = []
    a = report.assessment if report is not None else None
    if decision is not None:
        basis = decision.basis
        src = CREASE_SENSOR if basis == "sensor" else PHYSICS
        call = "WIDE — " + "; ".join(decision.wide_reasons) if decision.call else \
            f"fair — {decision.wide_margin_m:.2f} m inside the nearest wide limit" + (" (borderline)" if decision.borderline else "")
        rows.append(StoryRow("Call at the batter", call, src, sim and basis == "sensor"))
        rows.append(StoryRow("Would hit the stumps", "yes" if decision.hits_stumps else "no", src, sim and basis == "sensor"))
        flag_list = decision.no_ball_flags
        if not decision.agrees_with_model:
            flags.append("The crease sensor's call differs from the physics model's prediction; the model's bounce needs checking.")
    elif a is not None:
        call = "WIDE — " + "; ".join(a.wide_reasons) if a.is_wide else f"fair — {a.wide_margin_m:.2f} m inside the nearest wide limit"
        rows.append(StoryRow("Call at the batter", call, PHYSICS))
        rows.append(StoryRow("Would hit the stumps", "yes" if a.hits_stumps else "no", PHYSICS))
        flag_list = a.no_ball_flags
    else:
        rows.append(StoryRow("Call at the batter", "not assessed", NOT_MEASURED))
        flag_list = []
    if report is None and decision is None:
        flag_list = None
    if flag_list is not None:
        rows.append(StoryRow("No-ball conditions (counted, not scored)", "; ".join(flag_list) if flag_list else "none", PHYSICS))
    sections.append(StorySection("Where it went at the batter", rows))

    # ---------------- 3. the batter (video) ----------------
    rows = []
    if video is None:
        rows.append(StoryRow("Batter's movement", "no camera reading for this delivery", NOT_MEASURED))
    else:
        if video.footage_verdict:
            rows.append(StoryRow("Footage quality", video.footage_verdict, VIDEO_POSE))
        if video.swing_verdict:
            rows.append(StoryRow("Batter's swing", f"{video.swing_verdict}" + (f" ({video.peak_hand_speed:.1f} torso-lengths/s)"
                                 if video.peak_hand_speed is not None else ""), VIDEO_RULE))
        if video.footwork_class:
            fw = video.footwork_class
            if video.front_stride is not None and video.footwork_class in ("front-foot", "back-foot", "minimal"):
                fw += f" (front foot {video.front_stride:+.2f} torso-lengths toward the bowler"
                fw += f", started {video.lead_time_s:.2f} s before the swing)" if video.lead_time_s is not None else ")"
            rows.append(StoryRow("Footwork", fw, VIDEO_RULE))
        if video.hip_shift is not None:
            rows.append(StoryRow("Weight transfer", f"hips {video.hip_shift:+.2f} torso-lengths toward the bowler", VIDEO_POSE))
        if video.shot_verdict == "shot named" and video.shot:
            rows.append(StoryRow("Shot (from the hands)", video.shot, VIDEO_RULE))
        elif video.shot_verdict == "cannot tell":
            rows.append(StoryRow("Shot (from the hands)", "cannot tell — " + " ".join(video.shot_reasons[:1]), NOT_MEASURED))
        elif video.shot_family:
            rows.append(StoryRow("Shot (from the hands)", f"family only: {video.shot_family}", VIDEO_RULE))
    if not rows:
        rows.append(StoryRow("Batter's movement", "the camera pipeline produced no reading for this delivery", NOT_MEASURED))
    sections.append(StorySection("The batter", rows))

    # ---------------- 4. contact and outcome ----------------
    rows = []
    outcome_text, scoring = _outcome_text(impact.outcome if impact else None)
    if impact is None:
        rows.append(StoryRow("Contact", "no impact-sensor reading for this delivery", NOT_MEASURED))
    elif impact.exit_speed_kmh is None:
        rows.append(StoryRow("Contact", "none — the ball missed the bat", IMPACT_SENSOR, impact.simulated))
    else:
        rows.append(StoryRow("Off the bat", f"{impact.exit_speed_kmh:.0f} km/h at {impact.elevation_deg:.0f}° "
                             f"toward {impact.region}" if impact.region else f"{impact.exit_speed_kmh:.0f} km/h at {impact.elevation_deg:.0f}°",
                             IMPACT_SENSOR, impact.simulated))
        if impact.shot:
            rows.append(StoryRow("Shot (from the ball)", f"{impact.shot} ({impact.shot_confidence})" if impact.shot_confidence else impact.shot,
                                 INFERRED, impact.simulated))
    if fused is not None and fused.shot:
        rows.append(StoryRow("Shot (both sources)", f"{fused.shot} — {fused.agreement}: {fused.confidence}", INFERRED))
    rows.append(StoryRow("Result", outcome_text + (f" — {impact.outcome_note}" if impact and impact.outcome_note else ""),
                         IMPACT_SENSOR if impact else NOT_MEASURED, bool(impact and impact.simulated)))
    if scoring is not None and impact and impact.exit_speed_kmh is not None and impact.outcome == "boundary":
        rows.append(StoryRow("Four or six?", "a six needs distance travelled, which is not measured; scored as 4 unless entered by a person", NOT_MEASURED))
    sections.append(StorySection("Contact and result", rows))

    # ---------------- observations (cross-source coaching prompts) ----------------
    if video is not None and report is not None and video.footwork_class in ("front-foot", "back-foot") and report.length_label:
        short = report.length_label in ("short of a length", "short (bouncer territory)")
        full = report.length_label in ("full toss", "yorker", "full")
        if video.footwork_class == "front-foot" and short:
            obs.append(f"Went forward to a {report.length_label} ball: the front-foot stride and the length do not match, which usually "
                       "means the ball was played from the wrong place.")
        elif video.footwork_class == "back-foot" and full:
            obs.append(f"Stayed back to a {report.length_label} ball: usually a sign of being late to get forward.")
        else:
            obs.append(f"Footwork ({video.footwork_class}) suited the length ({report.length_label}).")
    if video is not None and video.lead_time_s is not None:
        if video.lead_time_s < LATE_FOOTWORK_S:
            obs.append(f"The front-foot stride began only {video.lead_time_s:.2f} s before the swing — late footwork.")
        elif video.lead_time_s > EARLY_FOOTWORK_S:
            obs.append(f"The front-foot stride began {video.lead_time_s:.2f} s before the swing — very early, which can commit the batter before the ball is read.")
    if impact is not None and impact.exit_speed_kmh is None and video is not None and video.swing_verdict == "swing":
        obs.append("A full swing with no contact: swung and missed.")
    if impact is not None and impact.exit_speed_kmh is None and video is not None and video.swing_verdict == "no swing":
        obs.append("No swing and no contact: the ball was left alone.")

    # ---------------- flags (cross-source disagreements) ----------------
    if impact is not None and impact.exit_speed_kmh is not None and video is not None and video.swing_verdict == "no swing":
        flags.append("The impact sensor felt bat on the ball but the camera saw no swing — likely a pose-tracking miss, or a very soft block.")
    if sync is not None:
        if not sync.matched:
            flags.append("This delivery could not be matched between the camera's and the sensors' clocks, so the camera and sensor "
                         "readings above may belong to different balls.")
        elif sync.residual_s is not None and abs(sync.residual_s) > SYNC_TOLERANCE_S:
            flags.append(f"The swing peak and the bat contact are {abs(sync.residual_s):.2f} s apart after aligning the clocks; treat the pairing with caution.")
    if video is not None and video.footage_verdict == "poor":
        flags.append("The footage quality is poor (the batter is small or poorly tracked), so the batter readings above are "
                     "indicative only, however precise the numbers look.")
    if video is not None:
        flags += [n for n in video.notes if n]

    provenance: Dict[str, int] = {}
    for r in (r for sec in sections for r in sec.rows):
        key = r.source_text()
        provenance[key] = provenance.get(key, 0) + 1

    story = DeliveryStory(sections, obs, flags, "", provenance)
    story.narrative = _narrate(report, decision, a, impact, video, fused, outcome_text, scoring)
    return story


def _narrate(report, decision, a, impact, video, fused, outcome_text, scoring) -> str:
    s: List[str] = []
    if report is None:
        s.append("A delivery seen only on video, so its pace, length and line are not known.")
    else:
        bowled = f"A {report.commanded_speed_kmh:.0f} km/h ball"
        if report.measured_speed_kmh is not None:
            bowled += f" (the release sensor measured {report.measured_speed_kmh:.0f})"
        bowled += (f", a {report.length_label}" if report.length_label != "full toss" else ", a full toss")
        bowled += f" {report.line_label}, {report.swing_label.lower()} by the model" if report.swing_label else ""
        s.append(bowled + ".")
    call = decision if decision is not None else a
    if call is not None:
        wide = decision.call if decision is not None else ("wide" if a.is_wide else None)
        hit = call.hits_stumps
        via = "the crease sensor" if (decision is not None and decision.basis == "sensor") else "the physics model"
        s.append(f"By {via}, it was {'a wide' if wide else 'a fair delivery'} and {'on' if hit else 'off'} target.")
    if video is not None:
        b = []
        if video.swing_verdict == "swing":
            b.append("swung")
        elif video.swing_verdict == "no swing":
            b.append("did not swing")
        if video.footwork_class in ("front-foot", "back-foot"):
            b.append(f"played off the {video.footwork_class.split('-')[0]} foot")
        if b:
            s.append(f"On video the batter {' and '.join(b)} (a rule-of-thumb reading from pose).")
    if impact is not None:
        sim = " (simulated)" if impact.simulated else ""
        if impact.exit_speed_kmh is None:
            s.append(f"The impact sensor{sim} found no contact.")
        else:
            shot = fused.shot if (fused is not None and fused.shot) else impact.shot
            s.append(f"The ball left the bat at {impact.exit_speed_kmh:.0f} km/h{sim}"
                     + (f" — {shot}" if shot else "") + (f", {fused.confidence}" if fused is not None and fused.shot else "") + ".")
        s.append(f"Result: {outcome_text}.")
    else:
        s.append("No sensor recorded what happened after the ball reached the batter.")
    return " ".join(s)
