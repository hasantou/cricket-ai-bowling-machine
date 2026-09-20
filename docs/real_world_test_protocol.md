# Real-world test protocol: body movement and shot reading

Purpose: find out, with real players, whether reading the batter's body from
video is worth building on, and if it is, make the algorithm stronger from the
same data. Written so the founders can agree the pass/fail lines **before** any
results exist. Everything marked *proposal* is a starting point to argue with,
not a finding.

## 0. What is already known (so it is not re-tested by accident)

- On the two real clips available (batter filmed from behind; a small side-on
  batter), the pose model was confident in **both hands in only 7% and 30% of
  frames**. Neither camera position can read a swing's shape.
- Smoothed, body-relative hand speed did separate real swings (about 7-9
  torso-lengths/s) from taking guard (about 1-3) on four deliveries of one
  clip, checked by eye. That is an observation, not a validation.
- The shot-naming rules have **never named a shot on real footage** (they
  refuse). Their accuracy is unknown.

## 1. Question 1 — where can the camera see the hands? (do this first, half a day)

The camera position decides whether everything else is possible.

| Position | Why test it |
|---|---|
| A. **At the machine / bowler's end**, behind and above the machine, looking at the batter | This is where a machine-mounted camera would actually sit; the batter faces it |
| B. Side-on, batter fills at least a third of the frame | Best view of feet and swing plane |
| C. Behind the batter (what we have now) | Baseline for comparison |

Filming: 1080p or better, 60 fps, fixed tripod (no panning), good even light, the
batter at least 30% of the frame height, no one else crossing in front. Film
**one batter, 20 deliveries, three positions**, same session.

Measure each clip with the app (or `footage_check.assess_footage`):
`hand_visibility` (both wrists confidently seen), person height, detection rate.

*Proposal:* a position passes if `hand_visibility` is at least 60% and the
batter is at least 30% of frame height on at least 90% of deliveries.
**If no position passes, stop: video shot-reading is not feasible with this
pose model, and the plan becomes sensors-only for shots.** Trying a stronger
pose model or a second camera is a separate decision.

## 2. Question 2 — is the swing verdict right? (one session, one position that passed)

Record 100 deliveries including deliberate leaves and blocks. A coach marks each
"swing / no swing" **without seeing the algorithm's output**. Compare with the
app's swing verdict.

*Proposal:* at least 90% agreement, and no more than 5% of real swings called
"no swing".

## 3. Question 3 — can it name the shot? (only if 1 and 2 pass)

1. Record **at least 200 deliveries** at a passing camera position, from at
   least 3 different batters (one batter would teach it that batter's habits).
   Include left-handers if the product must serve them; the algorithm mirrors
   sides but that has not been checked.
2. **Two coaches label each delivery independently** with a shot from
   `shot_vocabulary.py` (or "Leave" / "Defensive push"). Record how often they
   agree. **The algorithm can't be expected to beat two coaches' agreement with
   each other** — if they only agree 70% of the time, 70% is the ceiling.
3. Label in the app ("Real-world trial: record what each shot really was") or
   write records to a trial log; keep every labelled delivery, including the
   ones the algorithm refused.
4. **Split before looking:** set aside a third of the deliveries (whole batters,
   not random deliveries) as a test set the tuning never sees.
5. Tune on the rest (`shot_calibration.fit_rules`); it reports held-out
   accuracy itself, but the reserved test set is the number that goes in the deck.

*Proposals* (agree them first): on the reserved test set, at least **70%
correct at shot-family level** (drive-type / horizontal-bat / defensive push /
leave) and **60% at exact shot name**, with refusals (the algorithm declining
to name) on **no more than 40%** of deliveries. Report all three numbers
together; a high accuracy achieved by refusing most deliveries is not a result.

## 4. Question 4 — do sensor and video together beat the sensor alone?

On the machine, with the release and impact sensors installed, log for each
delivery: the sensor's shot name, the video's shot name, and the coaches' label.
`shot_fusion.fuse_shot` combines them. Report how often each source alone is
right, how often they agree, and — the useful number — **accuracy when they
agree versus when they disagree**. If agreement doesn't make the answer more
reliable, the fusion adds nothing.

## 5. Question 5 — does it make players improve faster? (the one that matters commercially)

Two matched groups of players over the same weeks: adaptation on **outcomes
only** vs on **outcomes plus body data**. Measure a skill outcome the coach
agrees in advance (e.g. runs / control rate against a fixed test set of
deliveries). Nothing above matters commercially if this is flat.

## 6. Sensor checks that belong in the same visit

- Release-speed sensor against an independent speed reference (a radar or timing
  gates), across the machine's speed range; record the error, not just "works".
- Impact / exit-velocity thresholds (`DEAD_BAT`, `WELL_STRUCK`, `SKIED`) against
  coach-judged shots; these are currently the author's estimates.
- Machine repeatability: same settings, 30 balls, where they land.

## 7. What to record for every trial, always

Camera position, batting hand, batter, date, lighting, clip filename and frame
range, both coaches' labels, the algorithm's output **including refusals and the
reason**, and the app version (commit). Refusals are data.

## 8. How to run it with the app

1. Upload the clip in "Estimate from a video clip", choose Batch, set **Where was
   the camera?** and **Batter bats**.
2. Read the footage-quality panel first: if it says the hands are not visible,
   the rest of that clip cannot answer Question 3.
3. Open "Real-world trial", pick the true shot per delivery, add to the log,
   **download the log after every session** (the hosted app forgets it).
4. Once 30 trusted labelled deliveries exist, "Tune the thresholds" gives a
   held-out accuracy; it adopts new thresholds only if they are clearly better.

## 9. Traps

- Tuning and testing on the same deliveries proves nothing.
- One batter, one camera, one day teaches the algorithm that batter's quirks.
- Labels made after seeing the algorithm's guess are contaminated.
- Counting only the deliveries it named hides how often it refused.
