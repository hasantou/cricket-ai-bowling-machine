import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from commentary_labels import (
    DOT_BALL, FOUR, SIX, WICKET, CommentaryEvent, TranscriptSegment, align_events_to_deliveries,
    find_outcome_events, transcribe_commentary,
)


def seg(start, text, end=None):
    return TranscriptSegment(start, end if end is not None else start + 3.0, text)


# ---- keyword matching, including the real mishearings found on a real clip ----
def test_finds_a_clean_six_and_four():
    events = find_outcome_events([seg(10.0, "That is a HUGE six over long on!"), seg(20.0, "Timed beautifully for four.")])
    assert [e.outcome for e in events] == [SIX, FOUR]


def test_round_the_wicket_and_over_the_wicket_are_bowling_technique_not_a_dismissal():
    """Found running this on the real transcript: 'he goes around the wicket' describes the
    bowler's angle of approach, a completely unrelated, extremely common piece of commentary
    that happens to contain the word 'wicket'."""
    for phrase in ("coming around the wicket now", "bowling over the wicket", "round the wicket for this one"):
        assert find_outcome_events([seg(0.0, phrase)]) == []


def test_regression_the_real_transcript_line_that_first_exposed_the_around_the_wicket_bug():
    """The exact real segment this bug was found on: Whisper's 'wicket'->'wicked' mishearing means
    the plain-spelling exclusion above is not enough on its own — it has to tolerate the same
    mishearing, or this exact real sentence slips back through as a false wicket."""
    text = "It goes around the wicked continues to go around the wicked and he's gone to Nasserie and that has gone full stop"
    assert find_outcome_events([seg(101.8, text)]) == []


def test_the_real_whisper_mishearing_of_wicket_as_wicked_is_still_caught():
    """Found running Whisper's base model on a real ICC broadcast clip: 'wicket' was transcribed
    as 'wicked' twice in a row. Matching only the correct spelling would silently miss real,
    correctly-occurring wickets on real commentary."""
    events = find_outcome_events([seg(5.0, "and the delay again brings a wicked Shinwari")])
    assert [e.outcome for e in events] == [WICKET]


def test_bowled_caught_stumped_and_lbw_all_read_as_wicket():
    for word in ("bowled him all ends up", "caught behind", "stumped by a mile", "given out lbw"):
        assert find_outcome_events([seg(0.0, word)])[0].outcome == WICKET


def test_dot_ball_phrasing_is_recognised():
    events = find_outcome_events([seg(0.0, "no run there, dot ball to end the over")])
    assert events[0].outcome == DOT_BALL


def test_an_incidental_out_in_unrelated_speech_does_not_crowd_out_a_more_specific_word_in_the_same_segment():
    events = find_outcome_events([seg(0.0, "he steps out and it's a wicket, bowled!")])
    assert events[0].outcome == WICKET


def test_regression_watch_out_is_not_a_wicket():
    """Found on a real clip: 'Watch out, delivery, this is.' was misread as a wicket because a bare
    'out' matched. A bare 'out' is no longer a WICKET pattern at all — genuine dismissals in real
    commentary have always said 'wicket', 'bowled', 'caught', 'stumped', 'lbw' or 'given/run out' in
    the same breath in every real clip checked so far."""
    assert find_outcome_events([seg(0.0, "Watch out, delivery, this is.")]) == []


def test_a_bare_out_with_no_other_wicket_word_is_not_a_wicket():
    for phrase in ("he was out of position for that one", "without a doubt the best shot today",
                   "as it turns out that was closer than it looked"):
        assert find_outcome_events([seg(0.0, phrase)]) == []


def test_hes_out_and_thats_out_are_still_recognised_as_wicket_calls():
    assert find_outcome_events([seg(0.0, "he's out, brilliant catch!")])[0].outcome == WICKET
    assert find_outcome_events([seg(0.0, "that's out, given straight away")])[0].outcome == WICKET


def test_ordinary_commentary_with_no_outcome_word_produces_no_event():
    events = find_outcome_events([seg(0.0, "he shapes up for the delivery, waiting"), seg(5.0, "steady over so far")])
    assert events == []


def test_word_boundaries_prevent_false_matches_inside_other_words():
    events = find_outcome_events([seg(0.0, "he was fourteen not out at the break, walking outside off")])
    # "fourteen" must not match FOUR, and "outside"/"walking out" must not match WICKET's \bout\b
    assert events == []


def test_at_most_one_event_per_segment():
    events = find_outcome_events([seg(0.0, "SIX! No wait, that was actually four, what a shot")])
    assert len(events) == 1


# ---- asymmetric alignment: commentary follows the action ----
def test_an_event_matches_the_nearest_preceding_delivery_within_the_window():
    swings = [10.0, 40.0, 70.0]
    events = [CommentaryEvent(73.0, SIX, "six", "that's a huge six")]     # 3s after the delivery at 70
    labels, unmatched = align_events_to_deliveries(swings, events)
    assert unmatched == []
    assert labels[2] is not None and labels[2].outcome == SIX and labels[2].lag_s == pytest.approx(3.0)
    assert labels[0] is None and labels[1] is None


def test_an_event_before_any_delivery_never_matches_one_in_the_future():
    labels, unmatched = align_events_to_deliveries([50.0], [CommentaryEvent(10.0, FOUR, "four", "four runs")])
    assert labels == [None] and len(unmatched) == 1


def test_an_event_too_long_after_a_delivery_is_left_unmatched_not_forced():
    labels, unmatched = align_events_to_deliveries([10.0], [CommentaryEvent(30.0, WICKET, "wicket", "he's out")], max_lag_s=8.0)
    assert labels == [None] and len(unmatched) == 1


def test_two_deliveries_close_together_each_get_their_own_nearest_event():
    swings = [10.0, 14.0]
    events = [CommentaryEvent(12.0, DOT_BALL, "dot", "dot ball"), CommentaryEvent(17.0, FOUR, "four", "four runs")]
    labels, unmatched = align_events_to_deliveries(swings, events)
    assert unmatched == []
    assert labels[0].outcome == DOT_BALL and labels[1].outcome == FOUR


def test_a_delivery_with_no_commentary_at_all_is_left_unlabelled():
    labels, unmatched = align_events_to_deliveries([10.0, 20.0], [CommentaryEvent(13.0, SIX, "six", "six")])
    assert labels[0] is not None and labels[1] is None
    assert unmatched == []


def test_each_event_claims_at_most_one_delivery():
    """Two events both fall in range of the same single delivery — only one may claim it; the other
    is reported as unmatched rather than silently dropped or double-counted."""
    labels, unmatched = align_events_to_deliveries(
        [10.0], [CommentaryEvent(12.0, FOUR, "four", "four"), CommentaryEvent(13.0, SIX, "six", "six")],
    )
    assert sum(l is not None for l in labels) == 1
    assert len(unmatched) == 1


# ---- transcribe_commentary: real audio, real model, skipped if unavailable ----
_HAS_TOOLS = shutil.which("ffmpeg") is not None
try:
    import whisper  # noqa: F401
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False


def test_transcribe_commentary_requires_ffmpeg_and_says_so_plainly(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="ffmpeg"):
        transcribe_commentary(str(tmp_path / "clip.mp4"))


@pytest.mark.skipif(not (_HAS_TOOLS and _HAS_WHISPER), reason="requires ffmpeg and the whisper package")
def test_transcribe_commentary_on_a_generated_silent_clip_returns_no_segments(tmp_path):
    """A real, if trivial, end-to-end run: ffmpeg extraction + a real Whisper pass on a clip with
    no speech (silence) should come back with nothing worth reporting, not an error."""
    import subprocess
    clip = str(tmp_path / "silent.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", "-shortest", clip],
        check=True, capture_output=True,
    )
    segments = transcribe_commentary(clip, model_size="base")
    assert find_outcome_events(segments) == []
