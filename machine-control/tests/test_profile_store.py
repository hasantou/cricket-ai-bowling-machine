import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "trajectory-engine"))

import pytest
from cricket_trajectory import BallProperties, Delivery, PlayerProfile
from machine_control.profile_store import (
    ProfileNameCollision,
    _slug,
    list_known_batters,
    load_named_profile,
    profile_summary,
    save_named_profile,
)

BALL = BallProperties()


def delivery(speed=30.0, seam=10.0, label="d"):
    return Delivery(speed_mps=speed, seam_angle_deg=seam, label=label)


# ---- _slug: the thing a naming collision would slip through if this got it wrong ----

def test_slug_is_lowercase_alnum_only():
    assert _slug("Sam Smith") == "sam_smith"
    assert _slug("  Priya   ") == "priya"
    assert _slug("O'Brien") == "o_brien"


def test_different_looking_names_can_slug_the_same_way():
    """The exact case ProfileNameCollision exists to catch -- confirmed here that it
    really does happen, not just theorised."""
    assert _slug("Sam") == _slug("SAM") == _slug("sam")


def test_an_empty_or_symbols_only_name_gets_a_safe_fallback_slug():
    assert _slug("!!!") == "unnamed"
    assert _slug("") == "unnamed"


# ---- save/load round trip, by name, no path management from the caller ----

def test_a_saved_profile_is_found_again_by_name_alone(tmp_path):
    profile = PlayerProfile(name="Priya", rating=1100.0)
    profile.record_outcome(delivery(), BALL, "boundary")
    save_named_profile(profile, tmp_path)

    loaded = load_named_profile("Priya", tmp_path)
    assert loaded is not None
    assert loaded.rating == profile.rating
    assert len(loaded.history) == 1
    assert loaded.skill_ratings == profile.skill_ratings


def test_an_unknown_batter_returns_none_not_an_error(tmp_path):
    assert load_named_profile("Nobody Yet", tmp_path) is None


def test_load_works_against_a_directory_that_does_not_exist_yet(tmp_path):
    assert load_named_profile("Anyone", tmp_path / "does_not_exist") is None


def test_saving_twice_for_the_same_batter_updates_in_place(tmp_path):
    p = PlayerProfile(name="Sam", rating=1000.0)
    save_named_profile(p, tmp_path)
    p.record_outcome(delivery(), BALL, "six")
    save_named_profile(p, tmp_path)

    loaded = load_named_profile("Sam", tmp_path)
    assert len(loaded.history) == 1
    assert loaded.rating == p.rating


# ---- the actual safety net: two different batters must never collide silently ----

def test_two_batters_whose_names_collide_when_slugged_raise_instead_of_overwriting(tmp_path):
    save_named_profile(PlayerProfile(name="Sam", rating=1000.0), tmp_path)
    with pytest.raises(ProfileNameCollision):
        save_named_profile(PlayerProfile(name="SAM", rating=1000.0), tmp_path)
    # the ORIGINAL Sam's data must be untouched by the failed attempt
    assert load_named_profile("Sam", tmp_path).rating == 1000.0


def test_re_saving_the_same_batter_repeatedly_is_not_treated_as_a_collision(tmp_path):
    """Sanity check the collision guard isn't so strict it also blocks the completely
    normal, expected case this whole module exists for: the SAME batter saving again."""
    for i in range(3):
        p = PlayerProfile(name="Priya", rating=1000.0 + i)
        save_named_profile(p, tmp_path)  # must not raise


# ---- list_known_batters: what a "pick a returning batter" UI would show ----

def test_list_known_batters_is_empty_for_a_fresh_directory(tmp_path):
    assert list_known_batters(tmp_path) == []


def test_list_known_batters_is_empty_for_a_directory_that_does_not_exist(tmp_path):
    assert list_known_batters(tmp_path / "nope") == []


def test_list_known_batters_returns_real_names_not_filename_slugs(tmp_path):
    save_named_profile(PlayerProfile(name="Priya Sharma", rating=1000.0), tmp_path)
    save_named_profile(PlayerProfile(name="Sam", rating=1000.0), tmp_path)
    assert sorted(list_known_batters(tmp_path)) == ["Priya Sharma", "Sam"]


def test_list_known_batters_skips_an_unrelated_json_file_rather_than_crashing(tmp_path):
    (tmp_path / "not_a_profile.json").write_text("[1, 2, 3]")
    save_named_profile(PlayerProfile(name="Sam", rating=1000.0), tmp_path)
    assert list_known_batters(tmp_path) == ["Sam"]


# ---- profile_summary: the long-term view, not just the current number ----

def test_summary_of_a_fresh_profile_has_no_trend_yet():
    p = PlayerProfile(name="Priya")
    summary = profile_summary(p, recent_n=10)
    assert summary["deliveries_faced"] == 0
    assert summary["rating_trend"] is None
    assert summary["weakest_skill"] == p.weakest_skill()


def test_summary_reports_no_trend_below_two_full_windows():
    p = PlayerProfile(name="Priya")
    for _ in range(15):  # 15 < 2 * recent_n(10) = 20
        p.record_outcome(delivery(), BALL, "defended")
    assert profile_summary(p, recent_n=10)["rating_trend"] is None


def test_summary_reports_a_real_trend_once_enough_history_exists():
    p = PlayerProfile(name="Priya")
    for _ in range(25):
        p.record_outcome(delivery(), BALL, "defended")
    summary = profile_summary(p, recent_n=10)
    assert summary["deliveries_faced"] == 25
    assert isinstance(summary["rating_trend"], float)


def test_summary_skill_ratings_is_a_copy_not_a_live_reference():
    p = PlayerProfile(name="Priya")
    summary = profile_summary(p)
    summary["skill_ratings"]["pace"] = -1.0
    assert p.skill_ratings["pace"] != -1.0
