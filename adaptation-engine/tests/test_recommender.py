import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommender import recommend_next_styles, explain_recommendation
from styles import STYLE_BY_KEY


def test_recommendation_excludes_mastered_style():
    recs = recommend_next_styles("fast-good-off-none", top_n=3)
    keys = [r.key for r in recs]
    assert "fast-good-off-none" not in keys
    assert len(keys) == 3


def test_recommendation_is_high_contrast():
    # A fast, seam-up, off-stump style mastered -> top pick should differ
    # in pace and/or movement, not just a trivial line/length tweak.
    recs = recommend_next_styles("fast-good-off-none", top_n=1)
    top = recs[0]
    mastered = STYLE_BY_KEY["fast-good-off-none"]
    assert top.pace_band != mastered.pace_band or top.movement != mastered.movement


def test_exclude_param_removes_recently_used_styles():
    recs = recommend_next_styles(
        "fast-good-off-none", top_n=5, exclude=["slow-good-off-offspin"]
    )
    keys = [r.key for r in recs]
    assert "slow-good-off-offspin" not in keys


def test_unknown_style_raises():
    try:
        recommend_next_styles("not-a-real-style")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_explanation_mentions_a_changed_attribute():
    recs = recommend_next_styles("fast-good-off-none", top_n=1)
    text = explain_recommendation("fast-good-off-none", recs[0])
    assert "Switching because" in text
