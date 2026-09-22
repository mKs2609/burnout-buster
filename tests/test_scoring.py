"""The model's promises: monotonic behaviour, safety rules, bands, explanations."""
import pandas as pd
import pytest

from scoring import FEATURES, MONOTONIC, assess, risk_from_score, safety_floor


@pytest.mark.parametrize("score,expected", [
    (0, "Low"), (33, "Low"), (34, "Medium"), (66, "Medium"), (67, "High"), (100, "High"),
])
def test_bands_match_the_score_guide(score, expected):
    assert risk_from_score(score) == expected


def test_score_stays_within_0_100(model_and_meta, healthy_answers):
    model, meta = model_and_meta
    worst = {f: 10 if d > 0 else 0 for f, d in MONOTONIC.items() if d}
    for answers in (healthy_answers, {**healthy_answers, **worst}):
        assert 0 <= assess(model, meta, {**healthy_answers, **answers})["score"] <= 100


@pytest.mark.parametrize("feature,direction", [(f, d) for f, d in MONOTONIC.items() if d])
def test_monotonic_features_never_move_the_wrong_way(model_and_meta, healthy_answers, feature, direction):
    """More sleep must never raise a score; more backlogs must never lower one."""
    model, _ = model_and_meta
    lo, hi = dict(healthy_answers), dict(healthy_answers)
    lo[feature], hi[feature] = 0, 12
    frame = pd.DataFrame([lo, hi])[FEATURES].astype(float)
    low_score, high_score = model.risk_score(frame)
    if direction > 0:
        assert high_score >= low_score - 1e-9
    else:
        assert high_score <= low_score + 1e-9


def test_severe_sleep_deprivation_is_never_thriving(model_and_meta, healthy_answers):
    model, meta = model_and_meta
    result = assess(model, meta, {**healthy_answers, "sleep_hours": 3})
    assert result["risk"] != "Low"
    assert result["safety_floor_applied"]
    assert any("4 hours" in reason for reason in result["safety_reasons"])


def test_three_severe_signals_force_at_risk(healthy_answers):
    floor, reasons = safety_floor({**healthy_answers, "sleep_hours": 4,
                                   "support_system": 2, "backlogs": 5})
    assert floor == 67 and len(reasons) == 3


def test_safety_rules_only_raise_never_lower(model_and_meta, healthy_answers):
    model, meta = model_and_meta
    struggling = {**healthy_answers, "sleep_hours": 4, "backlogs": 6, "confidence": 1,
                  "support_system": 1, "assignments_per_week": 12}
    result = assess(model, meta, struggling)
    assert result["score"] >= result["model_score"]


def test_healthy_answers_score_low(model_and_meta, healthy_answers):
    model, meta = model_and_meta
    assert assess(model, meta, healthy_answers)["risk"] == "Low"


def test_explanations_name_the_answer_that_changed(model_and_meta, healthy_answers):
    model, meta = model_and_meta
    result = assess(model, meta, {**healthy_answers, "sleep_hours": 4})
    assert "sleep_hours" in [f for f, _ in result["drivers"]]


def test_explanations_still_work_at_a_saturated_score(model_and_meta, healthy_answers):
    """Log-odds explanations must keep ranking factors even at score 100."""
    model, meta = model_and_meta
    worst = {**healthy_answers, "sleep_hours": 3, "backlogs": 8, "confidence": 1,
             "support_system": 1, "assignments_per_week": 12, "rejection_sensitivity": 10,
             "fomo_score": 10, "social_media_hrs": 12}
    result = assess(model, meta, worst)
    assert result["score"] == 100
    assert len(result["drivers"]) == 3


def test_strong_support_shows_up_as_protective(model_and_meta, healthy_answers):
    model, meta = model_and_meta
    struggling = {**healthy_answers, "assignments_per_week": 12, "sleep_hours": 4,
                  "confidence": 2, "support_system": 10}
    result = assess(model, meta, struggling)
    assert "support_system" in [f for f, _ in result["protective"]]


def test_probabilities_are_a_distribution(model_and_meta, healthy_answers):
    model, meta = model_and_meta
    proba = assess(model, meta, healthy_answers)["proba"]
    assert set(proba) == {"Low", "Medium", "High"}
    assert sum(proba.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= p <= 1 for p in proba.values())
