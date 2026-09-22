"""
scoring.py — the burnout model and everything that turns its output into a result.

Model: burnout risk is ordered (Low < Medium < High), so instead of one multiclass
classifier we train two binary gradient-boosting models (the Frank & Hall ordinal
approach):  P(risk ≥ Medium)  and  P(risk ≥ High).

Each binary model has monotonic constraints, which multiclass boosting can't have.
That guarantees common-sense behaviour: e.g. more sleep or more support can never
raise a student's score, and more backlogs can never lower it.

Score (0-100) = 50·P(≥Medium) + 50·P(≥High)  — the expected risk level scaled to 100.
Bands: 0-33 Low (Thriving), 34-66 Medium (Needs Attention), 67-100 High (At Risk).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = [
    "exams_per_month","assignments_per_week","attendance_pressure","cgpa",
    "backlogs","study_hours_per_day","fomo_score","peer_pressure",
    "family_expectations","social_media_hrs","rejection_sensitivity",
    "sleep_hours","exercise_days","diet_quality","confidence",
    "support_system","mental_health_visits",
]

# +1 = higher value can only raise risk, -1 = can only lower it, 0 = unconstrained.
# Study hours is U-shaped (too few and too many both hurt); counselor visits are
# ambiguous (a sign of struggle, but also of getting help).
MONOTONIC = {
    "exams_per_month": 1, "assignments_per_week": 1, "attendance_pressure": 1,
    "cgpa": -1, "backlogs": 1, "study_hours_per_day": 0, "fomo_score": 1,
    "peer_pressure": 1, "family_expectations": 1, "social_media_hrs": 1,
    "rejection_sensitivity": 1, "sleep_hours": -1, "exercise_days": -1,
    "diet_quality": -1, "confidence": -1, "support_system": -1,
    "mental_health_visits": 0,
}

LEVELS = ["Low", "Medium", "High"]

FEATURE_LABELS = {
    "exams_per_month": "Exams per month", "assignments_per_week": "Assignments per week",
    "attendance_pressure": "Attendance pressure", "cgpa": "CGPA", "backlogs": "Active backlogs",
    "study_hours_per_day": "Study hours per day", "fomo_score": "FOMO",
    "peer_pressure": "Peer pressure", "family_expectations": "Family expectations",
    "social_media_hrs": "Social media hours", "rejection_sensitivity": "Rejection sensitivity",
    "sleep_hours": "Sleep", "exercise_days": "Exercise days", "diet_quality": "Diet quality",
    "confidence": "Self-confidence", "support_system": "Support from friends/family",
    "mental_health_visits": "Counselor visits",
}


class OrdinalMonotonicClassifier(ClassifierMixin, BaseEstimator):
    """Two monotonic binary boosters: P(y ≥ Medium) and P(y ≥ High)."""

    def __init__(self, features=tuple(FEATURES), monotonic=None, max_depth=3,
                 learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42):
        self.features = features
        self.monotonic = monotonic          # None = use MONOTONIC
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _booster(self):
        mono = self.monotonic if self.monotonic is not None else MONOTONIC
        return HistGradientBoostingClassifier(
            monotonic_cst=[mono.get(f, 0) for f in self.features],
            max_depth=self.max_depth, learning_rate=self.learning_rate,
            max_iter=self.max_iter, l2_regularization=self.l2_regularization,
            random_state=self.random_state)

    def fit(self, X, y):
        X = self._matrix(X)
        rank = pd.Series(np.asarray(y)).map({l: i for i, l in enumerate(LEVELS)}).to_numpy()
        self.classes_ = np.array(LEVELS)
        self.ge_medium_ = self._booster().fit(X, (rank >= 1).astype(int))
        self.ge_high_   = self._booster().fit(X, (rank >= 2).astype(int))
        return self

    def _matrix(self, X):
        if isinstance(X, pd.DataFrame):
            return X[list(self.features)].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def cumulative_proba(self, X):
        X = self._matrix(X)
        p_med  = self.ge_medium_.predict_proba(X)[:, 1]
        p_high = np.minimum(self.ge_high_.predict_proba(X)[:, 1], p_med)  # keep P(≥High) ≤ P(≥Medium)
        return p_med, p_high

    def predict_proba(self, X):
        """Columns in LEVELS order: Low, Medium, High."""
        p_med, p_high = self.cumulative_proba(X)
        return np.column_stack([1 - p_med, p_med - p_high, p_high])

    def risk_score(self, X):
        """0-100 burnout score."""
        p_med, p_high = self.cumulative_proba(X)
        return 50 * p_med + 50 * p_high

    def risk_logit(self, X):
        """Sum of the two boosters' log-odds. Monotonic like the score, but it doesn't
        saturate at 0/100, so it still ranks factors for very high or very low scores."""
        X = self._matrix(X)
        return self.ge_medium_.decision_function(X) + self.ge_high_.decision_function(X)

    def predict(self, X):
        return np.array([risk_from_score(s) for s in self.risk_score(X)])


def risk_from_score(s):
    """The score is the single source of truth for the risk level."""
    return "Low" if s <= 33 else "Medium" if s <= 66 else "High"


# ── SAFETY RULES ──────────────────────────────────────────────────────────────
# Transparent rules for answers that should never be scored as "fine", whatever the
# model says. They can only RAISE a score (to the bottom of a band), never lower it.
SEVERE_SIGNALS = [
    ("sleep_hours",           lambda v: v <= 4,  "sleeping 4 hours or less"),
    ("support_system",        lambda v: v <= 2,  "very little support from friends/family"),
    ("confidence",            lambda v: v <= 2,  "very low self-confidence"),
    ("backlogs",              lambda v: v >= 4,  "4 or more active backlogs"),
    ("rejection_sensitivity", lambda v: v >= 9,  "very high rejection sensitivity"),
    ("social_media_hrs",      lambda v: v >= 9,  "9+ hours a day on social media"),
]

def safety_floor(features: dict):
    """Returns (minimum score, reasons)."""
    hits = [text for f, test, text in SEVERE_SIGNALS if test(float(features[f]))]
    if len(hits) >= 3:
        return 67, hits
    if hits:
        return 34, hits
    return 0, []


# ── ASSESSMENT ────────────────────────────────────────────────────────────────
MIN_EFFECT = 0.25   # log-odds; smaller effects aren't worth mentioning

def impact_label(effect):
    return "Major" if effect >= 1.5 else "Moderate" if effect >= 0.6 else "Minor"

def assess(model, meta: dict, features: dict) -> dict:
    """Score one survey. Returns score, risk level, per-level probabilities, the
    factors pushing the score up (drivers) / down (protective), and any safety rule."""
    x = pd.DataFrame([features])[FEATURES].astype(float)
    model_score = float(model.risk_score(x)[0])
    proba = dict(zip(LEVELS, model.predict_proba(x)[0]))

    floor, reasons = safety_floor(features)
    score = int(round(max(model_score, floor)))

    # Explanation: how much would the model's risk change if this one answer were a
    # typical "Thriving" student's answer (median of the Low class in training data)?
    # Measured in log-odds so it still works when the score is saturated near 0 or 100.
    reference = meta.get("reference_profile", {})
    variants = []
    for f in FEATURES:
        if f in reference and float(features[f]) != float(reference[f]):
            v = x.copy(); v[f] = float(reference[f]); variants.append((f, v))
    effects = []
    if variants:
        base = float(model.risk_logit(x)[0])
        alt = model.risk_logit(pd.concat([v for _, v in variants], ignore_index=True))
        effects = [(f, base - a) for (f, _), a in zip(variants, alt)]
    drivers    = sorted([e for e in effects if e[1] >=  MIN_EFFECT], key=lambda e: -e[1])[:3]
    protective = sorted([e for e in effects if e[1] <= -MIN_EFFECT], key=lambda e: e[1])[:2]

    return {
        "score": score,
        "risk": risk_from_score(score),
        "model_score": round(model_score, 1),
        "proba": proba,
        "drivers": [(f, impact_label(d)) for f, d in drivers],
        "protective": [(f, impact_label(-d)) for f, d in protective],
        "safety_floor_applied": floor > model_score,
        "safety_reasons": reasons,
    }
