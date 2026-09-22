"""
generate_dataset.py — builds the synthetic training set (burnout_dataset.csv).

This is SIMULATED data, not real student records. It is designed to behave like
survey data rather than to be easy to classify:

  • Answers are driven by shared hidden factors (general strain, academic load), so
    features correlate the way real answers do (e.g. poor sleep goes with high FOMO).
  • Every feature uses exactly the same range as the survey slider in the app.
  • The burnout label comes from a hidden burnout index — a weighted mix of risk and
    protective factors, a sleep-deprivation penalty and a workload × low-support
    interaction — plus noise. Classes therefore overlap, as they would in real life.

Replace this with real, consented survey data (and counselor-verified outcomes) as
soon as it is available; train_model.py works unchanged on any CSV with these columns.
"""
import numpy as np
import pandas as pd

N_STUDENTS    = 5000
NOISE_SD      = 0.9       # label noise — controls how much the classes overlap
CARELESS_FRAC = 0.05      # share of careless responders (see below)
SEED          = 42

# (min, max) of each survey slider in app.py
RANGES = {
    "exams_per_month": (1, 8),       "assignments_per_week": (1, 12),
    "attendance_pressure": (1, 10),  "cgpa": (4.0, 10.0),
    "backlogs": (0, 8),              "study_hours_per_day": (1, 12),
    "fomo_score": (1, 10),           "peer_pressure": (1, 10),
    "family_expectations": (1, 10),  "social_media_hrs": (0, 12),
    "rejection_sensitivity": (1, 10),"sleep_hours": (3, 10),
    "exercise_days": (0, 7),         "diet_quality": (1, 10),
    "confidence": (1, 10),           "support_system": (1, 10),
    "mental_health_visits": (0, 5),
}

def burnout_index(d: pd.DataFrame) -> np.ndarray:
    """Hidden 'true' burnout level. Positive weights raise risk, negative protect."""
    return (
        0.25 * (d.exams_per_month - 4)
        + 0.18 * (d.assignments_per_week - 5.5)
        + 0.20 * (d.attendance_pressure - 5.5)
        - 0.35 * (d.cgpa - 7.3)
        + 0.35 * d.backlogs
        + 0.25 * np.maximum(0, d.study_hours_per_day - 8)   # overwork
        + 0.15 * np.maximum(0, 3 - d.study_hours_per_day)   # disengagement
        + 0.22 * (d.fomo_score - 5)
        + 0.15 * (d.peer_pressure - 5)
        + 0.15 * (d.family_expectations - 6)
        + 0.18 * (d.social_media_hrs - 3.5)
        + 0.25 * (d.rejection_sensitivity - 5)
        - 0.45 * (d.sleep_hours - 7)
        + 0.60 * np.maximum(0, 5 - d.sleep_hours)          # severe sleep deprivation
        - 0.15 * (d.exercise_days - 3)
        - 0.12 * (d.diet_quality - 6)
        - 0.30 * (d.confidence - 6)
        - 0.30 * (d.support_system - 6.5)
        + 0.10 * d.mental_health_visits
        + 0.05 * (d.assignments_per_week - 5.5) * np.maximum(0, 5 - d.support_system)
    )

def generate(n=N_STUDENTS, noise_sd=NOISE_SD, seed=SEED, careless_frac=CARELESS_FRAC) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)   # general strain
    a = rng.normal(0, 1, n)   # academic load
    noise = lambda sd: rng.normal(0, sd, n)

    raw = {
        "exams_per_month":       4.0 + 1.2*a + 0.3*z + noise(1.2),
        "assignments_per_week":  5.5 + 1.8*a + 0.3*z + noise(1.8),
        "attendance_pressure":   5.5 + 1.2*a + 0.6*z + noise(1.5),
        "cgpa":                  7.3 - 0.5*a - 0.4*z + noise(0.8),
        "backlogs":              rng.poisson(np.exp(-0.8 + 0.6*a + 0.5*z)),
        "study_hours_per_day":   5.0 + 1.0*a + noise(1.8),
        "fomo_score":            5.0 + 1.3*z + noise(1.6),
        "peer_pressure":         5.0 + 1.1*z + noise(1.7),
        "family_expectations":   6.0 + 0.9*z + noise(1.8),
        "social_media_hrs":      3.5 + 1.2*z + noise(1.8),
        "rejection_sensitivity": 5.0 + 1.3*z + noise(1.6),
        "sleep_hours":           6.8 - 0.8*z - 0.4*a + noise(1.0),
        "exercise_days":         3.0 - 0.8*z + noise(1.6),
        "diet_quality":          6.0 - 1.0*z + noise(1.6),
        "confidence":            6.0 - 1.4*z + noise(1.5),
        "support_system":        6.5 - 1.2*z + noise(1.8),
        "mental_health_visits":  rng.poisson(np.exp(-1.5 + 0.6*z)),
    }
    df = pd.DataFrame({
        k: (np.round(np.clip(v, *RANGES[k]), 1) if k == "cgpa"
            else np.clip(np.round(v), *RANGES[k]).astype(int))
        for k, v in raw.items()
    })

    index = burnout_index(df) + rng.normal(0, noise_sd, n)
    low_cut, high_cut = np.quantile(index, [0.40, 0.75])   # ~40% Low, 35% Medium, 25% High
    df["burnout_risk"] = np.where(index >= high_cut, "High",
                          np.where(index >= low_cut, "Medium", "Low"))

    # Careless responders: every real survey has people who click straight down the
    # middle without reading. Their answers stop matching their true state, so these
    # rows keep their label but get flat mid-scale answers — noise no model can learn.
    n_careless = int(round(careless_frac * n))
    if n_careless:
        idx = rng.choice(n, n_careless, replace=False)
        for col, (lo, hi) in RANGES.items():
            mid = (lo + hi) / 2
            jitter = rng.integers(-1, 2, n_careless)
            df.loc[idx, col] = (np.round(np.clip(mid + jitter, lo, hi), 1) if col == "cgpa"
                                else np.clip(np.round(mid + jitter), lo, hi).astype(int))
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("burnout_dataset.csv", index=False)
    print(f"Saved {len(df)} rows:", df["burnout_risk"].value_counts().to_dict())
