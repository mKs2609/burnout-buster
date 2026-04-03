"""
generate_dataset.py
Generates a realistic synthetic student burnout dataset
simulating past college records from engineering colleges in India.
Run this once to create burnout_dataset.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 300  # number of student records

def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi).astype(int)

# ── Academic features ──────────────────────────────────────────────────────────
exams_per_month       = clamp(np.random.normal(4, 1.5, N), 1, 8)
assignments_per_week  = clamp(np.random.normal(5, 2,   N), 1, 12)
attendance_pressure   = clamp(np.random.normal(6, 2,   N), 1, 10)
cgpa                  = np.round(np.clip(np.random.normal(7.2, 1.0, N), 4.0, 10.0), 1)
backlogs              = clamp(np.random.poisson(1.2, N), 0, 8)
study_hours_per_day   = clamp(np.random.normal(5, 2,   N), 1, 12)

# ── Social / psychological features ──────────────────────────────────────────
fomo_score            = clamp(np.random.normal(6, 2,   N), 1, 10)
peer_pressure         = clamp(np.random.normal(6, 2,   N), 1, 10)
family_expectations   = clamp(np.random.normal(7, 1.5, N), 1, 10)
social_media_hrs      = clamp(np.random.normal(4, 2,   N), 0, 12)
rejection_sensitivity = clamp(np.random.normal(5, 2,   N), 1, 10)

# ── Lifestyle features ─────────────────────────────────────────────────────────
sleep_hours           = clamp(np.random.normal(6, 1.5, N), 3, 10)
exercise_days         = clamp(np.random.normal(2, 1.5, N), 0, 7)
diet_quality          = clamp(np.random.normal(5, 2,   N), 1, 10)

# ── Emotional / support features ─────────────────────────────────────────────
confidence            = clamp(np.random.normal(5, 2,   N), 1, 10)
support_system        = clamp(np.random.normal(5, 2,   N), 1, 10)
mental_health_visits  = clamp(np.random.poisson(0.5,   N), 0, 5)

# ── Derive burnout risk label ─────────────────────────────────────────────────
# Higher score → higher burnout risk
risk_score = (
    exams_per_month * 0.8
    + assignments_per_week * 0.7
    + attendance_pressure * 0.6
    + (10 - cgpa) * 1.0          # low CGPA → more stress
    + backlogs * 1.5
    + fomo_score * 0.7
    + peer_pressure * 0.8
    + family_expectations * 0.6
    + social_media_hrs * 0.4
    + rejection_sensitivity * 0.5
    + (10 - sleep_hours) * 1.2   # less sleep → more risk
    + (7 - exercise_days) * 0.3  # no exercise → more risk
    + (10 - diet_quality) * 0.3
    + (10 - confidence) * 0.9
    + (10 - support_system) * 0.7
    + np.random.normal(0, 2, N)  # noise
)

# Normalise to [0, 100] and bin into Low / Medium / High
risk_norm = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100

def label(r):
    if r < 38:   return "Low"
    elif r < 68: return "Medium"
    else:         return "High"

burnout_risk = [label(r) for r in risk_norm]

df = pd.DataFrame({
    "exams_per_month":       exams_per_month,
    "assignments_per_week":  assignments_per_week,
    "attendance_pressure":   attendance_pressure,
    "cgpa":                  cgpa,
    "backlogs":              backlogs,
    "study_hours_per_day":   study_hours_per_day,
    "fomo_score":            fomo_score,
    "peer_pressure":         peer_pressure,
    "family_expectations":   family_expectations,
    "social_media_hrs":      social_media_hrs,
    "rejection_sensitivity": rejection_sensitivity,
    "sleep_hours":           sleep_hours,
    "exercise_days":         exercise_days,
    "diet_quality":          diet_quality,
    "confidence":            confidence,
    "support_system":        support_system,
    "mental_health_visits":  mental_health_visits,
    "burnout_risk":          burnout_risk,
})

df.to_csv("burnout_dataset.csv", index=False)
print(f"✅ Dataset saved: {len(df)} rows")
print(df["burnout_risk"].value_counts())
print(df.head())
