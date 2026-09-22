"""
train_model.py — trains the burnout model and writes its model card.

  python generate_dataset.py   # (re)build the synthetic training data
  python train_model.py        # train, evaluate, save burnout_model.pkl + model_meta.json

Evaluation is honest by design: 5-fold cross-validation for every candidate, a
held-out test set that the model never sees during training, a confusion matrix,
and sanity checks that fail loudly if the model behaves against common sense.
"""
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scoring import FEATURES, LEVELS, MONOTONIC, OrdinalMonotonicClassifier, assess

df = pd.read_csv("burnout_dataset.csv")
X, y = df[FEATURES], df["burnout_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# ── 1. Compare candidates with cross-validation on the training split ─────────
candidates = {
    "Majority class (baseline)":   DummyClassifier(strategy="most_frequent"),
    "Logistic regression":         make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random forest":               RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient boosting":           HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=42),
    "Ordinal monotonic boosting":  OrdinalMonotonicClassifier(),
}
comparison = {}
print("5-fold CV on the training split")
for name, est in candidates.items():
    r = cross_validate(est, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    comparison[name] = {"cv_accuracy": round(r["test_accuracy"].mean() * 100, 1),
                        "cv_accuracy_std": round(r["test_accuracy"].std() * 100, 1),
                        "cv_f1_macro": round(r["test_f1_macro"].mean() * 100, 1)}
    print(f"  {name:28s} acc {comparison[name]['cv_accuracy']:5.1f}% ± {comparison[name]['cv_accuracy_std']:.1f}"
          f"   macro-F1 {comparison[name]['cv_f1_macro']:5.1f}%")

# The ordinal monotonic model is used even if another is marginally more accurate:
# its guarantees (e.g. more sleep never raises the score) matter more here.
CHOSEN = "Ordinal monotonic boosting"
model = OrdinalMonotonicClassifier().fit(X_train, y_train)

# ── 2. Held-out test set ──────────────────────────────────────────────────────
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
f1  = f1_score(y_test, pred, average="macro")
cm  = confusion_matrix(y_test, pred, labels=LEVELS)
rank = {l: i for i, l in enumerate(LEVELS)}
off_by_two = float(np.mean(np.abs(y_test.map(rank).to_numpy() - pd.Series(pred).map(rank).to_numpy()) == 2))
print(f"\nHeld-out test ({len(y_test)} students): accuracy {acc*100:.1f}%, macro-F1 {f1*100:.1f}%, "
      f"Low<->High mistakes {off_by_two*100:.1f}%")
print(classification_report(y_test, pred, labels=LEVELS))
print("Confusion matrix (rows = actual, cols = predicted):", LEVELS)
print(cm)

# ── 3. What the model relies on (permutation importance on the test set) ─────
perm = permutation_importance(model, X_test, y_test, scoring="accuracy",
                              n_repeats=10, random_state=42, n_jobs=-1)
feat_imp = dict(sorted(((f, round(float(max(v, 0)), 4)) for f, v in zip(FEATURES, perm.importances_mean)),
                       key=lambda kv: -kv[1]))

# Reference profile for per-student explanations: a typical "Thriving" student
reference = X_train[y_train == "Low"].median().to_dict()
reference = {f: (round(float(v), 1) if f == "cgpa" else int(round(v))) for f, v in reference.items()}

# ── 4. Refit on all data for deployment ───────────────────────────────────────
model = OrdinalMonotonicClassifier().fit(X, y)

# ── 5. Sanity checks — fail loudly if the model breaks common sense ──────────
meta_tmp = {"reference_profile": reference}
typical = {f: reference[f] for f in FEATURES}
def score_with(**changes):
    return assess(model, meta_tmp, {**typical, **changes})["score"]

sample = X.sample(300, random_state=0)
checks = {}
for f, direction in [("sleep_hours", -1), ("support_system", -1), ("backlogs", 1), ("social_media_hrs", 1)]:
    lo, hi = sample.copy(), sample.copy()
    lo[f], hi[f] = X[f].min(), X[f].max()
    diff = model.risk_score(hi) - model.risk_score(lo)
    checks[f"more {f} never {'raises' if direction < 0 else 'lowers'} the score"] = bool(np.all(diff * direction >= -1e-9))
checks["3h sleep is not 'Thriving'"]           = score_with(sleep_hours=3) >= 34
checks["12h social media is not 'Thriving'"]   = score_with(social_media_hrs=12) >= 34
worst = {f: (X[f].max() if d > 0 else X[f].min()) for f, d in MONOTONIC.items() if d}
best  = {f: (X[f].min() if d > 0 else X[f].max()) for f, d in MONOTONIC.items() if d}
checks["worst answers -> At Risk"] = score_with(**worst) >= 67
checks["best answers -> Thriving"] = score_with(**best) <= 33
print("\nSanity checks:")
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
assert all(checks.values()), "Model failed a sanity check — not saving it."

joblib.dump(model, "burnout_model.pkl")
meta = {
    "model": CHOSEN,
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "sklearn_version": sklearn.__version__,
    "data": "Synthetic (simulated) survey data — see generate_dataset.py",
    "n_samples": int(len(df)),
    "n_test": int(len(y_test)),
    "features": FEATURES,
    "classes": LEVELS,
    "accuracy": round(acc * 100, 1),
    "f1_macro": round(f1 * 100, 1),
    "low_high_confusion_pct": round(off_by_two * 100, 1),
    "confusion_matrix": cm.tolist(),
    "comparison": comparison,
    "sanity_checks": checks,
    "feature_importances": feat_imp,
    "reference_profile": reference,
}
with open("model_meta.json", "w") as fp:
    json.dump(meta, fp, indent=2)
print("\nSaved: burnout_model.pkl, model_meta.json")
