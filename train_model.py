"""
train_model.py
Trains the Random Forest model and saves artefacts.
Also logs the training run to MLflow if available.
"""

import pandas as pd
import numpy as np
import joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv("burnout_dataset.csv")
print(f"Dataset: {df.shape}  |  Classes: {df['burnout_risk'].value_counts().to_dict()}")

FEATURES = [c for c in df.columns if c != "burnout_risk"]
X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df["burnout_risk"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
cw = dict(zip(classes, weights))

model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=2,
    class_weight=cw, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cv     = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="accuracy")

print(f"\n✅ Test Accuracy : {acc*100:.1f}%")
print(f"   CV Accuracy   : {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

importances = model.feature_importances_
feat_imp    = sorted(zip(FEATURES, importances), key=lambda x: -x[1])
print("\nTop 5 Features:")
for f, i in feat_imp[:5]:
    print(f"  {f:<28} {i:.4f}")

joblib.dump(model, "burnout_model.pkl")
joblib.dump(le,    "label_encoder.pkl")
meta = {
    "features": FEATURES,
    "classes":  list(le.classes_),
    "accuracy": round(acc * 100, 1),
    "cv_accuracy": round(cv.mean() * 100, 1),
    "feature_importances": {f: round(float(i), 4) for f, i in feat_imp},
}
with open("model_meta.json", "w") as fp:
    json.dump(meta, fp, indent=2)

# ── MLflow logging (optional) ────────────────────────────────────────────────
try:
    from mlflow_tracker import log_model_training
    log_model_training(model, FEATURES, acc, cv.mean(), len(df))
    print("\n✅ MLflow run logged. Run `mlflow ui` to view dashboard.")
except Exception:
    pass

print("\n✅ Saved: burnout_model.pkl · label_encoder.pkl · model_meta.json")
