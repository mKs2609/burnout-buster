"""
mlflow_tracker.py
Logs every student prediction to MLflow for experiment tracking.
Run `mlflow ui` in the project folder to view the dashboard at localhost:5000
"""

import json
from datetime import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def log_prediction(student_name: str, roll_number: str,
                   features: dict, prediction: str,
                   probabilities: dict, model_accuracy: float):
    """
    Log a single prediction as an MLflow run.

    Parameters
    ----------
    student_name  : str
    roll_number   : str
    features      : dict  feature_name → value
    prediction    : str   "Low" | "Medium" | "High"
    probabilities : dict  {"High": float, "Low": float, "Medium": float}
    model_accuracy: float  e.g. 99.0
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_experiment("BurnoutBuster_Predictions")

        with mlflow.start_run(run_name=f"{roll_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("student_name", student_name)
            mlflow.set_tag("roll_number",  roll_number)
            mlflow.set_tag("prediction",   prediction)
            mlflow.set_tag("timestamp",    datetime.now().isoformat())

            # Feature params
            for k, v in features.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            # Metrics
            mlflow.log_metric("prob_high",       probabilities.get("High",   0))
            mlflow.log_metric("prob_medium",     probabilities.get("Medium", 0))
            mlflow.log_metric("prob_low",        probabilities.get("Low",    0))
            mlflow.log_metric("model_accuracy",  model_accuracy)
            mlflow.log_metric("is_high_risk",    1 if prediction == "High"   else 0)
            mlflow.log_metric("is_medium_risk",  1 if prediction == "Medium" else 0)

    except Exception:
        pass  # Never crash the main app due to tracking failure


def log_model_training(model, feature_names: list, accuracy: float,
                       cv_score: float, n_samples: int):
    """
    Log a full model training run to MLflow.
    Call this from train_model.py after fitting.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_experiment("BurnoutBuster_Training")

        with mlflow.start_run(run_name=f"RF_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Params
            mlflow.log_param("n_estimators",  model.n_estimators)
            mlflow.log_param("max_depth",     model.max_depth)
            mlflow.log_param("n_features",    len(feature_names))
            mlflow.log_param("n_samples",     n_samples)
            mlflow.log_param("algorithm",     "RandomForestClassifier")

            # Metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("cv_mean",        cv_score)

            # Feature importances as JSON artifact
            feat_imp = dict(zip(feature_names, model.feature_importances_))
            with open("feature_importances.json", "w") as f:
                json.dump(feat_imp, f, indent=2)
            mlflow.log_artifact("feature_importances.json")

    except Exception:
        pass
