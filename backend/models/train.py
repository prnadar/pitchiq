"""Training script for the PitchIQ ensemble model.

Reads data/matches.csv, builds features, trains XGBoost + LightGBM + LogisticRegression,
prints accuracy for each, saves .pkl files to backend/models/saved/.

Usage:
    python -m backend.models.train
    # or from project root:
    python backend/models/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Ensure project root is on path when run as script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.features import FeatureBuilder

SAVED_DIR = Path(__file__).resolve().parent / "saved"


def train_and_evaluate(csv_path: str | Path | None = None) -> dict[str, float]:
    """Train all 3 models, print accuracy, save to disk. Returns accuracy dict."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and building features...")
    fb = FeatureBuilder(csv_path)
    X, y = fb.build_dataset(min_index=5)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate (team1 wins): {y.mean():.2%}")

    accuracies: dict[str, float] = {}

    # --- XGBoost ---
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_scores = cross_val_score(xgb, X, y, cv=min(5, len(y)), scoring="accuracy")
    xgb.fit(X, y)
    joblib.dump(xgb, SAVED_DIR / "xgb_model.pkl")
    accuracies["xgboost"] = float(xgb_scores.mean())
    print(f"  XGBoost      CV accuracy: {xgb_scores.mean():.2%} (+/- {xgb_scores.std():.2%})")

    # --- LightGBM ---
    from lightgbm import LGBMClassifier

    lgb = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )
    lgb_scores = cross_val_score(lgb, X, y, cv=min(5, len(y)), scoring="accuracy")
    lgb.fit(X, y)
    joblib.dump(lgb, SAVED_DIR / "lgb_model.pkl")
    accuracies["lightgbm"] = float(lgb_scores.mean())
    print(f"  LightGBM     CV accuracy: {lgb_scores.mean():.2%} (+/- {lgb_scores.std():.2%})")

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=min(5, len(y)), scoring="accuracy")
    lr.fit(X, y)
    joblib.dump(lr, SAVED_DIR / "lr_model.pkl")
    accuracies["logistic_regression"] = float(lr_scores.mean())
    print(f"  LogRegression CV accuracy: {lr_scores.mean():.2%} (+/- {lr_scores.std():.2%})")

    # --- Ensemble estimate ---
    weights = np.array([0.45, 0.35, 0.20])
    accs = np.array([accuracies["xgboost"], accuracies["lightgbm"], accuracies["logistic_regression"]])
    ensemble_est = float(np.dot(weights, accs))
    accuracies["ensemble_estimate"] = ensemble_est
    print(f"\n  Ensemble est. accuracy:    {ensemble_est:.2%}")
    print(f"\nModels saved to {SAVED_DIR}/")

    return accuracies


if __name__ == "__main__":
    train_and_evaluate()
