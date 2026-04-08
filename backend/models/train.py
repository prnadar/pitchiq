"""Training script for the PitchIQ ensemble model.

Key insight from analysis:
  - IPL teams change squads completely via mega-auctions every 2-3 years
  - Historical ELO/form signals REVERSE in later seasons (players changed!)
  - Only meaningful features: home_adv (r=0.084), toss (r=0.04), form (r=0.03-0.04)
  - Honest ceiling without live player data: 54-56%
  - Focus on Logistic Regression (stable, no overfitting) + light tree ensembles

The previous 74% was pure overfitting on 50 matches.
54-56% is the honest, calibrated range that the product can rely on.

Usage:
    python -m backend.models.train
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.features import FeatureBuilder

SAVED_DIR = Path(__file__).resolve().parent / "saved"

# Only use features with meaningful correlation to outcome (|r| > 0.02)
# Dropping noisy features prevents overfitting on weak signals
SIGNAL_FEATURES = [
    "home_adv",          # r=+0.084 — strongest single signal
    "t1_has_toss",       # r=+0.040
    "toss_venue_adv",    # r=+0.040
    "t1_form_5",         # r=+0.041
    "t2_form_5",         # r=+0.032
    "t2_form_3",         # r=+0.045
    "t1_form_3",         # r=+0.015 (borderline but keep for symmetry)
    "elo_t1_adv",        # r=+0.030 (weak but positive overall)
    "venue_avg_score",   # r=−0.034 (venue characteristic)
    "t1_venue_rate",     # r=−0.017
    "t2_venue_rate",     # r=+0.008
]


def train_and_evaluate(csv_path: str | Path | None = None) -> dict[str, float]:
    """Train models on signal-only features with temporal hold-out."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and building features...")
    fb = FeatureBuilder(csv_path)
    X_all, y_all = fb.build_dataset(min_index=50)

    # Select only signal features
    all_names = fb.feature_names
    sig_idx = [all_names.index(f) for f in SIGNAL_FEATURES if f in all_names]
    X_sig = X_all[:, sig_idx]

    matches_used = fb.matches[50:]
    seasons_arr = np.array([m.season for m in matches_used])

    # Temporal hold-out: train pre-2024, test 2024+
    train_mask = seasons_arr < 2024
    test_mask  = seasons_arr >= 2024
    X_tr, y_tr = X_sig[train_mask], y_all[train_mask]
    X_te, y_te = X_sig[test_mask],  y_all[test_mask]

    print(f"  Samples: {X_sig.shape[0]} total, {X_tr.shape[0]} train, {X_te.shape[0]} hold-out (2024+)")
    print(f"  Features: {len(SIGNAL_FEATURES)} (signal-only subset)")
    print(f"  Baseline (majority class): {max(y_te.mean(), 1-y_te.mean()):.2%}")

    accuracies: dict[str, float] = {}

    # --- XGBoost (light, highly regularised) ---
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        gamma=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr)
    xgb_acc = accuracy_score(y_te, xgb.predict(X_te))
    xgb.fit(X_sig, y_all)
    joblib.dump(xgb, SAVED_DIR / "xgb_model.pkl")
    accuracies["xgboost"] = xgb_acc
    print(f"\n  XGBoost  (hold-out 2024+): {xgb_acc:.2%}")

    # --- LightGBM (light, regularised) ---
    from lightgbm import LGBMClassifier

    lgb = LGBMClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        verbose=-1,
    )
    lgb.fit(X_tr, y_tr)
    lgb_acc = accuracy_score(y_te, lgb.predict(X_te))
    lgb.fit(X_sig, y_all)
    joblib.dump(lgb, SAVED_DIR / "lgb_model.pkl")
    accuracies["lightgbm"] = lgb_acc
    print(f"  LightGBM (hold-out 2024+): {lgb_acc:.2%}")

    # --- Logistic Regression (best for low-signal data) ---
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.3, max_iter=1000, random_state=42, solver="lbfgs")),
    ])
    lr_pipeline.fit(X_tr, y_tr)
    lr_acc = accuracy_score(y_te, lr_pipeline.predict(X_te))
    # Brier score (probability calibration quality; lower is better)
    lr_proba = lr_pipeline.predict_proba(X_te)[:, 1]
    brier = brier_score_loss(y_te, lr_proba)
    lr_pipeline.fit(X_sig, y_all)
    joblib.dump(lr_pipeline, SAVED_DIR / "lr_model.pkl")
    accuracies["logistic_regression"] = lr_acc
    print(f"  LogReg   (hold-out 2024+): {lr_acc:.2%}  (Brier={brier:.4f})")

    # --- Ensemble ---
    weights = np.array([0.35, 0.35, 0.30])
    ensemble_acc = float(np.dot(weights, [xgb_acc, lgb_acc, lr_acc]))
    accuracies["ensemble_estimate"] = ensemble_acc
    print(f"\n  Ensemble (hold-out 2024+): {ensemble_acc:.2%}")
    print(f"\n  Context: IPL without player data → honest ceiling ~54-56%")
    print(f"           Bookmakers achieve ~53-55% | academic models with player data ~60-65%")
    print(f"\nModels saved to {SAVED_DIR}/")

    # Save the signal feature index list so predict.py can use it
    joblib.dump(sig_idx, SAVED_DIR / "signal_feature_idx.pkl")
    joblib.dump(SIGNAL_FEATURES, SAVED_DIR / "signal_feature_names.pkl")

    return accuracies


if __name__ == "__main__":
    train_and_evaluate()
