"""Ensemble predictor: XGBoost (0.35) + LightGBM (0.35) + LogisticRegression (0.30).

Only uses the 11 features with confirmed predictive signal (loaded from
signal_feature_names.pkl produced by train.py). All 19 features are still
computed by FeatureBuilder; the predictor selects the relevant subset.

Honest accuracy: ~54% on 2024-26 hold-out (best achievable without live player data).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

SAVED_DIR = Path(__file__).resolve().parent / "saved"

WEIGHTS = {"xgb": 0.35, "lgb": 0.35, "lr": 0.30}

# Fallback signal feature names if pkl not found (matches train.py SIGNAL_FEATURES)
_DEFAULT_SIGNAL_FEATURES = [
    "home_adv", "t1_has_toss", "toss_venue_adv",
    "t1_form_5", "t2_form_5", "t2_form_3", "t1_form_3",
    "elo_t1_adv", "venue_avg_score", "t1_venue_rate", "t2_venue_rate",
]


@dataclass(frozen=True)
class PredictionResult:
    t1_win_prob: float
    t2_win_prob: float
    confidence: float
    confidence_label: str
    model_breakdown: dict[str, float]
    score_range_low: int
    score_range_high: int


class EnsemblePredictor:
    """Loads 3 trained models and produces ensemble predictions."""

    def __init__(self, model_dir: Optional[str | Path] = None) -> None:
        self._dir = Path(model_dir) if model_dir else SAVED_DIR
        self._models: dict = {}
        self._signal_features: list[str] = []
        self._loaded = False

    def load(self) -> None:
        self._models["xgb"] = joblib.load(self._dir / "xgb_model.pkl")
        self._models["lgb"] = joblib.load(self._dir / "lgb_model.pkl")
        self._models["lr"] = joblib.load(self._dir / "lr_model.pkl")

        # Load the signal feature names saved during training
        sig_names_path = self._dir / "signal_feature_names.pkl"
        if sig_names_path.exists():
            self._signal_features = joblib.load(sig_names_path)
        else:
            self._signal_features = _DEFAULT_SIGNAL_FEATURES

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _individual_probs(self, X: np.ndarray) -> dict[str, float]:
        probs = {}
        for name, model in self._models.items():
            p = model.predict_proba(X)[0]
            probs[name] = float(p[1])
        return probs

    def _confidence_score(self, t1_prob: float) -> tuple[float, str]:
        """Confidence based on how far the probability is from 50/50."""
        # Distance from 0.5, scaled to [0, 1]
        dist = abs(t1_prob - 0.5) * 2  # 0 at 50/50, 1 at certainty
        if dist >= 0.25:
            label = "High"
        elif dist >= 0.10:
            label = "Medium"
        else:
            label = "Low"
        return round(dist, 4), label

    def _score_range(self, venue_avg_score: float) -> tuple[int, int]:
        spread = venue_avg_score * 0.15
        low = max(80, int(venue_avg_score - spread))
        high = int(venue_avg_score + spread)
        return low, high

    def predict(self, features: dict[str, float]) -> PredictionResult:
        """Run ensemble prediction on a feature dict (all 19 features from FeatureBuilder).

        Internally selects the 11 signal features before running models.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call .load() first.")

        # Build feature vector using only signal features (in training order)
        X = np.array([[features.get(k, 0.5) for k in self._signal_features]])

        individual = self._individual_probs(X)

        # Weighted ensemble
        t1_prob = sum(individual[k] * WEIGHTS[k] for k in WEIGHTS)
        t1_prob = round(max(0.01, min(0.99, t1_prob)), 4)
        t2_prob = round(1.0 - t1_prob, 4)

        confidence, label = self._confidence_score(t1_prob)

        score_low, score_high = self._score_range(features.get("venue_avg_score", 160.0))

        return PredictionResult(
            t1_win_prob=t1_prob,
            t2_win_prob=t2_prob,
            confidence=confidence,
            confidence_label=label,
            model_breakdown={
                "xgboost": round(individual["xgb"], 4),
                "lightgbm": round(individual["lgb"], 4),
                "logistic_regression": round(individual["lr"], 4),
            },
            score_range_low=score_low,
            score_range_high=score_high,
        )
