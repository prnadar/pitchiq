"""Ensemble predictor: XGBoost (0.45) + LightGBM (0.35) + LogisticRegression (0.20).

Loads trained .pkl files, returns win probabilities, confidence, and model breakdown.
Also provides score range prediction based on venue average.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

SAVED_DIR = Path(__file__).resolve().parent / "saved"

WEIGHTS = {"xgb": 0.45, "lgb": 0.35, "lr": 0.20}


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
        self._loaded = False

    def load(self) -> None:
        self._models["xgb"] = joblib.load(self._dir / "xgb_model.pkl")
        self._models["lgb"] = joblib.load(self._dir / "lgb_model.pkl")
        self._models["lr"] = joblib.load(self._dir / "lr_model.pkl")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _individual_probs(self, X: np.ndarray) -> dict[str, float]:
        """Get P(team1 wins) from each model."""
        probs = {}
        for name, model in self._models.items():
            p = model.predict_proba(X)[0]
            probs[name] = float(p[1])  # probability of class 1 (team1 wins)
        return probs

    def _confidence_score(self, probs: dict[str, float]) -> tuple[float, str]:
        """Model agreement score: 1 - std of individual predictions, scaled 0-1."""
        vals = list(probs.values())
        agreement = 1.0 - min(np.std(vals) * 3, 1.0)  # scale std to 0-1 range
        confidence = max(0.0, min(1.0, agreement))

        if confidence >= 0.75:
            label = "High"
        elif confidence >= 0.45:
            label = "Medium"
        else:
            label = "Low"

        return round(confidence, 4), label

    def _score_range(self, venue_avg_score: float) -> tuple[int, int]:
        """Predict first-innings score range based on venue average +/- 15%."""
        spread = venue_avg_score * 0.15
        low = max(80, int(venue_avg_score - spread))
        high = int(venue_avg_score + spread)
        return low, high

    def predict(self, features: dict[str, float]) -> PredictionResult:
        """Run ensemble prediction on a feature dict.

        Returns PredictionResult with win probabilities, confidence, and score range.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call .load() first.")

        feature_order = [
            "t1_form_5", "t2_form_5", "t1_form_3", "t2_form_3",
            "h2h_t1_rate", "t1_venue_rate", "t2_venue_rate",
            "toss_venue_adv", "t1_has_toss", "toss_bat_1st",
            "form_diff", "venue_avg_score",
        ]
        X = np.array([[features[k] for k in feature_order]])

        individual = self._individual_probs(X)

        t1_prob = sum(individual[k] * WEIGHTS[k] for k in WEIGHTS)
        t1_prob = round(max(0.01, min(0.99, t1_prob)), 4)
        t2_prob = round(1.0 - t1_prob, 4)

        confidence, label = self._confidence_score(individual)

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
