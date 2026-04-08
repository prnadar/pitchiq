"""Tests for the ML feature builder and predictor."""
import pytest
from pathlib import Path


def test_feature_builder_loads():
    csv = Path(__file__).parents[1] / "data" / "matches.csv"
    if not csv.exists():
        pytest.skip("No matches.csv")
    from backend.models.features import FeatureBuilder
    fb = FeatureBuilder(csv)
    assert len(fb.matches) > 100


def test_feature_builder_new_match():
    csv = Path(__file__).parents[1] / "data" / "matches.csv"
    if not csv.exists():
        pytest.skip("No matches.csv")
    from backend.models.features import FeatureBuilder
    fb = FeatureBuilder(csv)
    features = fb.build_features_for_new_match("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium")
    assert isinstance(features, dict)
    assert "home_adv" in features
    assert "elo_t1_adv" in features
    assert 0.0 <= features.get("home_adv", 0.5) <= 1.0


def test_predictor_loads():
    saved = Path(__file__).parents[1] / "backend" / "models" / "saved"
    if not (saved / "xgb_model.pkl").exists():
        pytest.skip("No saved models")
    from backend.models.predict import EnsemblePredictor
    p = EnsemblePredictor(saved)
    p.load()
    assert p.is_loaded


def test_predictor_output():
    saved = Path(__file__).parents[1] / "backend" / "models" / "saved"
    if not (saved / "xgb_model.pkl").exists():
        pytest.skip("No saved models")
    from backend.models.features import FeatureBuilder
    from backend.models.predict import EnsemblePredictor
    csv = Path(__file__).parents[1] / "data" / "matches.csv"
    fb = FeatureBuilder(csv)
    p = EnsemblePredictor(saved)
    p.load()
    features = fb.build_features_for_new_match("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium")
    result = p.predict(features)
    assert 0 < result.t1_win_prob < 1
    assert abs(result.t1_win_prob + result.t2_win_prob - 1.0) < 0.01


def test_score_range():
    csv = Path(__file__).parents[1] / "data" / "matches.csv"
    if not csv.exists():
        pytest.skip("No matches.csv")
    from backend.models.features import FeatureBuilder
    fb = FeatureBuilder(csv)
    lo, hi = fb.estimate_score_range("Mumbai Indians", "Wankhede Stadium")
    assert 100 < lo < 200
    assert lo < hi
    assert hi - lo <= 40
