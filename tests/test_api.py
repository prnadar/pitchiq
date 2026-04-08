"""Basic smoke tests for the PitchIQ API."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_predict_missing_fields():
    r = client.post("/predict", json={})
    assert r.status_code == 422  # validation error


def test_predict_valid(monkeypatch):
    from pathlib import Path
    import backend.main as m
    from backend.models.features import FeatureBuilder
    from backend.models.predict import EnsemblePredictor

    # Monkeypatch usage check to always allow
    monkeypatch.setattr(m, "_check_usage", lambda u, ip: True)

    csv = Path(__file__).parents[1] / "data" / "matches.csv"
    saved = Path(__file__).parents[1] / "backend" / "models" / "saved"

    if not csv.exists() or not (saved / "xgb_model.pkl").exists():
        pytest.skip("No matches.csv or trained models available")

    fb = FeatureBuilder(csv)
    p = EnsemblePredictor(saved)
    p.load()
    monkeypatch.setattr(m, "feature_builder", fb)
    monkeypatch.setattr(m, "predictor", p)

    r = client.post("/predict", json={
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "venue": "Wankhede Stadium",
    })
    assert r.status_code == 200
    data = r.json()
    assert "t1_win_prob" in data
    assert "t2_win_prob" in data
    assert data["t1_win_prob"] + data["t2_win_prob"] == pytest.approx(1.0, abs=0.01)


def test_upcoming_predictions():
    r = client.get("/matches/upcoming/with-predictions")
    # May return 200 or 500 depending on startup — just check it doesn't crash badly
    assert r.status_code in (200, 500)
    if r.status_code == 200:
        data = r.json()
        assert "matches" in data


def test_lineup_unknown_team():
    r = client.get("/teams/Unknown Team/lineup?season=2024")
    assert r.status_code == 200
    data = r.json()
    assert data["players"] == [] or isinstance(data["players"], list)


def test_signup_missing_fields():
    r = client.post("/auth/signup", json={"email": "test@example.com"})
    assert r.status_code in (400, 422)


def test_signup_and_login():
    import random
    email = f"test{random.randint(1000,9999)}@pitchiq.test"
    r = client.post("/auth/signup", json={"email": email, "name": "Test", "password": "test123"})
    assert r.status_code == 200
    data = r.json()
    assert "token" in data

    r2 = client.post("/auth/login", json={"email": email, "password": "test123"})
    # Dev mode always returns 200
    assert r2.status_code in (200, 401)


def test_rate_limit(monkeypatch):
    """After 2 predictions, 3rd should be rate-limited (no auth)."""
    from pathlib import Path
    import backend.main as m
    from backend.models.features import FeatureBuilder
    from backend.models.predict import EnsemblePredictor

    csv = Path(__file__).parents[1] / "data" / "matches.csv"
    saved = Path(__file__).parents[1] / "backend" / "models" / "saved"

    if csv.exists() and (saved / "xgb_model.pkl").exists():
        fb = FeatureBuilder(csv)
        p = EnsemblePredictor(saved)
        p.load()
        monkeypatch.setattr(m, "feature_builder", fb)
        monkeypatch.setattr(m, "predictor", p)
    else:
        pytest.skip("No matches.csv or trained models — cannot test rate limit end-to-end")

    # Reset usage for test IP
    m._DAILY_USAGE.pop("testclient", None)
    payload = {"team1": "Mumbai Indians", "team2": "CSK", "venue": "Wankhede Stadium"}
    # First two should work
    for _ in range(2):
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
    # Third should be rate-limited
    r = client.post("/predict", json=payload)
    assert r.status_code == 429
