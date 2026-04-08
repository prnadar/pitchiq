"""PitchIQ FastAPI backend.

Endpoints: /predict, /matches, /odds, /auth, /health, /train
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr

from backend.auth.auth import create_jwt, decode_jwt, hash_password, verify_password
from backend.models.features import FeatureBuilder
from backend.models.predict import EnsemblePredictor
from backend.models.reasoning import ReasoningEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pitchiq")

app = FastAPI(
    title="PitchIQ",
    description="IPL Match Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state ---
feature_builder: Optional[FeatureBuilder] = None
predictor: Optional[EnsemblePredictor] = None
reasoning_engine: ReasoningEngine = ReasoningEngine()

# --- Rate limiting ---
_DAILY_USAGE: dict[str, dict[str, int]] = {}

# --- Saved predictions store ---
_saved_predictions: dict[str, list[dict]] = {}


def _check_usage(user: Optional[dict], client_ip: str) -> bool:
    """Returns True if allowed, False if daily limit exceeded."""
    if user and user.get("plan") in ("trial", "pro", "expert"):
        return True  # unlimited for paid
    key = user["id"] if user else client_ip
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    counts = _DAILY_USAGE.setdefault(key, {})
    # Purge old dates
    for d in list(counts.keys()):
        if d != today:
            del counts[d]
    if counts.get(today, 0) >= 2:
        return False
    counts[today] = counts.get(today, 0) + 1
    return True


@app.on_event("startup")
async def startup() -> None:
    global feature_builder, predictor
    csv_path = Path(__file__).resolve().parents[1] / "data" / "matches.csv"
    if csv_path.exists():
        feature_builder = FeatureBuilder(csv_path)
        logger.info("FeatureBuilder loaded with %d matches", len(feature_builder.matches))

    saved_dir = Path(__file__).resolve().parent / "models" / "saved"
    if (saved_dir / "xgb_model.pkl").exists():
        predictor = EnsemblePredictor(saved_dir)
        predictor.load()
        logger.info("EnsemblePredictor loaded")
    else:
        logger.warning("No trained models found at %s — /predict will be unavailable", saved_dir)


# --- Request/Response models ---

class PredictRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str = ""
    toss_decision: str = ""
    odds_t1: float = 0.0
    odds_t2: float = 0.0


class SignupRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# --- Auth helper ---

def _get_current_user(authorization: str = Header(default="")) -> Optional[dict]:
    if not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    return decode_jwt(token)


# --- Endpoints ---

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": predictor is not None and predictor.is_loaded,
        "matches_loaded": feature_builder is not None,
        "match_count": len(feature_builder.matches) if feature_builder else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
async def predict(req: PredictRequest, request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
    user = _get_current_user(authorization)
    client_ip = request.client.host if request.client else "unknown"
    if not _check_usage(user, client_ip):
        raise HTTPException(429, "Daily limit reached. Create a free account for unlimited access.")
    if not predictor or not feature_builder:
        raise HTTPException(503, "Models not loaded. Train first.")

    features = feature_builder.build_features_for_new_match(
        team1=req.team1,
        team2=req.team2,
        venue=req.venue,
        toss_winner=req.toss_winner,
        toss_decision=req.toss_decision,
    )
    result = predictor.predict(features)

    reasoning = reasoning_engine.generate(
        team1=req.team1,
        team2=req.team2,
        venue=req.venue,
        features=features,
        result=result,
        fb=feature_builder,
        odds_t1=req.odds_t1,
        odds_t2=req.odds_t2,
    )

    return {
        "team1": req.team1,
        "team2": req.team2,
        "venue": req.venue,
        "t1_win_prob": result.t1_win_prob,
        "t2_win_prob": result.t2_win_prob,
        "confidence": result.confidence,
        "confidence_label": result.confidence_label,
        "model_breakdown": result.model_breakdown,
        "score_range_low": result.score_range_low,
        "score_range_high": result.score_range_high,
        "reasoning": reasoning,
    }


@app.get("/matches/today")
async def matches_today() -> dict[str, Any]:
    """Returns today's IPL matches. Falls back to cached predictions if no API key."""
    try:
        from backend.services.cricket import get_todays_matches
        matches = await get_todays_matches()
        return {"matches": matches, "source": "live"}
    except RuntimeError:
        # No API key — return empty
        return {"matches": [], "source": "unavailable", "message": "Set CRICKET_API_KEY env var"}


@app.get("/matches/upcoming")
async def matches_upcoming() -> dict[str, Any]:
    """Returns upcoming IPL matches (next 7 days)."""
    try:
        from backend.services.cricket import get_upcoming_matches
        matches = await get_upcoming_matches(days=7)
        return {"matches": matches, "source": "live"}
    except RuntimeError:
        return {"matches": [], "source": "unavailable", "message": "Set CRICKET_API_KEY env var"}


@app.get("/matches/upcoming/with-predictions")
async def matches_upcoming_with_predictions() -> dict[str, Any]:
    """Returns upcoming IPL matches with pre-computed predictions and reasoning.

    Uses sample fixture data when no live cricket API key is configured.
    Odds are generated from model probabilities with a ~5% bookmaker margin.
    """
    SAMPLE_FIXTURES = [
        {
            "match_id": "sample-001",
            "team1": "Mumbai Indians",
            "team2": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
            "match_time": "19:30",
        },
        {
            "match_id": "sample-002",
            "team1": "Royal Challengers Bangalore",
            "team2": "Kolkata Knight Riders",
            "venue": "M Chinnaswamy Stadium",
            "match_time": "15:30",
        },
        {
            "match_id": "sample-003",
            "team1": "Rajasthan Royals",
            "team2": "Sunrisers Hyderabad",
            "venue": "Sawai Mansingh Stadium",
            "match_time": "19:30",
        },
        {
            "match_id": "sample-004",
            "team1": "Delhi Capitals",
            "team2": "Punjab Kings",
            "venue": "Arun Jaitley Stadium",
            "match_time": "19:30",
        },
        {
            "match_id": "sample-005",
            "team1": "Gujarat Titans",
            "team2": "Lucknow Super Giants",
            "venue": "Narendra Modi Stadium",
            "match_time": "19:30",
        },
    ]

    # Try live cricket API first; fall back to sample fixtures
    try:
        from backend.services.cricket import get_upcoming_matches
        live = await get_upcoming_matches(days=7)
        fixtures = [
            {
                "match_id": m["match_id"] or f"live-{i}",
                "team1": m["team1"],
                "team2": m["team2"],
                "venue": m["venue"] or "TBC",
                "match_time": m["match_time"] or "TBD",
            }
            for i, m in enumerate(live[:5])
            if m["team1"] and m["team2"]
        ] or SAMPLE_FIXTURES
    except Exception:
        fixtures = SAMPLE_FIXTURES

    results = []

    for fixture in fixtures:
        t1 = fixture["team1"]
        t2 = fixture["team2"]
        venue = fixture["venue"]

        # Build features and predict (requires models to be loaded)
        if feature_builder and predictor and predictor.is_loaded:
            features = feature_builder.build_features_for_new_match(
                team1=t1,
                team2=t2,
                venue=venue,
            )
            result = predictor.predict(features)
            t1_prob = result.t1_win_prob
            t2_prob = result.t2_win_prob
            confidence = result.confidence
            confidence_label = result.confidence_label
        else:
            # Fallback when models are not loaded
            t1_prob = 0.5
            t2_prob = 0.5
            confidence = 0.0
            confidence_label = "Low"
            features = {
                "t1_form_5": 0.5, "t2_form_5": 0.5,
                "t1_form_3": 0.5, "t2_form_3": 0.5,
                "t1_form_10": 0.5, "t2_form_10": 0.5,
                "t1_season_form": 0.5, "t2_season_form": 0.5,
                "h2h_t1_rate": 0.5,
                "t1_venue_rate": 0.5, "t2_venue_rate": 0.5,
                "toss_venue_adv": 0.5,
                "t1_has_toss": 0.0, "toss_bat_1st": 0.0,
                "form_diff": 0.0, "venue_avg_score": 160.0,
                "pitch_type": 1.0, "elo_t1_adv": 0.0, "home_adv": 0.0,
            }

            from types import SimpleNamespace
            result = SimpleNamespace(
                t1_win_prob=t1_prob,
                t2_win_prob=t2_prob,
                confidence=confidence,
                confidence_label=confidence_label,
            )

        # Score range estimates
        score_range_t1 = feature_builder.estimate_score_range(t1, venue) if feature_builder else (150, 178)
        score_range_t2 = feature_builder.estimate_score_range(t2, venue) if feature_builder else (150, 178)

        # Generate odds with ~5% bookmaker margin (overround)
        MARGIN = 1.05
        odds_t1 = round(MARGIN / t1_prob, 2) if t1_prob > 0 else 2.0
        odds_t2 = round(MARGIN / t2_prob, 2) if t2_prob > 0 else 2.0

        # Try to merge real bookmaker odds
        try:
            from backend.services.odds import get_live_odds
            live_odds_list = await get_live_odds()
            for lo in live_odds_list:
                if lo["team1"] == t1 and lo["team2"] == t2 and lo["best_odds_t1"] > 0:
                    odds_t1 = lo["best_odds_t1"]
                    odds_t2 = lo["best_odds_t2"]
                    break
        except Exception:
            pass  # keep model-generated odds

        # Determine value bet: AI probability > implied odds probability by 3%+
        implied_t1 = 1.0 / odds_t1 if odds_t1 > 0 else 0.5
        is_value_bet = (t1_prob - implied_t1) > 0.03

        # Extract last 5 form results for each team
        form_t1: list[int] = []
        form_t2: list[int] = []
        if feature_builder:
            for m in reversed(feature_builder.matches):
                if len(form_t1) < 5 and (m.team1 == t1 or m.team2 == t1):
                    form_t1.append(1 if m.winner == t1 else 0)
                if len(form_t2) < 5 and (m.team1 == t2 or m.team2 == t2):
                    form_t2.append(1 if m.winner == t2 else 0)
                if len(form_t1) >= 5 and len(form_t2) >= 5:
                    break
        form_t1 = list(reversed(form_t1))
        form_t2 = list(reversed(form_t2))

        # Generate reasoning
        reasoning = reasoning_engine.generate(
            team1=t1,
            team2=t2,
            venue=venue,
            features=features,
            result=result,
            fb=feature_builder,
            odds_t1=odds_t1,
            odds_t2=odds_t2,
        )

        results.append({
            "match_id": fixture["match_id"],
            "team1": t1,
            "team2": t2,
            "venue": venue,
            "match_time": fixture["match_time"],
            "status": "upcoming",
            "t1_win_prob": t1_prob,
            "t2_win_prob": t2_prob,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "odds_t1": odds_t1,
            "odds_t2": odds_t2,
            "odds_book": "Sample",
            "is_value_bet": is_value_bet,
            "form_t1": form_t1,
            "form_t2": form_t2,
            "score_range_t1": score_range_t1,
            "score_range_t2": score_range_t2,
            "reasoning": reasoning,
        })

    return {"matches": results}


@app.post("/predictions/save")
async def save_prediction(req: dict, authorization: str = Header(default="")) -> dict:
    """Save a prediction the user has viewed."""
    user = _get_current_user(authorization)
    if not user:
        raise HTTPException(401, "Login required to save predictions")
    uid = user["id"]
    preds = _saved_predictions.setdefault(uid, [])
    # Avoid duplicates by match_id
    mid = req.get("match_id", "")
    if not any(p.get("match_id") == mid for p in preds):
        preds.insert(0, {**req, "saved_at": datetime.now(timezone.utc).isoformat()})
        if len(preds) > 50:
            preds.pop()
    return {"saved": True}


@app.get("/predictions/mine")
async def my_predictions(authorization: str = Header(default="")) -> dict:
    """Get this user's saved predictions."""
    user = _get_current_user(authorization)
    if not user:
        raise HTTPException(401, "Login required")
    return {"predictions": _saved_predictions.get(user["id"], [])}


@app.get("/teams/{team_name}/lineup")
async def team_lineup(team_name: str, season: int = 2025) -> dict[str, Any]:
    """Return the top players for a team in a given season from player_season_stats.csv.

    Returns up to 11 players sorted by total contribution (runs + wickets proxy).
    Each player includes batting and bowling stats.
    """
    if not feature_builder or not feature_builder._player_stats.available:
        return {"team": team_name, "season": season, "players": [], "source": "unavailable"}

    ps = feature_builder._player_stats
    # Try requested season, fall back 1-2 seasons
    rows: list[dict] = []
    for offset in (0, 1, 2):
        rows = ps._get_team_rows(team_name, season - offset)
        if rows:
            actual_season = season - offset
            break

    if not rows:
        return {"team": team_name, "season": season, "players": [], "source": "no_data"}

    def _score(r: dict) -> float:
        """Composite score: batting runs + wickets*15 (rough all-round value)."""
        runs = float(r.get("bat_runs") or 0)
        wkts = float(r.get("bowl_wickets") or 0)
        return runs + wkts * 15

    # Sort by composite score, take top 11
    players = sorted(rows, key=_score, reverse=True)[:11]

    def _role(r: dict) -> str:
        bat_runs = float(r.get("bat_runs") or 0)
        bowl_wkts = float(r.get("bowl_wickets") or 0)
        bat_balls = float(r.get("bat_balls") or 0)
        bowl_balls = float(r.get("bowl_balls") or 0)
        if bat_balls > 0 and bowl_balls > 0 and bat_runs > 100 and bowl_wkts > 5:
            return "AR"
        if bat_balls > 0 and bat_runs > 50:
            return "BAT"
        if bowl_balls > 0 and bowl_wkts > 0:
            return "BOWL"
        return "BAT"

    out = []
    for r in players:
        bat_sr = float(r.get("bat_sr") or 0)
        bat_avg = float(r.get("bat_avg") or 0)
        bowl_eco = float(r.get("bowl_eco") or 0)
        bowl_avg = float(r.get("bowl_avg") or 0)
        bat_runs = int(float(r.get("bat_runs") or 0))
        bowl_wkts = int(float(r.get("bowl_wickets") or 0))

        out.append({
            "name": r["player"],
            "role": _role(r),
            "bat_runs": bat_runs,
            "bat_avg": round(bat_avg, 1),
            "bat_sr": round(bat_sr, 1),
            "bowl_wickets": bowl_wkts,
            "bowl_eco": round(bowl_eco, 2),
            "bowl_avg": round(bowl_avg, 1),
        })

    return {
        "team": team_name,
        "season": actual_season,
        "players": out,
        "source": "player_stats",
    }


@app.get("/matches/{match_id}/players")
async def match_players(match_id: str) -> dict[str, Any]:
    """Return key players for both teams in a given match.

    Uses cricapi.com if CRICKET_API_KEY is set, otherwise returns curated
    static squad data for the 10 current IPL franchises.
    """
    # Static fallback squads (2024-25 season)
    STATIC_SQUADS: dict[str, list[dict[str, str]]] = {
        "Mumbai Indians": [
            {"name": "Rohit Sharma", "role": "BAT"},
            {"name": "Ishan Kishan", "role": "WK"},
            {"name": "Suryakumar Yadav", "role": "BAT"},
            {"name": "Tilak Varma", "role": "BAT"},
            {"name": "Hardik Pandya", "role": "AR", "captain": "1"},
            {"name": "Jasprit Bumrah", "role": "BOWL"},
        ],
        "Chennai Super Kings": [
            {"name": "Ruturaj Gaikwad", "role": "BAT", "captain": "1"},
            {"name": "Devon Conway", "role": "WK"},
            {"name": "Shivam Dube", "role": "AR"},
            {"name": "Ravindra Jadeja", "role": "AR"},
            {"name": "MS Dhoni", "role": "WK"},
            {"name": "Deepak Chahar", "role": "BOWL"},
        ],
        "Royal Challengers Bangalore": [
            {"name": "Virat Kohli", "role": "BAT"},
            {"name": "Faf du Plessis", "role": "BAT", "captain": "1"},
            {"name": "Glenn Maxwell", "role": "AR"},
            {"name": "Dinesh Karthik", "role": "WK"},
            {"name": "Mohammed Siraj", "role": "BOWL"},
            {"name": "Josh Hazlewood", "role": "BOWL"},
        ],
        "Kolkata Knight Riders": [
            {"name": "Shreyas Iyer", "role": "BAT", "captain": "1"},
            {"name": "Nitish Rana", "role": "BAT"},
            {"name": "Rinku Singh", "role": "BAT"},
            {"name": "Andre Russell", "role": "AR"},
            {"name": "Sunil Narine", "role": "AR"},
            {"name": "Varun Chakravarthy", "role": "BOWL"},
        ],
        "Delhi Capitals": [
            {"name": "David Warner", "role": "BAT"},
            {"name": "Prithvi Shaw", "role": "BAT"},
            {"name": "Rishabh Pant", "role": "WK", "captain": "1"},
            {"name": "Axar Patel", "role": "AR"},
            {"name": "Anrich Nortje", "role": "BOWL"},
            {"name": "Kuldeep Yadav", "role": "BOWL"},
        ],
        "Rajasthan Royals": [
            {"name": "Jos Buttler", "role": "WK"},
            {"name": "Yashasvi Jaiswal", "role": "BAT"},
            {"name": "Sanju Samson", "role": "WK", "captain": "1"},
            {"name": "Shimron Hetmyer", "role": "BAT"},
            {"name": "Ravichandran Ashwin", "role": "AR"},
            {"name": "Trent Boult", "role": "BOWL"},
        ],
        "Sunrisers Hyderabad": [
            {"name": "Aiden Markram", "role": "BAT", "captain": "1"},
            {"name": "Harry Brook", "role": "BAT"},
            {"name": "Heinrich Klaasen", "role": "WK"},
            {"name": "Pat Cummins", "role": "AR"},
            {"name": "Bhuvneshwar Kumar", "role": "BOWL"},
            {"name": "T Natarajan", "role": "BOWL"},
        ],
        "Punjab Kings": [
            {"name": "Shikhar Dhawan", "role": "BAT", "captain": "1"},
            {"name": "Jonny Bairstow", "role": "WK"},
            {"name": "Liam Livingstone", "role": "AR"},
            {"name": "Sam Curran", "role": "AR"},
            {"name": "Arshdeep Singh", "role": "BOWL"},
            {"name": "Kagiso Rabada", "role": "BOWL"},
        ],
        "Gujarat Titans": [
            {"name": "Shubman Gill", "role": "BAT", "captain": "1"},
            {"name": "David Miller", "role": "BAT"},
            {"name": "Matthew Wade", "role": "WK"},
            {"name": "Rashid Khan", "role": "AR"},
            {"name": "Mohammad Shami", "role": "BOWL"},
            {"name": "Alzarri Joseph", "role": "BOWL"},
        ],
        "Lucknow Super Giants": [
            {"name": "KL Rahul", "role": "WK", "captain": "1"},
            {"name": "Quinton de Kock", "role": "WK"},
            {"name": "Marcus Stoinis", "role": "AR"},
            {"name": "Nicholas Pooran", "role": "WK"},
            {"name": "Ravi Bishnoi", "role": "BOWL"},
            {"name": "Mark Wood", "role": "BOWL"},
        ],
    }

    # Try live API first
    api_key = os.environ.get("CRICKET_API_KEY", "")
    if api_key:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.cricapi.com/v1/match_squad",
                    params={"apikey": api_key, "id": match_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "success" and data.get("data"):
                        return {"match_id": match_id, "source": "live", "data": data["data"]}
        except Exception as e:
            logger.warning("Failed to fetch live squad: %s", e)

    # Fallback: return static squad data for both teams (look up by match ID in feature builder)
    if feature_builder:
        # Try to find the match in our historical data
        for m in feature_builder.matches:
            if str(m.id) == match_id:
                t1_squad = STATIC_SQUADS.get(m.team1, [])
                t2_squad = STATIC_SQUADS.get(m.team2, [])
                return {
                    "match_id": match_id,
                    "source": "static",
                    "team1": {"name": m.team1, "players": t1_squad},
                    "team2": {"name": m.team2, "players": t2_squad},
                }

    # Generic fallback
    return {
        "match_id": match_id,
        "source": "static",
        "team1": {"name": "Team 1", "players": []},
        "team2": {"name": "Team 2", "players": []},
    }


@app.get("/odds/live")
async def odds_live(authorization: str = Header(default="")) -> dict[str, Any]:
    """Fetch live bookmaker odds. Requires Pro/Expert plan."""
    user = _get_current_user(authorization)
    if not user or user.get("plan") not in ("pro", "expert", "trial"):
        raise HTTPException(403, "Pro plan required for live odds")

    try:
        from backend.services.odds import get_live_odds
        odds = await get_live_odds()
        return {"odds": odds}
    except RuntimeError as e:
        return {"odds": [], "message": str(e)}


@app.post("/auth/signup")
async def signup(req: SignupRequest) -> dict[str, Any]:
    """Create a new user account. Starts 7-day free trial."""
    if not req.email or not req.password or len(req.password) < 6:
        raise HTTPException(400, "Email and password (min 6 chars) required")

    pw_hash = hash_password(req.password)

    # Try Supabase if configured, otherwise use in-memory for dev
    try:
        from backend.db.supabase import create_user
        user = await create_user(req.email, req.name, pw_hash)
        user_id = user.get("id", "")
    except RuntimeError:
        # No Supabase — dev mode
        import uuid
        user_id = str(uuid.uuid4())
        user = {"id": user_id, "email": req.email, "name": req.name, "plan": "trial"}
        logger.info("Dev mode signup: %s", req.email)
    except ValueError as e:
        raise HTTPException(409, str(e))

    token = create_jwt(user_id, req.email, "trial")

    # Try sending welcome email (non-blocking)
    try:
        from backend.services.email import send_welcome_email
        await send_welcome_email(req.email, req.name)
    except Exception as e:
        logger.warning("Failed to send welcome email: %s", e)

    return {
        "token": token,
        "user": {"id": user_id, "email": req.email, "name": req.name, "plan": "trial"},
    }


@app.post("/auth/login")
async def login(req: LoginRequest) -> dict[str, Any]:
    """Authenticate user and return JWT."""
    if not req.email or not req.password:
        raise HTTPException(400, "Email and password required")

    try:
        from backend.db.supabase import get_user_by_email
        user = await get_user_by_email(req.email)
    except RuntimeError:
        # Dev mode — accept any login
        import uuid
        token = create_jwt(str(uuid.uuid4()), req.email, "trial")
        return {"token": token, "user": {"email": req.email, "plan": "trial"}}

    if not user:
        raise HTTPException(401, "Invalid email or password")

    if not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(401, "Invalid email or password")

    token = create_jwt(str(user["id"]), user["email"], user.get("plan", "free"))
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name", ""),
            "plan": user.get("plan", "free"),
        },
    }


@app.get("/user/plan")
async def user_plan(authorization: str = Header(default="")) -> dict[str, Any]:
    """Returns current user tier and remaining predictions."""
    user = _get_current_user(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")

    plan = user.get("plan", "free")
    limits = {"free": 2, "trial": 100, "pro": 100, "expert": 1000, "cancelled": 2}

    return {
        "plan": plan,
        "daily_limit": limits.get(plan, 2),
        "email": user.get("email", ""),
    }


@app.post("/train")
async def train_models(authorization: str = Header(default="")) -> dict[str, Any]:
    """Retrain all models. Admin only (check via JWT or simple key)."""
    admin_key = os.environ.get("ADMIN_KEY", "pitchiq-admin-dev")
    user = _get_current_user(authorization)

    # Allow if admin key matches or if it's dev mode
    if authorization != f"Bearer {admin_key}" and (not user or user.get("plan") != "expert"):
        raise HTTPException(403, "Admin access required")

    from backend.models.train import train_and_evaluate
    accuracies = train_and_evaluate()

    # Reload models
    global predictor, feature_builder
    csv_path = Path(__file__).resolve().parents[1] / "data" / "matches.csv"
    feature_builder = FeatureBuilder(csv_path)
    saved_dir = Path(__file__).resolve().parent / "models" / "saved"
    predictor = EnsemblePredictor(saved_dir)
    predictor.load()

    return {"status": "trained", "accuracies": accuracies}


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> dict[str, str]:
    """Handle Stripe webhook events (payment success/failure/cancel)."""
    # In production, verify Stripe signature
    body = await request.json()
    event_type = body.get("type", "")

    logger.info("Stripe webhook: %s", event_type)

    if event_type == "checkout.session.completed":
        customer_email = body.get("data", {}).get("object", {}).get("customer_email", "")
        if customer_email:
            try:
                from backend.db.supabase import get_user_by_email, update_user_plan
                user = await get_user_by_email(customer_email)
                if user:
                    await update_user_plan(str(user["id"]), "pro")
                    logger.info("Upgraded %s to pro", customer_email)
            except Exception as e:
                logger.error("Failed to update plan: %s", e)

    elif event_type in ("customer.subscription.deleted", "invoice.payment_failed"):
        customer_email = body.get("data", {}).get("object", {}).get("customer_email", "")
        if customer_email:
            try:
                from backend.db.supabase import get_user_by_email, update_user_plan
                user = await get_user_by_email(customer_email)
                if user:
                    await update_user_plan(str(user["id"]), "cancelled")
                    logger.info("Downgraded %s to cancelled", customer_email)
            except Exception as e:
                logger.error("Failed to update plan: %s", e)

    return {"status": "received"}


# --- Serve frontend ---
_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if _frontend_dir.exists():
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_index() -> FileResponse:
        return FileResponse(_frontend_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")
