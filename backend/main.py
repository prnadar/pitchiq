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
async def predict(req: PredictRequest) -> dict[str, Any]:
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
