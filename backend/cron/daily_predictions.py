"""Daily predictions cron job.

Fetches today's IPL matches from CricketData API, runs predictions for each,
fetches odds from The Odds API, detects value bets, upserts to predictions_cache,
and queues value_alert emails for pro users.

Usage:
    python backend/cron/daily_predictions.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pitchiq.cron")

VALUE_BET_THRESHOLD = 0.05  # 5% edge


async def run_daily_predictions() -> None:
    """Main cron logic."""
    from backend.models.features import FeatureBuilder
    from backend.models.predict import EnsemblePredictor

    # 1. Load models
    logger.info("Loading feature builder and models...")
    fb = FeatureBuilder()
    predictor = EnsemblePredictor()
    predictor.load()
    logger.info("Models loaded. %d historical matches available.", len(fb.matches))

    # 2. Fetch today's matches
    logger.info("Fetching today's matches from CricketData API...")
    try:
        from backend.services.cricket import get_todays_matches
        matches = await get_todays_matches()
    except RuntimeError as e:
        logger.warning("CricketData API unavailable: %s. Using fallback.", e)
        matches = _fallback_matches()

    if not matches:
        logger.info("No IPL matches today. Exiting.")
        return

    logger.info("Found %d match(es) today.", len(matches))

    # 3. Fetch odds
    odds_map: dict[str, dict] = {}
    try:
        from backend.services.odds import get_live_odds
        odds_list = await get_live_odds()
        for o in odds_list:
            key = f"{o['team1']}|{o['team2']}"
            odds_map[key] = o
            odds_map[f"{o['team2']}|{o['team1']}"] = o
        logger.info("Fetched odds for %d events.", len(odds_list))
    except RuntimeError as e:
        logger.warning("Odds API unavailable: %s. Predictions will skip odds.", e)

    # 4. Run predictions for each match
    predictions: list[dict] = []
    value_bets: list[dict] = []

    for match in matches:
        team1 = match["team1"]
        team2 = match["team2"]
        venue = match["venue"]

        logger.info("Predicting: %s vs %s @ %s", team1, team2, venue)

        features = fb.build_features_for_new_match(
            team1=team1,
            team2=team2,
            venue=venue,
        )
        result = predictor.predict(features)

        # Look up odds
        odds_key = f"{team1}|{team2}"
        odds = odds_map.get(odds_key, {})
        best_odds_t1 = odds.get("best_odds_t1", 0.0)
        best_odds_t2 = odds.get("best_odds_t2", 0.0)
        best_bookie = odds.get("best_bookie_t1", "")

        # Calculate model edge
        implied_prob_t1 = (1.0 / best_odds_t1) if best_odds_t1 > 0 else 0.0
        model_edge = result.t1_win_prob - implied_prob_t1 if implied_prob_t1 > 0 else 0.0
        is_value_bet = model_edge > VALUE_BET_THRESHOLD

        prediction = {
            "match_id": match.get("match_id", f"{team1}-{team2}-{match['match_date']}"),
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "match_date": match["match_date"],
            "t1_win_prob": result.t1_win_prob,
            "t2_win_prob": result.t2_win_prob,
            "confidence": result.confidence,
            "score_range_low": result.score_range_low,
            "score_range_high": result.score_range_high,
            "best_odds_t1": best_odds_t1,
            "best_odds_t2": best_odds_t2,
            "best_bookie": best_bookie,
            "model_edge": round(model_edge, 4),
            "is_value_bet": is_value_bet,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        predictions.append(prediction)

        if is_value_bet:
            value_bets.append(prediction)
            logger.info(
                "  VALUE BET: %s (model: %.1f%%, implied: %.1f%%, edge: +%.1f%%)",
                team1, result.t1_win_prob * 100,
                implied_prob_t1 * 100, model_edge * 100,
            )
        else:
            logger.info(
                "  %s: %.1f%% | %s: %.1f%% | Confidence: %s",
                team1, result.t1_win_prob * 100,
                team2, result.t2_win_prob * 100,
                result.confidence_label,
            )

    # 5. Upsert to predictions_cache (if Supabase configured)
    try:
        from backend.db.supabase import upsert_prediction
        for pred in predictions:
            await upsert_prediction(pred)
        logger.info("Upserted %d prediction(s) to Supabase.", len(predictions))
    except RuntimeError as e:
        logger.warning("Supabase unavailable: %s. Predictions not cached.", e)

    # 6. Send value bet alerts to pro users
    if value_bets:
        logger.info("Found %d value bet(s). Sending alerts to pro users...", len(value_bets))
        try:
            from backend.db.supabase import get_pro_users_for_alerts
            from backend.services.email import send_value_alert

            pro_users = await get_pro_users_for_alerts()
            logger.info("Sending alerts to %d pro user(s).", len(pro_users))

            for user in pro_users:
                try:
                    await send_value_alert(
                        email=user["email"],
                        name=user.get("name", ""),
                        matches=value_bets,
                    )
                    logger.info("  Alert sent to %s", user["email"])
                except Exception as e:
                    logger.error("  Failed to send alert to %s: %s", user["email"], e)
        except RuntimeError as e:
            logger.warning("Cannot send alerts: %s", e)
    else:
        logger.info("No value bets found today. No alerts to send.")

    # Summary
    logger.info("=== Daily Predictions Summary ===")
    logger.info("Matches processed: %d", len(predictions))
    logger.info("Value bets found: %d", len(value_bets))
    for p in predictions:
        vb = " [VALUE]" if p["is_value_bet"] else ""
        logger.info(
            "  %s vs %s: %.0f%%/%.0f%% (edge: %+.1f%%)%s",
            p["team1"], p["team2"],
            p["t1_win_prob"] * 100, p["t2_win_prob"] * 100,
            p["model_edge"] * 100, vb,
        )


def _fallback_matches() -> list[dict]:
    """Return hardcoded sample matches when API is unavailable."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return [
        {
            "match_id": f"mi-csk-{today}",
            "team1": "Mumbai Indians",
            "team2": "Chennai Super Kings",
            "venue": "Wankhede Stadium, Mumbai",
            "match_date": today,
            "match_time": "19:30",
        },
        {
            "match_id": f"kkr-rcb-{today}",
            "team1": "Kolkata Knight Riders",
            "team2": "Royal Challengers Bangalore",
            "venue": "Eden Gardens, Kolkata",
            "match_date": today,
            "match_time": "15:30",
        },
    ]


if __name__ == "__main__":
    asyncio.run(run_daily_predictions())
