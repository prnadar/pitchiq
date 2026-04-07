"""The Odds API client for fetching bookmaker odds.

Uses ODDS_API_KEY env var. Free tier: 500 req/month.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "cricket_ipl"
REGIONS = "uk"
MARKETS = "h2h"


def _get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY env var not set")
    return key


def _normalise_team(name: str) -> str:
    """Basic normalisation for odds API team names."""
    aliases: dict[str, str] = {
        "Mumbai Indians": "Mumbai Indians",
        "Chennai Super Kings": "Chennai Super Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bangalore",
        "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
        "Kolkata Knight Riders": "Kolkata Knight Riders",
        "Delhi Capitals": "Delhi Capitals",
        "Rajasthan Royals": "Rajasthan Royals",
        "Sunrisers Hyderabad": "Sunrisers Hyderabad",
        "Punjab Kings": "Punjab Kings",
        "Gujarat Titans": "Gujarat Titans",
        "Lucknow Super Giants": "Lucknow Super Giants",
    }
    return aliases.get(name, name)


async def get_live_odds() -> list[dict[str, Any]]:
    """Fetch live IPL odds from The Odds API."""
    api_key = _get_api_key()
    url = f"{BASE_URL}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "decimal",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        events = resp.json()

    results: list[dict[str, Any]] = []
    for event in events:
        home = _normalise_team(event.get("home_team", ""))
        away = _normalise_team(event.get("away_team", ""))

        best_t1_odds = 0.0
        best_t2_odds = 0.0
        best_bookie_t1 = ""
        best_bookie_t2 = ""
        bookmakers_data: list[dict[str, Any]] = []

        for bookie in event.get("bookmakers", []):
            bookie_name = bookie.get("title", "")
            for market in bookie.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team_name = _normalise_team(outcome.get("name", ""))
                    price = float(outcome.get("price", 0))
                    if team_name == home:
                        bookmakers_data.append({"bookie": bookie_name, "team": home, "odds": price})
                        if price > best_t1_odds:
                            best_t1_odds = price
                            best_bookie_t1 = bookie_name
                    elif team_name == away:
                        bookmakers_data.append({"bookie": bookie_name, "team": away, "odds": price})
                        if price > best_t2_odds:
                            best_t2_odds = price
                            best_bookie_t2 = bookie_name

        results.append({
            "event_id": event.get("id", ""),
            "team1": home,
            "team2": away,
            "commence_time": event.get("commence_time", ""),
            "best_odds_t1": best_t1_odds,
            "best_odds_t2": best_t2_odds,
            "best_bookie_t1": best_bookie_t1,
            "best_bookie_t2": best_bookie_t2,
            "bookmakers": bookmakers_data,
        })

    return results
