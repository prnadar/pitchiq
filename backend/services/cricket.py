"""CricketData.org API client for live match schedules.

Uses CRICKET_API_KEY env var. Returns normalised match dicts.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

BASE_URL = "https://api.cricapi.com/v1"

# Standard team name mapping for IPL teams
TEAM_ALIASES: dict[str, str] = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "PBKS": "Punjab Kings",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
}


def _normalise_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _get_api_key() -> str:
    key = os.environ.get("CRICKET_API_KEY", "")
    if not key:
        raise RuntimeError("CRICKET_API_KEY env var not set")
    return key


def _parse_match(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a CricketData.org match into normalised format."""
    if not raw.get("matchType", "").lower() in ("t20", "ipl"):
        series = raw.get("series", "") or raw.get("name", "")
        if "IPL" not in series.upper() and "INDIAN PREMIER LEAGUE" not in series.upper():
            return None

    teams = raw.get("teams", []) or raw.get("teamInfo", [])
    if len(teams) < 2:
        return None

    if isinstance(teams[0], dict):
        team1 = _normalise_team(teams[0].get("name", teams[0].get("shortname", "")))
        team2 = _normalise_team(teams[1].get("name", teams[1].get("shortname", "")))
    else:
        team1 = _normalise_team(str(teams[0]))
        team2 = _normalise_team(str(teams[1]))

    date_str = raw.get("date", raw.get("dateTimeGMT", ""))
    match_date = ""
    match_time = ""
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            match_date = dt.strftime("%Y-%m-%d")
            match_time = dt.strftime("%H:%M")
        except (ValueError, TypeError):
            match_date = str(date_str)[:10]

    return {
        "match_id": raw.get("id", ""),
        "team1": team1,
        "team2": team2,
        "venue": raw.get("venue", ""),
        "match_date": match_date,
        "match_time": match_time,
        "status": raw.get("status", ""),
        "match_started": raw.get("matchStarted", False),
        "match_ended": raw.get("matchEnded", False),
    }


async def get_todays_matches() -> list[dict[str, Any]]:
    """Fetch today's IPL matches from CricketData.org."""
    api_key = _get_api_key()
    url = f"{BASE_URL}/currentMatches"
    params = {"apikey": api_key, "offset": 0}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") != "success":
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matches: list[dict[str, Any]] = []
    for raw in data.get("data", []):
        parsed = _parse_match(raw)
        if parsed and parsed["match_date"] == today:
            matches.append(parsed)

    return matches


async def get_upcoming_matches(days: int = 7) -> list[dict[str, Any]]:
    """Fetch upcoming IPL matches (next *days* days) from CricketData.org."""
    api_key = _get_api_key()
    url = f"{BASE_URL}/matches"
    params = {"apikey": api_key, "offset": 0}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") != "success":
        return []

    today = datetime.now(timezone.utc).date()
    matches: list[dict[str, Any]] = []
    for raw in data.get("data", []):
        parsed = _parse_match(raw)
        if not parsed:
            continue
        try:
            md = datetime.strptime(parsed["match_date"], "%Y-%m-%d").date()
            delta = (md - today).days
            if -1 <= delta <= days:
                matches.append(parsed)
        except (ValueError, TypeError):
            continue

    matches.sort(key=lambda m: m["match_date"])
    return matches
