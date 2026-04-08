"""Download IPL ball-by-ball data from cricsheet.org and build per-player
per-season batting and bowling statistics.

Output: data/player_season_stats.csv

Columns:
    player, team, season, bat_runs, bat_balls, bat_innings, bat_outs,
    bowl_balls, bowl_runs, bowl_wickets, bowl_innings,
    bat_avg, bat_sr, bowl_eco, bowl_avg

Usage:
    python3 -m scripts.build_player_stats
    # or
    python3 scripts/build_player_stats.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "player_season_stats.csv"
CACHE_ZIP = DATA_DIR / "ipl_csv2.zip"

IPL_ZIP_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"

# Wicket types that count as a dismissed batsman
DISMISSAL_TYPES = {
    "caught", "bowled", "lbw", "run out", "stumped",
    "hit wicket", "obstructing the field", "timed out",
    "handled the ball", "hit the ball twice",
}


@dataclass
class BattingAccumulator:
    runs: int = 0
    balls: int = 0
    innings: int = 0
    outs: int = 0
    in_innings: bool = False  # whether player has batted this innings


@dataclass
class BowlingAccumulator:
    balls: int = 0
    runs: int = 0
    wickets: int = 0
    innings: int = 0
    in_innings: bool = False  # whether player has bowled this innings


def _download_zip() -> None:
    """Download the cricsheet IPL zip to the local cache path."""
    print(f"Downloading {IPL_ZIP_URL} ...")
    req = Request(IPL_ZIP_URL, headers={"User-Agent": "PitchIQ/1.0"})
    try:
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
    except URLError as exc:
        sys.exit(f"ERROR: Failed to download IPL data: {exc}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ZIP.write_bytes(data)
    print(f"Saved {len(data) / 1_048_576:.1f} MB to {CACHE_ZIP}")


def _get_zip_bytes() -> bytes:
    if CACHE_ZIP.exists():
        print(f"Using cached zip: {CACHE_ZIP}")
        return CACHE_ZIP.read_bytes()
    _download_zip()
    return CACHE_ZIP.read_bytes()


def _parse_int(val: str, default: int = 0) -> int:
    try:
        return int(val) if val.strip() != "" else default
    except (ValueError, AttributeError):
        return default


def build_stats() -> None:
    zip_bytes = _get_zip_bytes()

    # Nested dict: [season][player] -> (BattingAccumulator, BowlingAccumulator, team_set)
    # Key: (season, player)
    bat_stats: dict[tuple, BattingAccumulator] = defaultdict(BattingAccumulator)
    bowl_stats: dict[tuple, BowlingAccumulator] = defaultdict(BowlingAccumulator)
    player_teams: dict[tuple, set] = defaultdict(set)

    # Per-match per-innings tracking to count innings
    # We track (match_id, innings, player) to avoid double-counting
    bat_seen: set[tuple] = set()   # (match_id, innings, player) already started batting
    bowl_seen: set[tuple] = set()  # (match_id, innings, bowler) already started bowling

    print("Parsing delivery files...")
    match_count = 0
    row_count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        # Delivery files are named {id}.csv (not {id}_info.csv)
        delivery_files = [n for n in names if n.endswith(".csv") and "_info" not in n]
        delivery_files.sort()
        total = len(delivery_files)

        for file_idx, fname in enumerate(delivery_files):
            if file_idx % 100 == 0:
                print(f"  {file_idx}/{total} files processed ...", end="\r")

            with zf.open(fname) as fh:
                text = fh.read().decode("utf-8", errors="replace")

            reader = csv.DictReader(io.StringIO(text))
            try:
                rows = list(reader)
            except csv.Error:
                continue

            if not rows:
                continue

            match_count += 1

            for row in rows:
                row_count += 1
                try:
                    match_id = row.get("match_id", "").strip()
                    season_raw = row.get("season", "").strip()
                    innings = row.get("innings", "").strip()
                    striker = row.get("striker", "").strip()
                    bowler = row.get("bowler", "").strip()
                    batting_team = row.get("batting_team", "").strip()
                    bowling_team = row.get("bowling_team", "").strip()

                    # Parse season — may be "2007/08" or "2008"
                    if "/" in season_raw:
                        # e.g. "2007/08" → take first year as int + 1 for ending year
                        # IPL started 2008; "2007/08" season = 2008
                        parts = season_raw.split("/")
                        try:
                            season = int(parts[0]) + 1
                        except ValueError:
                            continue
                    else:
                        try:
                            season = int(season_raw)
                        except ValueError:
                            continue

                    if not striker or not bowler or not match_id:
                        continue

                    runs_off_bat = _parse_int(row.get("runs_off_bat", "0"))
                    extras = _parse_int(row.get("extras", "0"))
                    wides = _parse_int(row.get("wides", "0"))
                    noballs = _parse_int(row.get("noballs", "0"))
                    byes = _parse_int(row.get("byes", "0"))
                    legbyes = _parse_int(row.get("legbyes", "0"))

                    wicket_type = row.get("wicket_type", "").strip().lower()
                    player_dismissed = row.get("player_dismissed", "").strip()

                    bat_key = (season, striker)
                    bowl_key = (season, bowler)

                    # Track team associations
                    if batting_team:
                        player_teams[bat_key].add(batting_team)
                    if bowling_team:
                        player_teams[bowl_key].add(bowling_team)

                    # --- BATTING ---
                    bat_inns_key = (match_id, innings, striker)
                    if bat_inns_key not in bat_seen:
                        bat_seen.add(bat_inns_key)
                        bat_stats[bat_key].innings += 1

                    # Balls faced: exclude wides (batsman doesn't face wide)
                    if wides == 0:
                        bat_stats[bat_key].balls += 1
                        bat_stats[bat_key].runs += runs_off_bat

                    # Dismissal: check if this striker was dismissed
                    if player_dismissed == striker and wicket_type in DISMISSAL_TYPES:
                        bat_stats[bat_key].outs += 1

                    # --- BOWLING ---
                    bowl_inns_key = (match_id, innings, bowler)
                    if bowl_inns_key not in bowl_seen:
                        bowl_seen.add(bowl_inns_key)
                        bowl_stats[bowl_key].innings += 1

                    # Legal balls: exclude wides and no-balls
                    if wides == 0 and noballs == 0:
                        bowl_stats[bowl_key].balls += 1

                    # Runs conceded: runs_off_bat + extras - byes - legbyes
                    # (byes and legbyes not charged to bowler)
                    bowl_stats[bowl_key].runs += runs_off_bat + extras - byes - legbyes

                    # Wickets: exclude run outs (not credited to bowler)
                    if (wicket_type and wicket_type != "run out"
                            and wicket_type in DISMISSAL_TYPES
                            and player_dismissed):
                        bowl_stats[bowl_key].wickets += 1

                except (KeyError, ValueError):
                    continue

    print(f"\nParsed {match_count} match files, {row_count:,} delivery rows")

    # Collect all unique (season, player) keys
    all_keys = set(bat_stats.keys()) | set(bowl_stats.keys())
    print(f"Unique (season, player) combos: {len(all_keys)}")

    # Write output CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []

    for key in sorted(all_keys, key=lambda k: (k[0], k[1])):
        season, player = key
        bat = bat_stats.get(key, BattingAccumulator())
        bowl = bowl_stats.get(key, BowlingAccumulator())

        # Determine team (use most frequent; fallback to first)
        teams = player_teams.get(key, set())
        team = sorted(teams)[0] if teams else ""

        # Derived batting stats
        bat_avg = (bat.runs / bat.outs) if bat.outs > 0 else 99.0
        bat_sr = (bat.runs / bat.balls * 100.0) if bat.balls > 0 else 100.0

        # Derived bowling stats
        if bowl.balls >= 6:
            bowl_eco = bowl.runs / (bowl.balls / 6.0)
        else:
            bowl_eco = 8.0
        bowl_avg = (bowl.runs / bowl.wickets) if bowl.wickets > 0 else 30.0

        output_rows.append({
            "player": player,
            "team": team,
            "season": season,
            "bat_runs": bat.runs,
            "bat_balls": bat.balls,
            "bat_innings": bat.innings,
            "bat_outs": bat.outs,
            "bowl_balls": bowl.balls,
            "bowl_runs": bowl.runs,
            "bowl_wickets": bowl.wickets,
            "bowl_innings": bowl.innings,
            "bat_avg": round(bat_avg, 2),
            "bat_sr": round(bat_sr, 2),
            "bowl_eco": round(bowl_eco, 2),
            "bowl_avg": round(bowl_avg, 2),
        })

    fieldnames = [
        "player", "team", "season",
        "bat_runs", "bat_balls", "bat_innings", "bat_outs",
        "bowl_balls", "bowl_runs", "bowl_wickets", "bowl_innings",
        "bat_avg", "bat_sr", "bowl_eco", "bowl_avg",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved {len(output_rows):,} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    build_stats()
