"""Download IPL 2008-2024 match data from cricsheet.org.

Parses the CSV2 format (one _info.csv + one deliveries .csv per match)
and writes the full dataset to data/matches.csv in the PitchIQ format:
  id, season, date, team1, team2, venue, toss_winner, toss_decision,
  winner, result_type, win_margin, team1_score, team2_score

Usage:
    python scripts/download_ipl_data.py
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "matches.csv"

# Cricsheet IPL CSV2 download URL
URL = "https://cricsheet.org/downloads/ipl_csv2.zip"


def _parse_season(raw: str) -> int | None:
    """Normalise season strings like '2007/08' -> 2008, '2024' -> 2024."""
    raw = raw.strip()
    try:
        if "/" in raw:
            # e.g. "2007/08"
            parts = raw.split("/")
            base = int(parts[0])
            suffix = parts[1]
            # suffix is 2-digit year: 08 -> 2008
            return base + 1 if len(suffix) == 2 else int(suffix)
        return int(raw)
    except (ValueError, TypeError):
        return None


def _scores_from_deliveries(zf: zipfile.ZipFile, match_file: str, team1: str, team2: str) -> tuple[int, int]:
    """Sum runs from deliveries CSV to get team1_score and team2_score."""
    if match_file not in zf.namelist():
        return 0, 0

    try:
        with zf.open(match_file) as f:
            content = f.read().decode("utf-8", errors="replace")

        innings_runs: dict[int, int] = defaultdict(int)
        innings_team: dict[int, str] = {}

        for row in csv.DictReader(io.StringIO(content)):
            try:
                inn = int(row.get("innings", 0) or 0)
                if inn > 2:
                    continue  # skip super overs
                bat_team = (row.get("batting_team") or "").strip()
                if inn not in innings_team and bat_team:
                    innings_team[inn] = bat_team
                runs = int(row.get("runs_off_bat", 0) or 0)
                extras = int(row.get("extras", 0) or 0)
                innings_runs[inn] += runs + extras
            except (ValueError, TypeError):
                continue

        if 1 not in innings_team:
            return 0, 0

        inn1_team = innings_team[1]
        r1 = innings_runs.get(1, 0)
        r2 = innings_runs.get(2, 0)

        if inn1_team == team1:
            return r1, r2
        else:
            return r2, r1

    except Exception:
        return 0, 0


def download_and_parse() -> list[dict]:
    print(f"Downloading {URL} ...")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "PitchIQ/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            zip_data = response.read()
    except Exception as e:
        print(f"ERROR downloading: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloaded {len(zip_data) / 1024 / 1024:.1f} MB. Parsing ...")

    matches: list[dict] = []

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        info_files = sorted(n for n in zf.namelist() if n.endswith("_info.csv"))
        print(f"Found {len(info_files)} match info files.")

        for info_file in info_files:
            match_id_str = info_file.replace("_info.csv", "")
            delivery_file = match_id_str + ".csv"

            # Parse info file
            try:
                with zf.open(info_file) as f:
                    raw_info = f.read().decode("utf-8", errors="replace")
            except Exception:
                continue

            info: dict[str, str] = {}
            teams: list[str] = []

            for row in csv.reader(io.StringIO(raw_info)):
                if len(row) < 2 or row[0] != "info":
                    continue
                key = row[1].strip()
                val = row[2].strip() if len(row) > 2 else ""
                if key == "team":
                    teams.append(val)
                elif key not in info:
                    info[key] = val

            if len(teams) < 2:
                continue

            # Skip non-result matches
            outcome = info.get("outcome", "normal").lower()
            if outcome in ("no result", "cancelled", "abandoned"):
                continue

            winner = info.get("winner", "").strip()
            if not winner:
                continue

            season = _parse_season(info.get("season", ""))
            if season is None or season < 2008:
                continue

            date = info.get("date", "").strip()
            venue = info.get("venue", "").strip()
            toss_winner = info.get("toss_winner", "").strip()
            toss_decision = info.get("toss_decision", "bat").strip()

            team1, team2 = teams[0], teams[1]

            # Result type + margin
            if outcome == "tie":
                result_type = "tie"
                win_margin = 0
            elif info.get("winner_runs", "").strip():
                result_type = "runs"
                try:
                    win_margin = int(info["winner_runs"])
                except ValueError:
                    win_margin = 0
            elif info.get("winner_wickets", "").strip():
                result_type = "wickets"
                try:
                    win_margin = int(info["winner_wickets"])
                except ValueError:
                    win_margin = 0
            else:
                result_type = "runs"
                win_margin = 0

            # Scores from deliveries
            t1_score, t2_score = _scores_from_deliveries(zf, delivery_file, team1, team2)

            matches.append({
                "season": season,
                "date": date,
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "result_type": result_type,
                "win_margin": win_margin,
                "team1_score": t1_score,
                "team2_score": t2_score,
            })

    return matches


def write_csv(matches: list[dict]) -> None:
    # Sort by date
    matches.sort(key=lambda m: (m["season"], m["date"]))

    # Assign sequential IDs
    for i, m in enumerate(matches, 1):
        m["id"] = i

    FIELDS = [
        "id", "season", "date", "team1", "team2", "venue",
        "toss_winner", "toss_decision", "winner",
        "result_type", "win_margin", "team1_score", "team2_score",
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matches)

    print(f"\nWrote {len(matches)} matches to {OUTPUT_CSV}")


def main() -> None:
    matches = download_and_parse()

    print(f"\nParsed {len(matches)} valid IPL matches")
    seasons = sorted({m["season"] for m in matches})
    print(f"Seasons: {seasons[0]}–{seasons[-1]}")

    write_csv(matches)

    # Season breakdown
    from collections import Counter
    season_counts = Counter(m["season"] for m in matches)
    for s in sorted(season_counts):
        print(f"  {s}: {season_counts[s]} matches")


if __name__ == "__main__":
    main()
