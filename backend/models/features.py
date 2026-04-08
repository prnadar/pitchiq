"""Feature engineering for IPL match prediction.

Builds 23 features from historical match data for the ensemble model.

v3 additions:
- home_advantage     : 1 if team1 is playing at their home venue
- t1_season_form     : team1 win rate in current season (before this match)
- t2_season_form     : team2 win rate in current season
- elo_t1_adv         : seasonal-decay ELO advantage (resets 50% each new season)
- pitch_type         : 0=flat, 1=balanced, 2=spin, 3=seam

v4 additions (player quality from previous season — no future leakage):
- t1_bat_quality     : team1 avg bat SR of top-5 batters (normalised SR/200)
- t2_bat_quality     : team2 avg bat SR of top-5 batters
- t1_bowl_quality    : team1 avg economy of top-4 bowlers (normalised (12-eco)/12)
- t2_bowl_quality    : team2 avg economy of top-4 bowlers
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "matches.csv"
DEFAULT_PLAYER_STATS_CSV = Path(__file__).resolve().parents[2] / "data" / "player_season_stats.csv"

# ---------------------------------------------------------------------------
# ELO settings — partial reset each season to account for squad changes
# ---------------------------------------------------------------------------
ELO_K = 32
ELO_DEFAULT = 1500.0
ELO_SEASON_REVERT = 0.40   # each new season: rating → (rating × 0.60) + (1500 × 0.40)


# ---------------------------------------------------------------------------
# Home venues for each IPL franchise
# ---------------------------------------------------------------------------
HOME_VENUES: dict[str, list[str]] = {
    "Mumbai Indians": ["Wankhede Stadium"],
    "Chennai Super Kings": ["MA Chidambaram Stadium", "Chepauk Stadium"],
    "Royal Challengers Bangalore": ["M Chinnaswamy Stadium", "M. Chinnaswamy Stadium",
                                     "M.Chinnaswamy Stadium"],
    "Royal Challengers Bengaluru": ["M Chinnaswamy Stadium", "M. Chinnaswamy Stadium",
                                     "M.Chinnaswamy Stadium"],
    "Kolkata Knight Riders": ["Eden Gardens"],
    "Delhi Capitals": ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
    "Delhi Daredevils": ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
    "Rajasthan Royals": ["Sawai Mansingh Stadium"],
    "Sunrisers Hyderabad": ["Rajiv Gandhi International Stadium",
                             "Rajiv Gandhi Intl. Cricket Stadium",
                             "Rajiv Gandhi International Cricket Stadium"],
    "Deccan Chargers": ["Rajiv Gandhi International Stadium"],
    "Punjab Kings": ["Punjab Cricket Association IS Bindra Stadium",
                     "PCA Stadium, Mohali"],
    "Kings XI Punjab": ["Punjab Cricket Association IS Bindra Stadium",
                        "PCA Stadium, Mohali"],
    "Gujarat Titans": ["Narendra Modi Stadium", "Narendra Modi Stadium, Ahmedabad"],
    "Gujarat Lions": ["Narendra Modi Stadium", "Sardar Patel Stadium, Motera"],
    "Lucknow Super Giants": ["Ekana Cricket Stadium", "Ekana Cricket Stadium, Lucknow",
                              "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium"],
    "Rising Pune Supergiants": ["Maharashtra Cricket Association Stadium"],
    "Rising Pune Supergiant": ["Maharashtra Cricket Association Stadium"],
    "Kochi Tuskers Kerala": [],
    "Pune Warriors": [],
}


def _is_home(team: str, venue: str) -> float:
    """Return 1.0 if venue is the team's recognised home ground."""
    home_list = HOME_VENUES.get(team, [])
    venue_lower = venue.lower()
    for h in home_list:
        if h.lower() in venue_lower or venue_lower in h.lower():
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Venue → Pitch type mapping (0=flat/batting, 1=balanced, 2=spin, 3=seam/pace)
# ---------------------------------------------------------------------------
PITCH_TYPE_MAP: dict[str, int] = {
    # FLAT / BATTING-FRIENDLY
    "Wankhede Stadium": 0,
    "Wankhede Stadium, Mumbai": 0,
    "M Chinnaswamy Stadium": 0,
    "M. Chinnaswamy Stadium": 0,
    "M.Chinnaswamy Stadium": 0,
    "Eden Gardens": 0,
    "Narendra Modi Stadium": 0,
    "Narendra Modi Stadium, Ahmedabad": 0,
    "Sardar Patel Stadium, Motera": 0,
    "Arun Jaitley Stadium": 0,
    "Feroz Shah Kotla": 0,
    "Brabourne Stadium": 0,
    "Brabourne Stadium, Mumbai": 0,
    "Dr DY Patil Sports Academy": 0,
    "Dr DY Patil Sports Academy, Mumbai": 0,
    # BALANCED
    "Rajiv Gandhi International Stadium": 1,
    "Rajiv Gandhi Intl. Cricket Stadium": 1,
    "Rajiv Gandhi International Cricket Stadium": 1,
    "Sawai Mansingh Stadium": 1,
    "Maharashtra Cricket Association Stadium": 1,
    "Maharashtra Cricket Association Stadium, Pune": 1,
    "JSCA International Stadium Complex": 1,
    "Barsapara Cricket Stadium": 1,
    "Ekana Cricket Stadium": 1,
    "Ekana Cricket Stadium, Lucknow": 1,
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": 1,
    # SPIN-FRIENDLY
    "MA Chidambaram Stadium": 2,
    "MA Chidambaram Stadium, Chepauk": 2,
    "Chepauk Stadium": 2,
    "Holkar Cricket Stadium": 2,
    "Holkar Cricket Stadium, Indore": 2,
    "Barabati Stadium": 2,
    "Subrata Roy Sahara Stadium": 2,
    # SEAM / PACE
    "Punjab Cricket Association IS Bindra Stadium": 3,
    "PCA Stadium, Mohali": 3,
    "Punjab Cricket Association IS Bindra Stadium, Mohali": 3,
    "HPCA Stadium": 3,
    "Himachal Pradesh Cricket Association Stadium": 3,
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": 3,
}

DEFAULT_PITCH_TYPE = 1


def _venue_pitch_type(venue: str) -> int:
    if not venue:
        return DEFAULT_PITCH_TYPE
    if venue in PITCH_TYPE_MAP:
        return PITCH_TYPE_MAP[venue]
    venue_lower = venue.lower()
    for canonical, ptype in PITCH_TYPE_MAP.items():
        if canonical.lower() in venue_lower or venue_lower in canonical.lower():
            return ptype
    return DEFAULT_PITCH_TYPE


class PlayerStatsLoader:
    """Loads player_season_stats.csv and provides team-level quality signals.

    All lookups use *previous-season* data to prevent future leakage.
    """

    def __init__(self, csv_path: Optional[str | Path] = None) -> None:
        self._path = Path(csv_path) if csv_path else DEFAULT_PLAYER_STATS_CSV
        # Nested: season → team → list of row dicts
        self._data: dict[int, dict[str, list[dict]]] = {}
        self._available = False
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    season = int(row["season"])
                    team = row["team"].strip()
                    if not team:
                        continue
                    if season not in self._data:
                        self._data[season] = {}
                    if team not in self._data[season]:
                        self._data[season][team] = []
                    self._data[season][team].append(row)
                except (ValueError, KeyError):
                    continue
        self._available = bool(self._data)

    @property
    def available(self) -> bool:
        return self._available

    def _get_team_rows(self, team: str, season: int) -> list[dict]:
        """Return rows for a team in a given season, empty list if not found."""
        return self._data.get(season, {}).get(team, [])

    def team_batting_quality(self, team: str, before_season: int) -> float:
        """Average bat_sr of top-5 batters by runs in the season before *before_season*.

        Falls back to 2 seasons prior if no data found. Returns 0.5 (normalised) as
        default (equivalent to SR=100, which is normalised to 100/200=0.5).
        """
        if not self._available:
            return 0.5

        for offset in (1, 2):
            lookup_season = before_season - offset
            rows = self._get_team_rows(team, lookup_season)
            if rows:
                # Sort by bat_runs descending, take top 5
                sorted_rows = sorted(
                    rows,
                    key=lambda r: int(r.get("bat_runs", 0) or 0),
                    reverse=True,
                )[:5]
                srs = []
                for r in sorted_rows:
                    try:
                        sr = float(r["bat_sr"])
                        balls = int(r.get("bat_balls", 0) or 0)
                        if balls >= 10:  # only include players with meaningful exposure
                            srs.append(sr)
                    except (ValueError, KeyError):
                        continue
                if srs:
                    avg_sr = sum(srs) / len(srs)
                    return min(1.0, max(0.0, avg_sr / 200.0))

        return 0.5

    def team_bowling_quality(self, team: str, before_season: int) -> float:
        """Average bowl_eco of top-4 bowlers by wickets in the season before *before_season*.

        Falls back to 2 seasons prior if no data found. Returns 0.5 as default
        (equivalent to eco=6, normalised as (12-6)/12=0.5).
        """
        if not self._available:
            return 0.5

        for offset in (1, 2):
            lookup_season = before_season - offset
            rows = self._get_team_rows(team, lookup_season)
            if rows:
                # Sort by bowl_wickets descending, take top 4
                sorted_rows = sorted(
                    rows,
                    key=lambda r: int(r.get("bowl_wickets", 0) or 0),
                    reverse=True,
                )[:4]
                ecos = []
                for r in sorted_rows:
                    try:
                        eco = float(r["bowl_eco"])
                        balls = int(r.get("bowl_balls", 0) or 0)
                        if balls >= 12:  # at least 2 overs bowled
                            ecos.append(eco)
                    except (ValueError, KeyError):
                        continue
                if ecos:
                    avg_eco = sum(ecos) / len(ecos)
                    normalised = (12.0 - avg_eco) / 12.0
                    return min(1.0, max(0.0, normalised))

        return 0.5


@dataclass(frozen=True)
class MatchRow:
    id: int
    season: int
    date: str
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str
    winner: str
    result_type: str
    win_margin: int
    team1_score: int
    team2_score: int


class FeatureBuilder:
    """Loads matches.csv and builds the 23 prediction features."""

    def __init__(
        self,
        csv_path: Optional[str | Path] = None,
        player_stats_path: Optional[str | Path] = None,
    ) -> None:
        self._csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
        self._matches: list[MatchRow] = []
        self._player_stats = PlayerStatsLoader(player_stats_path)
        self._load()

    def _load(self) -> None:
        with open(self._csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self._matches.append(
                        MatchRow(
                            id=int(row["id"]),
                            season=int(row["season"]),
                            date=row["date"],
                            team1=row["team1"],
                            team2=row["team2"],
                            venue=row["venue"],
                            toss_winner=row["toss_winner"],
                            toss_decision=row["toss_decision"],
                            winner=row["winner"],
                            result_type=row["result_type"],
                            win_margin=int(row.get("win_margin") or 0),
                            team1_score=int(row.get("team1_score") or 0),
                            team2_score=int(row.get("team2_score") or 0),
                        )
                    )
                except (ValueError, KeyError):
                    continue

    @property
    def matches(self) -> list[MatchRow]:
        return list(self._matches)

    # ------------------------------------------------------------------
    # ELO with seasonal decay
    # ------------------------------------------------------------------

    def _build_elo_up_to(self, before_index: int) -> dict[str, float]:
        """Compute ELO ratings from matches[0:before_index] with season decay."""
        ratings: dict[str, float] = {}
        prev_season: Optional[int] = None

        for m in self._matches[:before_index]:
            # Apply seasonal decay on season boundary
            if prev_season is not None and m.season != prev_season:
                for team in list(ratings.keys()):
                    ratings[team] = (
                        ratings[team] * (1 - ELO_SEASON_REVERT)
                        + ELO_DEFAULT * ELO_SEASON_REVERT
                    )
            prev_season = m.season

            r1 = ratings.get(m.team1, ELO_DEFAULT)
            r2 = ratings.get(m.team2, ELO_DEFAULT)
            expected1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))

            if m.winner == m.team1:
                ratings[m.team1] = r1 + ELO_K * (1 - expected1)
                ratings[m.team2] = r2 + ELO_K * (0 - (1 - expected1))
            else:
                ratings[m.team1] = r1 + ELO_K * (0 - expected1)
                ratings[m.team2] = r2 + ELO_K * (1 - expected1)

        return ratings

    def _elo_advantage(self, team1: str, team2: str, before_index: int) -> float:
        """Normalised ELO rating difference (team1 − team2) / 400."""
        ratings = self._build_elo_up_to(before_index)
        r1 = ratings.get(team1, ELO_DEFAULT)
        r2 = ratings.get(team2, ELO_DEFAULT)
        return (r1 - r2) / 400.0

    # ------------------------------------------------------------------
    # Form helpers
    # ------------------------------------------------------------------

    def _team_form(self, team: str, before_index: int, window: int) -> float:
        relevant: list[bool] = []
        for m in reversed(self._matches[:before_index]):
            if m.team1 == team or m.team2 == team:
                relevant.append(m.winner == team)
                if len(relevant) == window:
                    break
        return sum(relevant) / len(relevant) if relevant else 0.5

    def _season_form(self, team: str, season: int, before_index: int) -> float:
        """Win rate in the *current* season so far (before this match)."""
        wins = total = 0
        for m in self._matches[:before_index]:
            if m.season == season and (m.team1 == team or m.team2 == team):
                total += 1
                if m.winner == team:
                    wins += 1
        return wins / total if total > 0 else 0.5

    def _h2h_rate(self, t1: str, t2: str, before_index: int) -> float:
        wins = total = 0
        for m in self._matches[:before_index]:
            if {m.team1, m.team2} == {t1, t2}:
                total += 1
                if m.winner == t1:
                    wins += 1
        return wins / total if total > 0 else 0.5

    def _venue_rate(self, team: str, venue: str, before_index: int) -> float:
        wins = total = 0
        for m in self._matches[:before_index]:
            if m.venue == venue and (m.team1 == team or m.team2 == team):
                total += 1
                if m.winner == team:
                    wins += 1
        return wins / total if total > 0 else 0.5

    def _toss_venue_advantage(self, venue: str, before_index: int) -> float:
        wins = total = 0
        for m in self._matches[:before_index]:
            if m.venue == venue:
                total += 1
                if m.toss_winner == m.winner:
                    wins += 1
        return wins / total if total > 0 else 0.5

    def _venue_avg_score(self, venue: str, before_index: int) -> float:
        scores = [m.team1_score for m in self._matches[:before_index]
                  if m.venue == venue and m.team1_score > 0]
        return float(np.mean(scores)) if scores else 160.0

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def build_features_for_match(self, index: int) -> dict[str, float]:
        """Build the 23-feature dict for the match at *index* using only prior data."""
        m = self._matches[index]

        t1_form_5 = self._team_form(m.team1, index, 5)
        t2_form_5 = self._team_form(m.team2, index, 5)
        t1_form_3 = self._team_form(m.team1, index, 3)
        t2_form_3 = self._team_form(m.team2, index, 3)
        t1_form_10 = self._team_form(m.team1, index, 10)
        t2_form_10 = self._team_form(m.team2, index, 10)
        t1_season_form = self._season_form(m.team1, m.season, index)
        t2_season_form = self._season_form(m.team2, m.season, index)
        h2h_t1_rate = self._h2h_rate(m.team1, m.team2, index)
        t1_venue_rate = self._venue_rate(m.team1, m.venue, index)
        t2_venue_rate = self._venue_rate(m.team2, m.venue, index)
        toss_venue_adv = self._toss_venue_advantage(m.venue, index)
        t1_has_toss = 1.0 if m.toss_winner == m.team1 else 0.0
        toss_bat_1st = 1.0 if (m.toss_winner == m.team1 and m.toss_decision == "bat") else 0.0
        form_diff = t1_form_5 - t2_form_5
        venue_avg_score = self._venue_avg_score(m.venue, index)
        pitch_type = float(_venue_pitch_type(m.venue))
        elo_t1_adv = self._elo_advantage(m.team1, m.team2, index)
        home_adv = _is_home(m.team1, m.venue) - _is_home(m.team2, m.venue)

        # Player quality features (previous season only — no leakage)
        t1_bat_quality = self._player_stats.team_batting_quality(m.team1, m.season)
        t2_bat_quality = self._player_stats.team_batting_quality(m.team2, m.season)
        t1_bowl_quality = self._player_stats.team_bowling_quality(m.team1, m.season)
        t2_bowl_quality = self._player_stats.team_bowling_quality(m.team2, m.season)

        return {
            "t1_form_5": t1_form_5,
            "t2_form_5": t2_form_5,
            "t1_form_3": t1_form_3,
            "t2_form_3": t2_form_3,
            "t1_form_10": t1_form_10,
            "t2_form_10": t2_form_10,
            "t1_season_form": t1_season_form,
            "t2_season_form": t2_season_form,
            "h2h_t1_rate": h2h_t1_rate,
            "t1_venue_rate": t1_venue_rate,
            "t2_venue_rate": t2_venue_rate,
            "toss_venue_adv": toss_venue_adv,
            "t1_has_toss": t1_has_toss,
            "toss_bat_1st": toss_bat_1st,
            "form_diff": form_diff,
            "venue_avg_score": venue_avg_score,
            "pitch_type": pitch_type,
            "elo_t1_adv": elo_t1_adv,
            "home_adv": home_adv,
            "t1_bat_quality": t1_bat_quality,
            "t2_bat_quality": t2_bat_quality,
            "t1_bowl_quality": t1_bowl_quality,
            "t2_bowl_quality": t2_bowl_quality,
        }

    def build_features_for_new_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: str = "",
        toss_decision: str = "",
        is_day_night: bool = True,
        season: Optional[int] = None,
    ) -> dict[str, float]:
        """Build features for a future match using all available historical data."""
        n = len(self._matches)
        if season is None:
            season = self._matches[-1].season if self._matches else 2025

        t1_form_5 = self._team_form(team1, n, 5)
        t2_form_5 = self._team_form(team2, n, 5)
        t1_form_3 = self._team_form(team1, n, 3)
        t2_form_3 = self._team_form(team2, n, 3)
        t1_form_10 = self._team_form(team1, n, 10)
        t2_form_10 = self._team_form(team2, n, 10)
        t1_season_form = self._season_form(team1, season, n)
        t2_season_form = self._season_form(team2, season, n)
        h2h_t1_rate = self._h2h_rate(team1, team2, n)
        t1_venue_rate = self._venue_rate(team1, venue, n)
        t2_venue_rate = self._venue_rate(team2, venue, n)
        toss_venue_adv = self._toss_venue_advantage(venue, n)
        t1_has_toss = 1.0 if toss_winner == team1 else 0.0
        toss_bat_1st = 1.0 if (toss_winner == team1 and toss_decision == "bat") else 0.0
        form_diff = t1_form_5 - t2_form_5
        venue_avg_score = self._venue_avg_score(venue, n)
        pitch_type = float(_venue_pitch_type(venue))
        elo_t1_adv = self._elo_advantage(team1, team2, n)
        home_adv = _is_home(team1, venue) - _is_home(team2, venue)

        # Player quality features (previous season only — no leakage)
        t1_bat_quality = self._player_stats.team_batting_quality(team1, season)
        t2_bat_quality = self._player_stats.team_batting_quality(team2, season)
        t1_bowl_quality = self._player_stats.team_bowling_quality(team1, season)
        t2_bowl_quality = self._player_stats.team_bowling_quality(team2, season)

        return {
            "t1_form_5": t1_form_5,
            "t2_form_5": t2_form_5,
            "t1_form_3": t1_form_3,
            "t2_form_3": t2_form_3,
            "t1_form_10": t1_form_10,
            "t2_form_10": t2_form_10,
            "t1_season_form": t1_season_form,
            "t2_season_form": t2_season_form,
            "h2h_t1_rate": h2h_t1_rate,
            "t1_venue_rate": t1_venue_rate,
            "t2_venue_rate": t2_venue_rate,
            "toss_venue_adv": toss_venue_adv,
            "t1_has_toss": t1_has_toss,
            "toss_bat_1st": toss_bat_1st,
            "form_diff": form_diff,
            "venue_avg_score": venue_avg_score,
            "pitch_type": pitch_type,
            "elo_t1_adv": elo_t1_adv,
            "home_adv": home_adv,
            "t1_bat_quality": t1_bat_quality,
            "t2_bat_quality": t2_bat_quality,
            "t1_bowl_quality": t1_bowl_quality,
            "t2_bowl_quality": t2_bowl_quality,
        }

    def estimate_score_range(self, team: str, venue: str) -> tuple[int, int]:
        """Estimate expected score range for a team batting at a venue."""
        n = len(self._matches)
        base = self._venue_avg_score(venue, n) or 160.0
        # Adjust by team batting quality (uses player stats if available)
        qual = 1.0
        if self._player_stats and self._player_stats.available:
            current_season = max(m.season for m in self.matches if m.season) if self.matches else 2025
            q = self._player_stats.team_batting_quality(team, current_season)
            qual = 0.85 + q * 0.30  # maps [0,1] -> [0.85, 1.15]
        est = base * qual
        return int(est - 14), int(est + 14)

    def build_dataset(self, min_index: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Build X, y arrays for training. Skips first *min_index* matches."""
        feature_names = list(self.build_features_for_match(min_index).keys())
        X_rows: list[list[float]] = []
        y_rows: list[int] = []

        for i in range(min_index, len(self._matches)):
            feats = self.build_features_for_match(i)
            X_rows.append([feats[k] for k in feature_names])
            y_rows.append(1 if self._matches[i].winner == self._matches[i].team1 else 0)

        return np.array(X_rows), np.array(y_rows)

    @property
    def feature_names(self) -> list[str]:
        return [
            "t1_form_5", "t2_form_5", "t1_form_3", "t2_form_3",
            "t1_form_10", "t2_form_10",
            "t1_season_form", "t2_season_form",
            "h2h_t1_rate", "t1_venue_rate", "t2_venue_rate",
            "toss_venue_adv", "t1_has_toss", "toss_bat_1st",
            "form_diff", "venue_avg_score",
            "pitch_type", "elo_t1_adv", "home_adv",
            "t1_bat_quality", "t2_bat_quality",
            "t1_bowl_quality", "t2_bowl_quality",
        ]
