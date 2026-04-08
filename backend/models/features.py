"""Feature engineering for IPL match prediction.

Builds 15 features from historical match data for the ensemble model.

v2 additions:
- pitch_type  : 0=flat/batting, 1=balanced, 2=spin-friendly, 3=seam/pace
- is_day_night: IPL is almost exclusively day/night; kept as a feature flag (0/1)
- elo_t1_adv  : ELO rating difference (team1 rating - team2 rating) before match
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "matches.csv"

# ---------------------------------------------------------------------------
# ELO settings
# ---------------------------------------------------------------------------
ELO_K = 32
ELO_DEFAULT = 1500.0


# ---------------------------------------------------------------------------
# Venue → Pitch type mapping (0=flat, 1=balanced, 2=spin, 3=seam/pace)
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

DEFAULT_PITCH_TYPE = 1  # balanced when unknown


def _venue_pitch_type(venue: str) -> int:
    """Look up pitch type for a venue; try partial key matches."""
    if not venue:
        return DEFAULT_PITCH_TYPE
    if venue in PITCH_TYPE_MAP:
        return PITCH_TYPE_MAP[venue]
    venue_lower = venue.lower()
    for canonical, ptype in PITCH_TYPE_MAP.items():
        if canonical.lower() in venue_lower or venue_lower in canonical.lower():
            return ptype
    return DEFAULT_PITCH_TYPE


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
    """Loads matches.csv and builds the 15 prediction features."""

    def __init__(self, csv_path: Optional[str | Path] = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
        self._matches: list[MatchRow] = []
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
    # ELO helpers
    # ------------------------------------------------------------------

    def _build_elo_up_to(self, before_index: int) -> dict[str, float]:
        """Compute ELO ratings for all teams from matches[0:before_index]."""
        ratings: dict[str, float] = {}
        for m in self._matches[:before_index]:
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
        """ELO rating difference (team1 - team2), normalised to [-1, 1]."""
        ratings = self._build_elo_up_to(before_index)
        r1 = ratings.get(team1, ELO_DEFAULT)
        r2 = ratings.get(team2, ELO_DEFAULT)
        # Raw difference; normalise by 400 (one ELO "step") to keep reasonable scale
        return (r1 - r2) / 400.0

    # ------------------------------------------------------------------
    # Standard form / venue helpers
    # ------------------------------------------------------------------

    def _team_form(self, team: str, before_index: int, window: int) -> float:
        relevant: list[bool] = []
        for m in reversed(self._matches[:before_index]):
            if m.team1 == team or m.team2 == team:
                relevant.append(m.winner == team)
                if len(relevant) == window:
                    break
        return sum(relevant) / len(relevant) if relevant else 0.5

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
        """Build the 15-feature dict for the match at *index* using only prior data."""
        m = self._matches[index]

        t1_form_5 = self._team_form(m.team1, index, 5)
        t2_form_5 = self._team_form(m.team2, index, 5)
        t1_form_3 = self._team_form(m.team1, index, 3)
        t2_form_3 = self._team_form(m.team2, index, 3)
        h2h_t1_rate = self._h2h_rate(m.team1, m.team2, index)
        t1_venue_rate = self._venue_rate(m.team1, m.venue, index)
        t2_venue_rate = self._venue_rate(m.team2, m.venue, index)
        toss_venue_adv = self._toss_venue_advantage(m.venue, index)
        t1_has_toss = 1.0 if m.toss_winner == m.team1 else 0.0
        toss_bat_1st = 1.0 if (m.toss_winner == m.team1 and m.toss_decision == "bat") else 0.0
        form_diff = t1_form_5 - t2_form_5
        venue_avg_score = self._venue_avg_score(m.venue, index)
        pitch_type = float(_venue_pitch_type(m.venue))
        is_day_night = 1.0
        elo_t1_adv = self._elo_advantage(m.team1, m.team2, index)

        return {
            "t1_form_5": t1_form_5,
            "t2_form_5": t2_form_5,
            "t1_form_3": t1_form_3,
            "t2_form_3": t2_form_3,
            "h2h_t1_rate": h2h_t1_rate,
            "t1_venue_rate": t1_venue_rate,
            "t2_venue_rate": t2_venue_rate,
            "toss_venue_adv": toss_venue_adv,
            "t1_has_toss": t1_has_toss,
            "toss_bat_1st": toss_bat_1st,
            "form_diff": form_diff,
            "venue_avg_score": venue_avg_score,
            "pitch_type": pitch_type,
            "is_day_night": is_day_night,
            "elo_t1_adv": elo_t1_adv,
        }

    def build_features_for_new_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: str = "",
        toss_decision: str = "",
        is_day_night: bool = True,
    ) -> dict[str, float]:
        """Build features for a future match using all available historical data."""
        n = len(self._matches)

        t1_form_5 = self._team_form(team1, n, 5)
        t2_form_5 = self._team_form(team2, n, 5)
        t1_form_3 = self._team_form(team1, n, 3)
        t2_form_3 = self._team_form(team2, n, 3)
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

        return {
            "t1_form_5": t1_form_5,
            "t2_form_5": t2_form_5,
            "t1_form_3": t1_form_3,
            "t2_form_3": t2_form_3,
            "h2h_t1_rate": h2h_t1_rate,
            "t1_venue_rate": t1_venue_rate,
            "t2_venue_rate": t2_venue_rate,
            "toss_venue_adv": toss_venue_adv,
            "t1_has_toss": t1_has_toss,
            "toss_bat_1st": toss_bat_1st,
            "form_diff": form_diff,
            "venue_avg_score": venue_avg_score,
            "pitch_type": pitch_type,
            "is_day_night": 1.0 if is_day_night else 0.0,
            "elo_t1_adv": elo_t1_adv,
        }

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
            "h2h_t1_rate", "t1_venue_rate", "t2_venue_rate",
            "toss_venue_adv", "t1_has_toss", "toss_bat_1st",
            "form_diff", "venue_avg_score",
            "pitch_type", "is_day_night", "elo_t1_adv",
        ]
