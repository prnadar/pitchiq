"""Feature engineering for IPL match prediction.

Builds 12 features from historical match data for the ensemble model.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "matches.csv"


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
    """Loads matches.csv and builds the 12 prediction features."""

    def __init__(self, csv_path: Optional[str | Path] = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
        self._matches: list[MatchRow] = []
        self._load()

    def _load(self) -> None:
        with open(self._csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                        win_margin=int(row["win_margin"]),
                        team1_score=int(row["team1_score"]),
                        team2_score=int(row["team2_score"]),
                    )
                )

    @property
    def matches(self) -> list[MatchRow]:
        return list(self._matches)

    def _team_form(self, team: str, before_index: int, window: int) -> float:
        """Win rate for *team* in the last *window* matches before index."""
        relevant: list[bool] = []
        for m in reversed(self._matches[:before_index]):
            if m.team1 == team or m.team2 == team:
                relevant.append(m.winner == team)
                if len(relevant) == window:
                    break
        if not relevant:
            return 0.5  # prior when no history
        return sum(relevant) / len(relevant)

    def _h2h_rate(self, t1: str, t2: str, before_index: int) -> float:
        """Head-to-head win rate of t1 vs t2 before index."""
        wins = 0
        total = 0
        for m in self._matches[:before_index]:
            if {m.team1, m.team2} == {t1, t2}:
                total += 1
                if m.winner == t1:
                    wins += 1
        return wins / total if total > 0 else 0.5

    def _venue_rate(self, team: str, venue: str, before_index: int) -> float:
        """Team's all-time win % at this venue before index."""
        wins = 0
        total = 0
        for m in self._matches[:before_index]:
            if m.venue == venue and (m.team1 == team or m.team2 == team):
                total += 1
                if m.winner == team:
                    wins += 1
        return wins / total if total > 0 else 0.5

    def _toss_venue_advantage(self, venue: str, before_index: int) -> float:
        """% of toss winners who won the match at this venue."""
        wins = 0
        total = 0
        for m in self._matches[:before_index]:
            if m.venue == venue:
                total += 1
                if m.toss_winner == m.winner:
                    wins += 1
        return wins / total if total > 0 else 0.5

    def _venue_avg_score(self, venue: str, before_index: int) -> float:
        """Average first-innings score at this venue."""
        scores: list[int] = []
        for m in self._matches[:before_index]:
            if m.venue == venue:
                scores.append(m.team1_score)
        return float(np.mean(scores)) if scores else 160.0  # IPL average default

    def build_features_for_match(self, index: int) -> dict[str, float]:
        """Build the 12-feature dict for the match at *index* using only prior data."""
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
        }

    def build_features_for_new_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: str = "",
        toss_decision: str = "",
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
        }

    def build_dataset(self, min_index: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Build X, y arrays for training. Skips first *min_index* matches (no history)."""
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
        ]
