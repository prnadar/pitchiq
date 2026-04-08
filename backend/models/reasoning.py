"""Generates human-readable reasoning for match predictions."""
from __future__ import annotations

from typing import Any, Optional


class ReasoningEngine:
    """Produces structured, human-readable reasoning for match predictions."""

    def generate(
        self,
        team1: str,
        team2: str,
        venue: str,
        features: dict[str, float],
        result: Any,
        fb: Any,
        odds_t1: float = 0.0,
        odds_t2: float = 0.0,
    ) -> dict[str, Any]:
        """Generate reasoning for a prediction.

        Args:
            team1: First team name.
            team2: Second team name.
            venue: Match venue.
            features: Feature dict from FeatureBuilder.
            result: PredictionResult from EnsemblePredictor.
            fb: FeatureBuilder instance (for H2H data).
            odds_t1: Bookmaker decimal odds for team1 (optional).
            odds_t2: Bookmaker decimal odds for team2 (optional).

        Returns:
            Dict with verdict, confidence_note, factors, and market_context.
        """
        t1_prob = result.t1_win_prob
        t2_prob = result.t2_win_prob
        confidence = result.confidence

        verdict = self._build_verdict(team1, team2, t1_prob, confidence)
        confidence_note = self._build_confidence_note(team1, team2, t1_prob, confidence)
        factors = self._build_factors(team1, team2, venue, features, fb)
        market_context = self._build_market_context(
            team1, team2, t1_prob, odds_t1, odds_t2
        )

        return {
            "verdict": verdict,
            "confidence_note": confidence_note,
            "factors": factors,
            "market_context": market_context,
        }

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    def _build_verdict(
        self, team1: str, team2: str, t1_prob: float, confidence: float
    ) -> str:
        """Build the headline verdict string."""
        # confidence = abs(t1_prob - 0.5) * 2
        if t1_prob >= 0.5:
            favoured = team1
            pct = int(round(t1_prob * 100))
        else:
            favoured = team2
            pct = int(round(t2_prob * 100)) if hasattr(locals(), 't2_prob') else int(round((1 - t1_prob) * 100))

        t2_prob = 1 - t1_prob

        if t1_prob >= 0.5:
            favoured = team1
            pct = int(round(t1_prob * 100))
        else:
            favoured = team2
            pct = int(round(t2_prob * 100))

        # confidence > 0.25 means prob > 0.625 or < 0.375
        if confidence > 0.25:
            return f"Strong lean towards {favoured} ({pct}%)"
        elif t1_prob >= 0.55 or t1_prob <= 0.45:
            return f"Slight lean towards {favoured} ({pct}%)"
        else:
            p1 = int(round(t1_prob * 100))
            p2 = int(round(t2_prob * 100))
            return f"Too close to call ({team1}={p1}% · {team2}={p2}%)"

    # ------------------------------------------------------------------
    # Confidence note
    # ------------------------------------------------------------------

    def _build_confidence_note(
        self, team1: str, team2: str, t1_prob: float, confidence: float
    ) -> str:
        if confidence > 0.25:
            t2_prob = 1 - t1_prob
            favoured = team1 if t1_prob >= 0.5 else team2
            return f"High confidence — model strongly favours {favoured}"
        elif confidence >= 0.10:
            return "Medium confidence — some signal present"
        else:
            return "Low confidence — both teams evenly matched"

    # ------------------------------------------------------------------
    # Factors
    # ------------------------------------------------------------------

    def _build_factors(
        self,
        team1: str,
        team2: str,
        venue: str,
        features: dict[str, float],
        fb: Any,
    ) -> list[dict[str, str]]:
        factors: list[dict[str, str]] = []

        factors.append(self._factor_home_adv(team1, team2, venue, features))
        factors.append(self._factor_recent_form(team1, team2, features))
        factors.append(self._factor_h2h(team1, team2, features, fb))
        factors.append(self._factor_venue(team1, team2, venue, features))

        toss_factor = self._factor_toss(team1, team2, features)
        if toss_factor is not None:
            factors.append(toss_factor)

        elo_factor = self._factor_elo(team1, team2, features)
        if elo_factor is not None:
            factors.append(elo_factor)

        return factors

    def _factor_home_adv(
        self,
        team1: str,
        team2: str,
        venue: str,
        features: dict[str, float],
    ) -> dict[str, str]:
        home_adv = features.get("home_adv", 0.0)
        t1_venue_rate = features.get("t1_venue_rate", 0.5)
        t2_venue_rate = features.get("t2_venue_rate", 0.5)

        if home_adv > 0:
            text = (
                f"{team1} playing at home — "
                f"{int(t1_venue_rate * 100)}% venue win rate"
            )
            impact = "positive"
        elif home_adv < 0:
            text = (
                f"{team2} has home advantage — "
                f"{int(t2_venue_rate * 100)}% venue win rate"
            )
            impact = "negative"
        else:
            text = f"Neutral venue — no home advantage for either team"
            impact = "neutral"

        return {"icon": "🏟️", "label": "Home advantage", "text": text, "impact": impact}

    def _factor_recent_form(
        self,
        team1: str,
        team2: str,
        features: dict[str, float],
    ) -> dict[str, str]:
        t1f = features.get("t1_form_5", 0.5)
        t2f = features.get("t2_form_5", 0.5)
        t1_wins = int(round(t1f * 5))
        t2_wins = int(round(t2f * 5))

        text = f"{team1} won {t1_wins}/5 last matches · {team2} won {t2_wins}/5"

        if t1f > t2f + 0.1:
            impact = "positive"
        elif t2f > t1f + 0.1:
            impact = "negative"
        else:
            impact = "neutral"

        return {"icon": "📈", "label": "Recent form", "text": text, "impact": impact}

    def _factor_h2h(
        self,
        team1: str,
        team2: str,
        features: dict[str, float],
        fb: Any,
    ) -> dict[str, str]:
        h2h_rate = features.get("h2h_t1_rate", 0.5)

        # Compute actual counts from FeatureBuilder matches
        t1_wins = t2_wins = total = 0
        if fb is not None and hasattr(fb, "_matches"):
            for m in fb._matches:
                if {m.team1, m.team2} == {team1, team2}:
                    total += 1
                    if m.winner == team1:
                        t1_wins += 1
                    else:
                        t2_wins += 1

        if total == 0:
            # Fallback to rate-based estimate
            text = "No historical H2H data available"
            impact = "neutral"
        elif t1_wins > t2_wins:
            text = f"{team1} leads H2H {t1_wins}-{t2_wins} all time"
            impact = "positive"
        elif t2_wins > t1_wins:
            text = f"{team2} leads H2H {t2_wins}-{t1_wins} all time"
            impact = "negative"
        else:
            text = f"Teams level H2H {t1_wins}-{t2_wins} all time"
            impact = "neutral"

        return {"icon": "⚔️", "label": "Head-to-head", "text": text, "impact": impact}

    def _factor_venue(
        self,
        team1: str,
        team2: str,
        venue: str,
        features: dict[str, float],
    ) -> dict[str, str]:
        t1_rate = features.get("t1_venue_rate", 0.5)
        t2_rate = features.get("t2_venue_rate", 0.5)

        text = (
            f"{team1} wins {int(t1_rate * 100)}% at this venue · "
            f"{team2} wins {int(t2_rate * 100)}%"
        )

        if t1_rate > t2_rate + 0.05:
            impact = "positive"
        elif t2_rate > t1_rate + 0.05:
            impact = "negative"
        else:
            impact = "neutral"

        return {"icon": "📍", "label": "Venue record", "text": text, "impact": impact}

    def _factor_toss(
        self,
        team1: str,
        team2: str,
        features: dict[str, float],
    ) -> Optional[dict[str, str]]:
        t1_has_toss = features.get("t1_has_toss", 0.0)
        toss_venue_adv = features.get("toss_venue_adv", 0.5)

        # Only show toss factor when toss winner is known (t1_has_toss is 0 or 1,
        # but 0 could mean team2 won toss OR toss not known; check toss_venue_adv > 0)
        if t1_has_toss == 0.0 and features.get("toss_bat_1st", 0.0) == 0.0:
            # Toss not yet decided — skip unless toss_venue_adv is meaningful
            return None

        if t1_has_toss == 1.0:
            winner = team1
            impact = "positive" if toss_venue_adv >= 0.5 else "neutral"
        else:
            winner = team2
            impact = "negative" if toss_venue_adv >= 0.5 else "neutral"

        text = (
            f"{winner} won the toss — toss winner wins "
            f"{int(toss_venue_adv * 100)}% at this venue"
        )

        return {"icon": "🪙", "label": "Toss", "text": text, "impact": impact}

    def _factor_elo(
        self,
        team1: str,
        team2: str,
        features: dict[str, float],
    ) -> Optional[dict[str, str]]:
        elo_adv = features.get("elo_t1_adv", 0.0)

        # Only show if the gap is meaningful (> 0.05 normalised, i.e. 20 raw ELO pts)
        if abs(elo_adv) <= 0.05:
            return None

        # elo_adv = (r1 - r2) / 400, so raw gap = elo_adv * 400
        raw_gap = int(abs(elo_adv) * 400)

        if elo_adv > 0:
            favoured = team1
            impact = "positive"
        else:
            favoured = team2
            impact = "negative"

        text = f"ELO gap: +{raw_gap} points in favour of {favoured}"

        return {"icon": "📊", "label": "ELO rating", "text": text, "impact": impact}

    # ------------------------------------------------------------------
    # Market context
    # ------------------------------------------------------------------

    def _build_market_context(
        self,
        team1: str,
        team2: str,
        t1_prob: float,
        odds_t1: float,
        odds_t2: float,
    ) -> str:
        t2_prob = 1 - t1_prob

        if odds_t1 > 1.0:
            # Use team1 odds as primary reference
            implied = round(1.0 / odds_t1 * 100, 1)
            ai_pct = round(t1_prob * 100, 1)
            gap = round(ai_pct - implied, 1)
            team = team1
            odds = odds_t1
        elif odds_t2 > 1.0:
            implied = round(1.0 / odds_t2 * 100, 1)
            ai_pct = round(t2_prob * 100, 1)
            gap = round(ai_pct - implied, 1)
            team = team2
            odds = odds_t2
        else:
            # No odds provided
            return "No market odds available for comparison"

        if abs(gap) < 3:
            edge_text = "AI agrees"
        elif gap > 0:
            edge_text = f"AI sees +{abs(gap):.1f}% edge"
        else:
            edge_text = f"AI sees -{abs(gap):.1f}% edge"

        return (
            f"Market prices {team} at {odds:.2f} ({implied}% implied) — {edge_text}"
        )
