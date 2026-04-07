"""Email service using Resend.com API.

Uses RESEND_API_KEY env var. Free tier: 3,000 emails/month.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

RESEND_URL = "https://api.resend.com/emails"
FROM_EMAIL = "PitchIQ <noreply@pitchiq.com>"


def _get_api_key() -> str:
    key = os.environ.get("RESEND_API_KEY", "")
    if not key:
        raise RuntimeError("RESEND_API_KEY env var not set")
    return key


async def send_email(to: str, subject: str, html: str) -> dict[str, Any]:
    """Send a single email via Resend."""
    api_key = _get_api_key()
    payload = {
        "from": FROM_EMAIL,
        "to": [to],
        "subject": subject,
        "html": html,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def send_welcome_email(email: str, name: str) -> dict[str, Any]:
    """Send welcome email after signup."""
    html = f"""
    <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; background: #07090f; color: #fff; padding: 32px;">
        <h1 style="color: #f0b429;">Welcome to PitchIQ, {name}!</h1>
        <p>Your 7-day free trial has started. You now have full access to:</p>
        <ul>
            <li>All match predictions with win probabilities</li>
            <li>Bookmaker odds comparison from 6 bookmakers</li>
            <li>Value bet alerts when our model finds an edge</li>
            <li>Predicted score ranges for every match</li>
        </ul>
        <p>Enjoy the IPL season with data-driven predictions.</p>
        <p style="color: #5a6380; font-size: 12px;">— The PitchIQ Team</p>
    </div>
    """
    return await send_email(email, "Welcome to PitchIQ — Your trial starts now", html)


async def send_value_alert(email: str, name: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Send value bet alert email."""
    rows = ""
    for m in matches:
        rows += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #1a2035;">{m['team1']} vs {m['team2']}</td>
            <td style="padding: 8px; border-bottom: 1px solid #1a2035;">{m.get('venue', '')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #1a2035; color: #10b981; font-weight: bold;">
                +{m.get('model_edge', 0):.1%} edge
            </td>
        </tr>
        """

    html = f"""
    <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; background: #07090f; color: #fff; padding: 32px;">
        <h1 style="color: #f0b429;">Value Bet Alert</h1>
        <p>Hi {name}, our model has found value bets for today:</p>
        <table style="width: 100%; color: #fff; border-collapse: collapse;">
            <tr style="color: #5a6380;">
                <th style="text-align: left; padding: 8px;">Match</th>
                <th style="text-align: left; padding: 8px;">Venue</th>
                <th style="text-align: left; padding: 8px;">Edge</th>
            </tr>
            {rows}
        </table>
        <p style="margin-top: 24px;">
            <a href="https://pitchiq.com" style="background: #f0b429; color: #07090f; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">
                View Full Predictions
            </a>
        </p>
        <p style="color: #5a6380; font-size: 12px; margin-top: 32px;">
            18+ | BeGambleAware.org | Predictions are for information only.
        </p>
    </div>
    """
    return await send_email(email, f"PitchIQ Value Alert — {len(matches)} bet(s) found today", html)
