"""Supabase client for PitchIQ.

Handles user management, predictions cache, and daily usage tracking.
Requires SUPABASE_URL and SUPABASE_KEY env vars.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

_SUPABASE_URL: str = ""
_SUPABASE_KEY: str = ""


def _get_config() -> tuple[str, str]:
    global _SUPABASE_URL, _SUPABASE_KEY
    if not _SUPABASE_URL:
        _SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
        _SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY env vars required")
    return _SUPABASE_URL, _SUPABASE_KEY


def _headers() -> dict[str, str]:
    url, key = _get_config()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def create_user(email: str, name: str, password_hash: str) -> dict[str, Any]:
    """Insert a new user row. Returns the created user dict."""
    url, _ = _get_config()
    payload = {
        "email": email,
        "name": name,
        "password_hash": password_hash,
        "plan": "trial",
        "trial_expires_at": None,  # set by trigger or app logic
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{url}/rest/v1/users",
            headers=_headers(),
            json=payload,
        )
        if resp.status_code == 409:
            raise ValueError("Email already registered")
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if isinstance(rows, list) else rows


async def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """Fetch a user by email. Returns None if not found."""
    url, _ = _get_config()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{url}/rest/v1/users",
            headers=_headers(),
            params={"email": f"eq.{email}", "select": "*", "limit": "1"},
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None


async def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """Fetch a user by ID."""
    url, _ = _get_config()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{url}/rest/v1/users",
            headers=_headers(),
            params={"id": f"eq.{user_id}", "select": "*", "limit": "1"},
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None


async def increment_daily_usage(user_id: str) -> dict[str, Any]:
    """Call the increment_usage RPC function.

    This increments the user's daily prediction count and returns the updated count.
    The RPC function handles the date check and reset logic server-side.
    """
    url, _ = _get_config()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{url}/rest/v1/rpc/increment_usage",
            headers=_headers(),
            json={"p_user_id": user_id},
        )
        resp.raise_for_status()
        return resp.json()


async def upsert_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Upsert a prediction into predictions_cache (on match_id conflict)."""
    url, _ = _get_config()
    headers = {**_headers(), "Prefer": "return=representation,resolution=merge-duplicates"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{url}/rest/v1/predictions_cache",
            headers=headers,
            json=prediction,
            params={"on_conflict": "match_id"},
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if isinstance(rows, list) else rows


async def get_cached_predictions(match_date: Optional[str] = None) -> list[dict[str, Any]]:
    """Fetch cached predictions, optionally filtered by date."""
    url, _ = _get_config()
    params: dict[str, str] = {"select": "*", "order": "match_date.asc"}
    if match_date:
        params["match_date"] = f"eq.{match_date}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{url}/rest/v1/predictions_cache",
            headers=_headers(),
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


async def get_pro_users_for_alerts() -> list[dict[str, Any]]:
    """Fetch all pro/expert users who should receive value bet alerts."""
    url, _ = _get_config()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{url}/rest/v1/users",
            headers=_headers(),
            params={
                "plan": "in.(pro,expert)",
                "select": "id,email,name",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def update_user_plan(user_id: str, plan: str) -> dict[str, Any]:
    """Update a user's subscription plan."""
    url, _ = _get_config()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.patch(
            f"{url}/rest/v1/users",
            headers=_headers(),
            params={"id": f"eq.{user_id}"},
            json={"plan": plan},
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if isinstance(rows, list) and rows else {}
