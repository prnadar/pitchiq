"""JWT authentication for PitchIQ.

Simple email+password auth with bcrypt hashing and JWT tokens.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Optional


def _get_secret() -> str:
    secret = os.environ.get("JWT_SECRET", "pitchiq-dev-secret-change-in-production")
    return secret


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with a salt. Simple but sufficient for MVP."""
    salt = os.urandom(16).hex()
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against its stored hash."""
    try:
        salt, hashed = stored_hash.split(":")
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        return hmac.compare_digest(hashed, check)
    except (ValueError, AttributeError):
        return False


def create_jwt(user_id: str, email: str, plan: str, expires_hours: int = 24) -> str:
    """Create a simple JWT token (HS256)."""
    secret = _get_secret()

    header = urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")

    payload_data = {
        "sub": user_id,
        "email": email,
        "plan": plan,
        "exp": int(time.time()) + expires_hours * 3600,
        "iat": int(time.time()),
    }
    payload = urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip("=")

    signature_input = f"{header}.{payload}".encode()
    sig = hmac.new(secret.encode(), signature_input, hashlib.sha256).digest()
    signature = urlsafe_b64encode(sig).decode().rstrip("=")

    return f"{header}.{payload}.{signature}"


def decode_jwt(token: str) -> Optional[dict]:
    """Decode and verify a JWT token. Returns payload dict or None if invalid."""
    secret = _get_secret()

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts

        # Verify signature
        signature_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(secret.encode(), signature_input, hashlib.sha256).digest()
        expected_b64 = urlsafe_b64encode(expected_sig).decode().rstrip("=")

        if not hmac.compare_digest(sig_b64, expected_b64):
            return None

        # Decode payload (add padding)
        padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(urlsafe_b64decode(padded))

        # Check expiry
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except (ValueError, json.JSONDecodeError, KeyError):
        return None
