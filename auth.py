"""Small signed-token boundary for the local demo.

This is intentionally dependency-free. It is suitable for a local portfolio
demo, not a replacement for the organization's identity provider.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_token(user_id: str, secret: str, ttl_seconds: int = 28800) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + ttl_seconds}
    body = _encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signature = _encode(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}"


def verify_token(token: str, secret: str) -> str | None:
    try:
        body, signature = token.split(".", 1)
        expected = _encode(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_decode(body).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        subject = str(payload.get("sub", "")).strip()
        return subject or None
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return None
