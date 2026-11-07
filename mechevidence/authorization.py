"""Re-check, before every actual model HTTP request of a v2 evaluation, that the caller may still send these
materials (option A of the permission ruling).

The service cannot know which sources a material came from or who may read them. The caller therefore sends an
opaque `authorization_ref` (scoped to one analysis, its materials, workspace and this service), and the service asks
a PRE-CONFIGURED endpoint - never an address taken from the request - whether the ref is still valid. Unreachable,
refusing, malformed or not configured all mean: do not send. The v1 path does not do this check; that is a
documented compatibility gap, not a guarantee of the new system.
"""
from __future__ import annotations

import os

import httpx

from .llm import SendBlocked


def configured() -> bool:
    return bool(os.environ.get("MECH_AUTHORIZATION_URL") and os.environ.get("MECH_AUTHORIZATION_TOKEN"))


def check_with_trusted_endpoint(ref: str, attempt: dict) -> None:
    """POST {service, authorization_ref, attempt} to MECH_AUTHORIZATION_URL (Bearer MECH_AUTHORIZATION_TOKEN);
    only HTTP 200 with {"valid": true} lets the request go. `attempt` names the request about to be sent (stage,
    attempt number, sha256 of its body, model, endpoint id, evaluator id) so the endpoint can log what it allowed."""
    url, token = os.environ.get("MECH_AUTHORIZATION_URL", ""), os.environ.get("MECH_AUTHORIZATION_TOKEN", "")
    if not url or not token:
        raise SendBlocked("authorization_not_configured", "MECH_AUTHORIZATION_URL / MECH_AUTHORIZATION_TOKEN not set")
    try:
        response = httpx.post(url, json={"service": "mechevidence", "authorization_ref": ref, "attempt": attempt},
                              headers={"Authorization": "Bearer " + token}, follow_redirects=False,
                              timeout=float(os.environ.get("MECH_AUTHORIZATION_TIMEOUT_SECONDS", "5")))
    except httpx.HTTPError as error:
        raise SendBlocked("authorization_unreachable", type(error).__name__) from None
    if response.status_code != 200:
        raise SendBlocked("authorization_refused", f"http_{response.status_code}")
    try:
        valid = response.json().get("valid") is True
    except (ValueError, AttributeError):
        raise SendBlocked("authorization_malformed", "the answer is not a JSON object") from None
    if not valid:
        raise SendBlocked("authorization_refused", "revoked or expired")
