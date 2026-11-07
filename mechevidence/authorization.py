"""Before every actual model HTTP request of a v2 evaluation, ask the caller's trusted endpoint whether this very
request is allowed, and after it, hand the endpoint the outcome (option A of the permission ruling; the wire format
is model_attempt_authorization.v1 / model_attempt_outcome.v1, decided with the radar).

The service cannot know which sources a material came from or who may read them. The caller sends an opaque
`authorization_ref`; the endpoint (its address and token come from the environment, never from a request) binds
that ref to the analysis, its lease, the evaluation key, the evaluator, the materials and an attempt budget. Per
attempt the service sends {authorization_ref, evaluation_key, evaluator_id, attempt{id, stage, index,
request_body_sha256, model, endpoint_id}} and sends the request only if the answer is HTTP 200 with
valid:true AND echoes this attempt id, evaluation key and evaluator id, names a decision, and has not expired.
A bare {valid:true}, a wrong echo, an expired decision, any other status, bad JSON, an unreachable endpoint or a
missing configuration all mean: do not send. The outcome of each counted request (raw answer and its hash,
truncation, delivery) is posted before the next request; if that cannot be recorded, nothing more is sent.
The v1 path does not do any of this; that is a documented compatibility gap, not a guarantee of the new system.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx

from .llm import SendBlocked

AUTHORIZE_V1 = "model_attempt_authorization.v1"
OUTCOME_V1 = "model_attempt_outcome.v1"


def _post(url: str, body: dict) -> httpx.Response:
    return httpx.post(url, json=body, headers={"Authorization": "Bearer " + os.environ.get("MECH_AUTHORIZATION_TOKEN", "")},
                      follow_redirects=False, timeout=float(os.environ.get("MECH_AUTHORIZATION_TIMEOUT_SECONDS", "5")))


def _expired(value) -> bool:
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return True
    return when.tzinfo is None or when <= datetime.now(UTC)


class TrustedEndpoint:
    """The real thing, configured from MECH_AUTHORIZATION_URL, MECH_AUTHORIZATION_TOKEN and MECH_OUTCOME_URL (which
    contains `{attempt_id}`). Tests pass any object with the same two methods to create_app."""

    def configured(self) -> bool:
        return bool(os.environ.get("MECH_AUTHORIZATION_URL") and os.environ.get("MECH_AUTHORIZATION_TOKEN")
                    and "{attempt_id}" in os.environ.get("MECH_OUTCOME_URL", ""))

    def authorize(self, ref: str, evaluation_key: str, evaluator_id: str, attempt: dict) -> dict:
        """Returns the decision; raises SendBlocked(code) for anything but a valid, matching, unexpired one."""
        if not self.configured():
            raise SendBlocked("authorization_not_configured")
        try:
            response = _post(os.environ["MECH_AUTHORIZATION_URL"], {
                "contract_version": AUTHORIZE_V1, "service": "mechevidence", "authorization_ref": ref,
                "evaluation_key": evaluation_key, "evaluator_id": evaluator_id, "attempt": attempt})
        except httpx.HTTPError:
            raise SendBlocked("authorization_unreachable") from None
        if response.status_code != 200:
            raise SendBlocked("authorization_refused")
        try:
            answer = response.json()
        except ValueError:
            raise SendBlocked("authorization_malformed") from None
        if not isinstance(answer, dict) or not isinstance(answer.get("decision_id"), str) or not answer["decision_id"]:
            raise SendBlocked("authorization_malformed")
        if answer.get("valid") is not True:
            raise SendBlocked("authorization_refused")
        echoed = (answer.get("contract_version"), answer.get("attempt_id"), answer.get("evaluation_key"),
                  answer.get("evaluator_id"))
        if echoed != (AUTHORIZE_V1, attempt["id"], evaluation_key, evaluator_id):
            raise SendBlocked("authorization_mismatch")
        if _expired(answer.get("expires_at")):
            raise SendBlocked("authorization_expired")
        return answer

    def settle(self, ref: str, decision: dict, entry: dict) -> None:
        """Hand over what happened to the request just made; raises if the endpoint did not record it."""
        url = os.environ.get("MECH_OUTCOME_URL", "").replace("{attempt_id}", decision["attempt_id"])
        response = _post(url, {
            "contract_version": OUTCOME_V1, "service": "mechevidence", "authorization_ref": ref,
            "decision_id": decision["decision_id"], "attempt_id": decision["attempt_id"],
            "index_in_call": entry["attempt"], "delivered": entry["delivered"], "status": entry["status"],
            "error": entry["error"], "response_sha256": entry["sha256"], "truncated": entry["truncated"],
            "response_text": entry["content"]})
        if response.status_code != 200 or response.json().get("recorded") is not True:
            raise RuntimeError("outcome not recorded")
