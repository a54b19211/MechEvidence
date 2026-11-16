"""POST /v2/evaluate: validate the binding, refuse another evaluator, authorize every model request with the trusted
endpoint (and hand it each outcome), evaluate, check the receipt against this service's own receipt rules, and answer
with it (it names evaluator, reuse key and bound materials). POST /v2/reuse-key answers the evaluator and reuse key of
a request without evaluating it, so a caller can look up its cache without knowing how the key is made."""
from __future__ import annotations

import uuid

from .assess import EvaluationFailed, evaluate
from .authorization import TrustedEndpoint
from .contract import (
    CONTRACT_V1,
    CONTRACT_V2,
    INCOMPLETE_CODES,
    ContractError,
    Parsed,
    adopt_wire,
    material_echo,
    parse_request,
    reuse_key,
)
from .identity import current_evaluator
from .llm import chat_json
from .models import EvaluateRequest


def _internal_request(parsed: Parsed) -> EvaluateRequest:
    request = parsed.request
    return EvaluateRequest(question=request.question, mechanism=request.mechanism, scope=request.scope.to_scope().model_dump(),
                           materials=[item.to_material() for item in request.materials],
                           claims=[item.model_dump() for item in request.claims] if request.claims is not None else None,
                           joins=[item.model_dump() for item in request.joins], max_llm_calls=request.max_llm_calls)


def new_attempt_id() -> str:
    return uuid.uuid4().hex


def _checked(raw) -> tuple[Parsed, dict]:
    """The validated request and the evaluator that would answer it; ContractError 422 / 409 evaluator_changed."""
    parsed = parse_request(raw)
    request, evaluator = parsed.request, current_evaluator()
    if request.expected_evaluator_id is not None and request.expected_evaluator_id != evaluator["evaluator_id"]:
        # one request, one determinate evaluator: nothing is evaluated and no model is called
        raise ContractError(409, "evaluator_changed",
                            "the evaluator is not the one this request was prepared for; ask POST /v2/reuse-key "
                            "for the current evaluator and key and send again", evaluator=evaluator,
                            expected_evaluator_id=request.expected_evaluator_id, llm_calls=[])
    return parsed, evaluator


def reuse_key_v2(raw) -> dict:
    """POST /v2/reuse-key: what POST /v2/evaluate would key this request by, and the material echo its receipt
    would carry. Nothing is evaluated and no model is called; the same ContractError as POST /v2/evaluate."""
    parsed, evaluator = _checked(raw)
    return {"contract_version": CONTRACT_V2, "evaluator": evaluator, "reuse_key": reuse_key(parsed, evaluator),
            "materials": material_echo(parsed)}


def evaluate_v2(raw, chat=chat_json, endpoint=None) -> dict:
    """Raises ContractError (422 invalid, 409 evaluator_changed) or EvaluationFailed (502: model failure, the
    endpoint refused a request or did not record an outcome, a raw model answer too large to archive, or a
    receipt that breaks this service's own receipt rules: receipt_invalid, never returned)."""
    endpoint = endpoint if endpoint is not None else TrustedEndpoint()
    parsed, evaluator = _checked(raw)
    request = parsed.request
    key = reuse_key(parsed, evaluator)
    seen: list[dict] = []
    sent = {"index": 0, "decision": None}   # index: request number over the whole evaluation, never reset

    def before_attempt(info: dict) -> None:
        sent["index"] += 1
        attempt = {"id": new_attempt_id(), "stage": info["stage"], "index": sent["index"],
                   "request_body_sha256": info["request_body_sha256"], "model": info["model"],
                   "endpoint_id": info["endpoint_id"]}
        sent["decision"] = None
        sent["decision"] = endpoint.authorize(request.authorization_ref, key, evaluator["evaluator_id"], attempt)

    def after_attempt(entry: dict) -> None:
        # the receipt maps every counted request to the endpoint's prepared row
        entry.update(index=sent["index"], attempt_id=sent["decision"]["attempt_id"],
                     decision_id=sent["decision"]["decision_id"])
        endpoint.settle(request.authorization_ref, sent["decision"], entry)

    def guarded(stage, system, payload, model_cls, **kwargs):
        result, record = chat(stage, system, payload, model_cls, before_attempt=before_attempt,
                              after_attempt=after_attempt, **kwargs)
        seen.append(record)
        error = str(record.get("error") or "")
        if error.startswith("authorization_"):
            raise EvaluationFailed("authorization_blocked", f"{error}: no further model request was sent", seen)
        if any(out.get("truncated") for out in record.get("outputs") or []):
            # fail closed: an answer that cannot be archived whole is not a result anybody can audit or adopt
            raise EvaluationFailed("output_oversize", "a raw model answer exceeds the archive limit; the result is "
                                   "not usable (audit incomplete)", seen)
        return result, record

    result = evaluate(_internal_request(parsed), guarded)
    reasons = sorted({note["code"] for note in result["material_notes"] if note["code"] in INCOMPLETE_CODES})
    receipt = {"contract_version": CONTRACT_V2, "evaluator": evaluator, "reuse_key": key,
               "complete": not reasons, "not_complete_because": reasons, "audit_complete": True,
               "scope": request.scope.normal(), "materials": material_echo(parsed), **result}
    # the producer enforces its own receipt rules: a receipt that breaks them is a failure, never an answer
    outcome = adopt_wire(parsed.wire, receipt, evaluator["evaluator_id"])["result"]
    if outcome["status"] != "completed":
        problems = outcome.get("invalid") or outcome.get("mismatch") or [outcome["reason"]]
        raise EvaluationFailed("receipt_invalid", f"the receipt breaks the v2 receipt rules: {str(problems)[:500]}",
                               seen)
    return receipt


def v1_receipt(result: dict) -> dict:
    """The compatibility answer: the evaluation itself unchanged, four fields saying what it is not."""
    calls = [{key: value for key, value in call.items() if key != "outputs"} for call in result["llm_calls"]]
    return {**result, "llm_calls": calls, "contract_version": CONTRACT_V1, "evaluator": current_evaluator(),
            "binding_state": "unbound", "reusable": False}
