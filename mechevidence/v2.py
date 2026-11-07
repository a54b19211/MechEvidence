"""POST /v2/evaluate: validate the binding, refuse another evaluator, authorize every model request with the trusted
endpoint (and hand it each outcome), evaluate, and answer with a receipt that names evaluator, reuse key and bound
materials."""
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
    parse_request,
    reuse_key,
)
from .identity import current_evaluator
from .llm import chat_json
from .models import EvaluateRequest


def _material_echo(parsed: Parsed) -> list[dict]:
    rows = []
    for material, value in zip(parsed.request.materials, parsed.binding_hashes):
        locators = [{"segment": index, "locators": segment.locators}
                    for index, segment in enumerate(material.binding.segments) if segment.locators]
        rows.append({"id": material.id, "binding_sha256": value, "binding_checked": "self_consistent",
                     "locators_returned": sum(len(item["locators"]) for item in locators), "locators": locators})
    return rows


def _internal_request(parsed: Parsed) -> EvaluateRequest:
    request = parsed.request
    return EvaluateRequest(question=request.question, mechanism=request.mechanism, scope=request.scope.to_scope().model_dump(),
                           materials=[item.to_material() for item in request.materials],
                           claims=[item.model_dump() for item in request.claims] if request.claims is not None else None,
                           joins=[item.model_dump() for item in request.joins], max_llm_calls=request.max_llm_calls)


def evaluate_v2(raw, chat=chat_json, endpoint=None) -> dict:
    """Raises ContractError (422 invalid, 409 evaluator_changed) or EvaluationFailed (502: model failure, the
    endpoint refused a request or did not record an outcome, or a raw model answer too large to archive)."""
    endpoint = endpoint if endpoint is not None else TrustedEndpoint()
    parsed = parse_request(raw)
    request, evaluator = parsed.request, current_evaluator()
    if request.expected_evaluator_id is not None and request.expected_evaluator_id != evaluator["evaluator_id"]:
        # one request, one determinate evaluator: nothing is evaluated and no model is called
        raise ContractError(409, "evaluator_changed",
                            "the evaluator is not the one this request was prepared for; fetch GET /v2/evaluator, "
                            "recompute the reuse key and send again", evaluator=evaluator,
                            expected_evaluator_id=request.expected_evaluator_id, llm_calls=[])
    key = reuse_key(parsed, evaluator)
    seen: list[dict] = []
    sent = {"index": 0, "decision": None}   # index: request number over the whole evaluation, never reset

    def before_attempt(info: dict) -> None:
        sent["index"] += 1
        attempt = {"id": uuid.uuid4().hex, "stage": info["stage"], "index": sent["index"],
                   "request_body_sha256": info["request_body_sha256"], "model": info["model"],
                   "endpoint_id": info["endpoint_id"]}
        sent["decision"] = None
        sent["decision"] = endpoint.authorize(request.authorization_ref, key, evaluator["evaluator_id"], attempt)

    def after_attempt(entry: dict) -> None:
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
    return {"contract_version": CONTRACT_V2, "evaluator": evaluator, "reuse_key": key,
            "complete": not reasons, "not_complete_because": reasons, "audit_complete": True,
            "scope": request.scope.normal(), "materials": _material_echo(parsed), **result}


def v1_receipt(result: dict) -> dict:
    """The compatibility answer: the evaluation itself unchanged, four fields saying what it is not."""
    calls = [{key: value for key, value in call.items() if key != "outputs"} for call in result["llm_calls"]]
    return {**result, "llm_calls": calls, "contract_version": CONTRACT_V1, "evaluator": current_evaluator(),
            "binding_state": "unbound", "reusable": False}
