"""POST /v2/evaluate: validate the binding, refuse another evaluator, check authorization before every model
request, evaluate, and answer with a receipt that names evaluator, reuse key and bound materials."""
from __future__ import annotations

from collections.abc import Callable

from .assess import EvaluationFailed, evaluate
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
from .llm import SendBlocked, chat_json
from .models import EvaluateRequest

Authorize = Callable[[str, dict], None]


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


def evaluate_v2(raw, chat=chat_json, authorize: Authorize | None = None) -> dict:
    """Raises ContractError (422 invalid, 409 evaluator_changed) or EvaluationFailed (502: model failure, the
    authorization refused a request, or a raw model answer too large to archive in full)."""
    parsed = parse_request(raw)
    request, evaluator = parsed.request, current_evaluator()
    if request.expected_evaluator_id is not None and request.expected_evaluator_id != evaluator["evaluator_id"]:
        # one request, one determinate evaluator: nothing is evaluated and no model is called
        raise ContractError(409, "evaluator_changed",
                            "the evaluator is not the one this request was prepared for; fetch GET /v2/evaluator, "
                            "recompute the reuse key and send again", evaluator=evaluator,
                            expected_evaluator_id=request.expected_evaluator_id, llm_calls=[])
    seen: list[dict] = []

    def guarded(stage, system, payload, model_cls, **kwargs):
        def before_attempt(attempt: dict) -> None:
            if authorize is None:
                raise SendBlocked("authorization_not_configured", "no authorization check is wired")
            authorize(request.authorization_ref, {**attempt, "evaluator_id": evaluator["evaluator_id"]})

        result, record = chat(stage, system, payload, model_cls, before_attempt=before_attempt, **kwargs)
        seen.append(record)
        if str(record.get("error") or "").startswith("authorization_"):
            raise EvaluationFailed("authorization_blocked", f"{record['error']}: no further model request was sent", seen)
        if any(out.get("truncated") for out in record.get("outputs") or []):
            # fail closed: an answer that cannot be archived whole is not a result anybody can audit or adopt
            raise EvaluationFailed("output_oversize", "a raw model answer exceeds the archive limit; the result is "
                                   "not usable (audit incomplete)", seen)
        return result, record

    result = evaluate(_internal_request(parsed), guarded)
    reasons = sorted({note["code"] for note in result["material_notes"] if note["code"] in INCOMPLETE_CODES})
    return {"contract_version": CONTRACT_V2, "evaluator": evaluator, "reuse_key": reuse_key(parsed, evaluator),
            "complete": not reasons, "not_complete_because": reasons, "audit_complete": True,
            "scope": request.scope.normal(), "materials": _material_echo(parsed), **result}


def v1_receipt(result: dict) -> dict:
    """The compatibility answer: the evaluation itself unchanged, four fields saying what it is not."""
    calls = [{key: value for key, value in call.items() if key != "outputs"} for call in result["llm_calls"]]
    return {**result, "llm_calls": calls, "contract_version": CONTRACT_V1, "evaluator": current_evaluator(),
            "binding_state": "unbound", "reusable": False}
