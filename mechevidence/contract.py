"""The versioned contract as pure functions: parsing and classifying a v2 request, the binding hash, the evaluator
id, the reuse key, and what a caller does with a receipt. No model, no HTTP, no environment - so the radar's tests
(and, later, a shared contract package) can run exactly the code the service runs.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .materials import IdentityError, source_identity
from .models_v2 import CONTRACT_V2, EvaluateRequestV2

CONTRACT_V1 = "mechevidence.evaluate.v1"
# The fields an evaluator id covers (code_version is audit only; a refactor must not void every cached result).
EVALUATOR_ID_FIELDS = ("batch_size", "contract_version", "model_config", "ordering_rule", "prompts", "rules_version",
                       "service")
# complete=false: the reviewer never read these materials, so nobody may adopt or reuse the answer as a whole
INCOMPLETE_CODES = ("pending_budget", "review_failed")


def canonical(body) -> str:
    """The one serialisation hashed and compared (same as extraction_contract.canonical)."""
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(body) -> str:
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


class ContractError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra):
        self.status, self.code, self.extra = status, code, extra
        super().__init__(message)


# ---- parsing a v2 request ----

_VALUE_CODES = {"material_binding_missing", "material_binding_mismatch", "scope_state_invalid"}


def _path(error: dict) -> str:
    return ".".join(map(str, error["loc"]))


def classify(errors: list[dict]) -> tuple[str, str]:
    """(error_code, message) of a failed v2 validation. An unknown field is always reported as such first."""
    extras = [_path(error) for error in errors if error["type"] == "extra_forbidden"]
    if extras:
        return "field_not_in_contract", (
            f"not in contract {CONTRACT_V2}: {', '.join(extras[:10])}. A field the contract does not name is "
            "rejected, never ignored.")
    for error in errors:
        match = re.match(r"Value error, ([a-z_]+): (.*)", error["msg"], re.DOTALL)
        if match and match.group(1) in _VALUE_CODES:
            return match.group(1), match.group(2)[:500]
    for error in errors:
        loc = error["loc"]
        if error["type"] == "missing" and len(loc) == 3 and loc[0] == "materials" and loc[2] == "binding":
            return "material_binding_missing", f"materials.{loc[1]}: a material without a binding names no fixed version"
        if error["type"] == "missing" and loc == ("authorization_ref",):
            return "authorization_required", "authorization_ref is required: it is what the service checks before each model request"
    details = "; ".join(f"{_path(error)}: {error['msg']}" for error in errors[:10])[:1000]
    first = errors[0]["loc"]
    if first and first[0] == "scope":
        return "scope_state_invalid", details
    if "binding" in first:
        return "material_binding_invalid", details
    return "input_invalid", details


@dataclass(frozen=True)
class Parsed:
    request: EvaluateRequestV2
    binding_hashes: list[str]


def binding_sha256(binding: dict) -> str:
    return digest(binding)


def parse_request(raw) -> Parsed:
    """Validate a v2 request body; raises ContractError(422, code, ...). Nothing is evaluated here."""
    if not isinstance(raw, dict):
        raise ContractError(422, "input_invalid", "the request body must be a JSON object")
    if raw.get("contract_version") != CONTRACT_V2:
        raise ContractError(422, "contract_version_required", f'contract_version must be "{CONTRACT_V2}"')
    try:
        request = EvaluateRequestV2.model_validate(raw)
    except ValidationError as error:
        code, message = classify(error.errors())
        raise ContractError(422, code, message) from None
    # the hash is over the binding as sent: every field of it is in the contract, so nothing is lost or added
    hashes = [binding_sha256(item["binding"]) for item in raw["materials"]]
    seen: dict[str, str] = {}
    for item, value in zip(request.materials, hashes):
        if value in seen:
            raise ContractError(422, "duplicate_material_binding",
                                f"materials {seen[value]} and {item.id} carry the same binding; send each text once")
        seen[value] = item.id
    return Parsed(request, hashes)


# ---- evaluator id and reuse key ----

def evaluator_id_of(evaluator: dict) -> str:
    return digest({name: evaluator[name] for name in EVALUATOR_ID_FIELDS})


def _identity(material) -> str | None:
    try:
        return source_identity(material.to_material())
    except IdentityError as error:
        return f"error:{error.code}"


def material_inputs(parsed: Parsed) -> list[dict]:
    """Every material field the evaluator uses, normalised, in request order. The model is sent title,
    source_kind, document_type and text_level; the source group comes from the identifiers; the passage ids it
    cites derive from the material id. tests/test_contract_v2.py perturbs each field to prove nothing is missing."""
    rows = []
    for material, value in zip(parsed.request.materials, parsed.binding_hashes):
        rows.append({"alias_id": material.id, "binding_sha256": value, "title": material.title,
                     "source_kind": material.source_kind, "document_type": material.document_type,
                     "text_level": material.text_level,
                     "identifiers": material.identifiers.model_dump(mode="json") if material.identifiers else None,
                     "identity": _identity(material)})
    return rows


def reuse_key(parsed: Parsed, evaluator: dict) -> str:
    """Phase 1, conservative: an ordered list of materials including their request alias ids, and the batch size.
    The same texts in another order or under other aliases are another key; that is deliberate until the evaluator
    itself orders and numbers its input canonically (ordering_rule != request_order). max_llm_calls is left out:
    an answer the budget cut short is not complete and is never reused."""
    request = parsed.request
    return digest({
        "contract_version": CONTRACT_V2,
        "proposition": {"question": request.question, "mechanism": request.mechanism,
                        "claims": [item.model_dump() for item in request.claims] if request.claims is not None else None,
                        "joins": [item.model_dump() for item in request.joins]},
        "scope": request.scope.normal(),
        "materials": material_inputs(parsed),
        "batch_size": evaluator["batch_size"],
        "ordering_rule": evaluator["ordering_rule"],
        "evaluator_id": evaluator["evaluator_id"]})


# ---- what a caller does with a receipt ----

def receipt_mismatches(parsed: Parsed, receipt: dict, expected_evaluator_id: str) -> list[dict]:
    """Differences between a receipt and the request it answers; empty = the receipt is for this request."""
    found = []

    def check(field, expected, received):
        if expected != received:
            found.append({"field": field, "expected": expected, "received": received})

    evaluator = receipt.get("evaluator") if isinstance(receipt.get("evaluator"), dict) else {}
    check("contract_version", CONTRACT_V2, receipt.get("contract_version"))
    check("evaluator.evaluator_id", expected_evaluator_id, evaluator.get("evaluator_id"))
    try:
        recomputed = evaluator_id_of(evaluator)
    except KeyError:
        recomputed = None
    check("evaluator.evaluator_id(recomputed from its fields)", evaluator.get("evaluator_id"), recomputed)
    try:
        expected_key = reuse_key(parsed, evaluator)
    except KeyError:
        expected_key = None
    check("reuse_key", expected_key, receipt.get("reuse_key"))
    echoed = [(item.get("id"), item.get("binding_sha256")) for item in receipt.get("materials") or []
              if isinstance(item, dict)]
    check("materials", [(item.id, value) for item, value in zip(parsed.request.materials, parsed.binding_hashes)],
          echoed)
    return found


def adopt(request: dict, receipt: dict, expected_evaluator_id: str) -> dict:
    """What the radar records for a 200 receipt: {"result": mechanism_result, "rejected": row or None}.
    A receipt that does not match its request, or whose raw model answers are not all archived, is `failed`; the
    raw receipt is kept as `rejected` (an audit row) and the conclusion is not usable anywhere."""
    parsed = parse_request(request)
    mismatch = receipt_mismatches(parsed, receipt, expected_evaluator_id)
    if mismatch:
        return {"result": {"status": "failed", "reason": "receipt_mismatch", "mismatch": mismatch},
                "rejected": {"response": receipt, "mismatch": mismatch}}
    outputs = [out for call in receipt.get("llm_calls") or [] for out in call.get("outputs") or []]
    if receipt.get("audit_complete") is not True or any(out.get("truncated") for out in outputs):
        return {"result": {"status": "failed", "reason": "audit_incomplete"},
                "rejected": {"response": receipt, "mismatch": []}}
    return {"result": {"status": "completed", "cacheable": receipt.get("complete") is True, "response": receipt},
            "rejected": None}
