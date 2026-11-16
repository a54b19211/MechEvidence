"""The versioned contract: parsing and classifying a v2 request with this service's models. The evaluator id, the
reuse key, the receipt rules and `adopt` are in v2_rules (pure functions over the wire dict); this module hands
them the validated request as a plain dict (`Parsed.wire`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from . import v2_rules as shared
from .models_v2 import CONTRACT_V2, EvaluateRequestV2
from .v2_rules import (  # noqa: F401 - re-exported: INCOMPLETE_CODES: complete=false is never adopted or reused as a whole
    EVALUATOR_ID_FIELDS,  # noqa: F401
    INCOMPLETE_CODES,
    canonical,
    digest,
    evaluator_id_of,
)

CONTRACT_V1 = "mechevidence.evaluate.v1"


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
    wire: dict   # the validated request as v2_rules reads it (defaults filled in, bindings as sent)


def binding_sha256(binding: dict) -> str:
    return digest(binding)


def _wire(request: EvaluateRequestV2, raw: dict) -> dict:
    materials = [{"id": item.id, "title": item.title, "source_kind": item.source_kind,
                  "document_type": item.document_type, "text_level": item.text_level,
                  "identifiers": item.identifiers.model_dump(mode="json") if item.identifiers else None,
                  "binding": sent["binding"]}
                 for item, sent in zip(request.materials, raw["materials"])]
    return {"question": request.question, "mechanism": request.mechanism, "scope": request.scope.normal(),
            "claims": [item.model_dump() for item in request.claims] if request.claims is not None else None,
            "joins": [item.model_dump() for item in request.joins], "materials": materials}


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
    return Parsed(request, hashes, _wire(request, raw))


# ---- evaluator id, reuse key and what a caller does with a receipt: v2_rules ----

def material_inputs(parsed: Parsed) -> list[dict]:
    """Every material field the evaluator uses, normalised, in request order. The model is sent title,
    source_kind, document_type and text_level; the source group comes from the identifiers; the passage ids it
    cites derive from the material id. tests/test_contract_v2.py perturbs each field to prove nothing is missing."""
    return shared.material_inputs(parsed.wire)


def reuse_key(parsed: Parsed, evaluator: dict) -> str:
    """Phase 1, conservative: an ordered list of materials including their request alias ids, and the batch size
    (rules in v2_rules). max_llm_calls is left out: an answer the budget cut short is not
    complete and is never reused."""
    return shared.reuse_key(parsed.wire, evaluator)


def material_echo(parsed: Parsed) -> list[dict]:
    """What a receipt says about the materials it evaluated: id, binding hash, how it was checked, locators."""
    return shared.material_echo(parsed.wire)


def receipt_mismatches(parsed: Parsed, receipt: dict, expected_evaluator_id: str) -> list[dict]:
    """Differences between a receipt and the request it answers; empty = the receipt is for this request."""
    return shared.receipt_mismatches(parsed.wire, receipt, expected_evaluator_id)


def adopt(request: dict, receipt, expected_evaluator_id: str) -> dict:
    """What a caller records for a 200 receipt: {"result": mechanism_result, "rejected": row or None}; the rules are
    v2_rules.adopt. Never raises on a malformed receipt (a malformed request raises
    ContractError)."""
    return shared.adopt(parse_request(request).wire, receipt, expected_evaluator_id)
