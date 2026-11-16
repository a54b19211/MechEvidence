"""MechEvidence's version-bound evaluation contract (mechevidence.evaluate.v2) as pure functions over the plain
dict of a v2 request: what an evaluator's id and a request's reuse key are, what a receipt echoes of its request,
and what a caller does with a 200 receipt (adopt). No model, no HTTP, no database, no third-party import.

mechevidence.contract hands these functions the request it validated with its models. These rules have one owner,
this service: it returns the evaluator id and reuse key (POST /v2/reuse-key, and in every receipt) and runs adopt
on its own receipt before answering (a receipt that fails is a 502 receipt_invalid, never returned). Callers keep
no copy: they validate a receipt against the JSON Schema of GET /v2/contract and compare it with what
POST /v2/reuse-key answered.

The request dict is the wire form: question, mechanism, scope, materials (each with its binding) and, for a
caller that sends them, claims and joins. A material may omit title, document_type and identifiers.
"""

import hashlib
import json

from .receipt import (
    INCOMPLETE_CODES,
    call_shape_problems,
    consistency_problems,
    shape_problems,
)
from .source_identity import IdentityError, source_identity

CONTRACT_V2 = "mechevidence.evaluate.v2"
# the fields an evaluator id covers (code_version is audit only: a refactor must not void every cached result)
EVALUATOR_ID_FIELDS = (
    "batch_size",
    "contract_version",
    "model_config",
    "ordering_rule",
    "prompts",
    "rules_version",
    "service",
)
IDENTIFIER_DEFAULTS = {
    "pmid": None,
    "doi": None,
    "url": None,
    "source_id": "",
    "record_id": "",
    "source_group": "",
}


def canonical(body) -> str:
    """The one serialisation hashed and compared (sorted keys, no spaces, UTF-8 kept)."""
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(body) -> str:
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def evaluator_id_of(evaluator: dict) -> str:
    return digest({name: evaluator[name] for name in EVALUATOR_ID_FIELDS})


def binding_hashes(request: dict) -> list[str]:
    """The sha256 of each material's binding as it is sent: every field of it is in the contract."""
    return [digest(item["binding"]) for item in request["materials"]]


def normal_scope(scope: dict) -> dict:
    """Each dimension as {state[, value]}: what the key and the receipt's echo are compared on."""
    dims = ("species", "system", "disease")
    return {name: {k: v for k, v in scope[name].items() if v is not None} for name in dims}


def _identity(identifiers: dict | None) -> str | None:
    try:
        return source_identity(identifiers)
    except IdentityError as error:
        return f"error:{error.code}"


def material_inputs(request: dict) -> list[dict]:
    """Every material field the evaluator uses, normalised, in request order (MechEvidence's
    contract.material_inputs): the model is sent title, kinds and level; the source group comes from the
    identifiers; the passage ids it cites derive from the material id."""
    rows = []
    for material, value in zip(request["materials"], binding_hashes(request)):
        identifiers = material.get("identifiers")
        rows.append(
            {
                "alias_id": material["id"],
                "binding_sha256": value,
                "title": material.get("title", ""),
                "source_kind": material["source_kind"],
                "document_type": material.get("document_type"),
                "text_level": material["text_level"],
                "identifiers": {**IDENTIFIER_DEFAULTS, **identifiers} if identifiers else None,
                "identity": _identity(identifiers),
            }
        )
    return rows


def reuse_key(request: dict, evaluator: dict) -> str:
    """Phase 1, conservative: an ordered list of materials including their request alias ids, and the batch
    size. The same texts in another order or under other aliases are another key. max_llm_calls and the
    authorization reference are not inputs: an answer the budget cut short is never reused."""
    return digest(
        {
            "contract_version": CONTRACT_V2,
            "proposition": {
                "question": request["question"],
                "mechanism": request["mechanism"],
                "claims": request.get("claims"),
                "joins": request.get("joins") or [],
            },
            "scope": normal_scope(request["scope"]),
            "materials": material_inputs(request),
            "batch_size": evaluator["batch_size"],
            "ordering_rule": evaluator["ordering_rule"],
            "evaluator_id": evaluator["evaluator_id"],
        }
    )


def material_echo(request: dict) -> list[dict]:
    """What a receipt says about its materials: id, binding hash, how it was checked, locators."""
    rows = []
    for material, value in zip(request["materials"], binding_hashes(request)):
        located = [
            {"segment": index, "locators": segment["locators"]}
            for index, segment in enumerate(material["binding"]["segments"])
            if segment.get("locators")
        ]
        rows.append(
            {
                "id": material["id"],
                "binding_sha256": value,
                "binding_checked": "self_consistent",
                "locators_returned": sum(len(item["locators"]) for item in located),
                "locators": located,
            }
        )
    return rows


def receipt_mismatches(request: dict, receipt: dict, expected_evaluator_id: str) -> list[dict]:
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
        expected_key = reuse_key(request, evaluator)
    except KeyError:
        expected_key = None
    check("reuse_key", expected_key, receipt.get("reuse_key"))
    check("materials", material_echo(request), receipt.get("materials"))
    return found


def _failed(receipt, reason: str, **detail) -> dict:
    return {
        "result": {"status": "failed", "reason": reason, **detail},
        "rejected": {
            "response": receipt,
            "mismatch": detail.get("mismatch", []),
            **({"invalid": detail["invalid"]} if "invalid" in detail else {}),
        },
    }


def adopt(request: dict, receipt, expected_evaluator_id: str) -> dict:
    """What the radar records for a 200 receipt: {"result": mechanism_result, "rejected": audit row or None}.
    Anything but a well-formed receipt for exactly this request with every raw model answer archived whole is
    `failed`: receipt_invalid (shape, then cross-field rules), receipt_mismatch (not the answer to this
    request), audit_incomplete. The raw receipt is kept as `rejected` and the conclusion is not usable or
    cacheable anywhere. Never raises on a malformed receipt. A completed result is not yet adopted by the
    radar: it must also agree with the radar's own attempt ledger (its mech.ledger.reconcile)."""
    invalid = shape_problems(receipt)
    if invalid:
        return _failed(receipt, "receipt_invalid", invalid=invalid)
    mismatch = receipt_mismatches(request, receipt, expected_evaluator_id)
    if mismatch:
        return _failed(receipt, "receipt_mismatch", mismatch=mismatch)
    invalid = consistency_problems(
        receipt, normal_scope(request["scope"]), [item["id"] for item in request["materials"]]
    )
    if invalid:
        return _failed(receipt, "receipt_invalid", invalid=invalid)
    if receipt["audit_complete"] is not True or any(
        out["truncated"] for call in receipt["llm_calls"] for out in call["outputs"]
    ):
        return _failed(receipt, "audit_incomplete")
    result = {"status": "completed", "cacheable": receipt["complete"], "response": receipt}
    return {"result": result, "rejected": None}


__all__ = [
    "CONTRACT_V2",
    "INCOMPLETE_CODES",
    "IdentityError",
    "adopt",
    "binding_hashes",
    "call_shape_problems",
    "canonical",
    "consistency_problems",
    "digest",
    "evaluator_id_of",
    "material_echo",
    "material_inputs",
    "normal_scope",
    "receipt_mismatches",
    "reuse_key",
    "shape_problems",
    "source_identity",
]
