"""Is this 200 receipt a well-formed v2 answer for its request? One validation layer, three passes, all pure:

1. shape_problems: every field against the SHAPE table - JSON types first and strict (a bool is not an int, a list
   is not an object), enums, hashes. Nothing else runs on a receipt that fails here, so no later check ever meets a
   value of the wrong type.
2. consistency_problems: cross-field rules - an archived HTTP answer must agree with itself (delivered / status /
   content / sha256 / error), the archive must cover every counted request with global, contiguous numbers, the
   verdict rows must reference observations and the request's materials that exist, `complete` must agree with the
   notes.
3. contract.adopt adds the comparison with the request (evaluator, reuse key, material echo).

A receipt that fails any pass is never adopted or cached; the caller keeps the raw receipt for audit. New receipt
fields are covered by adding them to SHAPE (and, if they refer to something, to the reference rules).

receipt_schema() is SHAPE as a JSON Schema, which GET /v2/contract publishes for callers (the radar validates
receipts and model calls against it instead of importing this module). The cross-field rules are enforced here,
by the service on its own receipt before it answers (v2.evaluate_v2); callers do not repeat them.
"""
from __future__ import annotations

import hashlib
import re

# A raw model answer is archived in the receipt up to this many bytes; a longer one is cut and marked truncated
# (its sha256 stays that of the whole answer). A truncated answer is an incomplete audit, never a usable result.
OUTPUT_ARCHIVE_LIMIT = 256 * 1024
INCOMPLETE_CODES = ("pending_budget", "review_failed")

OVERALL = {"supported", "partial", "insufficient"}
CLAIM_VERDICT = {"mixed", "supported", "refuted", "insufficient"}
JOIN_VERDICT = {"supported", "compatible_only", "unestablished"}
RELATION = {"supports", "refutes", "compatible"}
BASIS = {"intervention", "association", "review", "claim"}
SCOPE_STATUS = {"matched", "mismatched", "unknown"}
ASSESSABILITY = {"assessable", "limited", "unassessable"}
STAGES = {"decompose", "review"}
DELIVERED = {"yes", "no", "unknown"}

# ---- the table: "str" "int" (not bool) "num" "bool" "hex" (sha256) "any" | ("null", spec) | ("enum", set)
# ---- | ("list", spec) | ("obj", {field: spec}[, optional field names])
_EVALUATOR = ("obj", {
    "service": "str", "contract_version": "str", "rules_version": "str", "ordering_rule": "str", "batch_size": "int",
    "prompts": ("obj", {"decompose_sha256": "hex", "review_sha256": "hex"}),
    "model_config": ("obj", {"provider": "str", "model": "str", "thinking": "bool", "temperature": "num",
                             "max_tokens": "int", "endpoint_id": "str"}),
    "code_version": "str", "evaluator_id": "hex"})
_SCOPE_DIM = ("obj", {"state": ("enum", {"stated", "unknown", "not_required"}), "value": "str"}, {"value"})
_ECHO = ("obj", {"id": "str", "binding_sha256": "hex", "binding_checked": ("enum", {"self_consistent"}),
                 "locators_returned": "int",
                 "locators": ("list", ("obj", {"segment": "int", "locators": ("list", "any")}))})
_ROW = {"id": "str", "text": "str", "verdict": "str", "supporting": ("list", "str"), "opposing": ("list", "str"),
        "compatible": ("list", "str"), "applicability_gaps": ("list", "any"), "source_groups": ("list", "str")}
_OUTPUT = ("obj", {"attempt": "int", "index": "int", "attempt_id": "str", "decision_id": "str",
                   "delivered": ("enum", DELIVERED), "status": ("null", "int"), "error": ("null", "str"),
                   "content": ("null", "str"), "sha256": ("null", "hex"), "truncated": "bool"})
_CALL = ("obj", {"stage": ("enum", STAGES), "model": "str", "finish_reason": "any", "usage": "any", "seconds": "num",
                 "error": ("null", "str"), "requests": "int", "outputs": ("list", _OUTPUT)})
SHAPE = ("obj", {
    "contract_version": "str", "evaluator": _EVALUATOR, "reuse_key": "hex", "complete": "bool",
    "not_complete_because": ("list", "str"), "audit_complete": "bool",
    "scope": ("obj", {"species": _SCOPE_DIM, "system": _SCOPE_DIM, "disease": _SCOPE_DIM}),
    "materials": ("list", _ECHO), "overall": ("enum", OVERALL), "rationale": "str",
    "claims": ("list", ("obj", {**_ROW, "kind": ("enum", {"causal", "association", "reported"}),
                                "verdict": ("enum", CLAIM_VERDICT)})),
    "joins": ("list", ("obj", {**_ROW, "from": "str", "to": "str", "verdict": ("enum", JOIN_VERDICT)})),
    "observations": ("list", ("obj", {
        "id": "str", "targets": ("list", "str"), "material": "str", "passage": "str", "quote": "str",
        "relation": ("enum", RELATION), "basis": ("enum", BASIS), "causal_test": "bool", "context": "any",
        "scope_match": "any", "scope_status": ("enum", SCOPE_STATUS), "causal_evidence": "bool", "source_group": "str"})),
    "conflicts": ("list", "any"), "unresolved_counterevidence": ("list", "any"), "gaps": ("list", "str"),
    "independence": ("obj", {"distinct_source_groups": "int", "note": "str"}),
    "material_notes": ("list", ("obj", {"material": ("null", "str"), "code": "str", "message": "str"})),
    "assessability": ("obj", {"status": ("enum", ASSESSABILITY), "limitations": ("list", "str")}),
    "llm_calls": ("list", _CALL)})


def _kind(value) -> str:
    return ("null" if value is None else "bool" if isinstance(value, bool) else "int" if isinstance(value, int)
            else "num" if isinstance(value, float) else "str" if isinstance(value, str)
            else "list" if isinstance(value, list) else "object" if isinstance(value, dict) else type(value).__name__)


def _walk(value, spec, path: str, problems: list[str]) -> None:
    if spec == "any":
        return
    if isinstance(spec, str):
        ok = {"str": isinstance(value, str), "bool": isinstance(value, bool),
              "int": isinstance(value, int) and not isinstance(value, bool),
              "num": isinstance(value, (int, float)) and not isinstance(value, bool),
              "hex": isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None}[spec]
        if not ok:
            problems.append(f"{path} is {_kind(value)}, expected {spec}")
        return
    kind = spec[0]
    if kind == "null":
        if value is not None:
            _walk(value, spec[1], path, problems)
    elif kind == "enum":
        if not isinstance(value, str) or value not in spec[1]:
            shown = value if isinstance(value, str) else _kind(value)
            problems.append(f"{path} is {shown}, expected one of {sorted(spec[1])}")
    elif kind == "list":
        if not isinstance(value, list):
            problems.append(f"{path} is {_kind(value)}, expected list")
            return
        for index, item in enumerate(value):
            _walk(item, spec[1], f"{path}[{index}]", problems)
    else:
        if not isinstance(value, dict):
            problems.append(f"{path} is {_kind(value)}, expected object")
            return
        optional = spec[2] if len(spec) > 2 else set()
        for name, inner in spec[1].items():
            if name in value:
                _walk(value[name], inner, f"{path}.{name}", problems)
            elif name not in optional:
                problems.append(f"{path}.{name} is missing")


def shape_problems(receipt) -> list[str]:
    """Empty list = every field has the type, enum and form the table says."""
    problems: list[str] = []
    _walk(receipt, SHAPE, "receipt", problems)
    return problems


def call_shape_problems(call) -> list[str]:
    """Empty list = this one model call (an `llm_calls` entry of a receipt or of a service error answer) has the
    type, enum and form the SHAPE table gives a call. A client that keeps the calls of an answer it does not
    adopt checks each one with this before it believes anything the call says."""
    problems: list[str] = []
    _walk(call, _CALL, "call", problems)
    return problems


# ---- cross-field rules (run only on a receipt whose shape is right) ----

def _answer(where: str, out: dict) -> list[str]:
    problems = []
    status, content, digest, error = out["status"], out["content"], out["sha256"], out["error"]
    if out["delivered"] == "yes":
        if status is None or content is None or digest is None:
            return [f"{where} was answered but lacks status, text or sha256"]
        if error != (None if 200 <= status < 300 else f"http_{status}"):
            problems.append(f"{where}.error {error!r} does not agree with HTTP status {status}")
        if out["truncated"]:
            if len(content.encode("utf-8")) > OUTPUT_ARCHIVE_LIMIT:
                problems.append(f"{where}.content is longer than the archive limit")
        elif hashlib.sha256(content.encode("utf-8")).hexdigest() != digest:
            problems.append(f"{where}.content does not hash to its sha256")
    else:
        if status is not None or content is not None or digest is not None or out["truncated"]:
            problems.append(f"{where} was not answered but carries a status, text, hash or truncation")
        if error is None:
            problems.append(f"{where} was not answered and names no error")
    return problems


def _archive(calls: list[dict]) -> list[str]:
    problems, expected_index, seen_ids = [], 1, set()
    for number, call in enumerate(calls):
        where, outputs = f"llm_calls[{number}]", call["outputs"]
        if len(outputs) != call["requests"]:
            problems.append(f"{where} counts {call['requests']} requests but archives {len(outputs)} outputs")
        for position, out in enumerate(outputs, 1):
            spot = f"{where}.outputs[{position - 1}]"
            if out["attempt"] != position:
                problems.append(f"{spot}.attempt is {out['attempt']}, expected {position} (contiguous from 1)")
            if out["index"] != expected_index:
                problems.append(f"{spot}.index is {out['index']}, expected {expected_index} (global, contiguous from 1)")
            expected_index += 1
            if not out["attempt_id"] or not out["decision_id"] or out["attempt_id"] in seen_ids:
                problems.append(f"{spot} lacks a unique attempt_id or a decision_id")
            seen_ids.add(out["attempt_id"])
            problems += _answer(spot, out)
        if call["error"] is None and not (outputs and outputs[-1]["delivered"] == "yes"
                                          and outputs[-1]["status"] == 200):
            problems.append(f"{where} succeeded but its last request was not answered 200")
    return problems


def _references(receipt: dict, material_ids: list[str]) -> list[str]:
    problems = []
    observations = {}
    for row in receipt["observations"]:
        if row["id"] in observations:
            problems.append(f"observation id {row['id']} is used twice")
        observations[row["id"]] = row
    targets = {}
    for name in ("claims", "joins"):
        for row in receipt[name]:
            if row["id"] in targets:
                problems.append(f"{name} id {row['id']} is used twice")
            targets[row["id"]] = row
    for row in receipt["joins"]:
        if row["from"] not in targets or row["to"] not in targets:
            problems.append(f"join {row['id']} links {row['from']} -> {row['to']}, which do not exist")
    for row in targets.values():
        for field in ("supporting", "opposing", "compatible"):
            problems += [f"{row['id']}.{field} cites observation {oid}, which does not exist"
                         for oid in row[field] if oid not in observations]
    for row in observations.values():
        if row["material"] not in material_ids:
            problems.append(f"observation {row['id']} cites material {row['material']}, which is not in the request")
        if row["passage"].split(".")[0] != row["material"]:
            problems.append(f"observation {row['id']} cites passage {row['passage']} of another material")
        problems += [f"observation {row['id']} targets {target}, which does not exist"
                     for target in row["targets"] if target not in targets]
    problems += [f"a note cites material {note['material']}, which is not in the request"
                 for note in receipt["material_notes"]
                 if note["material"] is not None and note["material"] not in material_ids]
    return problems


def consistency_problems(receipt: dict, scope: dict, material_ids: list[str]) -> list[str]:
    """Cross-field problems of a receipt whose shape is right. `scope` is the request's normalised scope, which
    the receipt must echo; `material_ids` are the request's material ids."""
    problems = []
    if receipt["scope"] != scope:
        problems.append("scope does not echo the request's scope")
    reasons = sorted({note["code"] for note in receipt["material_notes"] if note["code"] in INCOMPLETE_CODES})
    if receipt["not_complete_because"] != reasons or receipt["complete"] is not (not reasons):
        problems.append("complete / not_complete_because disagree with material_notes")
    for row in receipt["materials"]:
        if row["locators_returned"] != sum(len(item["locators"]) for item in row["locators"]):
            problems.append(f"material {row['id']}: locators_returned does not count its locators")
    return problems + _archive(receipt["llm_calls"]) + _references(receipt, material_ids)


# ---- the published form of SHAPE (GET /v2/contract) ----

# bump when SHAPE changes in a way a caller validating against the schema would notice
RECEIPT_SCHEMA_VERSION = "mechevidence.evaluate.v2.receipt.1"
_HEX = {"type": "string", "pattern": "^[0-9a-f]{64}$", "minLength": 64, "maxLength": 64}
_SIMPLE = {"str": {"type": "string"}, "int": {"type": "integer"}, "num": {"type": "number"},
           "bool": {"type": "boolean"}, "hex": _HEX, "any": {}}


def _json_schema(spec, top: bool = False) -> dict:
    if isinstance(spec, str):
        return dict(_SIMPLE[spec])
    if spec is _CALL and not top:
        return {"$ref": "#/$defs/llm_call"}
    kind = spec[0]
    if kind == "null":
        return {"anyOf": [{"type": "null"}, _json_schema(spec[1])]}
    if kind == "enum":
        return {"type": "string", "enum": sorted(spec[1])}
    if kind == "list":
        return {"type": "array", "items": _json_schema(spec[1])}
    optional = spec[2] if len(spec) > 2 else set()
    return {"type": "object", "properties": {name: _json_schema(inner) for name, inner in spec[1].items()},
            "required": [name for name in spec[1] if name not in optional]}


REUSE_KEY_SHAPE = ("obj", {"contract_version": "str", "evaluator": _EVALUATOR, "reuse_key": "hex",
                           "materials": ("list", _ECHO)})


def reuse_key_schema() -> dict:
    """The POST /v2/reuse-key answer as a JSON Schema (draft 2020-12)."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **_json_schema(REUSE_KEY_SHAPE)}


def receipt_schema() -> dict:
    """SHAPE as a JSON Schema (draft 2020-12); one model call is its $defs/llm_call. "integer" means a JSON
    integer, never a float such as 1.0 (the table's "int"). The cross-field rules are not part of it."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": RECEIPT_SCHEMA_VERSION,
            **_json_schema(SHAPE), "$defs": {"llm_call": _json_schema(_CALL, top=True)}}
