"""Is this 200 receipt a well-formed v2 answer? Pure checks of structure, verdict domains, the archive of raw model
answers against the request counts, and the hashes of the archived text. A receipt that fails is never adopted or
cached; the caller keeps the raw receipt for audit (contract.adopt)."""
from __future__ import annotations

import hashlib
import re

OVERALL = {"supported", "partial", "insufficient"}
CLAIM_VERDICT = {"mixed", "supported", "refuted", "insufficient"}
JOIN_VERDICT = {"supported", "compatible_only", "unestablished"}
RELATION = {"supports", "refutes", "compatible"}
BASIS = {"intervention", "association", "review", "claim"}
SCOPE_STATUS = {"matched", "mismatched", "unknown"}
ASSESSABILITY = {"assessable", "limited", "unassessable"}
STAGES = {"decompose", "review"}
DELIVERED = {"yes", "no", "unknown"}
INCOMPLETE_CODES = ("pending_budget", "review_failed")

_TYPES = {"contract_version": str, "evaluator": dict, "reuse_key": str, "complete": bool,
          "not_complete_because": list, "audit_complete": bool, "scope": dict, "materials": list, "overall": str,
          "rationale": str, "claims": list, "joins": list, "observations": list, "conflicts": list,
          "unresolved_counterevidence": list, "gaps": list, "independence": dict, "material_notes": list,
          "assessability": dict, "llm_calls": list}


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hex(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _calls(calls: list, problems: list[str]) -> None:
    for number, call in enumerate(calls):
        where = f"llm_calls[{number}]"
        if not isinstance(call, dict):
            problems.append(f"{where} is not an object")
            continue
        outputs = call.get("outputs")
        if call.get("stage") not in STAGES or not _int(call.get("requests")) or call["requests"] < 0 \
                or not isinstance(outputs, list) or not isinstance(call.get("usage"), dict):
            problems.append(f"{where} lacks stage/requests/outputs/usage")
            continue
        if len(outputs) != call["requests"]:
            problems.append(f"{where} counts {call['requests']} requests but archives {len(outputs)} outputs")
        for position, out in enumerate(outputs, 1):
            problems += _output(f"{where}.outputs[{position - 1}]", position, out)
        if call.get("error") is None and outputs and not (isinstance(outputs[-1], dict)
                                                          and outputs[-1].get("status") == 200):
            problems.append(f"{where} succeeded but its last request was not answered 200")


def _output(where: str, position: int, out) -> list[str]:
    if not isinstance(out, dict):
        return [f"{where} is not an object"]
    problems = []
    if out.get("attempt") != position:
        problems.append(f"{where}.attempt is {out.get('attempt')!r}, expected {position} (contiguous from 1)")
    if out.get("delivered") not in DELIVERED or not isinstance(out.get("truncated"), bool):
        problems.append(f"{where} lacks delivered/truncated")
    elif out["delivered"] == "yes":
        content = out.get("content")
        if not isinstance(content, str) or not _int(out.get("status")) or not _hex(out.get("sha256")):
            problems.append(f"{where} was answered but has no archived text, status or sha256")
        elif not out["truncated"] and hashlib.sha256(content.encode("utf-8")).hexdigest() != out["sha256"]:
            problems.append(f"{where}.content does not hash to its sha256")
    elif out.get("content") is not None or out.get("sha256") is not None:
        problems.append(f"{where} was not answered but carries an answer")
    return problems


def _verdicts(receipt: dict, problems: list[str]) -> None:
    if receipt["overall"] not in OVERALL:
        problems.append(f"overall {receipt['overall']!r} is not one of {sorted(OVERALL)}")
    for name, domain in (("claims", CLAIM_VERDICT), ("joins", JOIN_VERDICT)):
        for row in receipt[name]:
            if not isinstance(row, dict) or row.get("verdict") not in domain or not isinstance(row.get("id"), str):
                problems.append(f"{name} row {row if not isinstance(row, dict) else row.get('id')!r}: verdict outside {sorted(domain)}")
    for row in receipt["observations"]:
        if not isinstance(row, dict) or row.get("relation") not in RELATION or row.get("basis") not in BASIS \
                or row.get("scope_status") not in SCOPE_STATUS:
            problems.append(f"observation {row.get('id') if isinstance(row, dict) else row!r}: relation/basis/scope_status outside their domain")
    assess = receipt["assessability"]
    if assess.get("status") not in ASSESSABILITY or not isinstance(assess.get("limitations"), list):
        problems.append("assessability lacks a status in its domain or a limitations list")


def receipt_problems(receipt, scope: dict) -> list[str]:
    """Empty list = well formed. `scope` is the request's normalised scope, which the receipt must echo."""
    if not isinstance(receipt, dict):
        return ["the receipt is not an object"]
    problems = [f"{name} is missing or not a {kind.__name__}" for name, kind in _TYPES.items()
                if not isinstance(receipt.get(name), kind)]
    if problems:
        return problems
    if not _hex(receipt["reuse_key"]):
        problems.append("reuse_key is not a sha256")
    if receipt["scope"] != scope:
        problems.append("scope does not echo the request's scope")
    _verdicts(receipt, problems)
    notes = receipt["material_notes"]
    if not all(isinstance(note, dict) and isinstance(note.get("code"), str) for note in notes):
        problems.append("material_notes has an entry without a code")
        return problems
    reasons = sorted({note["code"] for note in notes if note["code"] in INCOMPLETE_CODES})
    if receipt["not_complete_because"] != reasons or receipt["complete"] is not (not reasons):
        problems.append("complete / not_complete_because disagree with material_notes")
    _calls(receipt["llm_calls"], problems)
    return problems
