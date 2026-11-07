"""examples/contract-v2/*.json: the exact request/receipt examples and the negative cases the radar's tests reuse.

The committed files must equal a fresh build (a change of prompts, rules or contract is then a visible diff).
Regenerate after an intended change:  MECH_REGEN_FIXTURES=1 python -m pytest tests/test_contract_fixtures.py
"""
import json
import os
from pathlib import Path

import pytest
from contract_cases import dumps, fixture_documents

from mechevidence.contract import adopt, parse_request

DIRECTORY = Path(__file__).resolve().parents[1] / "examples" / "contract-v2"


def test_the_committed_fixtures_are_the_ones_the_code_produces(monkeypatch):
    documents = fixture_documents(monkeypatch)
    if os.environ.get("MECH_REGEN_FIXTURES") == "1":
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        for name, document in documents.items():
            (DIRECTORY / name).write_text(dumps(document), encoding="utf-8", newline="\n")
    assert sorted(path.name for path in DIRECTORY.glob("*.json")) == sorted(documents)
    for name, document in documents.items():
        assert json.loads((DIRECTORY / name).read_text(encoding="utf-8")) == json.loads(dumps(document)), name


def load(name):
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


def test_the_committed_request_is_valid_and_its_receipt_belongs_to_it():
    raw, receipt = load("request.json"), load("receipt.json")
    parsed = parse_request(raw)
    assert [item["binding_sha256"] for item in receipt["materials"]] == parsed.binding_hashes
    assert receipt["evaluator"]["evaluator_id"] == raw["expected_evaluator_id"] == load("evaluator.json")["evaluator"]["evaluator_id"]


def outcome_cases():
    return load("receipt_outcomes.json")["cases"]


INVALID = ("failed", "receipt_invalid")
EXPECTED = {
    "exact": ("completed", None),
    "reuse_key_differs": ("failed", "receipt_mismatch"),
    "binding_echo_swapped": ("failed", "receipt_mismatch"),
    "binding_echo_other_text": ("failed", "receipt_mismatch"),
    "evaluator_id_differs": ("failed", "receipt_mismatch"),
    "evaluator_fields_edited_id_kept": ("failed", "receipt_mismatch"),
    "not_complete_budget_cut": ("completed", None),
    "audit_incomplete": ("failed", "audit_incomplete"),
    "truncated_answer": ("failed", "audit_incomplete"),
    "outputs_missing": INVALID,
    "output_text_edited_hash_kept": INVALID,
    "output_hash_not_hex": INVALID,
    "requests_more_than_outputs": INVALID,
    "outputs_more_than_requests": INVALID,
    "attempt_numbers_not_contiguous": INVALID,
    "unanswered_attempt_carries_an_answer": INVALID,
    "successful_call_not_answered_200": INVALID,
    "llm_calls_not_a_list": INVALID,
    "invalid_overall_verdict": INVALID,
    "invalid_claim_verdict": INVALID,
    "invalid_join_verdict": INVALID,
    "invalid_observation_relation": INVALID,
    "complete_false_without_a_reason": INVALID,
    "complete_true_with_a_budget_note": INVALID,
    "scope_echo_changed": INVALID,
    "required_field_missing": INVALID,
    "assessability_status_invalid": INVALID,
    "stage_is_a_list": INVALID, "delivered_is_a_list": INVALID, "attempt_is_a_bool": INVALID,
    "attempt_is_a_float": INVALID, "claim_verdict_is_a_list": INVALID, "observation_relation_is_an_object": INVALID,
    "observation_cites_a_material_outside_the_request": INVALID, "observation_passage_of_another_material": INVALID,
    "observation_targets_a_missing_claim": INVALID, "claim_cites_a_missing_observation": INVALID,
    "echo_has_an_extra_non_object_element": INVALID,
    "echo_has_an_extra_object_element": ("failed", "receipt_mismatch"),
    "echo_locators_changed": ("failed", "receipt_mismatch"),
    "success_with_an_unanswered_output": INVALID, "answered_200_but_error_set": INVALID,
    "answered_without_text": INVALID, "global_index_not_contiguous": INVALID, "attempt_id_reused": INVALID,
    "decision_id_empty": INVALID, "attempt_id_missing": INVALID, "reuse_key_is_a_number": INVALID,
    "receipt_is_a_list": INVALID, "receipt_is_a_string": INVALID, "receipt_is_null": INVALID,
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_what_the_radar_does_with_each_receipt(name):
    """The radar's rule (r219 N17): a receipt that does not match its request is failed/receipt_mismatch, the raw
    receipt is kept as an audit row, and no conclusion is usable; an incomplete audit is failed too; a receipt
    with complete=false is adoptable for this analysis but never cacheable."""
    document = load("receipt_outcomes.json")
    case = next(item for item in document["cases"] if item["name"] == name)
    outcome = adopt(document["request"], case["receipt"], document["expected_evaluator_id"])
    assert json.loads(json.dumps(outcome)) == case["outcome"]
    status, reason = EXPECTED[name]
    assert outcome["result"]["status"] == status and outcome["result"].get("reason") == reason
    if status == "failed":
        assert outcome["rejected"]["response"] == case["receipt"] and "response" not in outcome["result"]
        if reason == "receipt_mismatch":
            assert outcome["rejected"]["mismatch"] == outcome["result"]["mismatch"] != []
        if reason == "receipt_invalid":
            assert outcome["rejected"]["invalid"] == outcome["result"]["invalid"] != []
    else:
        assert outcome["rejected"] is None
        assert outcome["result"]["cacheable"] is (name == "exact")


def test_every_outcome_case_is_asserted():
    assert {case["name"] for case in outcome_cases()} == set(EXPECTED)

