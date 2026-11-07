"""The committed reuse-key and service-error fixtures say what the radar must see; here they are checked for sense."""
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / "examples" / "contract-v2"


def load(name):
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


def test_a_key_differs_exactly_when_an_input_of_the_evaluation_differs():
    document = load("reuse_keys.json")
    base = document["base"]["reuse_key"]
    assert len(document["cases"]) == 11
    for case in document["cases"]:
        assert (case["reuse_key"] != base) is case["differs_from_base"], case["name"]
    same = {case["name"] for case in document["cases"] if not case["differs_from_base"]}
    assert same == {"same_request_again", "budget_and_authorization_are_not_inputs"}
    upgraded = next(case for case in document["cases"] if case["name"] == "evaluator_upgraded")
    assert upgraded["evaluator_id"] != document["base"]["evaluator_id"] and upgraded["request"] == document["base"]["request"]


def test_the_service_errors_after_a_valid_request():
    errors = {item["name"]: item for item in load("service_errors.json")}
    changed = errors["evaluator_changed"]
    assert changed["status"] == 409 and changed["body"]["error_code"] == "evaluator_changed"
    assert changed["body"]["llm_calls"] == [] and "overall" not in changed["body"]
    assert changed["body"]["evaluator"]["evaluator_id"] == errors["current_evaluator"]["body"]["evaluator"]["evaluator_id"]
    revoked = errors["authorization_revoked_before_the_second_request"]
    assert revoked["status"] == 502 and revoked["body"]["error_code"] == "authorization_blocked"
    assert [call["requests"] for call in revoked["body"]["llm_calls"]] == [1, 0]
    oversize = errors["raw_answer_over_256_KiB"]
    out = oversize["body"]["llm_calls"][0]["outputs"][0]
    assert oversize["status"] == 502 and oversize["body"]["error_code"] == "output_oversize"
    assert out["truncated"] is True and len(out["sha256"]) == 64 and "overall" not in oversize["body"]
