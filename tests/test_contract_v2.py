"""POST /v2/evaluate: the version-bound contract. Positive paths, the refusals, the evaluator id, the reuse key and
the check before every model request. The model is a stub; nothing touches the network."""
import copy

import pytest
from contract_cases import (
    Authorizer,
    HookedStub,
    bound,
    key_of,
    negative_requests,
    oversize_outputs,
    request,
    sha,
    two_material_request,
)
from fastapi.testclient import TestClient
from stub import Stub

from mechevidence import identity, llm
from mechevidence.contract import canonical, evaluator_id_of, parse_request, reuse_key
from mechevidence.identity import current_evaluator
from mechevidence.models_v2 import MaterialV2
from mechevidence.service import create_app
from mechevidence.v2 import evaluate_v2

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "test-token")
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT_ID", "MECH_AUTHORIZATION_URL", "MECH_AUTHORIZATION_TOKEN",
                 "MECH_OUTCOME_URL"):
        monkeypatch.delenv(name, raising=False)


def client(stub=None, authorizer=None):
    return TestClient(create_app(stub or HookedStub(), authorizer or Authorizer()))


# ---- positive ----

def test_a_bound_request_gets_a_receipt_that_names_evaluator_key_and_materials():
    stub, authorizer = HookedStub(), Authorizer()
    response = client(stub, authorizer).post("/v2/evaluate", json=request(), headers=AUTH)
    body = response.json()
    assert response.status_code == 200 and body["overall"] == "supported"
    assert body["contract_version"] == "mechevidence.evaluate.v2"
    assert body["evaluator"] == current_evaluator()
    assert body["complete"] is True and body["audit_complete"] is True and body["not_complete_because"] == []
    assert body["reuse_key"] == key_of(request(), current_evaluator())
    echo = body["materials"][0]
    assert echo["id"] == "E1" and echo["binding_checked"] == "self_consistent" and echo["locators_returned"] == 1
    assert echo["locators"][0]["segment"] == 1 and echo["locators"][0]["locators"][0]["json_pointer"] == "/abstractText"
    assert body["scope"]["disease"] == {"state": "not_required"}
    # the raw model answer of every call travels back, whole
    assert [len(call["outputs"]) for call in body["llm_calls"]] == [1, 1]
    assert all(out["truncated"] is False and out["sha256"] == sha(out["content"]) for call in body["llm_calls"]
               for out in call["outputs"])


def test_the_evaluation_is_the_v1_evaluation_of_the_same_input():
    from stub import case

    from mechevidence import evaluate

    v2 = evaluate_v2(request(), HookedStub(), Authorizer())
    v1 = evaluate(case(scope={"species": "mouse", "system": "S", "disease": None}, materials=[{
        **{key: value for key, value in bound().items() if key != "binding"}}]), Stub())
    for field in ("overall", "rationale", "claims", "joins", "observations", "gaps", "conflicts", "assessability"):
        assert v2[field] == v1[field]


def test_get_evaluator_and_health_agree_and_health_needs_no_token():
    http = client()
    body = http.get("/v2/evaluator", headers=AUTH).json()
    assert body["evaluator"]["evaluator_id"] == current_evaluator()["evaluator_id"]
    assert body["authorization"] == {"mode": "per_attempt_trusted_endpoint", "configured": True}
    unconfigured = TestClient(create_app(HookedStub())).get("/v2/evaluator", headers=AUTH).json()
    assert unconfigured["authorization"]["configured"] is False
    health = http.get("/health").json()
    assert health == {"status": "ok", "contracts": ["mechevidence.evaluate.v1", "mechevidence.evaluate.v2"],
                      "evaluator_id": body["evaluator"]["evaluator_id"]}
    assert http.get("/v2/evaluator").status_code == 401


def test_a_request_with_a_decomposition_and_no_material_calls_no_model_and_needs_no_authorization():
    stub, authorizer = HookedStub(), Authorizer()
    raw = request(materials=[], claims=[{"text": "A changes M", "kind": "causal"}])
    result = evaluate_v2(raw, stub, authorizer)
    assert result["overall"] == "insufficient" and result["llm_calls"] == [] and authorizer.checks == []
    assert result["complete"] is True


def test_a_budget_cut_answer_is_not_complete():
    result = evaluate_v2(request(max_llm_calls=1), HookedStub(), Authorizer())
    assert result["complete"] is False and result["not_complete_because"] == ["pending_budget"]


# ---- evaluator identity ----

PLACEHOLDER_FIELDS = {
    "batch_size": 3, "contract_version": "mechevidence.evaluate.v2",
    "model_config": {"endpoint_id": "deepseek-public", "max_tokens": 4096, "model": "deepseek-v4-pro",
                     "provider": "deepseek", "temperature": 0, "thinking": False},
    "ordering_rule": "mechevidence.material_order.v1",
    "prompts": {"decompose_sha256": "0" * 64, "review_sha256": "1" * 64},
    "rules_version": "mechevidence.aggregate.v1", "service": "mechevidence"}


def test_evaluator_id_matches_the_design_test_vector():
    assert canonical(PLACEHOLDER_FIELDS).startswith('{"batch_size":3,"contract_version":"mechevidence.evaluate.v2"')
    assert evaluator_id_of(PLACEHOLDER_FIELDS) == "e0876a1faac6f3a9588e0caad5bf14825fac496fad8156eb176965f27722c7b2"
    # code_version and the id itself are not part of it
    assert evaluator_id_of({**PLACEHOLDER_FIELDS, "code_version": "abc", "evaluator_id": "x"}) == evaluator_id_of(PLACEHOLDER_FIELDS)


def test_evaluator_id_follows_prompts_rules_model_and_endpoint_not_the_build(monkeypatch):
    base = current_evaluator()
    assert base["rules_version"] == "mechevidence.aggregate.v1" and base["ordering_rule"] == "request_order"
    assert base["prompts"]["decompose_sha256"] != base["prompts"]["review_sha256"]
    monkeypatch.setenv("MECHEVIDENCE_CODE_VERSION", "other-build")
    assert current_evaluator()["evaluator_id"] == base["evaluator_id"]  # a refactor does not void the cache
    for name, value in (("LLM_MODEL", "dev-qwen"), ("LLM_PROVIDER", "vllm"), ("LLM_ENDPOINT_ID", "dev-vllm-qwen")):
        with monkeypatch.context() as scoped:
            scoped.setenv(name, value)
            assert current_evaluator()["evaluator_id"] != base["evaluator_id"], name
    for attribute, value in (("REVIEW", identity.REVIEW + " "), ("DECOMPOSE", identity.DECOMPOSE + " "),
                             ("RULES_VERSION", "mechevidence.aggregate.v2"), ("BATCH_SIZE", 4)):
        with monkeypatch.context() as scoped:
            scoped.setattr(identity, attribute, value)
            assert current_evaluator()["evaluator_id"] != base["evaluator_id"], attribute


def test_an_evaluator_other_than_the_expected_one_is_refused_before_anything_is_evaluated(monkeypatch):
    stub, authorizer = HookedStub(), Authorizer()
    http = client(stub, authorizer)
    wrong = sha("an older evaluator")
    response = http.post("/v2/evaluate", json=request(expected_evaluator_id=wrong), headers=AUTH)
    body = response.json()
    assert response.status_code == 409 and body["error_code"] == "evaluator_changed"
    assert body["evaluator"]["evaluator_id"] == current_evaluator()["evaluator_id"] and body["expected_evaluator_id"] == wrong
    assert body["llm_calls"] == [] and stub.attempts == [] and authorizer.checks == []
    ok = http.post("/v2/evaluate", json=request(expected_evaluator_id=current_evaluator()["evaluator_id"]), headers=AUTH)
    assert ok.status_code == 200
    # an upgrade between fetching the identity and sending: same request, new model -> 409
    expected = current_evaluator()["evaluator_id"]
    monkeypatch.setenv("LLM_MODEL", "upgraded-model")
    assert http.post("/v2/evaluate", json=request(expected_evaluator_id=expected), headers=AUTH).status_code == 409


# ---- the reuse key ----

def key(raw):
    return key_of(raw, current_evaluator())


def test_the_key_is_stable_and_ignores_what_does_not_change_the_answer():
    base = key(request())
    assert key(copy.deepcopy(request())) == base
    for same in (request(max_llm_calls=3), request(authorization_ref="another-ref"),
                 request(expected_evaluator_id=sha("x"))):
        assert key(same) == base


@pytest.mark.parametrize("name,change", [
    ("question", lambda raw: raw.update(question="Does A cause E in human system S?")),   # N4
    ("mechanism", lambda raw: raw.update(mechanism="A changes M.")),
    ("species_scope", lambda raw: raw["scope"].update(species={"state": "stated", "value": "human"})),   # N3
    ("scope_unknown_vs_not_required", lambda raw: raw["scope"].update(disease={"state": "unknown"})),
    ("claims_given", lambda raw: raw.update(claims=[{"text": "A changes M", "kind": "causal"}])),
    ("title", lambda raw: raw["materials"][0].update(title="Another title")),             # N13
    ("alias", lambda raw: raw["materials"][0].update(id="E7")),
    ("source_kind", lambda raw: raw["materials"][0].update(source_kind="review")),
    ("document_type", lambda raw: raw["materials"][0].update(document_type="conference_abstract")),
    ("text_level", lambda raw: raw["materials"][0].update(text_level="full")),
    ("identifiers_added", lambda raw: raw["materials"][0].update(identifiers={"pmid": "123456"})),   # N14
    ("another_text", lambda raw: raw.update(materials=[bound(text="A changed M in mouse S cells.")])),
    ("another_processor", lambda raw: raw.update(materials=[bound(processor="datafabric.clean_text.v13")])),   # N2
])
def test_every_input_the_evaluator_uses_is_in_the_key(name, change):
    raw = copy.deepcopy(request())
    change(raw)
    assert key(raw) != key(request()), name


def test_order_is_part_of_the_key_in_phase_one():                                            # N15
    first, second = two_material_request(), two_material_request()
    second["materials"].reverse()
    assert key(first) != key(second)


def test_the_scope_strings_map_to_unknown_never_to_not_required():                           # N18
    strings = request(scope={"species": "", "system": "  ", "disease": "unknown"})
    explicit = request(scope={dim: {"state": "unknown"} for dim in ("species", "system", "disease")})
    assert key(strings) == key(explicit)
    assert key(strings) != key(request(scope={dim: {"state": "not_required"} for dim in ("species", "system", "disease")}))
    result = evaluate_v2(strings, HookedStub(), Authorizer())
    assert result["scope"] == {dim: {"state": "unknown"} for dim in ("species", "system", "disease")}
    assert result["overall"] != "supported" and all(row["scope_status"] != "matched" for row in result["observations"])


def test_the_key_covers_every_field_of_a_material():
    """A field added to MaterialV2 without a perturbation here makes this test fail, so it cannot slip out of the key."""
    perturbed = {"id", "title", "text", "source_kind", "document_type", "text_level", "identifiers", "binding"}
    assert set(MaterialV2.model_fields) == perturbed
    base = request()
    shifts = {
        "id": lambda m: m.update(id="E5"), "title": lambda m: m.update(title="t"),
        "source_kind": lambda m: m.update(source_kind="review"), "document_type": lambda m: m.update(document_type=None),
        "text_level": lambda m: m.update(text_level="excerpt"), "identifiers": lambda m: m.update(identifiers={"doi": "10.1000/xyz"}),
    }
    for field, shift in shifts.items():
        raw = copy.deepcopy(base)
        shift(raw["materials"][0])
        assert key(raw) != key(base), field
    other = request(materials=[bound(text="A changed M in mouse S cells. Blocking M prevented it.")])
    assert key(other) != key(base)  # text and binding together


def test_keys_of_the_two_material_request_use_the_service_group_rule():
    parsed = parse_request(two_material_request())
    assert reuse_key(parsed, current_evaluator()) == key(two_material_request())


# ---- refusals (also committed as examples/contract-v2/negative_requests.json) ----

@pytest.mark.parametrize("case", negative_requests(), ids=lambda case: case["name"])
def test_the_service_refuses_what_the_contract_forbids(case):
    stub, authorizer = HookedStub(), Authorizer()
    response = client(stub, authorizer).post("/v2/evaluate", json=case["request"], headers=AUTH)
    body = response.json()
    assert response.status_code == case["status"] and body["error_code"] == case["error_code"], body
    assert case.get("message_contains", "") in body["message"]
    assert "overall" not in body and stub.attempts == [] and authorizer.checks == []   # nothing evaluated, nothing sent


def test_a_json_syntax_error_is_input_invalid():
    response = client().post("/v2/evaluate", content=b"{not json", headers=AUTH | {"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["error_code"] == "input_invalid"


# ---- raw answers that cannot be archived whole ----

def test_an_answer_over_the_archive_limit_fails_closed_with_the_whole_text_hash():
    stub, authorizer = HookedStub(outputs=oversize_outputs()), Authorizer()
    response = client(stub, authorizer).post("/v2/evaluate", json=request(), headers=AUTH)
    body = response.json()
    assert response.status_code == 502 and body["error_code"] == "output_oversize"
    assert "overall" not in body and "reuse_key" not in body and "complete" not in body
    out = body["llm_calls"][0]["outputs"][0]
    assert out["truncated"] is True and out["sha256"] == sha("x" * (llm.OUTPUT_ARCHIVE_LIMIT + 10))
    assert stub.attempts == ["decompose"]                  # nothing further was sent after it



# ---- the service owns the key and its receipt rules ----

def test_reuse_key_answers_what_evaluate_would_key_and_echo_without_a_model_call():
    stub = HookedStub()
    response = client(stub).post("/v2/reuse-key", json=request(), headers=AUTH)
    body = response.json()
    assert response.status_code == 200 and stub.attempts == []
    assert body["evaluator"] == current_evaluator() and body["reuse_key"] == key_of(request(), current_evaluator())
    receipt = client().post("/v2/evaluate", json=request(), headers=AUTH).json()
    assert (body["reuse_key"], body["materials"]) == (receipt["reuse_key"], receipt["materials"])
    assert client().post("/v2/reuse-key", json={**request(), "extra": 1}, headers=AUTH).status_code == 422


def test_a_receipt_that_breaks_the_receipt_rules_is_never_returned(monkeypatch):
    from mechevidence import v2

    original = v2.material_echo
    monkeypatch.setattr(v2, "material_echo", lambda parsed: [dict(row, locators_returned=99) for row in original(parsed)])
    response = client().post("/v2/evaluate", json=request(), headers=AUTH)
    assert response.status_code == 502 and response.json()["error_code"] == "receipt_invalid"
