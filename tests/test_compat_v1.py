"""POST /v1/evaluate stays a compatibility path: the same evaluation as before, unknown fields no longer dropped in
silence, and the answer says what it is not (unbound, not reusable)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from stub import Stub, case, material

from mechevidence import evaluate
from mechevidence.identity import current_evaluator
from mechevidence.service import create_app

AUTH = {"Authorization": "Bearer test-token"}
GOLDEN = json.loads((Path(__file__).parent / "golden" / "v1_a4624ca.json").read_text(encoding="utf-8"))
V1_ONLY = {"contract_version", "evaluator", "binding_state", "reusable"}


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "test-token")


def post(body, stub=None):
    return TestClient(create_app(stub or Stub())).post("/v1/evaluate", json=body, headers=AUTH)


def stubs():
    from stub import good_observations
    return {"radar_body": Stub(), "two_materials": Stub([good_observations(), []])}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_the_evaluation_is_the_one_a4624ca_computed(name):
    """GOLDEN was captured from the code before this contract existed, for the request body the radar sends today."""
    request, expected = GOLDEN[name]["request"], GOLDEN[name]["response"]
    assert json.loads(json.dumps(evaluate(request, stubs()[name]))) == expected
    body = post(request, stubs()[name]).json()
    assert {key: value for key, value in body.items() if key not in V1_ONLY} == expected


def test_the_v1_answer_labels_itself_a_compatibility_result():
    body = post(case()).json()
    assert body["contract_version"] == "mechevidence.evaluate.v1"
    assert body["binding_state"] == "unbound" and body["reusable"] is False
    assert body["evaluator"] == current_evaluator()
    assert all("outputs" not in call for call in body["llm_calls"])   # raw answers are a v2 receipt feature


@pytest.mark.parametrize("where", ["top", "material", "scope", "claim", "join"])
def test_a_field_the_contract_does_not_name_is_rejected_not_dropped(where):
    request = case(claims=[{"text": "A changes M"}, {"text": "M is necessary for E"}],
                   joins=[{"text": "A acts through M", "from_claim": 1, "to_claim": 2}])
    target = {"top": request, "material": request["materials"][0], "scope": request["scope"],
              "claim": request["claims"][0], "join": request["joins"][0]}[where]
    target["version_id"] = "a" * 64
    stub = Stub()
    response = post(request, stub)
    body = response.json()
    assert response.status_code == 422 and body["error_code"] == "field_not_in_contract"
    assert "version_id" in body["message"] and "/v2/evaluate" in body["message"]
    assert stub.calls == []


def test_the_in_process_entry_point_rejects_them_too():
    with pytest.raises(ValueError, match="version"):
        evaluate(case(materials=[material(version="original-v1")]), Stub())


def test_model_outputs_stay_lenient_about_extra_keys():
    """Only request models forbid extras: a model that adds a key to a claim or to an observation's context must not
    void the whole answer."""
    from stub import DECOMPOSITION, good_observations
    decomposition = {**DECOMPOSITION, "claims": [{**claim, "confidence": "high"} for claim in DECOMPOSITION["claims"]]}
    observations = good_observations()
    observations[0]["context"]["strain"] = "C57BL/6"
    assert evaluate(case(), Stub([observations], decomposition))["overall"] == "supported"


def test_health_lists_both_contracts():
    health = TestClient(create_app(Stub())).get("/health").json()
    assert health["contracts"] == ["mechevidence.evaluate.v1", "mechevidence.evaluate.v2"]
