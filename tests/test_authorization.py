"""The check before every model request and the hand-over of every outcome (model_attempt_authorization.v1 /
model_attempt_outcome.v1): the real endpoint client against a stand-in, and whole evaluations through the app."""
import json

import httpx
import pytest
from contract_cases import (
    AUTH_REF,
    FakeEndpoint,
    HookedStub,
    key_of,
    request,
    two_material_request,
)
from fastapi.testclient import TestClient
from stub import DECOMPOSITION, good_observations
from test_llm import FakePost, use

from mechevidence import authorization
from mechevidence.identity import current_evaluator
from mechevidence.llm import SendBlocked, chat_json
from mechevidence.service import create_app
from mechevidence.v2 import evaluate_v2

AUTH = {"Authorization": "Bearer test-token"}
ATTEMPT = {"id": "a" * 32, "stage": "review", "index": 2, "request_body_sha256": "b" * 64, "model": "m", "endpoint_id": "e"}
KEY, EVALUATOR = "c" * 64, "d" * 64


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "test-token")
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MECH_AUTHORIZATION_URL", "https://radar.internal/internal/v1/model-attempts/authorize")
    monkeypatch.setenv("MECH_AUTHORIZATION_TOKEN", "t")
    monkeypatch.setenv("MECH_OUTCOME_URL", "https://radar.internal/internal/v1/model-attempts/{attempt_id}/outcome")


def good_answer(**changes):
    return {"contract_version": "model_attempt_authorization.v1", "valid": True, "decision_id": "dec-1",
            "attempt_id": ATTEMPT["id"], "evaluation_key": KEY, "evaluator_id": EVALUATOR,
            "expires_at": "2999-01-01T00:00:00Z", **changes}


class Wire:
    """httpx.post stand-in: records requests, answers with `reply` (a dict, a Response or an exception)."""

    def __init__(self, reply):
        self.reply, self.posts = reply, []

    def __call__(self, url, json, headers, **kwargs):
        self.posts.append((url, json, headers, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, httpx.Response) else httpx.Response(200, json=self.reply)


def authorize(monkeypatch, reply):
    wire = Wire(reply)
    monkeypatch.setattr(authorization.httpx, "post", wire)
    return wire, authorization.TrustedEndpoint().authorize("ref-1", KEY, EVALUATOR, ATTEMPT)


def test_a_decision_that_echoes_this_attempt_is_accepted_and_the_request_is_exactly_the_agreed_one(monkeypatch):
    wire, decision = authorize(monkeypatch, good_answer())
    assert decision["decision_id"] == "dec-1"
    url, sent, headers, kwargs = wire.posts[0]
    assert url == "https://radar.internal/internal/v1/model-attempts/authorize"
    assert sent == {"contract_version": "model_attempt_authorization.v1", "service": "mechevidence",
                    "authorization_ref": "ref-1", "evaluation_key": KEY, "evaluator_id": EVALUATOR, "attempt": ATTEMPT}
    assert headers == {"Authorization": "Bearer t"} and kwargs["follow_redirects"] is False


@pytest.mark.parametrize("reply,code", [
    ({"valid": True}, "authorization_malformed"),                                         # a bare valid:true
    (good_answer(decision_id=""), "authorization_malformed"),
    (good_answer(valid=False), "authorization_refused"),
    (good_answer(valid="yes"), "authorization_refused"),
    (good_answer(attempt_id="e" * 32), "authorization_mismatch"),                          # another attempt
    (good_answer(evaluation_key="f" * 64), "authorization_mismatch"),                      # another evaluation
    (good_answer(evaluator_id="f" * 64), "authorization_mismatch"),
    (good_answer(contract_version="model_attempt_authorization.v0"), "authorization_mismatch"),
    (good_answer(expires_at="2000-01-01T00:00:00Z"), "authorization_expired"),
    (good_answer(expires_at="2999-01-01T00:00:00"), "authorization_expired"),             # no timezone: not trusted
    (good_answer(expires_at="tomorrow"), "authorization_expired"),
    (good_answer(expires_at=None), "authorization_expired"),
    ([1], "authorization_malformed"),
    (httpx.Response(200, content=b"nope"), "authorization_malformed"),
    (httpx.Response(500), "authorization_refused"),
    (httpx.Response(403, json=good_answer()), "authorization_refused"),
    (httpx.ReadTimeout("slow"), "authorization_unreachable"),
    (httpx.ConnectError("down"), "authorization_unreachable"),
])
def test_anything_but_a_valid_matching_unexpired_decision_blocks_the_request(monkeypatch, reply, code):
    with pytest.raises(SendBlocked) as caught:
        authorize(monkeypatch, reply)
    assert caught.value.reason == code


@pytest.mark.parametrize("missing", ["MECH_AUTHORIZATION_URL", "MECH_AUTHORIZATION_TOKEN", "MECH_OUTCOME_URL"])
def test_an_endpoint_that_is_not_fully_configured_never_authorizes(monkeypatch, missing):
    monkeypatch.delenv(missing)
    wire = Wire(good_answer())
    monkeypatch.setattr(authorization.httpx, "post", wire)
    endpoint = authorization.TrustedEndpoint()
    assert endpoint.configured() is False
    with pytest.raises(SendBlocked) as caught:
        endpoint.authorize("ref-1", KEY, EVALUATOR, ATTEMPT)
    assert caught.value.reason == "authorization_not_configured" and wire.posts == []


ENTRY = {"attempt": 1, "delivered": "yes", "status": 200, "error": None, "content": "{}", "sha256": "9" * 64,
         "truncated": False}


def test_the_outcome_is_posted_to_the_attempt_s_own_url_and_must_be_recorded(monkeypatch):
    wire = Wire({"recorded": True})
    monkeypatch.setattr(authorization.httpx, "post", wire)
    authorization.TrustedEndpoint().settle("ref-1", good_answer(), ENTRY)
    url, sent, _, _ = wire.posts[0]
    assert url == f"https://radar.internal/internal/v1/model-attempts/{ATTEMPT['id']}/outcome"
    assert sent["contract_version"] == "model_attempt_outcome.v1" and sent["decision_id"] == "dec-1"
    assert sent["attempt_id"] == ATTEMPT["id"] and sent["response_sha256"] == "9" * 64 and sent["truncated"] is False
    for reply in ({"recorded": False}, {}, httpx.Response(500), httpx.ReadTimeout("slow")):
        monkeypatch.setattr(authorization.httpx, "post", Wire(reply))
        with pytest.raises(Exception):  # noqa: B017 - any failure to record is a failure
            authorization.TrustedEndpoint().settle("ref-1", good_answer(), ENTRY)


# ---- whole evaluations ----

def test_every_request_is_authorized_for_this_evaluation_and_settled_before_the_next():
    endpoint = FakeEndpoint()
    events = []
    original_authorize, original_settle = endpoint.authorize, endpoint.settle
    endpoint.authorize = lambda *args: (events.append("authorize"), original_authorize(*args))[1]
    endpoint.settle = lambda *args: (events.append("settle"), original_settle(*args))[1]
    result = evaluate_v2(two_material_request(), HookedStub(), endpoint)
    assert events == ["authorize", "settle", "authorize", "settle"]
    ids = [check[3]["id"] for check in endpoint.checks]
    assert len(set(ids)) == 2 and [check[3]["index"] for check in endpoint.checks] == [1, 2]
    assert [check[3]["stage"] for check in endpoint.checks] == ["decompose", "review"]
    for ref, key, evaluator_id, attempt in endpoint.checks:
        assert (ref, key, evaluator_id) == (AUTH_REF, result["reuse_key"], current_evaluator()["evaluator_id"])
        assert key == key_of(two_material_request(), current_evaluator()) and len(attempt["request_body_sha256"]) == 64
    assert [(decision["attempt_id"], entry["attempt"]) for _, decision, entry in endpoint.settled] == [(i, 1) for i in ids]


def test_the_format_retry_is_its_own_attempt_with_the_next_index(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    review = json.dumps({"observations": good_observations()})
    fake = FakePost((json.dumps(DECOMPOSITION), "stop"), ("not json", "stop"), (review, "stop"))
    use(monkeypatch, fake)
    endpoint = FakeEndpoint()
    result = evaluate_v2(request(), chat_json, endpoint)
    assert result["overall"] == "supported" and len(fake.bodies) == 3
    assert [(check[3]["stage"], check[3]["index"]) for check in endpoint.checks] == [("decompose", 1), ("review", 2), ("review", 3)]
    assert len({check[3]["id"] for check in endpoint.checks}) == 3 and len(endpoint.settled) == 3
    assert [sum(call["requests"] for call in result["llm_calls"])] == [3]


def test_a_refusal_before_the_second_request_stops_it_and_returns_no_verdict():
    stub, endpoint = HookedStub(), FakeEndpoint(refuse_from=2)
    body = TestClient(create_app(stub, endpoint)).post("/v2/evaluate", json=request(), headers=AUTH).json()
    assert body["error_code"] == "authorization_blocked" and "overall" not in body
    assert stub.attempts == ["decompose"]                  # the review was never sent
    assert [call["error"] for call in body["llm_calls"]] == [None, "authorization_refused"]
    assert body["llm_calls"][1]["requests"] == 0 and len(endpoint.settled) == 1


def test_an_outcome_that_is_not_recorded_stops_the_evaluation_after_that_request():
    stub, endpoint = HookedStub(), FakeEndpoint(settle_fails_at=1)
    response = TestClient(create_app(stub, endpoint)).post("/v2/evaluate", json=request(), headers=AUTH)
    body = response.json()
    assert response.status_code == 502 and body["error_code"] == "authorization_blocked"
    assert "authorization_outcome_failed" in body["message"] and "overall" not in body
    assert stub.attempts == ["decompose"] and len(endpoint.checks) == 1       # the second request never asked
    assert body["llm_calls"][0]["requests"] == 1 and len(body["llm_calls"][0]["outputs"]) == 1   # the sent one is kept


def test_the_real_endpoint_unconfigured_sends_no_model_request(monkeypatch):
    monkeypatch.delenv("MECH_AUTHORIZATION_URL")
    stub = HookedStub()
    response = TestClient(create_app(stub)).post("/v2/evaluate", json=request(), headers=AUTH)
    assert response.status_code == 502 and response.json()["error_code"] == "authorization_blocked"
    assert stub.attempts == [] and response.json()["llm_calls"][0]["error"] == "authorization_not_configured"


def test_the_endpoint_comes_from_the_environment_never_from_the_request(monkeypatch):
    wire = Wire(lambda: None)
    monkeypatch.setattr(authorization.httpx, "post", wire)
    raw = request(authorization_callback="https://evil.example/")
    response = TestClient(create_app(HookedStub())).post("/v2/evaluate", json=raw, headers=AUTH)
    assert response.status_code == 422 and response.json()["error_code"] == "field_not_in_contract" and wire.posts == []


def test_a_provider_answer_without_a_readable_envelope_is_a_handleable_502_that_keeps_the_sent_call(monkeypatch):
    """The probe that found the 500: HTTP 200 with choices=[7]. One request was sent; it must be reported."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    use(monkeypatch, FakePost(httpx.Response(200, json={"choices": [7], "usage": {"total_tokens": 2}})))
    endpoint = FakeEndpoint()
    response = TestClient(create_app(chat_json, endpoint), raise_server_exceptions=False).post(
        "/v2/evaluate", json=request(), headers=AUTH)
    body = response.json()
    assert response.status_code == 502 and body["error_code"] == "decomposition_failed"
    [call] = body["llm_calls"]
    assert call["error"] == "response_malformed" and call["requests"] == 1 and len(call["outputs"]) == 1
    assert json.loads(call["outputs"][0]["content"]) == {"choices": [7], "usage": {"total_tokens": 2}}
    assert len(endpoint.checks) == 1 and len(endpoint.settled) == 1       # authorized and settled like any request


def test_the_same_through_the_v1_path_keeps_the_error_and_the_call(monkeypatch):
    from stub import case

    monkeypatch.setenv("LLM_API_KEY", "k")
    use(monkeypatch, FakePost(httpx.Response(200, json={"choices": [7]})))
    response = TestClient(create_app(chat_json), raise_server_exceptions=False).post("/v1/evaluate", json=case(), headers=AUTH)
    body = response.json()
    assert response.status_code == 502 and body["error_code"] == "decomposition_failed"
    assert body["llm_calls"][0]["error"] == "response_malformed" and body["llm_calls"][0]["requests"] == 1
    assert "outputs" not in body["llm_calls"][0]
