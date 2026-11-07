"""What chat_json archives for a v2 receipt (the raw answer of every attempt) and the hook that runs before every
actual HTTP request."""
import json

import httpx
import pytest
from stub import DECOMPOSITION, case, good_observations
from test_llm import FakePost

from mechevidence import evaluate, llm
from mechevidence.models import Decomposition


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")


GOOD = json.dumps({"claims": [{"text": "A changes M"}], "joins": []})
BAD = '{"claims": []}'


def run(monkeypatch, *replies, **kwargs):
    fake = FakePost(*replies)
    monkeypatch.setattr(llm.httpx, "post", fake)
    result, call = llm.chat_json("decompose", "system", {"x": 1}, Decomposition, **kwargs)
    return result, call, fake


def test_the_raw_answer_of_every_attempt_is_archived_whole(monkeypatch):
    _, call, _ = run(monkeypatch, (BAD, "stop"), (GOOD, "stop"))
    assert [(out["attempt"], out["content"], out["truncated"]) for out in call["outputs"]] == [
        (1, BAD, False), (2, GOOD, False)]
    assert call["outputs"][1]["sha256"] == llm.hashlib.sha256(GOOD.encode()).hexdigest()


def test_a_long_answer_is_cut_keeps_the_hash_of_the_whole_and_is_not_retried(monkeypatch):
    long = "y" * (llm.OUTPUT_ARCHIVE_LIMIT + 5)
    result, call, fake = run(monkeypatch, (long, "stop"), (GOOD, "stop"))
    out = call["outputs"][0]
    assert result is None and call["error"] == "output_oversize" and len(fake.bodies) == 1
    assert out["truncated"] is True and len(out["content"].encode()) == llm.OUTPUT_ARCHIVE_LIMIT
    assert out["sha256"] == llm.hashlib.sha256(long.encode()).hexdigest()
    exact = "z" * llm.OUTPUT_ARCHIVE_LIMIT
    _, call, _ = run(monkeypatch, (exact, "stop"), (GOOD, "stop"))
    assert call["outputs"][0]["truncated"] is False and call["error"] is None


def test_the_cut_never_splits_a_character():
    out = llm.archived(1, "é" * (llm.OUTPUT_ARCHIVE_LIMIT // 2 + 1))
    assert out["truncated"] is True and out["content"].encode().decode("utf-8") == out["content"]


def test_a_malformed_http_answer_is_archived_too(monkeypatch):
    class Odd:
        def __call__(self, url, json, **_):
            return httpx.Response(200, content=b"<html>gateway</html>")

    monkeypatch.setattr(llm.httpx, "post", Odd())
    result, call = llm.chat_json("decompose", "s", {}, Decomposition)
    assert result is None and call["error"] == "response_malformed"
    assert call["outputs"][0]["content"] == "<html>gateway</html>"


def test_the_hook_runs_before_every_request_the_format_retry_included(monkeypatch):
    seen = []

    def hook(info):
        seen.append((dict(info), len(fake.bodies)))

    fake = FakePost((BAD, "stop"), (GOOD, "stop"))
    monkeypatch.setattr(llm.httpx, "post", fake)
    monkeypatch.setenv("LLM_ENDPOINT_ID", "deepseek-public")
    result, call = llm.chat_json("decompose", "s", {}, Decomposition, before_attempt=hook)
    assert result is not None and call["requests"] == 2
    assert [(info["attempt_in_call"], info["requests_before"], sent) for info, sent in seen] == [(1, 0, 0), (2, 1, 1)]
    assert seen[0][0]["endpoint_id"] == "deepseek-public" and seen[0][0]["model"] == call["model"]
    assert seen[0][0]["request_body_sha256"] != seen[1][0]["request_body_sha256"]   # the retry's body carries the error


@pytest.mark.parametrize("fail_on,sent,requests", [(1, 0, 0), (2, 1, 1)])
def test_a_hook_that_raises_stops_that_request_and_all_after_it(monkeypatch, fail_on, sent, requests):
    count = []

    def hook(info):
        count.append(1)
        if len(count) == fail_on:
            raise llm.SendBlocked("authorization_refused", "revoked")

    result, call, fake = run(monkeypatch, (BAD, "stop"), (GOOD, "stop"), before_attempt=hook)
    assert result is None and call["error"] == "authorization_refused"
    assert len(fake.bodies) == sent and call["requests"] == requests


def test_any_exception_in_the_hook_blocks_the_request(monkeypatch):
    def hook(info):
        raise RuntimeError("database down")           # a failed check is not a licence to send

    result, call, fake = run(monkeypatch, (GOOD, "stop"), before_attempt=hook)
    assert result is None and call["error"] == "authorization_error" and fake.bodies == [] and call["requests"] == 0


def test_an_unconfigured_call_does_not_even_ask_the_hook(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    asked = []
    _, call, _ = run(monkeypatch, before_attempt=asked.append)
    assert call["error"] == "llm_not_configured" and asked == []


def test_a_whole_evaluation_through_the_real_transport_checks_each_request(monkeypatch):
    review = json.dumps({"observations": good_observations()})
    fake = FakePost((json.dumps(DECOMPOSITION), "stop"), ("not json", "stop"), (review, "stop"))
    monkeypatch.setattr(llm.httpx, "post", fake)
    checks = []
    original = llm.chat_json
    out = evaluate(case(), lambda *a, **k: original(*a, before_attempt=lambda info: checks.append(
        (info["stage"], info["attempt_in_call"])), **k))
    assert checks == [("decompose", 1), ("review", 1), ("review", 2)] and len(fake.bodies) == 3
    assert [len(call["outputs"]) for call in out["llm_calls"]] == [1, 2]
