"""What chat_json archives for a v2 receipt (the raw HTTP answer of every counted request), what it does with an
answer it cannot read, and the hooks that run before and after every actual HTTP request."""
import hashlib
import json

import httpx
import pytest
from stub import DECOMPOSITION, case, good_observations
from test_llm import FakePost, use

from mechevidence import evaluate, llm
from mechevidence.models import Decomposition


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")


GOOD = json.dumps({"claims": [{"text": "A changes M"}], "joins": []})
BAD = '{"claims": []}'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(monkeypatch, *replies, **kwargs):
    fake = FakePost(*replies)
    use(monkeypatch, fake)
    result, call = llm.chat_json("decompose", "system", {"x": 1}, Decomposition, **kwargs)
    return result, call, fake


def body_of(content, finish="stop"):
    return json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}],
                       "usage": {"total_tokens": 10}}, ensure_ascii=False, separators=(",", ":"))  # as httpx writes it


def test_the_raw_http_answer_of_every_counted_request_is_archived_whole(monkeypatch):
    _, call, _ = run(monkeypatch, (BAD, "stop"), (GOOD, "stop"))
    assert [(out["attempt"], out["delivered"], out["status"], out["content"], out["truncated"])
            for out in call["outputs"]] == [(1, "yes", 200, body_of(BAD), False), (2, "yes", 200, body_of(GOOD), False)]
    assert call["outputs"][1]["sha256"] == sha(body_of(GOOD).encode()) and call["requests"] == 2


def test_a_long_answer_is_cut_keeps_the_hash_of_the_whole_and_is_not_retried(monkeypatch):
    long = "y" * (llm.OUTPUT_ARCHIVE_LIMIT + 5)
    result, call, fake = run(monkeypatch, (long, "stop"), (GOOD, "stop"))
    out = call["outputs"][0]
    assert result is None and call["error"] == "output_oversize" and len(fake.bodies) == 1 and call["requests"] == 1
    assert out["truncated"] is True and len(out["content"].encode()) == llm.OUTPUT_ARCHIVE_LIMIT
    assert out["sha256"] == sha(body_of(long).encode())
    exact = "z" * (llm.OUTPUT_ARCHIVE_LIMIT - len(body_of("")))
    assert len(body_of(exact).encode()) == llm.OUTPUT_ARCHIVE_LIMIT
    _, call, _ = run(monkeypatch, (exact, "stop"), (GOOD, "stop"))
    assert call["outputs"][0]["truncated"] is False and call["error"] is None


def test_the_cut_never_splits_a_character():
    out = llm.archived(1, "é" * (llm.OUTPUT_ARCHIVE_LIMIT // 2 + 1))
    assert out["truncated"] is True and out["content"].encode().decode("utf-8") == out["content"]


@pytest.mark.parametrize("answer", [
    {"choices": [7]}, {"choices": []}, {"choices": "x"}, {"choices": [{"message": 5}]}, [1, 2],
    {"choices": [{"message": {"content": ["a"]}}]},
])
def test_an_envelope_that_cannot_be_read_ends_the_call_without_losing_the_request(monkeypatch, answer):
    result, call, fake = run(monkeypatch, httpx.Response(200, json=answer))
    assert result is None and call["error"] == "response_malformed" and len(fake.bodies) == 1
    assert call["requests"] == 1 and len(call["outputs"]) == 1
    assert json.loads(call["outputs"][0]["content"]) == answer and call["outputs"][0]["status"] == 200


def test_a_non_json_answer_is_archived_too(monkeypatch):
    result, call, _ = run(monkeypatch, httpx.Response(200, content=b"<html>gateway</html>"))
    assert result is None and call["error"] == "response_malformed"
    assert call["outputs"][0]["content"] == "<html>gateway</html>" and call["requests"] == 1


def test_an_answer_with_another_status_is_archived_and_not_retried(monkeypatch):
    result, call, fake = run(monkeypatch, httpx.Response(503, content=b"busy"), (GOOD, "stop"))
    assert result is None and call["error"] == "http_503" and len(fake.bodies) == 0 + 1
    assert (call["outputs"][0]["status"], call["outputs"][0]["content"]) == (503, "busy")


def test_a_timeout_is_a_counted_request_with_no_answer(monkeypatch):
    _, call, _ = run(monkeypatch, httpx.ReadTimeout("slow"))
    assert call["error"] == "timeout" and call["requests"] == 1
    assert call["outputs"] == [{"attempt": 1, "delivered": "unknown", "status": None, "error": "timeout",
                                "content": None, "sha256": None, "truncated": False}]


def test_the_hook_runs_before_every_request_the_format_retry_included(monkeypatch):
    seen = []

    def hook(info):
        seen.append((dict(info), len(fake.bodies)))

    fake = FakePost((BAD, "stop"), (GOOD, "stop"))
    use(monkeypatch, fake)
    monkeypatch.setenv("LLM_ENDPOINT_ID", "deepseek-public")
    result, call = llm.chat_json("decompose", "s", {}, Decomposition, before_attempt=hook)
    assert result is not None and call["requests"] == 2
    assert [(info["index_in_call"], info["requests_before"], sent) for info, sent in seen] == [(1, 0, 0), (2, 1, 1)]
    assert seen[0][0]["endpoint_id"] == "deepseek-public" and seen[0][0]["model"] == call["model"]
    assert seen[0][0]["request_body_sha256"] != seen[1][0]["request_body_sha256"]   # the retry's body carries the error


@pytest.mark.parametrize("fail_on,sent,requests", [(1, 0, 0), (2, 1, 1)])
def test_a_hook_that_raises_stops_that_request_and_all_after_it(monkeypatch, fail_on, sent, requests):
    count = []

    def hook(info):
        count.append(1)
        if len(count) == fail_on:
            raise llm.SendBlocked("authorization_refused")

    result, call, fake = run(monkeypatch, (BAD, "stop"), (GOOD, "stop"), before_attempt=hook)
    assert result is None and call["error"] == "authorization_refused"
    assert len(fake.bodies) == sent and call["requests"] == requests == len(call["outputs"])


def test_any_exception_in_the_hook_blocks_the_request(monkeypatch):
    def hook(info):
        raise RuntimeError("database down")           # a failed check is not a licence to send

    result, call, fake = run(monkeypatch, (GOOD, "stop"), before_attempt=hook)
    assert result is None and call["error"] == "authorization_error" and fake.bodies == [] and call["requests"] == 0


def test_the_outcome_is_handed_over_after_each_request_and_before_the_next(monkeypatch):
    log = []
    fake = FakePost((BAD, "stop"), (GOOD, "stop"))
    use(monkeypatch, fake)
    result, _ = llm.chat_json(
        "decompose", "s", {}, Decomposition, before_attempt=lambda info: log.append(("before", len(fake.bodies))),
        after_attempt=lambda entry: log.append(("after", entry["attempt"], entry["status"], len(fake.bodies))))
    assert result is not None
    assert log == [("before", 0), ("after", 1, 200, 1), ("before", 1), ("after", 2, 200, 2)]


def test_an_outcome_that_cannot_be_kept_stops_everything_but_keeps_the_request_counted(monkeypatch):
    def after(entry):
        raise RuntimeError("database down")

    result, call, fake = run(monkeypatch, (BAD, "stop"), (GOOD, "stop"), after_attempt=after)
    assert result is None and call["error"] == "authorization_outcome_failed"
    assert len(fake.bodies) == 1 and call["requests"] == 1 and len(call["outputs"]) == 1


def test_an_unconfigured_call_does_not_even_ask_the_hook(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    asked = []
    _, call, _ = run(monkeypatch, before_attempt=asked.append)
    assert call["error"] == "llm_not_configured" and asked == []


def test_a_bad_base_url_sends_nothing(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "ftp://nowhere")
    _, call, fake = run(monkeypatch, (GOOD, "stop"))
    assert call["error"] == "llm_base_url_invalid" and call["requests"] == 0 and fake.bodies == []


def test_a_whole_evaluation_through_the_real_transport_checks_each_request(monkeypatch):
    review = json.dumps({"observations": good_observations()})
    fake = FakePost((json.dumps(DECOMPOSITION), "stop"), ("not json", "stop"), (review, "stop"))
    use(monkeypatch, fake)
    checks = []
    original = llm.chat_json
    out = evaluate(case(), lambda *a, **k: original(*a, before_attempt=lambda info: checks.append(
        (info["stage"], info["index_in_call"])), **k))
    assert checks == [("decompose", 1), ("review", 1), ("review", 2)] and len(fake.bodies) == 3
    assert [len(call["outputs"]) for call in out["llm_calls"]] == [1, 2]
