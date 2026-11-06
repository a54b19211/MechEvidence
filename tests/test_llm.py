"""模型调用真实出过的三种失败（思考吃额度、格式错误整份作废、超时重发），以及每次调用实际发出的请求数。"""
import json

import httpx
import pytest
from stub import DECOMPOSITION, case, good_observations

from mechevidence import evaluate, llm
from mechevidence.models import Decomposition


class FakePost:
    def __init__(self, *replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, url, json, **_):
        self.bodies.append(json)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, finish = reply
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                         "usage": {"total_tokens": 10}})


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")


def run(monkeypatch, *replies):
    fake = FakePost(*replies)
    monkeypatch.setattr(llm.httpx, "post", fake)
    result, call = llm.chat_json("decompose", "system", {"x": 1}, Decomposition)
    return result, call, fake


def test_local_vllm_gets_the_per_call_thinking_switch_its_template_reads(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    good = json.dumps({"claims": [{"text": "A changes M"}], "joins": []})
    _, _, fake = run(monkeypatch, (good, "stop"))
    assert fake.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking" not in fake.bodies[0]


def test_thinking_disabled_and_short_field_summary(monkeypatch):
    good = json.dumps({"claims": [{"text": "A changes M"}], "joins": []})
    result, call, fake = run(monkeypatch, (good, "stop"))
    body = fake.bodies[0]
    assert body["thinking"] == {"type": "disabled"} and body["temperature"] == 0 and body["max_tokens"] == 4096
    assert "properties" not in body["messages"][0]["content"] and "claims (list" in body["messages"][0]["content"]
    assert result.claims[0].text == "A changes M" and call["error"] is None and call["usage"]["total_tokens"] == 10


def test_invalid_output_retried_once_with_error_then_fails(monkeypatch):
    result, call, fake = run(monkeypatch, ('{"claims": []}', "stop"), ("not json", "stop"))
    assert result is None and call["error"] == "output_invalid" and len(fake.bodies) == 2
    assert "未通过校验" in fake.bodies[1]["messages"][1]["content"]
    good = json.dumps({"claims": [{"text": "A"}]})
    result, call, _ = run(monkeypatch, ('{"claims": []}', "stop"), (good, "stop"))
    assert result is not None and call["error"] is None and call["usage"]["total_tokens"] == 20


def test_empty_after_length_and_timeout_are_not_retried(monkeypatch):
    result, call, fake = run(monkeypatch, ("", "length"))
    assert result is None and call["error"] == "output_empty_after_limit" and len(fake.bodies) == 1
    result, call, fake = run(monkeypatch, httpx.ReadTimeout("slow"))
    assert result is None and call["error"] == "timeout" and len(fake.bodies) == 1


def test_each_call_reports_the_http_requests_it_sent(monkeypatch):
    good = json.dumps({"claims": [{"text": "A"}]})
    _, call, fake = run(monkeypatch, (good, "stop"))
    assert call["requests"] == len(fake.bodies) == 1
    _, call, fake = run(monkeypatch, ("{\"claims\": []}", "stop"), (good, "stop"))
    assert call["requests"] == len(fake.bodies) == 2 and call["error"] is None  # the format retry is a request too
    _, call, fake = run(monkeypatch, ("{\"claims\": []}", "stop"), ("not json", "stop"))
    assert call["requests"] == len(fake.bodies) == 2 and call["error"] == "output_invalid"
    _, call, fake = run(monkeypatch, httpx.ReadTimeout("slow"))
    assert call["requests"] == 1 and call["error"] == "timeout"  # sent and maybe billed, never resent


def test_an_unconfigured_call_sends_no_request(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    fake = FakePost()
    monkeypatch.setattr(llm.httpx, "post", fake)
    result, call = llm.chat_json("decompose", "system", {"x": 1}, Decomposition)
    assert result is None and call["error"] == "llm_not_configured" and call["requests"] == 0 and fake.bodies == []


def test_an_evaluation_reports_every_request_of_its_calls(monkeypatch):
    review = json.dumps({"observations": good_observations()})
    fake = FakePost((json.dumps(DECOMPOSITION), "stop"), ("not json", "stop"), (review, "stop"))
    monkeypatch.setattr(llm.httpx, "post", fake)
    out = evaluate(case())
    assert [(c["stage"], c["requests"]) for c in out["llm_calls"]] == [("decompose", 1), ("review", 2)]
    assert sum(c["requests"] for c in out["llm_calls"]) == len(fake.bodies) == 3
