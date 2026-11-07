"""唯一的模型调用入口：OpenAI 兼容 /chat/completions，返回 (对象或 None, 调用记录)。

HTTP 一侧（请求体、发送、送达结果、请求计数、响应信封）由共享的 llm_transport 完成；提示词说明、格式重试、
错误码和默认值留在这里。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import get_args

import httpx
import llm_transport as lt
from llm_transport import SendBlocked
from extraction_contract.mech_v2.receipt import OUTPUT_ARCHIVE_LIMIT
from pydantic import BaseModel, ValidationError

__all__ = ["OutputNotRecorded", "SendBlocked", "archived", "chat_json", "describe", "make_client"]

# What the two structured stages ask of the model; the evaluator identity (identity.py) states the same values.
STRUCTURED_MAX_TOKENS = 4096
STRUCTURED_TEMPERATURE = 0
STRUCTURED_THINKING = False
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PROVIDER = "deepseek"


class OutputNotRecorded(Exception):
    """Raised by an after_attempt hook that could not keep an attempt's outcome: nothing further is sent."""


def make_client() -> httpx.Client:
    return httpx.Client()


def _entry(index, raw: bytes | None, delivered: str, status, error) -> dict:
    """One counted request as the receipt archives it. `content` is the raw HTTP answer (cut at the limit),
    `sha256` the sha256 of the whole answer's bytes; no answer (timeout, refused connection) is content None."""
    return {"attempt": index, "delivered": delivered, "status": status, "error": error,
            "content": None if raw is None else raw[:OUTPUT_ARCHIVE_LIMIT].decode("utf-8", errors="ignore"),
            "sha256": None if raw is None else hashlib.sha256(raw).hexdigest(),
            "truncated": raw is not None and len(raw) > OUTPUT_ARCHIVE_LIMIT}


def archived(index: int, text: str) -> dict:
    """The entry of an answered request whose body is `text` (fixtures and tests)."""
    return _entry(index, text.encode("utf-8"), "yes", 200, None)


def describe(model_cls: type[BaseModel]) -> str:
    """简短字段说明（不是整份 JSON Schema），附在 system 末尾。"""
    lines = []
    for name, field in model_cls.model_fields.items():
        kind = getattr(field.annotation, "__name__", None) or str(field.annotation).replace("typing.", "")
        options = get_args(field.annotation)
        if options and all(isinstance(item, str) for item in options):
            kind = "|".join(options)
        need = "必填" if field.is_required() else "可省略"
        lines.append(f"- {name} ({kind}, {need}): {field.description or ''}".rstrip(": "))
    return "\n".join(lines)


def _provider(model: str) -> lt.Provider:
    name = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    return lt.Provider(
        kind=name if name in ("deepseek", "vllm") else "openai_compatible",
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"), model=model,
        api_key=os.environ.get("LLM_API_KEY", ""), timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "180")),
        endpoint_id=os.environ.get("LLM_ENDPOINT_ID", "unnamed"))


def _hooks(call: dict, provider: lt.Provider, model: str, before_attempt, after_attempt):
    """The transport's before/after callbacks for one logical call: authorization hook in front, record behind."""
    stage = call["stage"]

    def before(start: lt.AttemptStart) -> None:
        if before_attempt is None:
            return
        try:
            before_attempt({"stage": stage, "index_in_call": start.index, "requests_before": start.sent_before,
                            "request_body_sha256": start.body_sha256, "model": model,
                            "endpoint_id": provider.endpoint_id})
        except lt.SendBlocked:
            raise
        except Exception as error:  # noqa: BLE001 - whatever stops the check also stops the request
            raise lt.SendBlocked(getattr(error, "code", None) or "authorization_error") from None

    def after(attempt: lt.Attempt) -> None:
        call["outputs"].append(_entry(attempt.index, attempt.response_bytes, attempt.delivered, attempt.status,
                                      attempt.error))
        if after_attempt is not None:
            try:
                after_attempt(call["outputs"][-1])
            except Exception:  # noqa: BLE001 - an outcome that is not kept stops the call
                raise OutputNotRecorded from None

    return before, after


def _send(call: dict, provider: lt.Provider, body, record: lt.CallRecord, client, before, after):
    """One transport send; None (with call["error"] set) when the call has to stop."""
    try:
        return lt.send(provider, body, record, client=client, before_attempt=before, after_attempt=after)
    except lt.SendBlocked as blocked:
        call["error"] = blocked.reason
    except OutputNotRecorded:
        call["error"] = "authorization_outcome_failed"
    except lt.RequestNotBuilt:
        call["error"] = "request_not_built"
    except Exception:  # noqa: BLE001 - settled by the transport as unknown; the call keeps its record
        call["error"] = "unexpected_error"
    return None


def _content(call: dict, sent) -> str | None:
    """The model's text of a sent request; None (with call["error"] set) when the call has to stop."""
    if sent.exception is not None:
        call["error"] = "timeout" if sent.timed_out else "network_error"  # 不自动重发：对方可能已经计费
        return None
    if sent.status != 200:
        call["error"] = f"http_{sent.status}"
        return None
    if call["outputs"][-1]["truncated"]:
        call["error"] = "output_oversize"  # cannot be archived whole: no format retry, no result
        return None
    found = sent.envelope()
    if found.problem is not None or not (found.content is None or isinstance(found.content, str)):
        call["error"] = "response_malformed"
        return None
    for name, value in found.usage.items():
        call["usage"][name] = call["usage"].get(name, 0) + value
    call["finish_reason"] = found.finish_reason
    content = found.content or ""
    if call["finish_reason"] == "length" and not content.strip():
        call["error"] = "output_empty_after_limit"  # 思考或长输入吃掉了额度
        return None
    return content


def chat_json(stage: str, system: str, payload: dict, model_cls: type[BaseModel],
              *, max_tokens: int = STRUCTURED_MAX_TOKENS, thinking: bool = STRUCTURED_THINKING,
              before_attempt: Callable[[dict], None] | None = None,
              after_attempt: Callable[[dict], None] | None = None) -> tuple[BaseModel | None, dict]:
    """before_attempt(info) runs before every actual HTTP request of this call, the format retry included; any
    exception it raises means that request is not sent (requests is not incremented, error is the exception's
    `reason`/`code` or "authorization_error") and nothing further is sent for the call.
    after_attempt(entry) runs after every counted request, once its raw answer is archived and before anything
    is parsed or resent; if it raises, error is "authorization_outcome_failed" and nothing further is sent.

    A request that was sent is never lost: the raw HTTP answer is archived first, the envelope is read afterwards,
    and a malformed one ends the call with error "response_malformed" (outputs and requests intact)."""
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    # requests: HTTP requests actually sent for this logical call (a format retry is a second one)
    # outputs: one entry per counted request (_entry), in order
    call = {"stage": stage, "model": model, "finish_reason": None, "usage": {}, "seconds": 0.0, "error": None,
            "requests": 0, "outputs": []}
    if not os.environ.get("LLM_API_KEY", ""):
        call["error"] = "llm_not_configured"
        return None, call
    try:
        provider = _provider(model)
    except ValueError:
        call["error"] = "llm_base_url_invalid"
        return None, call
    record = lt.CallRecord(stage=stage, endpoint_id=provider.endpoint_id, model=model)
    user = json.dumps(payload, ensure_ascii=False)
    messages = [{"role": "system", "content": system + "\n\n只输出一个 JSON 对象，字段：\n" + describe(model_cls)},
                {"role": "user", "content": user}]
    before, after = _hooks(call, provider, model, before_attempt, after_attempt)

    started = time.monotonic()
    try:
        with make_client() as client:
            for _ in range(2):
                body = lt.build_body(provider, messages, max_tokens=max_tokens, thinking=thinking)
                sent = _send(call, provider, body, record, client, before, after)
                content = _content(call, sent) if sent is not None else None
                if content is None:
                    return None, call
                try:
                    result = model_cls.model_validate(json.loads(content))
                    call["error"] = None
                    return result, call
                except (ValueError, ValidationError) as error:
                    call["error"] = "output_invalid"
                    messages[1] = {"role": "user", "content": user + "\n\n上一次输出未通过校验，请修正后重新输出完整 JSON：\n"
                                   + str(error)[:2000]}
            return None, call
    finally:
        call["requests"] = record.requests
        call["seconds"] = round(time.monotonic() - started, 2)
