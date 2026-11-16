"""唯一的模型调用入口：OpenAI 兼容 /chat/completions（官方 openai 库），返回 (对象或 None, 调用记录)。

每个请求发出前过授权钩子、发出后先存档原始 HTTP 回答再解析，所以 openai 库自己的重发关掉（max_retries=0）：
库内部的重发绕不过钩子，也不会被计数。提示词说明、格式重试、错误码和默认值都在这里。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import get_args

import httpx
import openai
from extraction_contract.mech_v2.receipt import OUTPUT_ARCHIVE_LIMIT
from pydantic import BaseModel, ValidationError

__all__ = ["OutputNotRecorded", "SendBlocked", "archived", "chat_json", "describe", "make_client"]

# What the two structured stages ask of the model; the evaluator identity (identity.py) states the same values.
STRUCTURED_MAX_TOKENS = 4096
STRUCTURED_TEMPERATURE = 0
STRUCTURED_THINKING = False
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PROVIDER = "deepseek"

# Failures raised before any byte of the request left this process: the provider cannot have seen it.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol)


class SendBlocked(Exception):
    """Raised by a before_attempt hook: this request may not go out; `reason` becomes the call's error."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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


def _api(client: httpx.Client) -> openai.OpenAI:
    """The client for LLM_BASE_URL; ValueError when that is not an http(s) URL with a host."""
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL:
        raise ValueError("LLM_BASE_URL is not a URL") from None
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError("LLM_BASE_URL must be an http or https URL with a host")
    return openai.OpenAI(base_url=base_url, api_key=os.environ["LLM_API_KEY"], max_retries=0,
                         timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "180")), http_client=client)


def _body(model: str, messages: list, max_tokens: int, thinking: bool) -> dict:
    """The request body: the provider's per-call thinking switch, temperature 0 when not thinking."""
    body = {"model": model, "messages": list(messages), "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}}
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    if provider == "deepseek":
        body["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif provider == "vllm":  # local vLLM serving Qwen reads the switch from its chat template
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    if not thinking:
        body["temperature"] = STRUCTURED_TEMPERATURE
    return body


def _post(api: openai.OpenAI, body: dict):
    """One HTTP request: (raw answer or None, delivered, status, error, timed_out)."""
    extra = {k: body[k] for k in ("thinking", "chat_template_kwargs") if k in body}
    plain = {k: v for k, v in body.items() if k not in extra}
    try:
        response = api.chat.completions.with_raw_response.create(**plain, extra_body=extra or None).http_response
    except openai.APIStatusError as error:
        response = error.response
    except openai.APIConnectionError as error:
        cause, timed_out = error.__cause__, isinstance(error, openai.APITimeoutError)
        if isinstance(cause, _NOT_SENT):
            return None, "no", None, "pool_timeout" if isinstance(cause, httpx.PoolTimeout) else "connect_failed", timed_out
        return None, "unknown", None, "timeout" if timed_out else "connection_lost", timed_out
    status = response.status_code
    return response.content, "yes", status, None if response.is_success else f"http_{status}", False


def _send(call: dict, api, body: dict, model: str, before_attempt, after_attempt):
    """One request with its hooks: (raw answer, status, timed_out), or None (call["error"] set) to stop."""
    if before_attempt is not None:
        try:
            before_attempt({"stage": call["stage"], "index_in_call": call["requests"] + 1,
                            "requests_before": call["requests"], "model": model,
                            "request_body_sha256": hashlib.sha256(json.dumps(
                                body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                            "endpoint_id": os.environ.get("LLM_ENDPOINT_ID", "unnamed")})
        except SendBlocked as blocked:
            call["error"] = blocked.reason
            return None
        except Exception as error:  # noqa: BLE001 - whatever stops the check also stops the request
            call["error"] = getattr(error, "code", None) or "authorization_error"
            return None
    call["requests"] += 1
    try:
        raw, delivered, status, error, timed_out = _post(api, body)
    except Exception:  # noqa: BLE001 - the request may have gone out: settled as unknown, the call stops
        raw, delivered, status, error, timed_out = None, "unknown", None, "unexpected_error", False
    call["outputs"].append(_entry(call["requests"], raw, delivered, status, error))
    if after_attempt is not None:
        try:
            after_attempt(call["outputs"][-1])
        except Exception:  # noqa: BLE001 - an outcome that is not kept stops the call
            call["error"] = "authorization_outcome_failed"
            return None
    if error == "unexpected_error":
        call["error"] = error
        return None
    return raw, status, timed_out


def _content(call: dict, sent) -> str | None:
    """The model's text of a sent request; None (with call["error"] set) when the call has to stop."""
    raw, status, timed_out = sent
    if raw is None:
        call["error"] = "timeout" if timed_out else "network_error"  # 不自动重发：对方可能已经计费
        return None
    if status != 200:
        call["error"] = f"http_{status}"
        return None
    if call["outputs"][-1]["truncated"]:
        call["error"] = "output_oversize"  # cannot be archived whole: no format retry, no result
        return None
    choice = None
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        content = choice["message"].get("content")
        usage = data.get("usage") or {}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        content = usage = None
    if not isinstance(content, (str, type(None))) or not isinstance(usage, dict) or not isinstance(choice, dict):
        call["error"] = "response_malformed"
        return None
    for name, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            call["usage"][name] = call["usage"].get(name, 0) + value
    call["finish_reason"] = choice.get("finish_reason")
    content = content or ""
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
    user = json.dumps(payload, ensure_ascii=False)
    messages = [{"role": "system", "content": system + "\n\n只输出一个 JSON 对象，字段：\n" + describe(model_cls)},
                {"role": "user", "content": user}]
    started = time.monotonic()
    try:
        with make_client() as client:
            try:
                api = _api(client)
            except ValueError:
                call["error"] = "llm_base_url_invalid"
                return None, call
            for _ in range(2):
                body = _body(model, messages, max_tokens, thinking)
                sent = _send(call, api, body, model, before_attempt, after_attempt)
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
        call["seconds"] = round(time.monotonic() - started, 2)
