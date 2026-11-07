"""唯一的模型调用入口：OpenAI 兼容 /chat/completions，返回 (对象或 None, 调用记录)。"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import get_args

import httpx
from pydantic import BaseModel, ValidationError

# What the two structured stages ask of the model; the evaluator identity (identity.py) states the same values.
STRUCTURED_MAX_TOKENS = 4096
STRUCTURED_TEMPERATURE = 0
STRUCTURED_THINKING = False
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PROVIDER = "deepseek"
# A raw model answer is archived in the receipt up to this many bytes; a longer one is cut and marked truncated
# (its sha256 stays that of the whole answer). A truncated answer is an incomplete audit, never a usable result.
OUTPUT_ARCHIVE_LIMIT = 256 * 1024


class SendBlocked(Exception):
    """Raised by a before_attempt hook: this HTTP request must not be sent. `code` becomes the call's error."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


def archived(attempt: int, content: str) -> dict:
    raw = content.encode("utf-8")
    return {"attempt": attempt, "content": raw[:OUTPUT_ARCHIVE_LIMIT].decode("utf-8", errors="ignore"),
            "sha256": hashlib.sha256(raw).hexdigest(), "truncated": len(raw) > OUTPUT_ARCHIVE_LIMIT}


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


def chat_json(stage: str, system: str, payload: dict, model_cls: type[BaseModel],
              *, max_tokens: int = STRUCTURED_MAX_TOKENS, thinking: bool = STRUCTURED_THINKING,
              before_attempt: Callable[[dict], None] | None = None) -> tuple[BaseModel | None, dict]:
    """before_attempt(info) runs before every actual HTTP request of this call, the format retry included. Any
    exception it raises means that request is not sent (requests is not incremented, error is the exception's
    `code` or "authorization_error") and nothing further is sent for this call."""
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    # requests: HTTP requests actually sent for this logical call (a format retry is a second one)
    # outputs: the raw model answer of every attempt that returned one (archived(); v2 receipts carry them)
    call = {"stage": stage, "model": model, "finish_reason": None, "usage": {}, "seconds": 0.0, "error": None,
            "requests": 0, "outputs": []}
    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        call["error"] = "llm_not_configured"
        return None, call
    url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "180"))
    user = json.dumps(payload, ensure_ascii=False)
    body = {"model": model, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system + "\n\n只输出一个 JSON 对象，字段：\n" + describe(model_cls)},
                         {"role": "user", "content": user}]}
    if provider == "deepseek":
        body["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif provider == "vllm":
        # local vLLM serving Qwen: the same per-call switch, as its chat template reads it
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    if not thinking:
        body["temperature"] = STRUCTURED_TEMPERATURE
    started = time.monotonic()
    try:
        for attempt in range(2):
            if before_attempt is not None:
                try:
                    before_attempt({
                        "stage": stage, "attempt_in_call": attempt + 1, "requests_before": call["requests"],
                        "request_body_sha256": hashlib.sha256(
                            json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                        "model": model, "endpoint_id": os.environ.get("LLM_ENDPOINT_ID", "unnamed")})
                except Exception as error:  # noqa: BLE001 - whatever stops the check also stops the request
                    call["error"] = getattr(error, "code", "authorization_error")
                    return None, call
            call["requests"] += 1
            try:
                response = httpx.post(url, json=body, timeout=timeout, headers={"Authorization": "Bearer " + key})
            except httpx.TimeoutException:
                call["error"] = "timeout"  # 不自动重发：对方可能已经计费
                return None, call
            except httpx.HTTPError:
                call["error"] = "network_error"
                return None, call
            if response.status_code != 200:
                call["error"] = f"http_{response.status_code}"
                return None, call
            try:
                data = response.json()
                choice = data["choices"][0]
            except (ValueError, KeyError, IndexError, TypeError):
                call["outputs"].append(archived(attempt + 1, response.text))
                call["error"] = "response_malformed"
                return None, call
            for name, value in (data.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    call["usage"][name] = call["usage"].get(name, 0) + value
            call["finish_reason"] = choice.get("finish_reason")
            content = (choice.get("message") or {}).get("content") or ""
            call["outputs"].append(archived(attempt + 1, content))
            if call["outputs"][-1]["truncated"]:
                call["error"] = "output_oversize"  # cannot be archived whole: no format retry, no result
                return None, call
            if call["finish_reason"] == "length" and not content.strip():
                call["error"] = "output_empty_after_limit"  # 思考或长输入吃掉了额度
                return None, call
            try:
                result = model_cls.model_validate(json.loads(content))
                call["error"] = None
                return result, call
            except (ValueError, ValidationError) as error:
                call["error"] = "output_invalid"
                body["messages"][1]["content"] = user + "\n\n上一次输出未通过校验，请修正后重新输出完整 JSON：\n" + str(error)[:2000]
        return None, call
    finally:
        call["seconds"] = round(time.monotonic() - started, 2)
