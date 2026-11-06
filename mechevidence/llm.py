"""唯一的模型调用入口：OpenAI 兼容 /chat/completions，返回 (对象或 None, 调用记录)。"""
from __future__ import annotations

import json
import os
import time
from typing import get_args

import httpx
from pydantic import BaseModel, ValidationError


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
              *, max_tokens: int = 4096, thinking: bool = False) -> tuple[BaseModel | None, dict]:
    model = os.environ.get("LLM_MODEL", "deepseek-v4-pro")
    # requests: HTTP requests actually sent for this logical call (a format retry is a second one)
    call = {"stage": stage, "model": model, "finish_reason": None, "usage": {}, "seconds": 0.0, "error": None,
            "requests": 0}
    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        call["error"] = "llm_not_configured"
        return None, call
    url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
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
        body["temperature"] = 0
    started = time.monotonic()
    try:
        for attempt in range(2):
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
                call["error"] = "response_malformed"
                return None, call
            for name, value in (data.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    call["usage"][name] = call["usage"].get(name, 0) + value
            call["finish_reason"] = choice.get("finish_reason")
            content = (choice.get("message") or {}).get("content") or ""
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
