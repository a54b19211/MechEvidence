"""HTTP 服务：GET /health，GET /v2/evaluator，GET /v2/contract，POST /v1/evaluate（兼容）和 POST /v2/evaluate（版本绑定契约）。
一次请求 = 一次同步评价，服务自己调模型。"""
from __future__ import annotations

import os
import secrets
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .assess import EvaluationFailed, evaluate
from .authorization import TrustedEndpoint
from .contract import CONTRACT_V1, CONTRACT_V2, ContractError
from .identity import current_evaluator
from .llm import chat_json
from .models import EvaluateRequest
from .receipt import RECEIPT_SCHEMA_VERSION, receipt_schema
from .v2 import evaluate_v2, v1_receipt

# the port MechEvidence is deployed on (the query planner has its own); README and AGENTS name the same one
DEFAULT_PORT = 18563


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message, **extra})


def _without_outputs(calls: list[dict]) -> list[dict]:
    return [{key: value for key, value in call.items() if key != "outputs"} for call in calls]


def create_app(chat=chat_json, endpoint=None) -> FastAPI:
    endpoint = endpoint if endpoint is not None else TrustedEndpoint()
    app = FastAPI(title="MechEvidence")

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        token = os.environ.get("SERVICE_TOKEN", "")
        if not token:
            return _error(503, "service_not_configured", "服务端未设置 SERVICE_TOKEN")
        if not secrets.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {token}".encode()):
            return _error(401, "unauthorized", "需要 Authorization: Bearer <SERVICE_TOKEN>")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, error: RequestValidationError):
        errors = error.errors()
        extras = [".".join(map(str, item["loc"])) for item in errors if item["type"] == "extra_forbidden"]
        if extras and request.url.path == "/v1/evaluate":
            # v1 used to drop these silently and answer 200 as if they had been checked
            return _error(422, "field_not_in_contract",
                          f"not in contract {CONTRACT_V1}: {', '.join(extras[:10])}. Version-binding fields "
                          f"exist only in POST /v2/evaluate ({CONTRACT_V2}); an unknown field is rejected, not ignored.")
        details = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in errors[:10])
        return _error(422, "input_invalid", details[:1000])

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, error: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(error.status_code, f"http_{error.status_code}")
        return _error(error.status_code, code, str(error.detail))

    @app.exception_handler(Exception)
    async def internal(_request, error: Exception):
        return _error(500, "internal_error", type(error).__name__)

    @app.get("/health")
    def health():
        return {"status": "ok", "contracts": [CONTRACT_V1, CONTRACT_V2], "evaluator_id": current_evaluator()["evaluator_id"]}

    @app.get("/v2/evaluator")
    def evaluator():
        return {"contract_version": CONTRACT_V2, "evaluator": current_evaluator(),
                "authorization": {"mode": "per_attempt_trusted_endpoint", "configured": endpoint.configured()}}

    @app.get("/v2/contract")
    def contract():
        """The receipt format of the v2 contract as a JSON Schema, for callers that validate receipts."""
        return {"contract_version": CONTRACT_V2, "schema_version": RECEIPT_SCHEMA_VERSION,
                "receipt_schema": receipt_schema()}

    @app.post("/v1/evaluate")
    def evaluate_route(body: EvaluateRequest):
        try:
            return v1_receipt(evaluate(body, chat))
        except EvaluationFailed as error:
            return _error(502, error.code, str(error), llm_calls=_without_outputs(error.llm_calls))

    @app.post("/v2/evaluate")
    def evaluate_v2_route(body: Annotated[Any, Body()]):
        try:
            return evaluate_v2(body, chat, endpoint)
        except ContractError as error:
            return _error(error.status, error.code, str(error), **error.extra)
        except EvaluationFailed as error:
            return _error(502, error.code, str(error), llm_calls=error.llm_calls)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", DEFAULT_PORT)))
