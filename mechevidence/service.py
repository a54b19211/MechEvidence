"""HTTP 服务：GET /health，POST /v1/evaluate。一次请求 = 一次同步评价，服务自己调模型。"""
from __future__ import annotations

import os
import secrets

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .assess import EvaluationFailed, evaluate
from .llm import chat_json
from .models import EvaluateRequest


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message, **extra})


def create_app(chat=chat_json) -> FastAPI:
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
    async def invalid_input(_request, error: RequestValidationError):
        details = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()[:10])
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
        return {"status": "ok"}

    @app.post("/v1/evaluate")
    def evaluate_route(body: EvaluateRequest):
        try:
            return evaluate(body, chat)
        except EvaluationFailed as error:
            return _error(502, error.code, str(error), llm_calls=error.llm_calls)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "18561")))
