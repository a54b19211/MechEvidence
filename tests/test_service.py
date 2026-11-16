"""HTTP 鉴权和统一错误格式；模型由替身代替，不联网。"""
from fastapi.testclient import TestClient
from stub import Stub, case

from mechevidence.service import create_app

AUTH = {"Authorization": "Bearer test-token"}


def client(monkeypatch, stub=None, token="test-token"):
    if token is None:
        monkeypatch.delenv("SERVICE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SERVICE_TOKEN", token)
    return TestClient(create_app(stub or Stub()))


def test_auth_and_uniform_errors(monkeypatch):
    http = client(monkeypatch)
    assert http.get("/health").json()["status"] == "ok"
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "test-token"}):
        response = http.post("/v1/evaluate", json=case(), headers=headers)
        assert response.status_code == 401 and response.json()["error_code"] == "unauthorized"
    response = http.post("/v1/evaluate", json={"question": "q"}, headers=AUTH)
    assert response.status_code == 422 and response.json()["error_code"] == "input_invalid"
    assert "mechanism" in response.json()["message"]
    response = http.post("/v1/evaluate", content=b"{not json", headers=AUTH | {"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["error_code"] == "input_invalid"
    assert http.get("/v1/capabilities", headers=AUTH).json()["error_code"] == "not_found"
    unset = client(monkeypatch, token=None)
    assert unset.post("/v1/evaluate", json=case(), headers=AUTH).json()["error_code"] == "service_not_configured"


def test_evaluate_returns_result_with_llm_calls(monkeypatch):
    response = client(monkeypatch).post("/v1/evaluate", json=case(), headers=AUTH)
    body = response.json()
    assert response.status_code == 200 and body["overall"] == "supported"
    assert [call["stage"] for call in body["llm_calls"]] == ["decompose", "review"]


def test_decomposition_failure_is_an_error_not_a_verdict(monkeypatch):
    response = client(monkeypatch, Stub(decomposition=None)).post("/v1/evaluate", json=case(), headers=AUTH)
    body = response.json()
    assert response.status_code == 502 and body["error_code"] == "decomposition_failed"
    assert "overall" not in body and body["llm_calls"][0]["error"] == "output_invalid"


def test_the_default_port_is_the_deployed_one_everywhere():
    """18563 is MechEvidence; 18561 is the query planner and must not be started by mistake."""
    import inspect
    from pathlib import Path

    from mechevidence import service

    assert service.DEFAULT_PORT == 18563
    assert "DEFAULT_PORT" in inspect.getsource(service).split("__main__")[1]  # the entry point uses it
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "docs/设计.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "18563" in text and "18561" not in text.replace("18561 是查询规划服务的", "")


def test_the_receipt_format_is_published_as_a_json_schema(monkeypatch):
    from mechevidence.receipt import SHAPE

    response = client(monkeypatch).get("/v2/contract", headers=AUTH)
    body = response.json()
    assert response.status_code == 200
    assert body["contract_version"] == "mechevidence.evaluate.v2"
    assert body["schema_version"] == "mechevidence.evaluate.v2.receipt.1"
    schema = body["receipt_schema"]
    assert schema["type"] == "object" and schema["required"] == list(SHAPE[1])
    assert schema["properties"]["llm_calls"] == {"type": "array", "items": {"$ref": "#/$defs/llm_call"}}
    assert "outputs" in schema["$defs"]["llm_call"]["required"]
    assert client(monkeypatch).get("/v2/contract").status_code == 401
