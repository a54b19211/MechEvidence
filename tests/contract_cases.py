"""Builders for v2 requests and the contract fixtures the radar's tests reuse (examples/contract-v2/*.json).

Everything here is deterministic: hashes derive from the ids, the model is `stub.Stub`. The fixtures are
regenerated with `MECH_REGEN_FIXTURES=1 python -m pytest tests/test_contract_fixtures.py` and committed; a test
compares the committed files with a fresh build, so a change to the prompts, rules or contract shows up as a diff.
"""
import copy
import hashlib
import json

from stub import TEXT, Stub, good_observations

from mechevidence import llm
from mechevidence.contract import adopt, parse_request
from mechevidence.llm import SendBlocked, archived

AUTH_REF = "auth-0001-analysis-7-workspace-3"
HEADER = "[Attachment text, version 3f9c] "


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bound(id="E1", text=TEXT, header=HEADER, processor="datafabric.clean_text.v12", **fields):
    """A material with a binding the way bioradar.material builds one: a literal header the radar wrote, then one
    slice [start, end) of a fixed version's cleaned text."""
    part_prefix = "ABSTRACT: "
    part = {"kind": "clean_text", "source": "europe_pmc", "record_id": f"PMC{id}", "version_id": sha("version:" + id),
            "raw_sha256": sha("raw:" + id), "is_demo": False, "processor_version": processor,
            "text_sha256": sha("part:" + part_prefix + text)}
    start = len(part_prefix)
    full = header + text
    segments = [{"literal": header},
                {"part": part, "start": start, "end": start + len(text), "sha256": sha(text),
                 "locators": [{"start": start, "end": start + len(text), "basis": "raw_json", "json_pointer": "/abstractText"}]}]
    return {"id": id, "title": "Controlled fixture", "text": full, "source_kind": "primary",
            "document_type": "research_article", "text_level": "abstract",
            "binding": {"sha256": sha(full), "chars": len(full), "segments": segments}, **fields}


def request(**fields):
    base = {"contract_version": "mechevidence.evaluate.v2",
            "question": "Does A cause E through M in mouse system S?", "mechanism": "A changes M; M is necessary for E.",
            "scope": {"species": {"state": "stated", "value": "mouse"}, "system": {"state": "stated", "value": "S"},
                      "disease": {"state": "not_required"}},
            "materials": [bound()], "authorization_ref": AUTH_REF, "max_llm_calls": 20}
    return {**base, **fields}


def two_material_request(**fields):
    other = bound("E2", "Other text without results.", identifiers={"pmid": "123456", "source_id": "pubmed",
                                                                   "record_id": "123456"})
    return request(materials=[bound(), other], **fields)


def key_of(raw: dict, evaluator: dict) -> str:
    from mechevidence.contract import reuse_key
    return reuse_key(parse_request(raw), evaluator)


class HookedStub(Stub):
    """Stub that behaves like chat_json towards before_attempt: one HTTP attempt per call, checked first."""

    def __init__(self, *args, outputs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts, self.outputs = [], outputs

    def __call__(self, stage, system, payload, model_cls, before_attempt=None, **kwargs):
        if before_attempt is not None:
            try:
                before_attempt({"stage": stage, "attempt_in_call": 1, "requests_before": 0,
                                "request_body_sha256": sha(stage + json.dumps(payload, sort_keys=True)),
                                "model": "stub", "endpoint_id": "stub"})
            except Exception as error:  # noqa: BLE001 - the same rule as llm.chat_json
                record = {"stage": stage, "model": "stub", "finish_reason": None, "usage": {}, "seconds": 0,
                          "requests": 0, "outputs": [], "error": getattr(error, "code", "authorization_error")}
                return None, record
        self.attempts.append(stage)
        result, record = super().__call__(stage, system, payload, model_cls, **kwargs)
        record["outputs"] = (self.outputs if self.outputs is not None
                             else [archived(1, result.model_dump_json() if result is not None else "")])
        return result, record


class Authorizer:
    """Records every check; refuses from the n-th check on when refuse_from is set."""

    def __init__(self, refuse_from=None, code="authorization_refused"):
        self.checks, self.refuse_from, self.code = [], refuse_from, code

    def __call__(self, ref, attempt):
        self.checks.append((ref, attempt))
        if self.refuse_from is not None and len(self.checks) >= self.refuse_from:
            raise SendBlocked(self.code, "revoked")


def oversize_outputs():
    return [archived(1, "x" * (llm.OUTPUT_ARCHIVE_LIMIT + 10))]


# ---- the documents committed under examples/contract-v2/ ----

def negative_requests() -> list[dict]:
    """Requests the service must refuse with 422, each with the code the radar's tests should expect."""
    def mutate(fn, base=None):
        raw = copy.deepcopy(base or request())
        fn(raw)
        return raw

    def drop_binding(raw):
        del raw["materials"][0]["binding"]

    def literal_only(raw):
        raw["materials"][0]["binding"]["segments"] = [{"literal": raw["materials"][0]["text"]}]

    def new_cleaner_text(raw):  # same version id, the text changed (processor v11 -> v12) but the old binding was sent
        raw["materials"][0]["text"] = raw["materials"][0]["text"].replace("prevented", "blocked")

    def wrong_literal(raw):
        raw["materials"][0]["binding"]["segments"][0]["literal"] = "【另一个页眉】\n\n"

    def short_segments(raw):
        raw["materials"][0]["binding"]["segments"].pop(0)

    def bad_version(raw):
        raw["materials"][0]["binding"]["segments"][1]["part"]["version_id"] = "9a1e"

    def no_processor(raw):
        del raw["materials"][0]["binding"]["segments"][1]["part"]["processor_version"]

    def twin(raw):
        second = copy.deepcopy(raw["materials"][0])
        second["id"] = "E2"
        raw["materials"].append(second)

    return [
        {"name": "missing_binding", "request": mutate(drop_binding), "status": 422, "error_code": "material_binding_missing"},
        {"name": "binding_without_part", "request": mutate(literal_only), "status": 422, "error_code": "material_binding_missing"},
        {"name": "same_version_different_text", "request": mutate(new_cleaner_text), "status": 422, "error_code": "material_binding_mismatch"},
        {"name": "literal_differs_from_text", "request": mutate(wrong_literal), "status": 422, "error_code": "material_binding_mismatch"},
        {"name": "segments_do_not_cover_text", "request": mutate(short_segments), "status": 422, "error_code": "material_binding_mismatch"},
        {"name": "version_id_not_64_hex", "request": mutate(bad_version), "status": 422, "error_code": "material_binding_invalid"},
        {"name": "part_without_processor", "request": mutate(no_processor), "status": 422, "error_code": "material_binding_invalid"},
        {"name": "duplicate_binding", "request": mutate(twin), "status": 422, "error_code": "duplicate_material_binding"},
        {"name": "unknown_field_on_material", "request": mutate(lambda raw: raw["materials"][0].update(version_id=sha("v"))),
         "status": 422, "error_code": "field_not_in_contract", "message_contains": "materials.0.version_id"},
        {"name": "unknown_top_level_field", "request": mutate(lambda raw: raw.update(binding={})),
         "status": 422, "error_code": "field_not_in_contract", "message_contains": "binding"},
        {"name": "unknown_field_in_part", "request": mutate(lambda raw: raw["materials"][0]["binding"]["segments"][1]["part"].update(cleaner="x")),
         "status": 422, "error_code": "field_not_in_contract", "message_contains": "part.cleaner"},
        {"name": "stated_without_value", "request": mutate(lambda raw: raw["scope"].update(species={"state": "stated"})),
         "status": 422, "error_code": "scope_state_invalid"},
        {"name": "unknown_with_value", "request": mutate(lambda raw: raw["scope"].update(species={"state": "unknown", "value": "mouse"})),
         "status": 422, "error_code": "scope_state_invalid"},
        {"name": "scope_dimension_missing", "request": mutate(lambda raw: raw["scope"].pop("disease")),
         "status": 422, "error_code": "scope_state_invalid"},
        {"name": "contract_version_missing", "request": mutate(lambda raw: raw.pop("contract_version")),
         "status": 422, "error_code": "contract_version_required"},
        {"name": "contract_version_v1", "request": mutate(lambda raw: raw.update(contract_version="mechevidence.evaluate.v1")),
         "status": 422, "error_code": "contract_version_required"},
        {"name": "authorization_missing", "request": mutate(lambda raw: raw.pop("authorization_ref")),
         "status": 422, "error_code": "authorization_required"},
        {"name": "text_hash_differs", "request": mutate(lambda raw: raw["materials"][0]["binding"].update(sha256=sha("other"))),
         "status": 422, "error_code": "material_binding_mismatch"},
        {"name": "body_is_not_an_object", "request": [1, 2], "status": 422, "error_code": "input_invalid"},
    ]


def _fixture_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("LLM_ENDPOINT_ID", "deepseek-public")
    monkeypatch.setenv("MECHEVIDENCE_CODE_VERSION", "fixture")


def receipt_cases(monkeypatch):
    """(request, receipt, evaluator id, outcomes): a good receipt and what the radar does with tampered ones."""
    from mechevidence.identity import current_evaluator
    from mechevidence.v2 import evaluate_v2

    _fixture_env(monkeypatch)
    evaluator = current_evaluator()
    raw = two_material_request(expected_evaluator_id=evaluator["evaluator_id"])
    receipt = json.loads(json.dumps(evaluate_v2(raw, HookedStub([good_observations(), []]), Authorizer())))
    outcomes = []

    def case(name, change):
        tampered = copy.deepcopy(receipt)
        change(tampered)
        outcomes.append({"name": name, "receipt": tampered,
                         "outcome": json.loads(json.dumps(adopt(raw, tampered, evaluator["evaluator_id"])))})

    case("exact", lambda r: None)
    case("reuse_key_differs", lambda r: r.update(reuse_key=sha("another")))
    case("binding_echo_swapped", lambda r: r["materials"].reverse())
    case("binding_echo_other_text", lambda r: r["materials"][0].update(binding_sha256=sha("other text")))
    case("evaluator_id_differs", lambda r: r["evaluator"].update(evaluator_id=sha("another evaluator")))
    case("evaluator_fields_edited_id_kept", lambda r: r["evaluator"]["model_config"].update(model="other-model"))
    case("not_complete", lambda r: r.update(complete=False, not_complete_because=["pending_budget"]))
    case("audit_incomplete", lambda r: r.update(audit_complete=False))
    case("truncated_answer", lambda r: r["llm_calls"][0]["outputs"][0].update(truncated=True))
    return raw, receipt, evaluator, outcomes


def key_cases(monkeypatch) -> dict:
    """The reuse key of a base request and of variants of it, each with whether it must differ (phase 1 key)."""
    from mechevidence.identity import current_evaluator

    _fixture_env(monkeypatch)
    evaluator = current_evaluator()
    base = two_material_request()

    def variant(name, change, differs, evaluator=evaluator):
        raw = copy.deepcopy(base)
        change(raw)
        return {"name": name, "request": raw, "evaluator_id": evaluator["evaluator_id"],
                "reuse_key": key_of(raw, evaluator), "differs_from_base": differs}

    def swapped(raw):
        raw["materials"].reverse()

    with monkeypatch.context() as scoped:
        scoped.setenv("LLM_PROVIDER", "vllm")
        scoped.setenv("LLM_MODEL", "qwen")
        scoped.setenv("LLM_ENDPOINT_ID", "zktest-vllm-qwen")
        upgraded = current_evaluator()
    cases = [
        variant("same_request_again", lambda raw: None, False),
        variant("budget_and_authorization_are_not_inputs",
                lambda raw: raw.update(max_llm_calls=3, authorization_ref="another-analysis-ref"), False),
        variant("scope_species_mouse_to_human",
                lambda raw: raw["scope"].update(species={"state": "stated", "value": "human"}), True),
        variant("scope_disease_not_required_to_unknown", lambda raw: raw["scope"].update(disease={"state": "unknown"}), True),
        variant("question_changed", lambda raw: raw.update(question="Does A cause E in human system S?"), True),
        variant("title_changed", lambda raw: raw["materials"][0].update(title="Another title"), True),
        variant("identifiers_changed", lambda raw: raw["materials"][1]["identifiers"].update(pmid="654321"), True),
        variant("same_texts_other_order", swapped, True),
        variant("same_version_new_processor",
                lambda raw: raw.update(materials=[bound(processor="datafabric.clean_text.v13"), raw["materials"][1]]), True),
        variant("alias_renamed", lambda raw: raw["materials"][0].update(id="E9"), True),
        variant("evaluator_upgraded", lambda raw: None, True, upgraded),
    ]
    return {"base": {"request": base, "evaluator_id": evaluator["evaluator_id"], "reuse_key": key_of(base, evaluator)},
            "cases": cases}


def service_errors(monkeypatch) -> list[dict]:
    """What the service answers (status and body) when it refuses or fails closed after a valid request."""
    from fastapi.testclient import TestClient

    from mechevidence.identity import current_evaluator
    from mechevidence.service import create_app

    _fixture_env(monkeypatch)
    monkeypatch.setenv("SERVICE_TOKEN", "fixture-token")
    headers = {"Authorization": "Bearer fixture-token"}

    def answer(name, raw, stub, authorizer):
        response = TestClient(create_app(stub, authorizer)).post("/v2/evaluate", json=raw, headers=headers)
        body = response.json()
        for call in body.get("llm_calls", []):
            for out in call.get("outputs", []):
                if out["truncated"]:
                    out["content"] = f"<{llm.OUTPUT_ARCHIVE_LIMIT} bytes elided from this fixture>"
        return {"name": name, "request": raw, "status": response.status_code, "body": body}

    stale = request(expected_evaluator_id=sha("the evaluator before an upgrade"))
    return [
        answer("evaluator_changed", stale, HookedStub(), Authorizer()),
        answer("authorization_revoked_before_the_second_request", request(), HookedStub(), Authorizer(refuse_from=2)),
        answer("raw_answer_over_256_KiB", request(), HookedStub(outputs=oversize_outputs()), Authorizer()),
        {"name": "current_evaluator", "status": 200, "body": {"evaluator": current_evaluator()}},
    ]


def fixture_documents(monkeypatch) -> dict:
    raw, receipt, evaluator, outcomes = receipt_cases(monkeypatch)
    return {
        "request.json": raw,
        "receipt.json": receipt,
        "evaluator.json": {"contract_version": "mechevidence.evaluate.v2", "evaluator": evaluator},
        "negative_requests.json": negative_requests(),
        "receipt_outcomes.json": {"request": raw, "expected_evaluator_id": evaluator["evaluator_id"],
                                  "reuse_key": receipt["reuse_key"], "cases": outcomes},
        "reuse_keys.json": key_cases(monkeypatch),
        "service_errors.json": service_errors(monkeypatch),
    }


def dumps(document) -> str:
    return json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True) + "\n"

