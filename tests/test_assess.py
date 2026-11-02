"""评级规则：范围、因果要求、冲突、整体结论；以及坏观察只丢一条。"""
import pytest

from mechevidence import EvaluationFailed, evaluate
from stub import Stub, case, good_observations, material, obs


def by_id(rows):
    return {row["id"]: row for row in rows}


def test_overall_supported_partial_insufficient():
    result = evaluate(case(), Stub())
    assert result["overall"] == "supported"
    assert [row["verdict"] for row in result["claims"]] == ["supported", "supported"]
    assert result["joins"][0]["verdict"] == "supported"
    assert result["joins"][0]["from"] == "C1" and result["joins"][0]["to"] == "C2"
    assert result["gaps"] == []

    # 两个原子命题都成立，也不能替代未检验的因果连接
    atomic = [obs(["C1"], "E1.1", "A changed M"), obs(["C2"], "E1.2", "Blocking M prevented the E response")]
    result = evaluate(case(), Stub([atomic]))
    assert result["overall"] == "partial"
    assert result["joins"][0]["verdict"] == "unestablished"
    assert any(gap.startswith("J1：") and "没有适用证据" in gap for gap in result["gaps"])

    result = evaluate(case(), Stub([[]]))
    assert result["overall"] == "insufficient"
    assert {gap.split("：")[0] for gap in result["gaps"]} == {"C1", "C2", "J1"}


def test_scope_species_alias_and_known_conflict():
    for target, observed in [("小鼠", "Mus musculus"), ("mice", "murine")]:
        observations = good_observations()
        for row in observations:
            row["context"]["species"] = observed
        assert evaluate(case(scope={"species": target, "system": "S"}), Stub([observations]))["overall"] == "supported"

    # 模型标了 matched，但已知物种明确不同：程序覆盖，不计入目标范围的支持
    result = evaluate(case(scope={"species": "human", "system": "S"}), Stub())
    assert result["overall"] == "insufficient"
    assert all(row["scope_status"] == "mismatched" for row in result["observations"])
    assert all(not row["supporting"] and row["applicability_gaps"] for row in result["claims"])
    assert [note["code"] for note in result["material_notes"]] == ["species_conflict", "species_conflict"]


def test_unknown_context_is_not_matched():
    observations = good_observations()
    observations[0]["context"]["system"] = None
    result = evaluate(case(), Stub([observations]))
    assert result["observations"][0]["scope_status"] == "unknown"
    assert by_id(result["claims"])["C1"]["verdict"] == "insufficient"
    assert result["overall"] == "partial"
    # 目标物种写成 unknown 不是“未限定”
    assert evaluate(case(scope={"species": "unknown", "system": "S"}), Stub())["overall"] == "insufficient"
    # 目标完全未限定时，只要观察报告了情境就算适用
    assert evaluate(case(scope={}), Stub())["overall"] == "supported"


def test_causal_claims_need_intervention_causal_test_and_original_result():
    for change in [{"basis": "association"}, {"causal_test": False}]:
        observations = [obs(**{**row, **change}) for row in good_observations()]
        result = evaluate(case(), Stub([observations]))
        assert result["overall"] == "insufficient"
        assert by_id(result["claims"])["C1"]["compatible"] == ["O1"]
    # 综述里的干预描述不是原始实验
    review = material(source_kind="review", document_type="review")
    assert evaluate(case(materials=[review]), Stub())["overall"] == "insufficient"
    # reported 命题可以由原始研究的非干预观察支持，但登记/方案不行
    reported = {"claims": [{"text": "M is expressed in S", "kind": "reported"}], "joins": []}
    association = [obs(["C1"], "E1.1", "A changed M", basis="association", causal_test=False)]
    assert evaluate(case(), Stub([association], reported))["overall"] == "supported"
    protocol = material(document_type="study_protocol")
    assert evaluate(case(materials=[protocol]), Stub([association], reported))["overall"] == "insufficient"


def test_conflict_only_when_contexts_comparable():
    refuting = obs(["C1"], "E1.1", "A changed M", relation="refutes")
    result = evaluate(case(), Stub([good_observations() + [refuting]]))
    assert [item["target"] for item in result["conflicts"]] == ["C1"]
    assert by_id(result["claims"])["C1"]["verdict"] == "mixed"
    assert result["overall"] == "partial"

    # 系统描述不同：不能确定可比 → 未解决的反证，不是冲突，也不能忽略
    refuting["context"]["system"] = "S in culture"
    result = evaluate(case(), Stub([good_observations() + [refuting]]))
    assert result["conflicts"] == [] and len(result["unresolved_counterevidence"]) == 1
    assert by_id(result["claims"])["C1"]["verdict"] == "supported"
    assert result["overall"] == "partial"
    assert any("不能确定可比" in gap for gap in result["gaps"])

    # 适用背景不明的反向结果同样保留为未解决
    refuting["context"] = {"species": None, "system": None, "disease": None}
    result = evaluate(case(), Stub([good_observations() + [refuting]]))
    assert result["conflicts"] == [] and result["unresolved_counterevidence"][0]["supporting"] is None

    # 物种不同的反向结果不反驳目标范围
    refuting["context"] = {"species": "human", "system": "S", "disease": None}
    result = evaluate(case(), Stub([good_observations() + [refuting]]))
    assert result["overall"] == "supported" and not result["conflicts"] and not result["unresolved_counterevidence"]
    assert by_id(result["claims"])["C1"]["applicability_gaps"][0]["status"] == "mismatched"


def test_refutation_without_support_gives_refuted_and_correction_gap():
    result = evaluate(case(), Stub([[obs(["C1"], "E1.1", "A changed M", relation="refutes")]]))
    assert by_id(result["claims"])["C1"]["verdict"] == "refuted"
    assert any(gap.startswith("C1：") and "修正主张" in gap for gap in result["gaps"])


def test_invalid_observations_are_dropped_not_fatal():
    bad = [obs(["C9"], "E1.1", "A changed M"),                    # 不存在的目标
           obs(["C1"], "E7.1", "A changed M"),                    # 不存在的材料
           obs(["C1"], "E1.99", "A changed M"),                   # 不存在的片段
           {"targets": ["C1"], "passage": "E1.1", "quote": "A changed M"},  # 缺 relation/basis
           obs(["C1"], "E1.1", "A changed M", relation="maybe"),  # 枚举错误
           "not an object"]
    result = evaluate(case(), Stub([bad + good_observations()]))
    assert result["overall"] == "supported"
    assert [row["id"] for row in result["observations"]] == ["O1", "O2"]
    codes = [note["code"] for note in result["material_notes"]]
    assert codes == ["target_unknown", "passage_unknown", "passage_unknown",
                     "observation_invalid", "observation_invalid", "observation_invalid"]


def test_quote_not_found_is_dropped():
    paraphrase = [obs(["C1"], "E1.1", "A increased M"), obs(["C2", "J1"], "E1.1", "Blocking M prevented the E response")]
    result = evaluate(case(), Stub([paraphrase]))
    assert result["observations"] == [] and result["overall"] == "insufficient"
    assert [note["code"] for note in result["material_notes"]] == ["quote_not_found", "quote_not_found"]
    # 空白差异不算改写
    spaced = [obs(["C1"], "E1.1", "A  changed\nM")]
    assert evaluate(case(), Stub([spaced]))["observations"][0]["quote"] == "A  changed\nM"


def test_abstract_only_material_can_support_but_is_limited():
    result = evaluate(case(), Stub())
    assert result["overall"] == "supported"
    assert result["assessability"]["status"] == "limited"
    assert any("摘要" in item for item in result["assessability"]["limitations"])
    assert evaluate(case(materials=[material(text_level="full")]), Stub())["assessability"]["status"] == "assessable"


def test_batches_of_three_budget_and_failed_batch():
    materials = [material(id=f"E{i}") for i in range(1, 8)]
    stub = Stub([None, [obs(["C1"], "E4.1", "A changed M")]])
    result = evaluate(case(materials=materials, max_llm_calls=3), stub)
    reviews = [payload for stage, payload in stub.calls if stage == "review"]
    assert [[m["id"] for m in p["materials"]] for p in reviews] == [["E1", "E2", "E3"], ["E4", "E5", "E6"]]
    assert [p["id"] for p in reviews[0]["materials"][0]["passages"]] == ["E1.1", "E1.2"]
    assert len(result["llm_calls"]) == 3
    notes = {note["material"]: note["code"] for note in result["material_notes"]}
    assert notes == {"E1": "review_failed", "E2": "review_failed", "E3": "review_failed", "E7": "pending_budget"}
    assert result["observations"][0]["material"] == "E4" and result["overall"] == "partial"
    assert any("未完成审阅" in item for item in result["assessability"]["limitations"])


def test_decomposition_numbering_and_failure():
    decomposition = {"claims": [{"text": "A changes M"}, {"text": "M is necessary for E"}],
                     "joins": [{"text": "bad", "from_claim": 1, "to_claim": 5}, {"text": "A→E via M", "from_claim": 1, "to_claim": 2}]}
    stub = Stub([[obs(["C2", "J1"], "E1.2", "Blocking M prevented the E response")]], decomposition)
    result = evaluate(case(), stub)
    assert [row["id"] for row in result["joins"]] == ["J1"] and result["joins"][0]["text"] == "A→E via M"
    assert result["material_notes"][0]["code"] == "join_invalid"
    assert [t["id"] for t in stub.calls[1][1]["targets"]] == ["C1", "C2", "J1"]
    with pytest.raises(EvaluationFailed) as error:
        evaluate(case(), Stub(decomposition=None))
    assert error.value.code == "decomposition_failed" and len(error.value.llm_calls) == 1


def test_no_usable_material_makes_no_model_call():
    stub = Stub()
    result = evaluate(case(materials=[material(document_type="press_release")]), stub)
    assert stub.calls == [] and result["overall"] == "insufficient"
    assert result["material_notes"][0]["code"] == "material_type_not_evaluable"
    assert result["assessability"]["status"] == "unassessable"
    assert any("没有提供允许使用的原文段落" in gap for gap in result["gaps"])
