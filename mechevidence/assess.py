"""拆机制 → 分批审阅材料 → 确定性评级。模型做有出处的语义判断，程序管覆盖、范围、因果要求和冲突。"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

from pydantic import ValidationError

from .llm import chat_json
from .materials import DESIGNS, assessability, original_results, prepare
from .models import Decomposition, EvaluateRequest, Observation, Review, Scope
from .prompts import DECOMPOSE, REVIEW

BATCH_SIZE = 3
# Bump RULES_VERSION with any change to what this module (and materials.admissibility / source grouping) concludes
# from the same observations - scope aliases, comparison, grading; docs/decisions/0002 says what counts.
RULES_VERSION = "mechevidence.aggregate.v1"
# Phase 1 reviews materials in the order the request lists them, in batches of BATCH_SIZE; a canonical order is a
# later rules version, and only then may a reuse key treat the materials as a set.
ORDERING_RULE = "request_order"
_BUDGET = object()


class EvaluationFailed(RuntimeError):
    """执行失败（不是科学结论）；带上已发生的模型调用记录。"""

    def __init__(self, code: str, message: str, llm_calls: list[dict]):
        self.code, self.llm_calls = code, llm_calls
        super().__init__(message)


# ---- 范围匹配 ----

def _known(value) -> bool:
    return value is not None and str(value).strip().casefold() not in {
        "", "unknown", "unspecified", "not reported", "not stated", "n/a", "未知", "未说明",
    }


def _normal(value) -> str:
    return " ".join(str(value or "").casefold().split())


_SPECIES_ALIASES = {
    "mouse": "mouse", "mice": "mouse", "murine": "mouse", "mus musculus": "mouse", "小鼠": "mouse",
    "human": "human", "humans": "human", "homo sapiens": "human", "人": "human", "人类": "human",
    "rat": "rat", "rats": "rat", "rattus norvegicus": "rat", "大鼠": "rat",
    "zebrafish": "zebrafish", "danio rerio": "zebrafish", "斑马鱼": "zebrafish",
}


def _species(value) -> str:
    normalized = _normal(value)
    return _SPECIES_ALIASES.get(normalized, normalized)


def _species_conflict(left, right) -> bool:
    left, right = _species(left), _species(right)
    return left in _SPECIES_ALIASES.values() and right in _SPECIES_ALIASES.values() and left != right


def _scope_status(observation: dict, scope: Scope) -> str:
    """已知物种冲突覆盖模型标记；目标维度已限定而观察未报告时是 unknown，unknown 不算匹配。"""
    context, match = observation["context"], observation["scope_match"]
    if _species_conflict(scope.species, context["species"]):
        return "mismatched"
    statuses = []
    for dimension in ("species", "system", "disease"):
        requested = getattr(scope, dimension)
        if requested is not None and str(requested).strip() and not _known(requested):
            statuses.append("unknown")
        elif _known(requested):
            reported = match[dimension]
            statuses.append(reported if _known(context[dimension])
                            and reported in {"matched", "mismatched"} else "unknown")
    if "mismatched" in statuses:
        return "mismatched"
    if "unknown" in statuses or not any(_known(value) for value in context.values()):
        return "unknown"
    return "matched"


def _comparison(left: dict, right: dict) -> str:
    """相反结果的情境：已知物种不同 → different；物种、系统、疾病都报告且一致 → comparable；其余 unknown。"""
    a, b = left["context"], right["context"]
    if _species_conflict(a["species"], b["species"]):
        return "different"
    same = (_known(a["species"]) and _known(b["species"]) and _species(a["species"]) == _species(b["species"])
            and _known(a["system"]) and _known(b["system"]) and _normal(a["system"]) == _normal(b["system"])
            and (_normal(a["disease"]) if _known(a["disease"]) else None)
            == (_normal(b["disease"]) if _known(b["disease"]) else None))
    return "comparable" if same else "unknown"


# ---- 确定性评级 ----

class _Buckets(NamedTuple):
    """Observation ids per target obligation, by how the observation bears on it (matched scope only)."""
    positive: defaultdict
    negative: defaultdict
    compatible: defaultdict
    unknown_opposition: defaultdict


def _annotate(scope: Scope, observations: list[dict], materials: dict, groups: dict) -> list[dict]:
    """Stamp each observation with scope status, causal-evidence flag and source group; return the species notes."""
    notes = []
    for item in observations:
        material = materials[item["material"]]
        item["scope_status"] = _scope_status(item, scope)
        # 因果证据 = 干预 + 直接检验了该因果问题 + 原始研究结果
        item["causal_evidence"] = (item["basis"] == "intervention" and item["causal_test"]
                                   and original_results(material))
        item["source_group"] = groups[item["material"]]
        if _species_conflict(scope.species, item["context"]["species"]) and item["scope_match"]["species"] != "mismatched":
            notes.append({"material": item["material"], "code": "species_conflict", "message": (
                f"{item['id']} 记录的物种 {item['context']['species']} 与目标 {scope.species} 明确不同，已覆盖模型的适用性标记；"
                "保留原观察，不计为目标范围的支持或反驳。")})
    return notes


def _bucket(claims: list[dict], observations: list[dict], materials: dict) -> _Buckets:
    """Sort every observation into supporting / refuting / compatible / unknown-context opposition per target."""
    claim_by_id = {item["id"]: item for item in claims}
    buckets = _Buckets(*(defaultdict(list) for _ in range(4)))
    for item in observations:
        material = materials[item["material"]]
        for target in item["targets"]:
            claim = claim_by_id.get(target)
            strict = claim is None or claim["kind"] == "causal"
            eligible = item["causal_evidence"] if strict else (claim["kind"] == "reported" or original_results(material))
            # reported 命题也可以断言结果；登记/方案不能借此绕过可评性
            eligible = eligible and material.document_type not in DESIGNS
            if item["scope_status"] == "unknown" and item["relation"] == "refutes" and eligible:
                buckets.unknown_opposition[target].append(item["id"])
            if item["scope_status"] != "matched":
                continue
            if item["relation"] == "supports" and eligible:
                buckets.positive[target].append(item["id"])
            elif item["relation"] == "refutes" and eligible:
                buckets.negative[target].append(item["id"])
            elif item["relation"] != "refutes":
                buckets.compatible[target].append(item["id"])
    return buckets


def _pair_opposition(required: list[dict], buckets: _Buckets, by_id: dict):
    """Compare every supporting with every opposing observation of a target: conflicts, unresolved, contexts."""
    conflicts, unresolved, conflicted, differing = [], [], set(), defaultdict(list)
    for target in [item["id"] for item in required]:
        for support in buckets.positive[target]:
            for opposition in buckets.negative[target]:
                relation = _comparison(by_id[support], by_id[opposition])
                pair = {"target": target, "supporting": support, "opposing": opposition}
                if relation == "comparable":
                    conflicts.append({**pair, "reason": "相反结果涉及同一义务，且实验背景可比较。"})
                    conflicted.add(target)
                elif relation == "unknown":
                    unresolved.append({**pair, "reason": "相反结果的实验背景尚不能确定可比；不能视为已解决的反证。"})
                else:
                    differing[target].append({"observation": opposition, "status": "different_evidence_context",
                                              "compared_with": support, "context": by_id[opposition]["context"]})
        if buckets.positive[target]:
            for opposition in buckets.unknown_opposition[target]:
                unresolved.append({"target": target, "supporting": None, "opposing": opposition,
                                   "reason": "潜在相反结果的适用背景不明；不能判为反驳，也不能忽略。"})
    return conflicts, unresolved, conflicted, differing


def _obligation_row(item: dict, is_claim: bool, observations: list[dict], by_id: dict, buckets: _Buckets,
                    conflicted: set, differing: dict) -> dict:
    key, supporting, opposing = item["id"], buckets.positive[item["id"]], buckets.negative[item["id"]]
    if is_claim:
        verdict = ("mixed" if key in conflicted else "supported" if supporting
                   else "refuted" if opposing else "insufficient")
    else:
        verdict = ("supported" if supporting and key not in conflicted
                   else "compatible_only" if buckets.compatible[key] and not opposing else "unestablished")
    relevant = [obs for obs in observations if key in obs["targets"]]
    return {**item, "verdict": verdict, "supporting": supporting, "opposing": opposing,
            "compatible": buckets.compatible[key],
            "applicability_gaps": [{"observation": obs["id"], "status": obs["scope_status"], "context": obs["context"]}
                                   for obs in relevant if obs["scope_status"] != "matched"] + differing[key],
            "source_groups": sorted({by_id[oid]["source_group"] for oid in supporting})}


def _overall(required: list[dict], buckets: _Buckets, by_id: dict, conflicts: list, unresolved: list):
    """Overall grade and whether one species-consistent witness covers every obligation."""
    positive = buckets.positive

    # 整体见证：每项义务都有适用支持；不同物种的结果不拼成一条因果链
    def observed_species(oid):
        value = by_id[oid]["context"]["species"]
        return _species(value) if _known(value) else None

    species = {observed_species(oid) for item in required for oid in positive[item["id"]]} - {None}
    witness = bool(required) and any(
        all(any(chosen is None or observed_species(oid) in {None, chosen} for oid in positive[item["id"]])
            for item in required)
        for chosen in (sorted(species) if len(species) > 1 else [None]))
    if witness and not conflicts and not unresolved:
        overall = "supported"
    elif any(positive[item["id"]] for item in required):
        overall = "partial"
    else:
        overall = "insufficient"
    return witness, overall


def _gaps(scope: Scope, required: list[dict], buckets: _Buckets, unresolved: list, witness: bool,
          has_material: bool) -> list[str]:
    """What an expert should look up next, one sentence per obligation or global shortfall."""
    positive, negative = buckets.positive, buckets.negative
    scope_text = "，".join(f"{key}={value}" for key, value in scope.model_dump().items() if _known(value)) or "未限定"
    gaps = []
    for item in required:
        if not positive[item["id"]]:
            gaps.append(f"{item['id']}：本材料包中已有适用的反向结果；应先修正候选主张或查清可比较条件。"
                        f"补查反向结果的原始方法、终点及适用条件，供专家核对并修正主张：{item['text']}"
                        if negative[item["id"]] else
                        f"{item['id']}：本材料包中没有适用证据确立这项具体义务。补查目标背景（{scope_text}）中与该义务对应的"
                        f"原始实验观察及方法、对照和终点信息，供专家审核：{item['text']}")
    everything = ",".join(item["id"] for item in required) or "整体"
    if not has_material:
        gaps.append(f"{everything}：没有提供允许使用的原文段落。获取原始报告中与候选机制相关的方法和实验结果，再评价机制支持程度。")
    elif not witness and required and all(positive[item["id"]] for item in required):
        gaps.append(f"{everything}：局部证据虽有覆盖，尚无背景一致的整体机制见证。补查在同一目标生物系统中报告指定中介依赖及相应终点的"
                    "原始证据，包括已有的中介阻断或救援结果，供专家审核；细胞或动物终点不替代患者终点证据。")
    texts = {item["id"]: item["text"] for item in required}
    for counter in unresolved:
        gaps.append(f"{counter['target']}：{counter['reason']}补查相反结果的物种、系统、剂量、时间及终点的原始记录，"
                    f"供专家判断该义务的证据是否可比：{texts[counter['target']]}")
    return gaps


_RATIONALE = {"supported": "存在覆盖全部义务且背景一致的有效见证。",
              "partial": "部分义务获得适用支持；整体机制仍有覆盖、衔接或相反结果问题。",
              "insufficient": "现有适用证据不足以确立所需义务。"}


def aggregate(scope: Scope, claims: list[dict], joins: list[dict], observations: list[dict],
              materials: dict, groups: dict, has_material: bool = True) -> dict:
    by_id = {item["id"]: item for item in observations}
    notes = _annotate(scope, observations, materials, groups)
    buckets = _bucket(claims, observations, materials)
    required = claims + joins
    conflicts, unresolved, conflicted, differing = _pair_opposition(required, buckets, by_id)
    witness, overall = _overall(required, buckets, by_id, conflicts, unresolved)
    gaps = _gaps(scope, required, buckets, unresolved, witness, has_material)

    rationale = _RATIONALE[overall]
    if notes:
        rationale += "部分观察记录的物种与目标明确不符，已从目标范围的支持和反驳中排除。"

    def rows(items: list[dict], is_claim: bool) -> list[dict]:
        return [_obligation_row(item, is_claim, observations, by_id, buckets, conflicted, differing) for item in items]

    return {"overall": overall, "rationale": rationale,
            "claims": rows(claims, True), "joins": rows(joins, False),
            "observations": observations, "conflicts": conflicts, "unresolved_counterevidence": unresolved,
            "gaps": list(dict.fromkeys(gaps)),
            "independence": {"distinct_source_groups": len({obs["source_group"] for obs in observations}),
                             "note": "来源去重不等于确认队列或实验相互独立。"},
            "species_notes": notes}


# ---- 两次模型调用 ----

def _number(decomposition, notes: list[dict]) -> tuple[list[dict], list[dict]]:
    claims = [{"id": f"C{index}", "text": item.text, "kind": item.kind}
              for index, item in enumerate(decomposition.claims, 1)]
    joins = []
    for item in decomposition.joins:
        if not (1 <= item.from_claim <= len(claims) and 1 <= item.to_claim <= len(claims)) or item.from_claim == item.to_claim:
            notes.append({"material": None, "code": "join_invalid",
                          "message": f"因果连接“{item.text}”的端点下标无效（{item.from_claim}→{item.to_claim}），已丢弃。"})
            continue
        joins.append({"id": f"J{len(joins) + 1}", "text": item.text,
                      "from": f"C{item.from_claim}", "to": f"C{item.to_claim}"})
    return claims, joins


def _squash(text: str) -> str:
    return " ".join(text.split())


def _passage_id(value: str, single: str | None) -> str:
    value = value.strip()
    match = re.fullmatch(r"([A-Za-z0-9_-]+)[.:]P?(\d+)", value, re.IGNORECASE)
    if match:
        return f"{match.group(1)}.{int(match.group(2))}"
    if value.isdigit() and single:
        return f"{single}.{int(value)}"
    return value


def _accept(raw, passages: dict, target_ids: set, single: str | None):
    """返回 (观察, None) 或 (None, (material, code, message))。"""
    cited = str(raw.get("passage", "")) if isinstance(raw, dict) else ""
    guess = _passage_id(cited, single).split(".")[0]
    guess = guess if guess in {p["material"] for p in passages.values()} else single
    try:
        observation = Observation.model_validate(raw)
    except ValidationError as error:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        return None, (guess, "observation_invalid", detail[:300])
    pid = _passage_id(observation.passage, single)
    if pid not in passages:
        return None, (guess, "passage_unknown", f"引用了不存在的片段 {observation.passage}")
    passage = passages[pid]
    unknown = [target for target in observation.targets if target not in target_ids]
    if unknown:
        return None, (passage["material"], "target_unknown", f"{pid} 的观察引用了不存在的目标 {unknown}")
    if _squash(observation.quote) not in _squash(passage["text"]):
        return None, (passage["material"], "quote_not_found", f"{pid} 中找不到短引：{observation.quote[:120]}")
    return {"targets": list(dict.fromkeys(observation.targets)), "material": passage["material"], "passage": pid,
            "quote": observation.quote, "relation": observation.relation, "basis": observation.basis,
            "causal_test": observation.causal_test, "context": observation.context.model_dump(),
            "scope_match": observation.scope_match.model_dump()}, None


Chat = Callable[..., tuple]


def _review_batches(req: EvaluateRequest, usable: list[dict], targets: list[dict], call, calls: list[dict],
                    notes: list[dict]) -> tuple[list[dict], set]:
    """Phase 1: review the usable materials batch by batch; returns the accepted observations and the ids of
    materials that were not reviewed (notes gets one entry per failure or rejected observation)."""
    target_ids = {item["id"] for item in targets}
    observations, failed = [], set()
    for start in range(0, len(usable) if targets else 0, BATCH_SIZE):
        batch = usable[start:start + BATCH_SIZE]
        ids = [item["material"].id for item in batch]
        passages = {p["id"]: {**p, "material": item["material"].id} for item in batch for p in item["passages"]}
        payload = {"question": req.question, "mechanism": req.mechanism, "scope": req.scope.model_dump(),
                   "targets": targets,
                   "materials": [{**item["material"].model_dump(include={"id", "title", "source_kind", "document_type", "text_level"}),
                                  "passages": item["passages"]} for item in batch]}
        review = call("review", REVIEW, payload, Review)
        if review is None or review is _BUDGET:
            failed.update(ids)
            code, message = (("pending_budget", f"已达 max_llm_calls={req.max_llm_calls}，未审阅，待核。")
                             if review is _BUDGET else
                             ("review_failed", f"材料审阅失败（{calls[-1].get('error')}），未审阅，待核。"))
            notes += [{"material": mid, "code": code, "message": message} for mid in ids]
            continue
        for raw in review.observations:
            observation, problem = _accept(raw, passages, target_ids, ids[0] if len(ids) == 1 else None)
            if problem:
                notes.append({"material": problem[0], "code": problem[1], "message": problem[2]})
            else:
                observations.append({"id": f"O{len(observations) + 1}", **observation})
    return observations, failed


def evaluate(request: EvaluateRequest | dict, chat: Chat = chat_json) -> dict:
    req = EvaluateRequest.model_validate(request)
    usable, notes = prepare(req.materials)
    calls: list[dict] = []

    def call(stage, system, payload, model_cls):
        if len(calls) >= req.max_llm_calls:
            return _BUDGET
        result, record = chat(stage, system, payload, model_cls)
        calls.append(record)
        return result

    if req.claims is not None:
        claims, joins = _number(Decomposition.model_construct(claims=req.claims, joins=req.joins), notes)
    elif usable:
        decomposition = call("decompose", DECOMPOSE, {"question": req.question, "mechanism": req.mechanism,
                                                      "scope": req.scope.model_dump()}, Decomposition)
        if decomposition is None:
            raise EvaluationFailed("decomposition_failed",
                                   f"机制拆解失败（{calls[-1].get('error')}）；未产生科学判断。", calls)
        claims, joins = _number(decomposition, notes)
    else:
        claims, joins = [], []  # 没有可评材料时不调模型

    targets = claims + joins
    observations, failed = _review_batches(req, usable, targets, call, calls, notes)

    result = aggregate(req.scope, claims, joins, observations,
                       {item["material"].id: item["material"] for item in usable},
                       {item["material"].id: item["group"] for item in usable}, has_material=bool(usable))
    notes += result.pop("species_notes")
    return {**result, "material_notes": notes, "assessability": assessability(usable, failed), "llm_calls": calls}
