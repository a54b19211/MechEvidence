"""材料可评性、文献身份（PMID/DOI 合并）和片段编号。"""
from __future__ import annotations

import re

from extraction_contract.mech_v2 import IdentityError
from extraction_contract.mech_v2 import source_identity as identity_of

from .models import Material

DECLARATIONS = {"press_release", "company_disclosure", "news", "social_post"}
DESIGNS = {"trial_registration", "study_protocol"}
_EXPECTED_KIND = {"research_article": "primary", "conference_abstract": "primary", "trial_results": "primary",
                  "trial_registration": "primary", "study_protocol": "primary", "review": "review", "graph": "graph"}


def admissibility(material: Material) -> tuple[str, str] | None:
    """返回 (code, message) 表示这份材料不进入评价；None 表示可评。"""
    kind = material.document_type
    if kind in DECLARATIONS:
        return ("material_type_not_evaluable",
                "新闻、新闻稿、公司声明或社媒属于检索线索，不是原始机制或研究记录；请雷达保留并取得原始研究的方法和结果。")
    if kind in _EXPECTED_KIND and material.source_kind != _EXPECTED_KIND[kind]:
        return ("material_type_conflict",
                f"document_type={kind} 与 source_kind={material.source_kind} 冲突；请按原始记录核对分类，不要为了入选而改标签。")
    return None


def original_results(material: Material) -> bool:
    """只有原始研究报告的结果能作为因果证据；登记/方案只是设计或状态背景。"""
    return material.source_kind == "primary" and material.document_type not in DESIGNS | DECLARATIONS


# ---- 文献身份：规则在共享契约包里（reuse key 依赖它），这里只把 Material 交给它 ----

def source_identity(material: Material) -> str | None:
    """同一篇文献的不同记录合并为一组（pmid:/doi:/group:/record:）；无标识返回 None。

    这是文献身份，不是队列或实验独立性。
    """
    return identity_of({name: getattr(material, name) for name in (
        "pmid", "doi", "url", "source_id", "record_id", "source_group")})


# ---- 片段：按换行和句末标点切分，每份材料编号 E1.1..E1.k ----

_BOUNDARY = re.compile(r"\n+|(?<=[.!?。！？])\s+(?=[A-Z0-9一-鿿Α-Ω])")


def split_passages(material: Material) -> list[dict]:
    text, passages = material.text, []
    bounds = [0] + [match.end() for match in _BOUNDARY.finditer(text)] + [len(text)]
    for start, end in zip(bounds, bounds[1:]):
        piece = text[start:end].strip()
        if piece:
            passages.append({"id": f"{material.id}.{len(passages) + 1}", "text": piece})
    return passages


def prepare(materials: list[Material]) -> tuple[list[dict], list[dict]]:
    """返回 (可评材料 [{material, group, passages}], 不可评材料的 material_notes)。"""
    usable, notes, by_text = [], [], {}
    for material in materials:
        problem = admissibility(material)
        try:
            group = source_identity(material)
        except IdentityError as error:
            problem = problem or (error.code, f"文献标识无效或互相矛盾（{error.kind}: {error.fields}）；请按原始记录更正，不猜测是否同一文献。")
        if problem:
            notes.append({"material": material.id, "code": problem[0], "message": problem[1]})
            continue
        # 无标识时，正文完全相同的材料视为同一来源，不因重复副本增加来源数
        group = group or "text:" + by_text.setdefault(" ".join(material.text.split()), material.id)
        usable.append({"material": material, "group": group, "passages": split_passages(material)})
    return usable, notes


def assessability(usable: list[dict], failed: set[str]) -> dict:
    materials = [item["material"] for item in usable if item["material"].id not in failed]
    primary = [item for item in materials if original_results(item)]
    status = ("unassessable" if not materials else
              "assessable" if any(item.text_level == "full" for item in primary) else "limited")
    limitations = []
    if any(item.document_type in DESIGNS for item in materials):
        limitations.append("登记与研究方案仅作设计或状态背景，不贡献机制结果支持、反驳或已实施研究/患者关系。")
    if not materials:
        limitations.append("没有可供评价的原文段落。")
    elif not primary:
        limitations.append("现有材料不含原始研究报告段落。")
    if any(item.text_level == "abstract" for item in materials):
        limitations.append("摘要报告的实验缺少完整方法和结果细节。")
    if any(item.text_level == "excerpt" for item in materials):
        limitations.append("选取段落不能证明已覆盖原报告的全部相关结果。")
    if failed:
        limitations.append(f"材料 {', '.join(sorted(failed))} 未完成审阅，结论未覆盖这些材料。")
    return {"status": status, "limitations": limitations}
