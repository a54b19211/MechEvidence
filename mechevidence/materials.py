"""材料可评性、文献身份（PMID/DOI 合并）和片段编号。"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

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


# ---- 文献身份：只认整字段标签、裸 DOI 或官方 URL，不从任意子串猜测 ----

class IdentityError(ValueError):
    def __init__(self, code: str, kind: str, fields: dict):
        self.code, self.kind, self.fields = code, kind, fields
        super().__init__(f"{code}: {kind} {fields}")


_PMID = r"[1-9][0-9]*"
_DOI = r'10\.[0-9]{4,9}/[^\s\x00-\x1f\x7f<>"\\]+'


def _identifier_value(value, kind: str, field: str) -> str:
    text = str(value).strip()
    if isinstance(value, bool) or not re.fullmatch(_PMID if kind == "pmid" else _DOI, text):
        raise IdentityError("material_identifier_invalid", kind, {field: value})
    # DOI 末尾的 ')' 或 '.' 属于标识符本身，不能剥掉
    return text if kind == "pmid" else text.casefold()


def _url_identifier(value: str) -> tuple[str, str] | None:
    if ("\\" in value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            or re.search(r"%(?![0-9a-fA-F]{2})", value)):
        return None
    try:
        url = urlsplit(value)
        if (url.scheme not in {"http", "https"} or url.username is not None or url.password is not None
                or url.port not in {None, 443 if url.scheme == "https" else 80}):
            return None
        routes = {
            "pubmed.ncbi.nlm.nih.gov": rf"/({_PMID})/?",
            "ncbi.nlm.nih.gov": rf"/pubmed/({_PMID})/?",
            "www.ncbi.nlm.nih.gov": rf"/pubmed/({_PMID})/?",
            "europepmc.org": rf"/article/MED/({_PMID})/?",
            "www.europepmc.org": rf"/article/MED/({_PMID})/?",
        }
        if url.hostname in routes:
            match = re.fullmatch(routes[url.hostname], url.path)
            return ("pmid", match.group(1)) if match else None
        if url.hostname in {"doi.org", "dx.doi.org"}:
            doi = unquote(url.path[1:], errors="strict") if url.path.startswith("/") else ""
            return ("doi", doi) if re.fullmatch(_DOI, doi) else None
    except (ValueError, UnicodeError):
        return None
    return None


def _tagged_identifier(value: str) -> tuple[str, str] | None:
    # 冒号或斜杠才声明标识符；普通分组标签保持原样
    for pattern, kind in [(r"(?:pmid|pubmed)\s*[:/]\s*(.*)", "pmid"), (rf"(?:pmid|pubmed)\s*({_PMID})", "pmid"),
                          (r"doi\s*[:/]\s*(.*)", "doi"), (rf"doi\s+({_DOI})", "doi")]:
        match = re.fullmatch(pattern, value, re.I)
        if match:
            return kind, match.group(1)
    if re.fullmatch(_DOI, value):
        return "doi", value
    return _url_identifier(value)


def source_identity(material: Material) -> str | None:
    """同一篇文献的不同记录合并为一组（pmid:/doi:/group:/record:）；无标识返回 None。

    这是文献身份，不是队列或实验独立性。
    """
    candidates = {"pmid": {}, "doi": {}}
    for kind in candidates:
        value = getattr(material, kind)
        if value is not None:
            candidates[kind][kind] = _identifier_value(value, kind, kind)
    source = material.source_id.strip().casefold().replace("-", "_")
    for field in ("source_group", "record_id", "url"):
        value = (getattr(material, field) or "").strip()
        if not value:
            continue
        parsed = _url_identifier(value) if field == "url" else _tagged_identifier(value)
        if parsed:
            candidates[parsed[0]][field] = _identifier_value(parsed[1], parsed[0], field)
        elif field == "record_id" and source == "pubmed":
            candidates["pmid"][field] = _identifier_value(value, "pmid", field)
        elif field == "record_id" and source in {"europe_pmc", "europepmc"} and value.upper().startswith("MED:"):
            candidates["pmid"][field] = _identifier_value(value[4:], "pmid", field)
    # 两类都检查完再取优先级，PMID 不能掩盖互相矛盾的 DOI
    for kind, fields in candidates.items():
        if len(set(fields.values())) > 1:
            raise IdentityError("material_identity_conflict", kind, fields)
    for kind in ("pmid", "doi"):
        if candidates[kind]:
            return kind + ":" + next(iter(candidates[kind].values()))
    if material.source_group:
        return "group:" + " ".join(material.source_group.casefold().split())
    if material.record_id:
        return "record:" + " ".join((material.source_id + ":" + material.record_id).casefold().split())
    return None


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
