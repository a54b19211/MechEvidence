"""复用 Data-Fabric 的只读历史接口，不在本项目重新采集、清洗或建图。"""
from __future__ import annotations

import html
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .model import read_settings


def fetch_records(terms: list[str], record_ids: list[str] | None = None,
                  limit: int = 50, source: str = "europe_pmc",
                  cutoff: str | None = None, cursor: str | None = None) -> dict:
    """返回完整响应，保留覆盖范围和下一页游标；不把一页数据说成检索全集。"""
    config = Path(os.environ.get("FABRIC_CONFIG", str(Path.home() / ".config/bioradar/runtime.env")))
    settings = read_settings(config) | dict(os.environ)
    url = settings.get("BIORADAR_DATA_FABRIC_URL", "").rstrip("/")
    token = settings.get("BIORADAR_DATA_FABRIC_HISTORY_READ_TOKEN", "")
    if not url or not token:
        raise ValueError("缺少 Fabric 只读配置：BIORADAR_DATA_FABRIC_URL / BIORADAR_DATA_FABRIC_HISTORY_READ_TOKEN")
    if not 1 <= limit <= 100:
        raise ValueError("limit 应在 1–100 之间")
    params = [("source_id", source), ("is_demo", "false"), ("mode", "current"),
              ("cutoff", cutoff or datetime.now(timezone.utc).isoformat()), ("limit", str(limit))]
    params += [("terms", term) for term in terms]
    params += [("source_record_ids", rid) for rid in (record_ids or [])]
    if cursor:
        params.append(("cursor", cursor))
    response = httpx.get(url + "/api/history/records", params=params,
                         headers={"Authorization": "Bearer " + token}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Fabric {source} 读取失败：HTTP {response.status_code}；请核对服务状态和读取权限")
    return response.json()


def to_material(record: dict, material_id: str = "M1", source_kind: str | None = None) -> dict:
    """原文位置以本次规范化文本为准；原记录及其版本由调用方一并保存。"""
    facts = record.get("facts", {})
    if source_kind is None:
        types = facts.get("publication_types", {})
        if isinstance(types, dict):
            types = types.get("pubType", [])
        labels = " ".join(types if isinstance(types, list) else [str(types)]).lower()
        if "review" in labels:
            source_kind = "review"
        elif "research-article" in labels or "clinical trial" in labels:
            source_kind = "primary"
        else:
            raise ValueError("资料类型不明确；阅读原文后明确指定 primary/review/graph，不能默认原始实验")
    text = record.get("clean_text") or facts.get("abstract") or ""
    if not text.strip():
        raise ValueError(f"{record.get('source_record_id')}: 没有可阅读文本")
    # 先移除真实标签，再解实体；p&lt;0.05 / &gt;90% 必须保留为比较符。
    text = html.unescape(re.sub(r"</?[A-Za-z][^>]*>", " ", text))
    text = re.sub(r"[ \t]+", " ", text).strip()
    group = ("PMID:" + str(facts["pmid"]) if facts.get("pmid") else
             "DOI:" + str(facts["doi"]).lower() if facts.get("doi") else
             record["source_id"] + ":" + record["source_record_id"])
    return {"id": material_id, "text": text, "source_id": record["source_id"],
            "record_id": record["source_record_id"], "version": record["version_id"],
            "title": record.get("title", ""), "source_group": group,
            "source_kind": source_kind, "text_level": "abstract", "url": record.get("source_url", ""),
            **{key: facts[key] for key in ("pmid", "doi") if facts.get(key) is not None}}
