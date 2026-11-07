"""请求和两次模型输出。编号（C/J/E1.3/O）全部由程序分配，模型只引用。"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    model_validator,
)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


Text = Annotated[str, AfterValidator(_nonblank)]
Match = Literal["matched", "mismatched", "unknown", "not_required"]
SourceKind = Literal["primary", "review", "graph"]
DocumentType = Literal["research_article", "conference_abstract", "trial_results", "trial_registration",
                       "study_protocol", "review", "press_release", "company_disclosure", "news", "social_post",
                       "graph"]
TextLevel = Literal["full", "abstract", "excerpt"]
# A request model rejects a field the contract does not name: ignoring it would answer 200 as if it had been
# checked. Model outputs (Scope as an observation context, ClaimIn in a decomposition) stay lenient.
FORBID_EXTRA = ConfigDict(extra="forbid")


class Scope(BaseModel):
    species: str | None = None
    system: str | None = None
    disease: str | None = None


class ScopeIn(Scope):
    model_config = FORBID_EXTRA


class Material(BaseModel):
    """雷达给定的一份材料；source_kind / document_type 必须来自原始记录，不由本服务推断。"""
    model_config = FORBID_EXTRA
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]
    title: str = ""
    text: Text
    source_kind: SourceKind
    document_type: DocumentType | None = None
    text_level: TextLevel
    # 以下仅用于把同一篇文献的多份记录合并为一个来源组
    pmid: str | StrictInt | None = None
    doi: str | None = None
    url: str | None = None
    source_id: str = ""
    record_id: str = ""
    source_group: str = ""


class ClaimIn(BaseModel):
    text: Text = Field(description="一个原子命题")
    kind: Literal["causal", "association", "reported"] = Field(
        "causal", description="causal=断言 X 改变/导致 Y；association=只断言相关或共现；reported=断言某事被观察到（如表达）")


class JoinIn(BaseModel):
    text: Text = Field(description="机制要求的因果连接，例如“C1 的作用通过 C2 产生”")
    from_claim: int = Field(description="上游命题在 claims 中的序号，从 1 开始")
    to_claim: int = Field(description="下游命题在 claims 中的序号，从 1 开始")


class ClaimReq(ClaimIn):
    model_config = FORBID_EXTRA


class JoinReq(JoinIn):
    model_config = FORBID_EXTRA


class Decomposition(BaseModel):
    """模型输出 ①：只给文本和下标，程序编 C1..Cn / J1..Jm。"""
    claims: list[ClaimIn] = Field(min_length=1, description="原子命题列表，每项 {text, kind: causal|association|reported}")
    joins: list[JoinIn] = Field(default_factory=list, description="因果连接列表，每项 {text, from_claim, to_claim}")


class EvaluateRequest(BaseModel):
    model_config = FORBID_EXTRA
    question: Text
    mechanism: Text
    scope: ScopeIn = Field(default_factory=ScopeIn)
    materials: list[Material] = Field(default_factory=list)
    # 已有拆解时可直接提供，省掉第一次模型调用
    claims: list[ClaimReq] | None = None
    joins: list[JoinReq] = Field(default_factory=list)
    max_llm_calls: int = Field(20, ge=1, le=100)

    @model_validator(mode="after")
    def unique_materials(self):
        ids = [item.id for item in self.materials]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate material ids")
        return self


class ContextMatch(BaseModel):
    species: Match = "unknown"
    system: Match = "unknown"
    disease: Match = "unknown"


class Observation(BaseModel):
    """模型输出 ② 中的一条观察；逐条校验，坏的一条只丢这一条。"""
    targets: list[str] = Field(min_length=1)
    passage: str
    quote: Text
    relation: Literal["supports", "refutes", "compatible"]
    basis: Literal["intervention", "association", "review", "claim"]
    causal_test: bool = False
    context: Scope = Field(default_factory=Scope)
    scope_match: ContextMatch = Field(default_factory=ContextMatch)


class Review(BaseModel):
    observations: list = Field(description=(
        "观察列表；没有相关结果时为 []。每项字段："
        "targets (list，目标编号如 C1/J1)、passage (片段编号如 E1.3)、quote (从该片段逐字复制的短引)、"
        "relation (supports|refutes|compatible)、basis (intervention|association|review|claim)、"
        "causal_test (bool，可省略=false)、context ({species, system, disease}，未报告填 null)、"
        "scope_match ({species, system, disease}，各取 matched|mismatched|unknown|not_required)"))
