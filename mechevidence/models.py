"""Inputs and the two semantic reviews; source text is never model-generated."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt, model_validator


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


Text = Annotated[str, AfterValidator(_nonblank)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Match = Literal["matched", "mismatched", "unknown", "not_required"]


def unique_ids(items: list, name: str) -> None:
    ids = [item.id if hasattr(item, "id") else item for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {name} IDs")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Scope(Record):
    species: str | None = None
    system: str | None = None
    disease: str | None = None


class Claim(Record):
    id: Identifier
    text: Text
    kind: Literal["causal", "association", "reported"] = "causal"


class Join(Record):
    id: Identifier
    text: Text
    claim_ids: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_members(self):
        unique_ids(self.claim_ids, "join member")
        return self


class Material(Record):
    id: Identifier
    text: Text
    source_id: str = ""
    record_id: str = ""
    version: str = ""
    title: str = ""
    source_group: str = ""
    pmid: str | StrictInt | None = None
    doi: str | None = None
    source_kind: Literal["primary", "review", "graph"]
    text_level: Literal["abstract", "full_text", "selected_passages"]
    url: str | None = None
    allowed_passage_ids: list[Identifier] | None = None

    @model_validator(mode="after")
    def valid_selection(self):
        if self.allowed_passage_ids is not None:
            unique_ids(self.allowed_passage_ids, "allowed passage")
        return self


class Decomposition(Record):
    claims: list[Claim] = Field(min_length=1)
    joins: list[Join]

    @model_validator(mode="after")
    def valid_structure(self):
        unique_ids(self.claims, "claim")
        unique_ids(self.joins, "join")
        claim_ids = {claim.id for claim in self.claims}
        if claim_ids & {join.id for join in self.joins}:
            raise ValueError("claim and join IDs must be distinct")
        for join in self.joins:
            if not set(join.claim_ids) <= claim_ids:
                raise ValueError(f"join {join.id} references an unknown claim")
        return self


class Case(Record):
    id: str = ""
    question: Text
    mechanism: Text
    scope: Scope = Field(default_factory=Scope)
    claims: list[Claim] = Field(default_factory=list)
    joins: list[Join] = Field(default_factory=list)
    materials: list[Material] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_structure(self):
        unique_ids(self.materials, "material")
        if self.claims:
            Decomposition(claims=self.claims, joins=self.joins)
        elif self.joins:
            raise ValueError("joins require nonempty structured claims")
        return self


class ContextMatch(Record):
    species: Match
    system: Match
    disease: Match


class Observation(Record):
    id: Identifier
    claim_ids: list[Identifier]
    join_ids: list[Identifier]
    relation: Literal["supports", "opposes", "compatible"]
    basis: Literal["intervention", "association", "statement"]
    causal_test: Literal["established", "not_established", "not_applicable"]
    passage_ids: list[Identifier] = Field(min_length=1)
    context: Scope
    applicability: ContextMatch
    rationale: Text
    limitations: list[Text]

    @model_validator(mode="after")
    def valid_references(self):
        if not self.claim_ids and not self.join_ids:
            raise ValueError("an observation must address a claim or causal join")
        for name in ("claim_ids", "join_ids", "passage_ids"):
            unique_ids(getattr(self, name), name)
        return self


class Gap(Record):
    claim_ids: list[Identifier]
    join_ids: list[Identifier]
    reason: Text
    missing_experiment: Text = Field(description=(
        "Missing experimental observations or study evidence not seen in this supplied material package, "
        "for upstream retrieval and expert review. Describe evidence needs, not operational protocols or "
        "recommendations to initiate patient interventions. Endpoints must be measurable in the stated "
        "study system; cell or animal evidence cannot substitute for patient endpoint evidence. "
        "Do not claim that a study has never been performed anywhere."
    ))


class LocalObservation(Observation):
    id: Annotated[str, Field(pattern=r"^L[0-9]+$")]
    claim_ids: list[Identifier] = Field(min_length=1)
    join_ids: list[Identifier] = Field(default_factory=list, max_length=0)


class LocalReview(Record):
    observations: list[LocalObservation]
    gaps: list[Gap]


class MechanismObservation(Observation):
    id: Annotated[str, Field(pattern=r"^M[0-9]+$")]


class Combination(Record):
    id: Identifier
    observation_ids: list[Identifier] = Field(min_length=1)
    context_coherence: Literal["matched", "mismatched", "unknown"]
    rationale: Text


class Comparison(Record):
    observation_ids: list[Identifier] = Field(min_length=2, max_length=2)
    context_relation: Literal["comparable", "different", "unknown"]
    rationale: Text


class MechanismReview(Record):
    additional_claims: list[Claim]
    additional_joins: list[Join]
    observations: list[MechanismObservation]
    combinations: list[Combination]
    comparisons: list[Comparison]
    whole_assessment: Literal["supported", "compatible_only", "unestablished"]
    rationale: Text
    gaps: list[Gap]


class DirectReview(Record):
    verdict: Literal["supported", "partial", "insufficient"]
    rationale: Text
    supporting_passage_ids: list[Identifier]
    opposing_passage_ids: list[Identifier]
    scope_limitations: list[Text]
    missing_experiments: list[Text]
