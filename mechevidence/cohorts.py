"""Bounded study/cohort relation review; publication grouping is not independence."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .assess import (
    CallJSON, EvaluationError, _call, _references, _original_results, prepare_materials,
    prepare_passages, source_identity,
)
from .models import Identifier, Material, Record, Text, unique_ids


Unit = Literal["study", "patient_cohort"]
Relation = Literal["same", "overlap", "independent", "unknown"]
Basis = Literal[
    "explicit_linkage", "sampling_details", "publication_identity",
    "graph", "time_only", "insufficient",
]


class CohortPair(Record):
    model_config = ConfigDict(extra="forbid", strict=True)

    left_material_id: Identifier
    right_material_id: Identifier
    unit: Unit

    @model_validator(mode="after")
    def distinct_members(self):
        if self.left_material_id == self.right_material_id:
            raise ValueError("a pair requires two distinct material IDs")
        return self


def _pair_key(pair: CohortPair) -> tuple:
    return (*sorted((pair.left_material_id, pair.right_material_id)), pair.unit)


class CohortArguments(Record):
    """Explicit pairs bound the review; no all-pairs expansion is performed."""

    model_config = ConfigDict(extra="forbid", strict=True)

    materials: list[Material] = Field(min_length=2, max_length=128)
    pairs: list[CohortPair] = Field(min_length=1, max_length=64)
    question: Text | None = None

    @model_validator(mode="after")
    def valid_pairs(self):
        unique_ids(self.materials, "material")
        material_ids = {item.id for item in self.materials}
        keys = set()
        for pair in self.pairs:
            if not {pair.left_material_id, pair.right_material_id} <= material_ids:
                raise ValueError("pair references an unknown material")
            key = _pair_key(pair)
            if key in keys:
                raise ValueError("duplicate pair for the same unit, including reversed pairs")
            keys.add(key)
        return self


class CohortJudgment(CohortPair):
    relation: Relation
    basis: Basis
    passage_ids: list[Identifier]
    rationale: Text
    uncertainty: list[Text]
    information_needs: list[Text]

    @model_validator(mode="after")
    def unique_passages(self):
        unique_ids(self.passage_ids, "passage")
        return self


class CohortReview(Record):
    model_config = ConfigDict(extra="forbid", strict=True)

    pairs: list[CohortJudgment] = Field(min_length=1, max_length=64)


_INSTRUCTIONS = """Review only the explicitly requested material pairs and unit.
Return exactly one judgment for every requested pair; never add unrequested pairs.
Use only the supplied permitted passages. Cite their IDs, not invented quotations.

For unit=study, compare the underlying study execution/data: same study, partially
shared study data/experiments, independently conducted studies, or unknown.
For unit=patient_cohort, compare actual included patients: the same patient set,
partially overlapping patients, no shared patients, or unknown. Do not infer one
unit's relationship from the other unit or from publication identity. A paper can
contain several cohorts; several papers can report the same study or patients.

For same/overlap/independent, relevant original study passages must describe BOTH
members and establish the requested relationship. Use basis=explicit_linkage for
an explicit statement connecting the actual studies/cohorts, or sampling_details
for original recruitment/participant/study-execution details that establish it.
Different PMIDs/DOIs/titles, trial IDs alone, author lists, hospitals, countries,
publication/recruitment times or follow-up lengths never by themselves establish
independence. Nor does absence of a reported link establish no patient overlap.
Shared registry labels alone also do not establish identical included patients.
When original participant/sampling details do not resolve this, return unknown.

Registry/protocol design cannot establish actual enrolment, execution or participant overlap.
Graph assertions and reviews provide retrieval leads, not original evidence for
determinate relations. Select basis=publication_identity, graph, time_only or
insufficient where appropriate and return unknown. Do not recast those sources as
primary. Cite only passages belonging to this pair, covering both members for a
determinate judgment. Do not use a relevant-looking but unrelated quotation.

Explain the evidence and limits in rationale and uncertainty. For unknown, state
which existing methods, cohort descriptions or supplementary material should be
read in information_needs; do not suggest performing new experiments. Do not
invent participant counts, overlap sizes, independent-study totals, probabilities
or an expert approval. Do not propagate relations to any unrequested pair.
"""


def assess_independence(arguments: CohortArguments | dict, call_json: CallJSON) -> dict:
    """Review bounded pairs with one injected four-argument model callback.

    Input errors remain Pydantic/identity errors. Malformed or incomplete model
    responses raise EvaluationError; scientific insufficiency returns unknown.
    """
    request = CohortArguments.model_validate(arguments)
    case = {
        "question": request.question or "Assess the requested study/cohort relationships.",
        "mechanism": "Assess study/cohort relationships from supplied original evidence.",
        "materials": request.materials,
    }
    # The existing boundary validates identities even for masked-out materials.
    passages = {item["id"]: item for item in prepare_passages(case)}
    materials = {item.id: item for item in request.materials}
    groups = {}
    for item in request.materials:
        groups.setdefault(source_identity(item), []).append(item.id)
    publication_groups = [
        {"source_group": group, "material_ids": ids} for group, ids in groups.items()
    ]
    payload = {
        "question": request.question,
        "materials": prepare_materials(case),
        "pairs": [pair.model_dump() for pair in request.pairs],
        "publication_groups": publication_groups,
    }
    review = _call(call_json, "cohort_review", _INSTRUCTIONS, payload, CohortReview)
    requested = {_pair_key(pair) for pair in request.pairs}
    judgments = {}
    for judgment in review.pairs:
        key = _pair_key(judgment)
        if key not in requested or key in judgments:
            raise EvaluationError("cohort_review contains an unrequested or duplicate pair")
        judgments[key] = judgment
        members = {judgment.left_material_id, judgment.right_material_id}
        for pid in judgment.passage_ids:
            if pid not in passages or passages[pid]["material_id"] not in members:
                raise EvaluationError("cohort_review cites an unavailable or out-of-pair passage")
    if judgments.keys() != requested:
        raise EvaluationError("cohort_review omitted a requested pair")

    pairs = []
    for pair in request.pairs:
        judgment = judgments[_pair_key(pair)]
        refs = _references(judgment.passage_ids, passages, materials)
        relation = judgment.relation
        uncertainty = list(judgment.uncertainty)
        needs = list(judgment.information_needs)
        issues = []
        if relation != "unknown":
            if judgment.basis not in {"explicit_linkage", "sampling_details"}:
                issues.append("Publication identity, graph assertions or time differences do not establish this relation.")
            for mid in (pair.left_material_id, pair.right_material_id):
                member_refs = [ref for ref in refs if ref["material_id"] == mid]
                if not member_refs:
                    issues.append(f"No permitted source passage covers material {mid}.")
                elif not _original_results(materials[mid]):
                    issues.append(f"Material {mid} is not an original observed study/cohort result; registration/protocol and secondary records are retrieval leads.")
                if member_refs and not materials[mid].version.strip():
                    issues.append(f"The cited material {mid} has no supplied source version.")
            if issues:
                relation = "unknown"
                uncertainty.extend(issues)
                needs.append("Read versioned original descriptions for both requested members that establish their study or patient relationship.")
        if relation == "unknown":
            if not uncertainty:
                uncertainty.append("The supplied material does not establish this pair's requested relationship.")
            if not needs:
                needs.append("Read the original study methods, participant selection and any explicit cohort-linkage statements for this pair.")
        pairs.append({
            **pair.model_dump(),
            "relation": relation,
            "proposed_relation": judgment.relation,
            "basis": judgment.basis,
            "passage_ids": list(judgment.passage_ids),
            "references": refs,
            "rationale": ("Cannot establish a determinate relation: " + " ".join(issues)
                          if issues else judgment.rationale),
            "uncertainty": list(dict.fromkeys(uncertainty)),
            "information_needs": list(dict.fromkeys(needs)),
        })
    return {
        "status": "completed",
        "question": request.question,
        "publication_groups": publication_groups,
        "pairs": pairs,
        "cohort_review": review.model_dump(),
        "limitations": [
            "Publication groups identify records of a paper; they do not identify patient cohorts or independent studies.",
            "Relations concern only requested pairs and units; no transitive closure or independent-study/patient total is computed.",
            "Semantic relation judgments come from the injected model review, not expert labels; program checks enforce source boundaries.",
        ],
    }
