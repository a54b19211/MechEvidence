"""材料可评性与文献身份（含协调人复现过的两个身份冲突）。"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mechevidence import evaluate
from mechevidence.materials import prepare, split_passages
from mechevidence.models import Material
from stub import TEXT, Stub, case, good_observations, material

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "examples/identity_contract.json").read_text(encoding="utf-8"))


def group(fields):
    usable, notes = prepare([Material(**(FIXTURE["base_material"] | fields))])
    return usable[0]["group"] if usable else notes[0]


@pytest.mark.parametrize("row", FIXTURE["accepted"], ids=lambda row: row["name"])
def test_accepted_identifiers(row):
    assert group(row["fields"]) == row["identity"]


@pytest.mark.parametrize("row", FIXTURE["unrecognized"], ids=lambda row: row["name"])
def test_pseudo_urls_and_free_text_are_not_identifiers(row):
    assert group(row["fields"]).startswith(row["identity_prefix"])


@pytest.mark.parametrize("row", FIXTURE["rejected"], ids=lambda row: row["name"])
def test_conflicting_or_invalid_identifiers_exclude_the_material(row):
    note = group(row["fields"])
    assert note["code"] == row["error_code"] and row["identifier_type"] in note["message"]


def test_cross_source_pairs_and_boolean_pmid():
    for row in FIXTURE["pairs"]:
        assert (group(row["left"]) == group(row["right"])) == row["same_group"], row["name"]
    with pytest.raises(ValidationError):
        Material(**(FIXTURE["base_material"] | {"pmid": True}))


def test_same_paper_from_two_databases_counts_one_source_group():
    materials = [material("E1", pmid="111", source_id="pubmed", record_id="111"),
                 material("E2", source_id="europe_pmc", record_id="MED:111"),
                 material("E3"), material("E4", text=TEXT.replace(" ", "  "))]  # 无标识的相同正文只算一组
    def observations(*ids):
        return [{**row, "passage": row["passage"].replace("E1", mid)} for mid in ids for row in good_observations()]
    result = evaluate(case(materials=materials), Stub([observations("E1", "E2", "E3"), observations("E4")]))
    groups = {row["material"]: row["source_group"] for row in result["observations"]}
    assert groups["E1"] == groups["E2"] == "pmid:111"
    assert groups["E3"] == groups["E4"] == "text:E3"
    assert result["independence"]["distinct_source_groups"] == 2


def test_declarations_and_type_conflicts_are_excluded_with_a_note():
    materials = [material("E1"), material("E2", document_type="news"),
                 material("E3", document_type="review"),  # source_kind 仍是 primary
                 material("E4", source_kind="graph", document_type="graph")]
    stub = Stub()
    result = evaluate(case(materials=materials), stub)
    sent = [m["id"] for m in stub.calls[1][1]["materials"]]
    assert sent == ["E1", "E4"]
    assert {n["material"]: n["code"] for n in result["material_notes"]} == {
        "E2": "material_type_not_evaluable", "E3": "material_type_conflict"}
    assert result["overall"] == "supported"


def test_passage_split_keeps_sentences_and_numbers_per_material():
    text = "First result. Second result in 3 mice.\n\nThird, e.g. lower case stays. 4 cells died."
    passages = split_passages(Material(**material("E9", text=text)))
    assert [(p["id"], p["text"]) for p in passages] == [
        ("E9.1", "First result."), ("E9.2", "Second result in 3 mice."),
        ("E9.3", "Third, e.g. lower case stays."), ("E9.4", "4 cells died.")]
