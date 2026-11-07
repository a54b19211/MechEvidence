"""A generated type-mutation matrix over the committed receipt: whatever JSON a receipt contains, adopt() answers with
a verdict (never an exception), and a field that changes JSON type is never adopted."""
import copy
import json
from pathlib import Path

import pytest
from extraction_contract.mech_v2.receipt import SHAPE, shape_problems

from mechevidence.contract import adopt

DIRECTORY = Path(__file__).resolve().parents[1] / "examples" / "contract-v2"
REQUEST = json.loads((DIRECTORY / "request.json").read_text(encoding="utf-8"))
RECEIPT = json.loads((DIRECTORY / "receipt.json").read_text(encoding="utf-8"))
EXPECTED = RECEIPT["evaluator"]["evaluator_id"]
# whose content the table deliberately leaves free (opaque to the contract), or that may legitimately be null/absent
FREE = ("locators", "usage", "context", "scope_match", "applicability_gaps", "conflicts", "unresolved_counterevidence",
        "finish_reason", ".error", ".value")
NEW_VALUES = [None, True, 7, 1.5, "zz", [], {}, [None], {"k": None}]


def category(value):
    return ("null" if value is None else "bool" if isinstance(value, bool) else "num" if isinstance(value, (int, float))
            else "str" if isinstance(value, str) else "list" if isinstance(value, list) else "object")


def nodes(value, path=()):
    """Every (path, value) of the receipt, containers included."""
    yield path, value
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from nodes(inner, path + (key,))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            yield from nodes(inner, path + (index,))


def shown(path):
    return "".join(f"[{key}]" if isinstance(key, int) else f".{key}" for key in path)


def at(receipt, path):
    for key in path:
        receipt = receipt[key]
    return receipt


def adopted(receipt):
    return adopt(REQUEST, receipt, EXPECTED)


def checked(path):
    if path[:1] == ("material_notes",) and path[-1] == "material":
        return False                                  # a note may name no material
    return bool(path) and not any(word in shown(path) for word in FREE)


MUTATIONS = [(path, value) for path, original in nodes(RECEIPT) if checked(path)
             for value in NEW_VALUES if category(value) != category(original)]
DELETIONS = [path for path, _ in nodes(RECEIPT) if checked(path) and isinstance(path[-1], str)]


def test_the_untouched_receipt_is_adopted():
    assert adopted(RECEIPT)["result"]["status"] == "completed" and shape_problems(RECEIPT) == []
    assert len(MUTATIONS) > 800 and len(DELETIONS) > 100          # the matrix is not accidentally empty


@pytest.mark.parametrize("path,value", MUTATIONS, ids=lambda item: shown(item) if isinstance(item, tuple) else repr(item)[:12])
def test_a_field_of_another_json_type_is_never_adopted_and_never_raises(path, value):
    mutated = copy.deepcopy(RECEIPT)
    if len(path) == 1:
        mutated[path[0]] = value
    else:
        at(mutated, path[:-1])[path[-1]] = value
    outcome = adopted(mutated)                      # must not raise
    assert outcome["result"]["status"] == "failed", shown(path)
    assert outcome["rejected"]["response"] == mutated and "response" not in outcome["result"]


@pytest.mark.parametrize("path", DELETIONS, ids=shown)
def test_a_missing_field_is_never_adopted_and_never_raises(path):
    mutated = copy.deepcopy(RECEIPT)
    del at(mutated, path[:-1])[path[-1]]
    outcome = adopted(mutated)
    assert outcome["result"]["status"] == "failed" and outcome["rejected"]["response"] == mutated


def test_the_table_covers_every_field_the_service_puts_in_a_receipt():
    """A field of the real receipt that SHAPE does not name is a field nobody validates."""
    def covered(value, spec):
        if spec == "any" or isinstance(spec, str):
            return []
        if spec[0] == "null":
            return covered(value, spec[1]) if value is not None else []
        if spec[0] == "list":
            return [m for item in value for m in covered(item, spec[1])]
        if spec[0] == "obj":
            missing = [key for key in value if key not in spec[1]]
            return missing + [m for key, inner in spec[1].items() if key in value for m in covered(value[key], inner)]
        return []
    assert covered(RECEIPT, SHAPE) == []


def test_bool_is_not_an_int_and_a_float_is_not_an_int():
    for value in (True, 1.0):
        mutated = copy.deepcopy(RECEIPT)
        mutated["llm_calls"][0]["outputs"][0]["attempt"] = value
        assert adopted(mutated)["result"]["reason"] == "receipt_invalid"
