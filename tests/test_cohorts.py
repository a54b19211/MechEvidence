"""Controlled callback fixtures test the protocol, not biomedical performance."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from mechevidence.assess import EvaluationError, IdentityInputError
from mechevidence.cohorts import CohortArguments, assess_independence


def material(mid, text, pmid):
    return {
        "id": mid, "text": text, "source_id": "europe_pmc",
        "record_id": f"MED:{pmid}", "pmid": pmid, "version": "fixture-v1",
        "source_kind": "primary", "text_level": "selected_passages",
    }


def arguments():
    return {
        "materials": [
            material("A", "This report reuses the patients enrolled in study B.", "111"),
            material("B", "These patients form the cohort reused in report A.", "222"),
        ],
        "pairs": [{"left_material_id": "A", "right_material_id": "B", "unit": "patient_cohort"}],
    }


def judgment(**changes):
    return {
        "left_material_id": "A", "right_material_id": "B", "unit": "patient_cohort",
        "relation": "same", "basis": "explicit_linkage",
        "passage_ids": ["A:P001", "B:P001"],
        "rationale": "Controlled fixture explicitly links the two included patient sets.",
        "uncertainty": [], "information_needs": [], **changes,
    }


class CohortReviewTests(unittest.TestCase):
    def run_review(self, data=None, rows=None):
        calls = []

        def callback(stage, instructions, payload, schema):
            calls.append((stage, instructions, payload, schema))
            return {"pairs": deepcopy(rows if rows is not None else [judgment()])}

        result = assess_independence(data if data is not None else arguments(), callback)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "cohort_review")
        return result, calls[0]

    def test_one_callback_quotes_versions_and_separates_publication_groups(self):
        data = arguments()
        result, call = self.run_review(data)
        self.assertEqual(result["pairs"][0]["relation"], "same")
        self.assertEqual(len(result["publication_groups"]), 2)
        self.assertEqual(call[2]["pairs"], data["pairs"])
        self.assertEqual(call[3]["properties"]["pairs"]["maxItems"], 64)
        for entry in call[2]["materials"]:
            self.assertNotIn("text", entry)
            self.assertNotIn("allowed_passage_ids", entry)
        for ref in result["pairs"][0]["references"]:
            source = next(m for m in data["materials"] if m["id"] == ref["material_id"])
            self.assertEqual(source["text"][ref["start"]:ref["end"]], ref["text"])
            self.assertEqual(ref["version"], source["version"])
            self.assertEqual(ref["record_id"], source["record_id"])

    def test_same_publication_never_automatically_means_same_cohort(self):
        data = arguments()
        data["materials"][1]["pmid"] = "111"
        data["materials"][1]["record_id"] = "MED:111"
        result, _ = self.run_review(data, [judgment(
            relation="unknown", basis="publication_identity", passage_ids=[],
            rationale="The same paper may report distinct patient cohorts.")])
        self.assertEqual(len(result["publication_groups"]), 1)
        self.assertEqual(result["pairs"][0]["relation"], "unknown")
        self.assertTrue(result["pairs"][0]["uncertainty"])
        self.assertTrue(result["pairs"][0]["information_needs"])

    def test_metadata_time_graph_and_single_sided_support_do_not_confirm(self):
        for changes in [
            {"basis": "publication_identity"}, {"basis": "time_only"},
            {"basis": "graph"}, {"passage_ids": ["A:P001"]},
        ]:
            with self.subTest(changes=changes):
                result, _ = self.run_review(rows=[judgment(relation="independent", **changes)])
                pair = result["pairs"][0]
                self.assertEqual(pair["relation"], "unknown")
                self.assertEqual(pair["proposed_relation"], "independent")
                self.assertTrue(pair["uncertainty"])
        for kind in ["graph", "review"]:
            data = arguments()
            data["materials"][0]["source_kind"] = kind
            result, _ = self.run_review(data)
            self.assertEqual(result["pairs"][0]["relation"], "unknown")

    def test_missing_version_does_not_yield_a_determinate_relation(self):
        data = arguments()
        data["materials"][0]["version"] = ""
        result, _ = self.run_review(data)
        self.assertEqual(result["pairs"][0]["relation"], "unknown")
        self.assertIn("source version", " ".join(result["pairs"][0]["uncertainty"]))

    def test_explicit_original_sampling_evidence_can_establish_independence(self):
        data = arguments()
        data["materials"][0]["text"] = "Participant matching confirmed no shared patients with cohort B."
        data["materials"][1]["text"] = "Cohort B excluded every patient enrolled in cohort A."
        result, _ = self.run_review(data, [judgment(
            relation="independent", basis="sampling_details",
            rationale="Controlled passages describe an explicitly disjoint included patient set.")])
        self.assertEqual(result["pairs"][0]["relation"], "independent")
        self.assertEqual(len(result["pairs"][0]["references"]), 2)

    def test_allowed_passages_are_the_only_text_sent_and_citable(self):
        data = arguments()
        data["materials"][0]["text"] += " Secret excluded sentence."
        data["materials"][0]["allowed_passage_ids"] = ["A:P001"]
        result, call = self.run_review(data)
        self.assertNotIn("Secret excluded sentence", json.dumps(call[2]))
        self.assertEqual(result["pairs"][0]["relation"], "same")
        with self.assertRaises(EvaluationError):
            self.run_review(data, [judgment(passage_ids=["A:P002", "B:P001"])])

    def test_unavailable_out_of_pair_missing_and_duplicate_responses_fail(self):
        for rows in [
            [judgment(passage_ids=["A:P999", "B:P001"])],
            [judgment(right_material_id="C")],
            [judgment(), judgment()], [],
        ]:
            with self.subTest(rows=rows), self.assertRaises(EvaluationError):
                self.run_review(rows=rows)
        data = arguments()
        data["materials"].append(material("C", "Other source text.", "333"))
        with self.assertRaises(EvaluationError):
            self.run_review(data, [judgment(passage_ids=["A:P001", "C:P001"])])

    def test_input_validation_and_identity_checks_happen_before_callback(self):
        bad = []
        data = arguments(); data["extra"] = True; bad.append(data)
        data = arguments(); data["pairs"][0]["right_material_id"] = "C"; bad.append(data)
        data = arguments(); data["pairs"][0]["right_material_id"] = "A"; bad.append(data)
        data = arguments(); data["pairs"][0]["unit"] = "paper"; bad.append(data)
        data = arguments(); data["pairs"].append({"left_material_id": "B", "right_material_id": "A", "unit": "patient_cohort"}); bad.append(data)
        data = arguments(); data["materials"].append(deepcopy(data["materials"][0])); bad.append(data)
        data = arguments(); data["pairs"] *= 65; bad.append(data)
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                assess_independence(data, lambda *_: self.fail("invalid input reached callback"))
        data = arguments()
        data["materials"][0].update(source_group="PMID:999", allowed_passage_ids=[])
        with self.assertRaises(IdentityInputError):
            assess_independence(data, lambda *_: self.fail("identity conflict reached callback"))

    def test_units_and_requested_pairs_are_not_closed_transitively(self):
        data = arguments()
        data["materials"].append(material("C", "This cohort overlaps with B.", "333"))
        data["pairs"].extend([
            {"left_material_id": "A", "right_material_id": "B", "unit": "study"},
            {"left_material_id": "B", "right_material_id": "C", "unit": "patient_cohort"},
        ])
        rows = [judgment(), judgment(unit="study", relation="unknown", passage_ids=[]),
                judgment(left_material_id="B", right_material_id="C", relation="overlap",
                         passage_ids=["B:P001", "C:P001"])]
        result, _ = self.run_review(data, rows)
        self.assertEqual(len(result["pairs"]), 3)
        self.assertEqual([p["relation"] for p in result["pairs"]], ["same", "unknown", "overlap"])
        self.assertNotIn("independent_cohorts", result)
        self.assertNotIn("independent_studies", result)
        with self.assertRaises(EvaluationError):
            self.run_review(data, rows[:1])

    def test_saved_real_materials_are_only_an_unknown_boundary_fixture(self):
        # True saved text, synthetic unknown callback: this is not a new scientific run.
        path = Path(__file__).resolve().parents[1] / "examples/investigation_feedback.json"
        packet = json.loads(path.read_text())
        first = deepcopy(packet["snapshots"]["Q001"]["case"]["materials"][0])
        second = deepcopy(packet["snapshots"]["Q020"]["case"]["materials"][0])
        first["id"], second["id"] = "A", "B"
        data = {"materials": [first, second], "pairs": arguments()["pairs"]}
        result, _ = self.run_review(data, [judgment(
            relation="unknown", basis="insufficient", passage_ids=[],
            rationale="These saved abstracts do not establish a shared or disjoint patient set.")])
        self.assertEqual(result["pairs"][0]["relation"], "unknown")

    def test_callback_failure_is_an_execution_error_with_original_cause(self):
        def failing(*_):
            raise TimeoutError("controlled transport failure")

        with self.assertRaises(EvaluationError) as caught:
            assess_independence(arguments(), failing)
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)


if __name__ == "__main__":
    unittest.main()
