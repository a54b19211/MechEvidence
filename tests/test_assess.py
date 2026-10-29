"""Boundary tests for provenance, causal coverage and conservative scope handling."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from mechevidence import (
    EvaluationError, Material, aggregate, direct_baseline, evaluate, prepare_materials, prepare_passages, source_identity,
)


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "EXPECTED_POSITIVE_VARIANT_MUST_NOT_LEAK",
            "question": "Does intervention A cause endpoint E through mediator M in system S?",
            "mechanism": "A changes M; M is necessary for E.",
            "scope": {"species": "mouse", "system": "S"},
            "claims": [{"id": "C1", "text": "A changes M"},
                       {"id": "C2", "text": "M is necessary for E"}],
            "joins": [{"id": "J1", "text": "A produces E through M", "claim_ids": ["C1", "C2"]}],
            "materials": [{"id": "m", "source_group": "PMID12345678", "source_id": "fabric-paper",
                           "record_id": "12345678", "version": "v3", "source_kind": "primary",
                           "text_level": "abstract", "text": "A changed M. Blocking M prevented the E response."}],
        }
        self.observation = {
            "id": "L1", "claim_ids": ["C1", "C2"], "join_ids": [], "relation": "supports",
            "basis": "intervention", "causal_test": "established", "passage_ids": ["m:P001", "m:P002"],
            "context": {"species": "mouse", "system": "S"},
            "applicability": {"species": "matched", "system": "matched", "disease": "not_required"},
            "rationale": "Intervention and mediator blockade are reported.", "limitations": ["Abstract methods."],
        }
        self.local = {"observations": [deepcopy(self.observation)], "gaps": []}
        bridge = {**deepcopy(self.observation), "id": "M1", "claim_ids": [], "join_ids": ["J1"]}
        self.review = {
            "additional_claims": [], "additional_joins": [], "observations": [bridge],
            "combinations": [{"id": "W1", "observation_ids": ["L1", "M1"],
                              "context_coherence": "matched", "rationale": "Same experimental system."}],
            "comparisons": [], "whole_assessment": "supported", "rationale": "Specified dependency is tested.",
            "gaps": [],
        }

    def test_no_material_is_unassessable_without_a_model_call(self):
        self.case["materials"] = []

        def unexpected_call(*_args):
            self.fail("No material should not invoke a model")

        result = evaluate(self.case, unexpected_call)
        self.assertEqual(result["overall"]["verdict"], "insufficient")
        self.assertEqual(result["material_assessability"]["status"], "unassessable")
        self.assertTrue(result["critical_gaps"])
        self.assertEqual(direct_baseline(self.case, unexpected_call)["verdict"], "insufficient")

    def test_evaluation_identity_never_enters_model_payload(self):
        stages = []

        def call(stage, _instructions, payload, schema):
            stages.append(stage)
            self.assertNotIn(self.case["id"], str(payload))
            self.assertNotIn("case_id", payload)
            self.assertIn("properties", schema)
            if stage == "local_review":
                self.assertNotIn("joins", payload)
                self.assertEqual(schema["$defs"]["LocalObservation"]["properties"]["join_ids"]["maxItems"], 0)
                self.assertEqual(schema["$defs"]["LocalObservation"]["properties"]["id"]["pattern"], "^L[0-9]+$")
            else:
                self.assertEqual(schema["$defs"]["MechanismObservation"]["properties"]["id"]["pattern"], "^M[0-9]+$")
            return deepcopy(self.local if stage == "local_review" else self.review)

        result = evaluate(self.case, call)
        self.assertEqual(stages, ["local_review", "mechanism_review"])
        self.assertEqual(result["overall"]["verdict"], "supported")
        self.assertEqual(result["material_assessability"]["status"], "limited")
        self.assertEqual(result["independence"]["distinct_source_groups"], 1)
        reference = result["evidence"][0]["references"][0]
        self.assertEqual((reference["source_id"], reference["record_id"], reference["version"]),
                         ("fabric-paper", "12345678", "v3"))

    def test_unknown_context_is_not_matching_or_compatible(self):
        self.local["observations"][0]["context"]["system"] = None
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["claims"][0]["verdict"], "insufficient")
        self.assertEqual(result["evidence"][0]["effective_applicability"], "unknown")
        self.assertNotEqual(result["overall"]["verdict"], "supported")
        self.assertEqual(aggregate(self.case, self.local, self.review, use_context=False)["overall"]["verdict"], "supported")
        self.case["scope"]["species"] = "unknown"
        self.assertNotEqual(aggregate(self.case, self.local, self.review)["overall"]["verdict"], "supported")
        gap = {"claim_ids": ["C1"], "join_ids": [], "reason": "本包未见所需观察。",
               "missing_experiment": "补查既有原始观察供专家审核。"}
        self.local["gaps"] = [deepcopy(gap)]
        self.review["gaps"] = [deepcopy(gap), {**gap, "claim_ids": ["C2"]},
                               {**gap, "missing_experiment": gap["missing_experiment"] + "另核对终点。"}]
        gaps = aggregate(self.case, self.local, self.review)["critical_gaps"]
        self.assertEqual(sum(item == gap for item in gaps), 1)
        self.assertEqual(sum(item["reason"] == gap["reason"] for item in gaps), 3)

    def test_all_counterevidence_is_checked_outside_the_witness(self):
        negative = {**deepcopy(self.observation), "id": "M2", "relation": "opposes"}
        self.review["observations"].append(negative)
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["overall"]["verdict"], "partial")
        self.assertFalse(result["overall"]["has_comparable_conflict"])
        self.assertEqual(result["claims"][0]["verdict"], "supported")
        self.assertTrue(result["unresolved_counterevidence"])
        self.review["comparisons"] = [{"observation_ids": ["L1", "M2"], "context_relation": "comparable",
                                       "rationale": "The intervention, controls, endpoint and context are comparable."}]
        result = aggregate(self.case, self.local, self.review)
        self.assertTrue(result["overall"]["has_comparable_conflict"])
        self.assertEqual(result["claims"][0]["verdict"], "mixed")
        negative["context"]["species"] = "human"
        negative["applicability"]["species"] = "mismatched"
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["overall"]["verdict"], "supported")
        self.assertFalse(result["conflicts"])
        negative["context"]["species"] = None
        negative["applicability"]["species"] = "unknown"
        result = aggregate(self.case, self.local, self.review)
        self.assertNotEqual(result["overall"]["verdict"], "supported")
        self.assertTrue(result["unresolved_counterevidence"])
        self.assertFalse(result["conflicts"])
        self.review["comparisons"] = []
        negative["context"] = {"species": "mouse", "system": "S in culture"}
        negative["applicability"]["species"] = "matched"
        result = aggregate(self.case, self.local, self.review)
        self.assertNotEqual(result["overall"]["verdict"], "supported")
        self.assertTrue(result["unresolved_counterevidence"])
        self.assertFalse(result["conflicts"])
        self.review["comparisons"] = [{"observation_ids": ["L1", "M2"], "context_relation": "comparable",
                                       "rationale": "The two system descriptions refer to the same preparation."}]
        self.assertTrue(aggregate(self.case, self.local, self.review)["conflicts"])
        self.local["observations"][0]["relation"] = "opposes"
        self.review["observations"] = []
        self.review["comparisons"] = []
        self.review["combinations"] = []
        self.review["whole_assessment"] = "unestablished"
        self.review["gaps"] = [{"claim_ids": [], "join_ids": ["J1"], "reason": "The mediator join is not established.",
                                "missing_experiment": "Clarify the tested mediator-to-endpoint dependency."}]
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["claims"][0]["verdict"], "refuted")
        gap = next(item for item in result["critical_gaps"] if item["claim_ids"] == ["C1"])
        self.assertIn("已有适用的反向结果", gap["reason"])
        self.assertIn("修正主张", gap["missing_experiment"])
        self.assertNotIn("没有适用证据", gap["reason"])

    def test_atomic_coverage_does_not_supply_a_missing_join(self):
        self.review["observations"] = []
        self.review["combinations"][0]["observation_ids"] = ["L1"]
        self.review["rationale"] = "L1 already establishes causal join J1."
        with self.assertRaisesRegex(EvaluationError, "response is incomplete.*J1"):
            aggregate(self.case, self.local, self.review)
        self.review["gaps"] = [{"claim_ids": [], "join_ids": ["J1"], "reason": "Only local atomic results are available.",
                                "missing_experiment": "Test mediator blockade while measuring the endpoint under A."}]
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["overall"]["verdict"], "partial")
        self.assertEqual(result["joins"][0]["verdict"], "unestablished")
        self.assertEqual(result["witnesses"][0]["missing_obligation_ids"], ["J1"])
        self.assertTrue(result["critical_gaps"])
        self.assertEqual(aggregate(self.case, self.local, self.review, use_joins=False)["overall"]["verdict"], "supported")
        self.review["additional_joins"] = [{"id": "J2", "text": "An omitted required dependency.",
                                            "claim_ids": ["C1", "C2"]}]
        with self.assertRaisesRegex(EvaluationError, "response is incomplete.*J2"):
            aggregate(self.case, self.local, self.review)

    def test_invalid_references_and_execution_failures_raise(self):
        invalid_local = deepcopy(self.local)
        invalid_local["observations"][0]["join_ids"] = ["J1"]
        with self.assertRaises(EvaluationError):
            aggregate(self.case, invalid_local, self.review)
        self.local["observations"][0]["passage_ids"] = ["invented:P001"]
        with self.assertRaises(EvaluationError):
            aggregate(self.case, self.local, self.review)
        with self.assertRaises(EvaluationError):
            evaluate(self.case, lambda *_args: {"wrong": "schema"})

        def unavailable(*_args):
            raise TimeoutError("upstream timed out")

        with self.assertRaisesRegex(EvaluationError, "TimeoutError"):
            evaluate(self.case, unavailable)

    def test_second_stage_cannot_reuse_a_local_id_for_a_different_finding(self):
        # The real batch failed when the second stage reused local IDs for its own records.
        self.review["observations"][0]["id"] = "L1"
        with self.assertRaises(EvaluationError):
            aggregate(self.case, self.local, self.review)

        def colliding_call(stage, _instructions, _payload, _schema):
            return deepcopy(self.local if stage == "local_review" else self.review)

        with self.assertRaisesRegex(EvaluationError, "mechanism_review response is invalid"):
            evaluate(self.case, colliding_call)
        self.review["observations"][0]["id"] = "M1"
        self.assertEqual(aggregate(self.case, self.local, self.review)["overall"]["verdict"], "supported")
        captured = []

        def echoed_call(stage, _instructions, payload, _schema):
            if stage == "local_review":
                return deepcopy(self.local)
            echoed = deepcopy(self.review)
            echoed["observations"].insert(0, deepcopy(payload["local_review"]["observations"][0]))
            captured.append((echoed, deepcopy(echoed)))
            return echoed

        result = evaluate(self.case, echoed_call)
        self.assertEqual(result["overall"]["verdict"], "supported")
        self.assertEqual([item["id"] for item in result["mechanism_review"]["observations"]], ["M1"])
        self.assertEqual(captured[0][0], captured[0][1])

        def edited_echo_call(stage, _instructions, payload, _schema):
            if stage == "local_review":
                return deepcopy(self.local)
            changed = deepcopy(self.review)
            local_copy = deepcopy(payload["local_review"]["observations"][0])
            local_copy["rationale"] += " One changed field."
            changed["observations"].insert(0, local_copy)
            return changed

        with self.assertRaisesRegex(EvaluationError, "mechanism_review response is invalid"):
            evaluate(self.case, edited_echo_call)

    def test_passage_selection_preserves_spans_and_hides_removed_text(self):
        original = self.case["materials"][0]["text"]
        for passage in prepare_passages(self.case):
            self.assertEqual(passage["text"], original[passage["start"]:passage["end"]])
        self.case["materials"][0]["allowed_passage_ids"] = ["m:P001"]
        packet = prepare_materials(self.case)
        self.assertEqual(len(packet[0]["passages"]), 1)
        self.assertNotIn("Blocking M", str(packet))
        self.assertNotIn("text", {key: value for key, value in packet[0].items() if key != "passages"})
        with self.assertRaises(EvaluationError):
            aggregate(self.case, self.local, self.review)
        self.case["materials"][0]["allowed_passage_ids"] = ["m:P999"]
        with self.assertRaises(ValueError):
            prepare_passages(self.case)

    def test_duplicate_publication_does_not_increase_source_count(self):
        duplicate = {**deepcopy(self.case["materials"][0]), "id": "copy"}
        self.case["materials"].append(duplicate)
        copy_observation = {**deepcopy(self.observation), "id": "L2", "passage_ids": ["copy:P001", "copy:P002"]}
        self.local["observations"].append(copy_observation)
        self.review["combinations"][0]["observation_ids"].append("L2")
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["independence"]["distinct_source_groups"], 1)
        self.assertIsNone(result["independence"]["independent_cohorts"])
        self.assertEqual(len(result["claims"][0]["distinct_supporting_source_groups"]), 1)
        self.assertEqual(aggregate(self.case, self.local, self.review, deduplicate=False)["independence"]["distinct_source_groups"], 2)
        self.case["materials"].reverse()
        self.assertEqual(aggregate(self.case, self.local, self.review)["overall"], result["overall"])

    def test_publication_identity_across_database_namespaces(self):
        base = deepcopy(self.case["materials"][0])
        base.update(source_group="", source_id="", record_id="", url=None)
        equivalents = [
            {"source_group": "PMID/32954360"},
            {"source_id": "pubmed", "record_id": "32954360"},
            {"source_id": "europe_pmc", "record_id": "MED:32954360"},
            {"url": "https://europepmc.org/article/MED/32954360"},
            {"url": "https://pubmed.ncbi.nlm.nih.gov/32954360/"},
            {"source_id": "pubmed", "record_id": "32954360", "source_group": "DOI:10.1158/example"},
            {"source_id": "europe_pmc", "record_id": "MED:32954360", "source_group": "DOI:10.1158/example"},
        ]
        for fields in equivalents:
            with self.subTest(fields=fields):
                self.assertEqual(source_identity(Material(**(base | fields))), "pmid:32954360")
        self.assertEqual(source_identity(Material(**(base | {"source_id": "pubmed", "record_id": "123"}))), "pmid:123")
        self.assertEqual(source_identity(Material(**(base | {"source_id": "europe_pmc", "record_id": "MED:123"}))), "pmid:123")
        self.assertEqual(source_identity(Material(**(base | {"source_id": "chictr", "record_id": "32954360"}))),
                         "record:chictr:32954360")
        self.assertEqual(source_identity(Material(**(base | {"source_id": "clinicaltrials", "source_group": "32954360"}))),
                         "group:32954360")
        # An unlinked DOI is kept as a DOI; this function does not guess its PMID.
        self.assertEqual(source_identity(Material(**(base | {"source_group": "DOI:10.1158/EXAMPLE"}))), "doi:10.1158/example")

    def test_pubmed_and_europe_pmc_keep_original_versions_but_count_one_paper(self):
        self.case["materials"][0].update(source_group="", source_id="pubmed", record_id="123", version="pubmed-v1")
        duplicate = {**deepcopy(self.case["materials"][0]), "id": "copy", "source_id": "europe_pmc",
                     "record_id": "MED:123", "version": "epmc-v2"}
        self.case["materials"].append(duplicate)
        self.local["observations"].append({**deepcopy(self.observation), "id": "L2",
                                           "passage_ids": ["copy:P001", "copy:P002"]})
        self.review["combinations"][0]["observation_ids"].append("L2")
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["independence"]["distinct_source_groups"], 1)
        self.assertIsNone(result["independence"]["independent_cohorts"])
        self.assertEqual(result["material_assessability"]["source_groups"], ["pmid:123"])
        self.assertEqual({m["source_group"] for m in prepare_materials(self.case)}, {"pmid:123"})
        self.assertEqual(result["claims"][0]["distinct_supporting_source_groups"], ["pmid:123"])
        references = [ref for evidence in result["evidence"] for ref in evidence["references"]]
        self.assertEqual({(r["source_id"], r["record_id"], r["version"]) for r in references},
                         {("pubmed", "123", "pubmed-v1"), ("europe_pmc", "MED:123", "epmc-v2")})

    def test_real_partial_handoff_preserves_complete_result_and_text_positions(self):
        packet = json.loads((Path(__file__).resolve().parents[1] / "examples/partial_handoff.json").read_text())
        case, saved = packet["case"], packet["evaluation"]
        stages = []

        def replay(stage, _instructions, _payload, _schema):
            stages.append(stage)
            return deepcopy(saved[stage])

        result = evaluate(case, replay)
        self.assertEqual(stages, ["local_review", "mechanism_review"])
        self.assertEqual(result, saved)
        self.assertEqual(result["overall"]["verdict"], "partial")
        self.assertTrue(all(row["verdict"] == "supported" for row in result["claims"]))
        self.assertEqual({row["id"]: row["verdict"] for row in result["joins"]},
                         {"J1": "unestablished", "J2": "supported"})
        for field in packet["consumer_checks"]["required_result_fields"]:
            self.assertIn(field, result)
        materials = {m["id"]: m for m in case["materials"]}
        for observation in result["evidence"]:
            for ref in observation["references"]:
                material = materials[ref["material_id"]]
                self.assertEqual(material["text"][ref["start"]:ref["end"]], ref["text"])
                self.assertEqual(ref["version"], material["version"])
        self.assertNotEqual(packet["provenance"]["fabric_abstract"]["version_id"],
                            packet["provenance"]["fulltext_supplement"]["version"])

    def test_primary_and_species_checks_override_an_overclaimed_combination(self):
        self.case["materials"][0]["source_kind"] = "review"
        self.assertEqual(aggregate(self.case, self.local, self.review)["overall"]["verdict"], "insufficient")
        review_source = deepcopy(self.case["materials"][0])
        review_source["id"] = "review"
        self.case["materials"][0]["source_kind"] = "primary"
        self.case["materials"].append(review_source)
        self.review["observations"][0]["passage_ids"].append("review:P002")
        self.assertNotEqual(aggregate(self.case, self.local, self.review)["overall"]["verdict"], "supported")
        self.review["observations"][0]["passage_ids"].remove("review:P002")
        self.case["scope"] = {}
        self.review["observations"][0]["context"] = {"species": "human", "system": "different tissue"}
        result = aggregate(self.case, self.local, self.review)
        self.assertNotEqual(result["overall"]["verdict"], "supported")
        self.assertFalse(result["witnesses"][0]["known_species_consistent"])
        self.review["observations"][0]["context"] = {"species": "mouse", "system": "S"}
        negative = {**deepcopy(self.observation), "id": "M2", "relation": "opposes",
                    "context": {"species": "human", "system": "different tissue"}}
        self.review["observations"].append(negative)
        self.review["comparisons"] = [{"observation_ids": ["L1", "M2"], "context_relation": "comparable",
                                       "rationale": "Deliberately overclaimed comparison."}]
        result = aggregate(self.case, self.local, self.review)
        self.assertFalse(result["conflicts"])
        self.assertEqual(result["claims"][0]["verdict"], "supported")
        self.assertTrue(result["claims"][0]["applicability_gaps"])
        self.review["observations"].pop()
        self.review["comparisons"] = []
        self.review["observations"][0]["context"]["species"] = "Mus musculus"
        result = aggregate(self.case, self.local, self.review)
        self.assertEqual(result["overall"]["verdict"], "supported")
        self.assertTrue(result["witnesses"][0]["known_species_consistent"])


if __name__ == "__main__":
    unittest.main()
