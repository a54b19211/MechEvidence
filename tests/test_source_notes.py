"""Controlled semantic-state fixtures; no model calls or HTTP requests."""
from copy import deepcopy
import unittest
import test_assess as legacy_cases
from mechevidence import aggregate, evaluate, EvaluationError, prepare_passages, generation_view
from mechevidence.service import execute


class SourceNoteTests(unittest.TestCase):
    def setUp(self):
        fixture = legacy_cases.AssessmentTests()
        fixture.setUp()
        self.case, self.local, self.review = fixture.case, fixture.local, fixture.review
        self.local["source_notes"] = [self.note("local.observations.L1.finding")]
        self.review["source_notes"] = deepcopy(self.local["source_notes"]) + [self.note("mechanism.observations.M1.finding")]

    def note(self, target, status="reported", original="", statement="所引段落报告该观察。", negative_kind=None):
        passage = prepare_passages(self.case)[0]
        return {"target": target, "original": original, "status": status, "statement": statement,
                "negative_kind": negative_kind, "anchors": [] if status == "not_reported" else [{
                    "passage_id": passage["id"], "start": passage["start"], "end": passage["end"], "quote": passage["text"]}]}

    def call(self, stage, *_):
        return deepcopy(self.local if stage == "local_review" else self.review)

    def test_live_requires_notes_and_legacy_replay_is_explicit(self):
        self.local.pop("source_notes");self.review.pop("source_notes")
        with self.assertRaises(EvaluationError): evaluate(self.case, self.call)
        old = evaluate(self.case, self.call, legacy_replay=True)
        self.assertEqual(old["source_reporting"]["status"], "legacy_unreviewed")
        self.assertEqual(old["overall"]["verdict"], "supported")

    def test_reporting_states_and_negative_result_are_distinct(self):
        states = [("reported", None, "The intervention was performed.", True),
                  ("not_reported", None, "The report describes a different measurement.", False),
                  ("explicit_negative", "method_not_performed", "The intervention was not performed.", False),
                  ("explicit_negative", "negative_result", "The intervention produced no response.", True)]
        for status, kind, text, eligible in states:
            with self.subTest(status=status, kind=kind):
                self.case["materials"][0]["text"] = text
                for obs in self.local["observations"] + self.review["observations"]:
                    obs["passage_ids"] = ["m:P001"]
                    obs["relation"] = "opposes" if status != "reported" else "supports"
                self.local["source_notes"] = [self.note("local.observations.L1.finding", status, negative_kind=kind)]
                self.review["source_notes"] = deepcopy(self.local["source_notes"]) + [
                    self.note("mechanism.observations.M1.finding", status, negative_kind=kind)]
                result = evaluate(self.case, self.call)
                self.assertEqual(result["evidence"][0]["causal_evidence_eligible"], eligible)
                self.assertEqual(bool(result["claims"][0]["opposing_observation_ids"]), status == "explicit_negative" and kind == "negative_result")
                if status == "not_reported": self.assertEqual(result["claims"][0]["verdict"], "insufficient")

    def test_mechanism_review_corrects_local_text_without_erasing_real_finding(self):
        bad = "研究没有实施方法乙。"
        self.local["observations"][0]["limitations"] = [bad]
        self.local["observations"][0]["rationale"] = "已报告干预结果。" + bad
        self.local["gaps"] = [{"claim_ids": ["C1"], "join_ids": [], "reason": bad, "missing_experiment": "补查原研究的方法记录。"}]
        targets = ["local.observations.L1.limitations.0", "local.observations.L1.rationale", "local.gaps.0.reason"]
        self.local["source_notes"] += [self.note(t, "explicit_negative", bad, bad, "method_not_performed") for t in targets]
        # Deliberately wrong local annotations; the controlled M audit corrects them from original material.
        self.review["source_notes"] += [self.note(t, "not_reported", bad, "是否实施方法乙") for t in targets]
        before = deepcopy(self.local)
        result = evaluate(self.case, self.call)
        self.assertEqual(self.local, before)
        self.assertEqual(result["overall"]["verdict"], "supported")
        self.assertTrue(result["claims"][0]["supporting_observation_ids"])
        for text in [result["evidence"][0]["limitations"][0], result["evidence"][0]["rationale"], result["local_review"]["gaps"][0]["reason"]]:
            self.assertNotIn(bad, text)
            self.assertIn("本次允许材料", text)
            self.assertIn("仍未知", text)
        raw = result["source_reporting"]["raw_reviews"]["local_review"]
        self.assertEqual(raw["observations"][0]["limitations"], [bad])
        self.assertEqual(result["evidence"][0]["relation"], self.local["observations"][0]["relation"])
        self.assertEqual(result["evidence"][0]["passage_ids"], self.local["observations"][0]["passage_ids"])

    def test_unknown_limitation_does_not_remove_reported_result(self):
        self.local["observations"][0]["limitations"] = ["方法乙的实施情况"]
        note = self.note("local.observations.L1.limitations.0", "not_reported", "方法乙的实施情况", "方法乙的实施情况")
        self.local["source_notes"].append(note)
        self.review["source_notes"].append(deepcopy(note))
        result = evaluate(self.case, self.call)
        self.assertEqual(result["overall"]["verdict"], "supported")
        self.assertTrue(result["evidence"][0]["causal_evidence_eligible"])

    def test_bad_anchors_targets_or_missing_recheck_are_rejected(self):
        for problem in ["quote", "position", "unrelated_passage", "protected_target", "missing_local_finding"]:
            with self.subTest(problem=problem):
                review = deepcopy(self.review)
                note = review["source_notes"][0]
                if problem == "quote": note["anchors"][0]["quote"] = "Invented source quotation."
                elif problem == "position": note["anchors"][0]["start"] += 1
                elif problem == "unrelated_passage": note["anchors"][0]["passage_id"] = "absent:P001"
                elif problem == "protected_target": note["target"] = "local.observations.L1.relation"
                else: review["source_notes"].pop(0)
                with self.assertRaises(EvaluationError): aggregate(self.case, self.local, review)

    def test_wrong_and_overlapping_revisions_are_rejected(self):
        self.local["observations"][0]["limitations"] = ["需要核对方法甲及方法乙。"]
        for originals in [["原文不存在的片段"], ["方法甲及方法乙", "方法乙"]]:
            with self.subTest(originals=originals):
                review = deepcopy(self.review)
                review["source_notes"] += [self.note("local.observations.L1.limitations.0", "not_reported", original, "实施情况") for original in originals]
                with self.assertRaises(EvaluationError): aggregate(self.case, self.local, review)

    def test_missing_disclosure_is_not_an_opposing_result_for_any_claim_kind(self):
        self.case["materials"][0]["text"] = "The report does not disclose the measurement results."
        for obs in self.local["observations"] + self.review["observations"]:
            obs["passage_ids"] = ["m:P001"]
            obs["relation"] = "opposes"
        self.local["source_notes"] = [self.note("local.observations.L1.finding", "explicit_negative", negative_kind="not_disclosed")]
        self.review["source_notes"] = deepcopy(self.local["source_notes"]) + [self.note("mechanism.observations.M1.finding", "explicit_negative", negative_kind="not_disclosed")]
        for negative_kind in ["not_disclosed", "method_not_performed"]:
            for kind in ["causal", "association", "reported"]:
                with self.subTest(kind=kind, negative_kind=negative_kind):
                    for note in self.local["source_notes"] + self.review["source_notes"]:
                        note["negative_kind"] = negative_kind
                    for claim in self.case["claims"]: claim["kind"] = kind
                    result = evaluate(self.case, self.call)
                    self.assertEqual(result["overall"]["verdict"], "insufficient")
                    self.assertFalse(result["claims"][0]["opposing_observation_ids"])
                    self.assertTrue(result["evidence"][0]["references"])

    def test_text_correction_can_read_other_passage_of_same_publication(self):
        self.case["materials"][0]["text"] = "A changed M. Method B was performed."
        for obs in self.local["observations"] + self.review["observations"]:
            obs["passage_ids"] = ["m:P001"]
        self.local["observations"][0]["limitations"] = ["方法B的实施情况"]
        unknown = self.note("local.observations.L1.limitations.0", "not_reported", "方法B的实施情况", "方法B的实施情况")
        self.local["source_notes"] = [self.note("local.observations.L1.finding"), unknown]
        self.review["source_notes"] = [self.note("local.observations.L1.finding"), self.note("mechanism.observations.M1.finding")]
        other = prepare_passages(self.case)[1]
        correction = self.note(unknown["target"], "reported", unknown["original"], "原文报告已实施方法B。")
        correction["anchors"] = [{"passage_id": other["id"], "start": other["start"], "end": other["end"], "quote": other["text"]}]
        self.review["source_notes"].append(correction)
        result = evaluate(self.case, self.call)
        self.assertEqual(result["evidence"][0]["limitations"], ["原文报告已实施方法B。"])
        self.assertEqual(result["evidence"][0]["passage_ids"], ["m:P001"])
        # The same broader passage cannot silently replace the finding's original evidence.
        self.review["source_notes"][0]["anchors"] = deepcopy(correction["anchors"])
        with self.assertRaises(EvaluationError): evaluate(self.case, self.call)

    def test_overlapping_occurrences_are_not_a_unique_original_fragment(self):
        self.local["observations"][0]["limitations"] = ["AAAA"]
        self.review["source_notes"].append(self.note("local.observations.L1.limitations.0", "not_reported", "AAA", "实施情况"))
        with self.assertRaises(EvaluationError): evaluate(self.case, self.call)

    def test_same_paper_supplement_retains_its_own_record_version(self):
        material = deepcopy(self.case["materials"][0])
        material.update(id="supplement", source_id="fulltext", record_id="fulltext-1", version="fulltext-v2", text="Method B was performed.")
        self.case["materials"].append(material)
        self.local["observations"][0]["limitations"] = ["方法B的实施情况"]
        passage = next(p for p in prepare_passages(self.case) if p["material_id"] == "supplement")
        correction = self.note("local.observations.L1.limitations.0", "reported", "方法B的实施情况", "原文报告已实施方法B。")
        correction["anchors"] = [{"passage_id": passage["id"], "start": passage["start"], "end": passage["end"], "quote": passage["text"]}]
        self.review["source_notes"].append(correction)
        result = evaluate(self.case, self.call)
        anchor = result["source_reporting"]["notes"][-1]["anchors"][0]
        self.assertEqual((anchor["material_id"], anchor["source_id"], anchor["record_id"], anchor["version"]),
                         ("supplement", "fulltext", "fulltext-1", "fulltext-v2"))
        for source in self.case["materials"]: source["source_group"] = "unverified-upstream-label"
        with self.assertRaises(EvaluationError): evaluate(self.case, self.call)

    def test_generation_view_excludes_raw_reviews_and_rejects_legacy(self):
        result = evaluate(self.case, self.call)
        view = generation_view(result)
        self.assertNotIn("raw_reviews", view["source_reporting"])
        self.assertIn("raw_reviews", result["source_reporting"])
        for field in ["claims", "joins", "evidence", "critical_gaps", "conflicts", "unresolved_counterevidence"]:
            self.assertEqual(view[field], result[field])
        view["local_review"]["observations"][0]["rationale"] = "changed consumer copy"
        self.assertNotEqual(view["local_review"], result["local_review"])
        with self.assertRaises(EvaluationError): evaluate(self.case, self.call, legacy_replay=True)
        self.local.pop("source_notes");self.review.pop("source_notes")
        with self.assertRaises(ValueError):
            generation_view(evaluate(self.case, self.call, legacy_replay=True))

    def test_unknown_finding_does_not_leave_a_definite_negative_rationale_active(self):
        bad = "研究没有实施该方法。"
        self.local["observations"][0]["rationale"] = bad
        self.local["source_notes"] = [self.note("local.observations.L1.finding", "not_reported", statement="该方法是否实施")]
        self.review["source_notes"] = deepcopy(self.local["source_notes"]) + [self.note("mechanism.observations.M1.finding")]
        result = evaluate(self.case, self.call)
        view = generation_view(result)
        self.assertIn("仍未知", view["evidence"][0]["rationale"])
        self.assertNotIn(bad, view["evidence"][0]["rationale"])
        self.assertIn("仍未知", view["local_review"]["observations"][0]["rationale"])
        self.assertEqual(result["source_reporting"]["raw_reviews"]["local_review"]["observations"][0]["rationale"], bad)

    def test_pure_rpc_replay_keeps_new_contract_without_http(self):
        request = {"schema_version": "bioradar.professional.v1", "operation_id": "source-notes-test",
                   "operation": "evaluate", "arguments": {"mode": "biomedical_research", "task": "mechanism", "case": self.case}, "model_receipts": []}
        for _ in range(3):
            response = execute(request)
            if response["status"] == "completed": break
            self.assertEqual(response["status"], "model_required")
            required = response["model_call"]
            self.assertIn("source_notes", required["schema"]["required"])
            request["provider_version"] = response["provider_version"]
            request["model_receipts"].append({"call_id": required["call_id"], "request_hash": required["request_hash"], "output": self.call(required["stage"])})
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["result"]["source_reporting"]["status"], "reviewed")
        for field in ["claims", "joins", "critical_gaps", "conflicts", "unresolved_counterevidence"]:
            self.assertIn(field, response["result"])
