"""Optional sealed-B5 regression. The private row is read in place, never copied or edited."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest

from mechevidence import evaluate, generation_view, EvaluationError, prepare_passages


@unittest.skipUnless(os.environ.get("MECHEVIDENCE_B5_ROW"), "Set the private B5 row path for offline historical regression")
class SavedB5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ["MECHEVIDENCE_B5_ROW"])
        raw = cls.path.read_bytes()
        cls.original_hash = hashlib.sha256(raw).hexdigest()
        if cls.original_hash != "2e3a9417a3171154b5e92898113000b1bfc54eaa378bb07aeaed7c991c5c6186":
            raise ValueError("This test expects the sealed B5 row, not a replacement input")
        row = json.loads(raw)
        cls.outputs = {c["stage"][5:]: c["output"] for c in row["progress"]["calls"] if c["stage"].startswith("mech_")}
        matches = [exchange for op in row["progress"]["professional_operations"].values()
                   for exchange in op.get("exchanges", [])
                   if exchange.get("request", {}).get("operation") == "evaluate"
                   and exchange.get("response", {}).get("provider") == "mech_evidence"
                   and exchange["response"].get("status") == "completed"]
        if len(matches) != 1:
            raise ValueError("Expected one completed saved Mech operation")
        cls.case = matches[0]["request"]["arguments"]["case"]
        cls.saved = matches[0]["response"]["result"]

    def test_legacy_failure_is_preserved_and_not_admitted_as_current_review(self):
        def replay(stage, *_): return deepcopy(self.outputs[stage])
        result = evaluate(self.case, replay, legacy_replay=True)
        self.assertEqual(result.pop("source_reporting"), {"status": "legacy_unreviewed"})
        self.assertTrue(result == self.saved)
        self.assertEqual(result["overall"]["verdict"], "partial")
        with self.assertRaises(EvaluationError): evaluate(self.case, replay)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.original_hash)

    def test_controlled_correction_of_saved_local_and_mechanism_wording(self):
        # An explicit reviewer correction tests application, not a new model run or scientific pass.
        outputs = deepcopy(self.outputs)
        passages = {p["id"]: p for p in prepare_passages(self.case)}
        finding_notes = []
        for prefix, key in [("local", "local_review"), ("mechanism", "mechanism_review")]:
            notes = []
            for obs in outputs[key]["observations"]:
                passage = passages[obs["passage_ids"][0]]
                notes.append({"target": f'{prefix}.observations.{obs["id"]}.finding', "original": "",
                    "status": "reported", "statement": "受控回归保留原观察的资格，不作为新医学标注。", "negative_kind": None,
                    "anchors": [{"passage_id": passage["id"], "start": passage["start"], "end": passage["end"], "quote": passage["text"]}]})
            if prefix == "local":
                outputs[key]["source_notes"] = deepcopy(notes)
            finding_notes += notes
        outputs["mechanism_review"]["source_notes"] = finding_notes
        cases = [("local_review", "local", "L5", 1), ("mechanism_review", "mechanism", "M2", 2),
                 ("mechanism_review", "mechanism", "M3", 3)]
        for key, prefix, oid, index in cases:
            obs = next(o for o in outputs[key]["observations"] if o["id"] == oid)
            outputs["mechanism_review"]["source_notes"].append({
                "target": f"{prefix}.observations.{oid}.limitations.{index}",
                "original": obs["limitations"][index], "status": "not_reported",
                "statement": "相关方法是否实施", "negative_kind": None, "anchors": []})
        result = evaluate(self.case, lambda stage, *_: deepcopy(outputs[stage]))
        self.assertEqual(result["overall"]["verdict"], self.saved["overall"]["verdict"])
        self.assertTrue(result["claims"] == self.saved["claims"])
        self.assertTrue(result["joins"] == self.saved["joins"])
        view = generation_view(result)
        self.assertNotIn("raw_reviews", view["source_reporting"])
        for key, prefix, oid, index in cases:
            obs = next(o for o in result["evidence"] if o["id"] == oid)
            self.assertIn("本次允许材料", obs["limitations"][index])
            self.assertIn("仍未知", obs["limitations"][index])
            old = next(o for o in self.saved["evidence"] if o["id"] == oid)
            self.assertTrue(obs["references"] == old["references"])
        self.assertTrue(result["source_reporting"]["raw_reviews"]["local_review"]["observations"] == self.outputs["local_review"]["observations"])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.original_hash)
