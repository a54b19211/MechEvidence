import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from mechevidence.fabric import to_material
from mechevidence.model import DeepSeek, ModelError


class InputAndCallTests(unittest.TestCase):
    def test_review_is_not_promoted_to_experimental_evidence(self):
        record = {"source_id": "europe_pmc", "source_record_id": "MED:12345678",
                  "version_id": "v1", "clean_text": "<p>A review.</p>",
                  "facts": {"pmid": "12345678", "publication_types": {"pubType": ["Review", "Journal Article"]}}}
        self.assertEqual(to_material(record)["source_kind"], "review")
        self.assertEqual(to_material(record)["source_group"], "PMID:12345678")
        record["facts"]["publication_types"] = {"pubType": ["Journal Article"]}
        with self.assertRaises(ValueError):
            to_material(record)

    def test_html_cleanup_preserves_statistical_inequalities(self):
        record = {"source_id": "europe_pmc", "source_record_id": "MED:12345678",
                  "version_id": "v1", "clean_text": "<p>Treatment p&lt;0.05; &gt;90% viable.</p>",
                  "facts": {"publication_types": {"pubType": ["research-article"]}}}
        self.assertEqual(to_material(record)["text"], "Treatment p<0.05; >90% viable.")

    def test_failed_or_truncated_calls_are_not_scientific_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("DEEPSEEK_API_KEY=test-not-a-real-key\n")
            call = DeepSeek(root, root / "calls")
            response = httpx.Response(200, json={"choices": [{"finish_reason": "length",
                "message": {"content": '{"verdict":"supported"}', "reasoning_content": "hidden"}}],
                "usage": {"total_tokens": 10}})
            with patch("mechevidence.model.httpx.post", return_value=response):
                with self.assertRaises(ModelError):
                    call("review", "Return JSON", {}, {})
            saved = (root / "calls/call-01.json").read_text()
            self.assertNotIn("test-not-a-real-key", saved)
            self.assertNotIn("reasoning_content", saved)
            self.assertEqual(json.loads(saved)["status"], "failed")
            self.assertEqual(json.loads(saved)["usage"]["total_tokens"], 10)

    def test_success_saves_only_final_content_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("DEEPSEEK_API_KEY=test-not-a-real-key\n")
            call = DeepSeek(root, root / "calls")
            response = httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                "message": {"content": '{"answer":"ok"}', "reasoning_content": "hidden"}}],
                "usage": {"total_tokens": 10}})
            with patch("mechevidence.model.httpx.post", return_value=response):
                self.assertEqual(call("review", "Return JSON", {}, {}), {"answer": "ok"})
            saved = json.loads((root / "calls/call-01.json").read_text())
            self.assertEqual(saved["content"], '{"answer":"ok"}')
            self.assertNotIn("reasoning_content", json.dumps(saved))


if __name__ == "__main__":
    unittest.main()
