"""Offline publication-identity contract, including the coordinator's two reproductions."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from mechevidence import (IdentityInputError, Material, direct_baseline, evaluate,
                         prepare_materials, source_identity)
from mechevidence.fabric import to_material


class IdentityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).resolve().parents[1] /
                                 "examples/identity_contract.json").read_text())

    def material(self, fields):
        return Material(**(self.fixture["base_material"] | fields))

    def test_accepted_identifiers_and_cross_source_pairs(self):
        for row in self.fixture["accepted"]:
            with self.subTest(row["name"]):
                material = self.material(row["fields"])
                before = material.model_dump()
                self.assertEqual(source_identity(material), row["identity"])
                self.assertEqual(material.model_dump(), before)
        for row in self.fixture["pairs"]:
            with self.subTest(row["name"]):
                self.assertEqual(source_identity(self.material(row["left"])) ==
                                 source_identity(self.material(row["right"])), row["same_group"])

    def test_pseudo_urls_and_free_text_do_not_assert_publication_identifiers(self):
        for row in self.fixture["unrecognized"]:
            with self.subTest(row["name"]):
                material = self.material(row["fields"])
                self.assertTrue(source_identity(material).startswith(row["identity_prefix"]))
                for field in ("source_id", "record_id", "version", "url"):
                    if field in row["fields"]:
                        self.assertEqual(getattr(material, field), row["fields"][field])

    def test_conflicts_and_invalid_identifiers_stop_before_any_model_call(self):
        for row in self.fixture["rejected"]:
            for function in (evaluate, direct_baseline):
                for selected in (None, []):
                    with self.subTest(row["name"], entry=function.__name__, selected=selected):
                        # A valid first material cannot make a later excluded conflict disappear.
                        bad = self.fixture["base_material"] | row["fields"] | {"id": "bad", "allowed_passage_ids": selected}
                        case = {"question": "A mechanism question", "mechanism": "A candidate",
                                "materials": [self.fixture["base_material"], bad]}
                        calls = []

                        def forbidden_call(*args):
                            calls.append(args)
                            self.fail("Invalid identity reached a model callback")

                        with self.assertRaises(IdentityInputError) as error:
                            function(case, forbidden_call)
                        details = error.exception.as_dict()
                        self.assertEqual(details["status"], "needs_input")
                        self.assertEqual(details["error_code"], row["error_code"])
                        self.assertEqual(details["identifier_type"], row["identifier_type"])
                        self.assertEqual(details["material_id"], "bad")
                        self.assertTrue(details["fields"])
                        self.assertEqual(calls, [])

    def test_new_fields_preserve_original_metadata_and_optional_output_shape(self):
        fields = {"pmid": "111", "doi": "10.1234/ABC", "source_id": "pubmed",
                  "record_id": "111", "version": "source-v7", "url": "https://pubmed.ncbi.nlm.nih.gov/111/"}
        case = {"question": "Question", "mechanism": "Candidate",
                "materials": [self.fixture["base_material"] | fields]}
        view = prepare_materials(case)[0]
        for field, value in fields.items():
            self.assertEqual(view[field], value)
        for material in (self.fixture["base_material"], self.fixture["base_material"] | {"pmid": None, "doi": None}):
            view = prepare_materials(case | {"materials": [material]})[0]
            self.assertNotIn("pmid", view)
            self.assertNotIn("doi", view)
        for value in (True, 1.5, [], {}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.material({"pmid": value})
        with self.assertRaises(ValidationError):
            self.material({"pubmed_id": "111"})

    def test_fabric_transmits_explicit_identifiers_without_resolving_them(self):
        record = {"source_id": "europe_pmc", "source_record_id": "MED:111", "version_id": "original-v1",
                  "clean_text": "Actual text.", "facts": {"pmid": "111", "doi": "10.1234/ABC"}}
        material = to_material(record, source_kind="primary")
        self.assertEqual((material["pmid"], material["doi"]), ("111", "10.1234/ABC"))
        self.assertEqual(source_identity(Material(**material)), "pmid:111")
        record = deepcopy(record)
        record["source_record_id"] = "MED:222"
        with self.assertRaises(IdentityInputError):
            source_identity(Material(**to_material(record, source_kind="primary")))


if __name__ == "__main__":
    unittest.main()
