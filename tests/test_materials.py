"""Material admission and context-only safeguards; saved/controlled outputs, no model API."""
from copy import deepcopy
from http.client import HTTPConnection
import json
from pathlib import Path
import threading
import unittest

from mechevidence import MaterialInputError, aggregate, direct_baseline, evaluate
from mechevidence.assess import EvaluationError, prepare_materials
from mechevidence.cohorts import assess_independence
from mechevidence.service import ProfessionalServer, capabilities, execute


class MaterialContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads((Path(__file__).resolve().parents[1] / 'examples/partial_handoff.json').read_text())

    def case(self, document_type=None):
        case = deepcopy(self.saved['case'])
        if document_type is not None:
            case['materials'][0]['document_type'] = document_type
        return case

    def request(self, case):
        return {'schema_version': 'bioradar.professional.v1', 'operation_id': 'material-check', 'operation': 'evaluate',
                'arguments': {'mode': 'biomedical_research', 'task': 'mechanism', 'case': case}, 'model_receipts': []}

    def test_missing_source_kind_is_not_inferred_and_feedback_identifies_each_material(self):
        case = self.case('research_article')
        case['materials'][0].pop('source_kind')
        second = deepcopy(case['materials'][0]);second['id'] = 'other';case['materials'].append(second)
        result = execute(self.request(case))
        self.assertEqual(result['status'], 'needs_input')
        self.assertNotIn('model_call', result)
        self.assertEqual(result['missing_fields'], ['materials.M1.source_kind', 'materials.other.source_kind'])
        self.assertEqual([x['material_id'] for x in result['material_feedback']], ['M1', 'other'])
        self.assertEqual(result['material_feedback'][0]['input_path'], ['arguments', 'case', 'materials', 0, 'source_kind'])
        case = self.case();case['materials'][0]['version'] = ''
        self.assertEqual(execute(self.request(case))['missing_fields'], ['materials.M1.version'])

    def test_declarations_cannot_be_promoted_to_primary_even_in_a_demo(self):
        for kind in ['press_release', 'company_disclosure', 'news', 'social_post']:
            for function in [evaluate, direct_baseline]:
                with self.subTest(kind=kind, entry=function.__name__):
                    case = self.case(kind);case['materials'][0]['is_demo'] = True
                    calls = []
                    with self.assertRaises(MaterialInputError) as error:
                        function(case, lambda *args: calls.append(args))
                    self.assertEqual(error.exception.as_dict()['error_code'], 'material_type_not_evaluable')
                    self.assertEqual(calls, [])
            response = execute(self.request(self.case(kind)))
            self.assertEqual(response['status'], 'needs_input')
            self.assertTrue(response['material_feedback'][0]['retain_as_radar_lead'])

    def test_explicit_secondary_or_graph_type_cannot_hide_behind_primary(self):
        for kind in ['review', 'graph']:
            with self.subTest(kind=kind):
                response = execute(self.request(self.case(kind)))
                self.assertEqual(response['error_code'], 'material_type_conflict')
                self.assertNotIn('model_call', response)

    def test_registration_and_protocol_are_context_even_for_reported_claims(self):
        saved = self.saved['evaluation']
        for kind in ['trial_registration', 'study_protocol']:
            for claim_kind in ['causal', 'association', 'reported']:
                with self.subTest(kind=kind, claim_kind=claim_kind):
                    case = self.case(kind)
                    for claim in case['claims']:
                        claim['kind'] = claim_kind
                    # Deliberately reuse favourable experimental observations to test the program boundary.
                    result = aggregate(case, saved['local_review'], saved['mechanism_review'])
                    self.assertEqual(result['overall']['verdict'], 'insufficient')
                    self.assertTrue(all(not row['supporting_observation_ids'] for row in result['claims'] + result['joins']))
                    self.assertTrue(all(not row['causal_evidence_eligible'] for row in result['evidence']))
                    self.assertTrue(all(ref['document_type'] == kind for row in result['evidence'] for ref in row['references']))
            response = execute(self.request(self.case(kind)))
            self.assertEqual(response['status'], 'model_required')  # Context remains visible; not silently discarded.

    def test_direct_baseline_cannot_use_design_as_mechanism_support(self):
        for kind in ['trial_registration', 'study_protocol']:
            case = self.case(kind)
            output = {'verdict': 'supported', 'rationale': 'Deliberately invalid protocol fixture.',
                      'supporting_passage_ids': ['M1:P001'], 'opposing_passage_ids': [],
                      'scope_limitations': [], 'missing_experiments': []}
            with self.subTest(kind=kind), self.assertRaises(EvaluationError):
                direct_baseline(case, lambda *_: deepcopy(output))
            output.update(verdict='insufficient', supporting_passage_ids=[], missing_experiments=['Retrieve actual results.'])
            self.assertEqual(direct_baseline(case, lambda *_: deepcopy(output))['verdict'], 'insufficient')

    def test_protocols_cannot_confirm_patient_relationships(self):
        materials = [deepcopy(self.case('study_protocol')['materials'][0]) for _ in range(2)]
        materials[0]['id'], materials[1]['id'] = 'a', 'b'
        arguments = {'materials': materials, 'pairs': [{'left_material_id': 'a', 'right_material_id': 'b', 'unit': 'patient_cohort'}]}
        output = {'pairs': [{'left_material_id': 'a', 'right_material_id': 'b', 'unit': 'patient_cohort',
                  'relation': 'independent', 'basis': 'sampling_details', 'passage_ids': ['a:P001', 'b:P001'],
                  'rationale': 'Deliberately overclaimed planned-cohort fixture.', 'uncertainty': [], 'information_needs': []}]}
        result = assess_independence(arguments, lambda *_: deepcopy(output))
        self.assertEqual(result['pairs'][0]['relation'], 'unknown')
        self.assertEqual(result['pairs'][0]['proposed_relation'], 'independent')
        self.assertTrue(result['pairs'][0]['information_needs'])

    def test_original_conference_and_trial_results_require_explicit_primary(self):
        for kind in ['research_article', 'conference_abstract', 'trial_results']:
            with self.subTest(kind=kind):
                case = self.case(kind)
                response = execute(self.request(case))
                self.assertEqual(response['status'], 'model_required')
                material = response['model_call']['payload']['materials'][0]
                self.assertEqual(material['document_type'], kind)
                self.assertEqual(material['source_kind'], 'primary')
                case['materials'][0]['source_kind'] = 'review'
                self.assertEqual(execute(self.request(case))['error_code'], 'material_type_conflict')
        self.assertIn('material_contract', capabilities())
        view = prepare_materials(self.case())[0]
        self.assertNotIn('document_type', view)
        self.assertNotIn('is_demo', view)

    def test_http_saved_original_mechanism_with_type_and_demo_metadata_round_trips(self):
        server = ProfessionalServer(('127.0.0.1', 0), 'isolated-material-token')
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        case = self.case('research_article');case['materials'][0]['is_demo'] = True
        request = self.request(case)
        try:
            for _ in range(3):
                client = HTTPConnection(*server.server_address, timeout=5)
                client.request('POST', '/v1/execute', json.dumps(request),
                               {'Authorization': 'Bearer isolated-material-token', 'Content-Type': 'application/json'})
                reply = client.getresponse();self.assertEqual(reply.status, 200)
                result = json.loads(reply.read());client.close()
                if result['status'] == 'completed':break
                self.assertEqual(result['status'], 'model_required')
                call = result['model_call']
                self.assertTrue(call['payload']['materials'][0]['is_demo'])
                request['provider_version'] = result['provider_version']
                request['model_receipts'].append({'call_id': call['call_id'], 'request_hash': call['request_hash'],
                                                 'output': deepcopy(self.saved['evaluation'][call['stage']])})
            self.assertEqual(result['status'], 'completed')
            native = evaluate(case, lambda stage, *_: deepcopy(self.saved['evaluation'][stage]))
            self.assertEqual(result['result'], native)
            self.assertEqual(native['overall'], self.saved['evaluation']['overall'])
            for observation in native['evidence']:
                for ref in observation['references']:
                    self.assertTrue(ref['is_demo'])
                    self.assertEqual(ref['document_type'], 'research_article')
                    self.assertEqual(ref['version'], case['materials'][0]['version'])
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
