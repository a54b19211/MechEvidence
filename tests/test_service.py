"""Real HTTP transport with saved/controlled receipts; never commercial model calls."""
from copy import deepcopy
from http.client import HTTPConnection
import json
from pathlib import Path
import threading
import unittest

from mechevidence import service


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = 'isolated-protocol-test-token'
        cls.server = service.ProfessionalServer(('127.0.0.1', 0), cls.token)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.snapshot = json.loads((Path(__file__).resolve().parents[1] /
                                  'examples/partial_handoff.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def http(self, method, path, value=None, authorized=True):
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        headers = {'Content-Type': 'application/json'}
        if authorized:
            headers['Authorization'] = 'Bearer ' + self.token
        body = json.dumps(value) if value is not None else None
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        status, data = response.status, json.loads(response.read())
        connection.close()
        return status, data

    def request(self):
        return {'schema_version': service.SCHEMA_VERSION, 'operation_id': 'saved-Q005', 'operation': 'evaluate',
                'provider_version': None, 'arguments': {'mode': 'biomedical_research', 'task': 'mechanism',
                    'case': deepcopy(self.snapshot['case']),
                    'computation_refs': [{'run_id': 'run-1', 'result_id': 'result-1', 'version': 'fixed-v1',
                                          'finding_ids': ['finding-1']}]}, 'model_receipts': []}

    def post(self, request):
        status, response = self.http('POST', '/v1/execute', request)
        self.assertEqual(status, 200)
        return response

    def add_receipt(self, request, response, output=None):
        call = response['model_call']
        request['provider_version'] = response['provider_version']
        request['model_receipts'].append({'call_id': call['call_id'], 'request_hash': call['request_hash'],
                                         'output': deepcopy(output if output is not None else self.snapshot['evaluation'][call['stage']])})

    def test_http_two_step_resume_and_saved_local_result_are_identical(self):
        request = self.request()
        first = self.post(request)
        self.assertEqual(first['status'], 'model_required')
        self.assertEqual(first['model_call']['stage'], 'local_review')
        self.assertEqual(self.post(request), first)  # Lost HTTP reply: identical request, no paid effect.
        self.add_receipt(request, first)
        second = self.post(request)
        self.assertEqual(second['status'], 'model_required')
        self.assertEqual(second['model_call']['stage'], 'mechanism_review')
        self.assertNotEqual(first['model_call']['call_id'], second['model_call']['call_id'])
        self.assertEqual(self.post(request), second)
        self.add_receipt(request, second)
        completed = self.post(request)
        self.assertEqual(completed['status'], 'completed')
        self.assertEqual(completed['result'], self.snapshot['evaluation'])
        self.assertEqual(completed['context']['computation_refs'], request['arguments']['computation_refs'])
        self.assertEqual(self.post(request), completed)

    def test_decomposition_also_pauses_through_core_exception_wrapper(self):
        request = self.request()
        case = request['arguments']['case']
        decomposition = {key: case.pop(key) for key in ('claims', 'joins')}
        first = self.post(request)
        self.assertEqual(first['status'], 'model_required')
        self.assertEqual(first['model_call']['stage'], 'decomposition')
        self.add_receipt(request, first, decomposition)  # Controlled protocol receipt, not a new model result.
        self.assertEqual(self.post(request)['model_call']['stage'], 'local_review')

    def test_mismatched_out_of_order_extra_and_unpinned_receipts_stop(self):
        request = self.request()
        first = self.post(request)
        self.add_receipt(request, first)
        for mutate, expected in [
            (lambda r: r.update(provider_version='other-version'), 'provider_version_mismatch'),
            (lambda r: r.update(provider_version=None), 'provider_version_required'),
            (lambda r: r['model_receipts'][0].update(request_hash='0' * 64), 'model_receipt_mismatch'),
            (lambda r: r['model_receipts'][0].update(call_id='other-call'), 'model_receipt_mismatch'),
            (lambda r: r.update(operation_id='different-operation'), 'model_receipt_mismatch'),
            (lambda r: r['arguments']['case'].update(question='Changed question'), 'model_receipt_mismatch'),
        ]:
            with self.subTest(expected=expected):
                changed = deepcopy(request); mutate(changed)
                self.assertEqual(self.post(changed)['error_code'], expected)
        second = self.post(request)
        self.add_receipt(request, second)
        reversed_receipts = deepcopy(request)
        reversed_receipts['model_receipts'].reverse()
        self.assertEqual(self.post(reversed_receipts)['error_code'], 'model_receipt_mismatch')
        request['model_receipts'].append(deepcopy(request['model_receipts'][-1]))
        self.assertEqual(self.post(request)['error_code'], 'unexpected_model_receipts')

    def test_unknown_failed_and_invalid_model_outputs_never_resume_as_success(self):
        request = self.request()
        first = self.post(request)
        self.add_receipt(request, first)
        for status in ('unknown', 'failed', 'not_sent'):
            bad = deepcopy(request); bad['model_receipts'][0]['status'] = status
            response = self.post(bad)
            self.assertEqual(response['status'], 'needs_input')
            self.assertEqual(response['error_code'], 'input_invalid')
        request['model_receipts'][0]['output'] = {'invalid': 'completed transport is not valid model content'}
        failed = self.post(request)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['error_code'], 'method_failed')
        self.assertNotIn('model_call', failed)

    def test_direction_qualification_identity_and_provenance_before_callback(self):
        for mode in ('business', 'public_health_metrics'):
            request = self.request();request['arguments'] = {'mode': mode, 'task': 'metrics'}
            self.assertEqual(self.post(request)['status'], 'not_applicable')
        request = self.request();request['arguments']['task'] = 'independence'
        self.assertEqual(self.post(request)['error_code'], 'operation_task_mismatch')
        request = self.request();request['arguments']['case']['materials'][0]['version'] = ''
        self.assertEqual(self.post(request)['error_code'], 'material_provenance_missing')
        request = self.request();request['arguments']['case']['materials'][0]['pmid'] = '111'
        self.assertEqual(self.post(request)['error_code'], 'material_identity_conflict')

    def test_cohort_operation_real_http_unknown_does_not_become_independence(self):
        request = self.request()
        material = deepcopy(self.snapshot['case']['materials'][0]);material['id'] = 'a'
        other = deepcopy(material);other['id'] = 'b'
        request.update(operation='assess_independence')
        request['arguments'] = {'mode': 'biomedical_research', 'task': 'independence',
            'materials': [material, other], 'pairs': [{'left_material_id': 'a', 'right_material_id': 'b', 'unit': 'patient_cohort'}]}
        first = self.post(request)
        self.assertEqual(first['status'], 'model_required')
        self.assertEqual(first['model_call']['stage'], 'cohort_review')
        output = {'pairs': [{'left_material_id': 'a', 'right_material_id': 'b', 'unit': 'patient_cohort',
            'relation': 'unknown', 'basis': 'publication_identity', 'passage_ids': [],
            'rationale': 'Protocol fixture: same mouse-neuron paper does not establish patient membership.',
            'uncertainty': ['No patient cohort descriptions supplied.'], 'information_needs': ['Read actual cohort descriptions if applicable.']}]}
        self.add_receipt(request, first, output)
        result = self.post(request)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['result']['publication_groups']), 1)
        self.assertEqual(result['result']['pairs'][0]['relation'], 'unknown')

    def test_authenticated_health_capabilities_and_transport_errors(self):
        self.assertEqual(self.http('GET', '/health', authorized=False)[0], 401)
        status, health = self.http('GET', '/health')
        self.assertEqual((status, health['status']), (200, 'ok'))
        status, caps = self.http('GET', '/v1/capabilities')
        self.assertEqual(status, 200)
        self.assertEqual(caps['provider_version'], health['provider_version'])
        self.assertEqual(caps['operations'], ['evaluate', 'assess_independence'])
        self.assertEqual(self.http('POST', '/v1/execute', ['not an object'])[1]['error_code'], 'input_invalid')
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        connection.request('POST', '/v1/execute', 'bad json', {'Authorization': 'Bearer ' + self.token})
        response = connection.getresponse()
        self.assertEqual(response.status, 400);response.read();connection.close()


if __name__ == '__main__':
    unittest.main()
