"""HTTP boundary for pure professional methods; the host owns every model call."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
from typing import Literal

from pydantic import Field, ValidationError

from .assess import EvaluationError, IdentityInputError, evaluate
from .cohorts import CohortArguments, assess_independence
from .models import Case, Identifier, Record, Text

SCHEMA_VERSION = 'bioradar.professional.v1'
PROVIDER = 'mech_evidence'
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_MODEL_CALLS = 8


def _digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _version() -> str:
    root = Path(__file__).parent
    names = ('models.py', 'assess.py', 'cohorts.py', 'service.py')
    return hashlib.sha256(b''.join(name.encode() + b'\0' + (root / name).read_bytes() for name in names)).hexdigest()


# Code and prompts stay fixed for this process. Restart to adopt a new version.
PROVIDER_VERSION = _version()


class ComputationRef(Record):
    run_id: Identifier
    result_id: Identifier
    version: Text
    finding_ids: list[Identifier] = Field(default_factory=list)


class Context(Record):
    mode: Literal['biomedical_research', 'business', 'public_health_metrics']
    task: Literal['mechanism', 'independence', 'metrics']
    computation_refs: list[ComputationRef] = Field(default_factory=list)


class EvaluateArguments(Context):
    task: Literal['mechanism']
    case: Case


class IndependenceArguments(Context, CohortArguments):
    task: Literal['independence']


class ModelReceipt(Record):
    call_id: Identifier
    request_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    output: dict


class ExecuteRequest(Record):
    schema_version: Literal['bioradar.professional.v1']
    operation_id: Identifier
    operation: Literal['evaluate', 'assess_independence']
    arguments: dict
    provider_version: str | None = None
    model_receipts: list[ModelReceipt] = Field(default_factory=list, max_length=MAX_MODEL_CALLS)


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ModelRequired(Exception):
    def __init__(self, call: dict):
        self.call = call
        super().__init__('Host model output required')


class Replay:
    """A request-local cursor over completed host receipts, never persistent state."""
    def __init__(self, request: ExecuteRequest):
        self.request, self.position = request, 0
        self.input_hash = _digest({'operation_id': request.operation_id, 'operation': request.operation,
                                   'arguments': request.arguments, 'provider_version': PROVIDER_VERSION})

    def call_json(self, stage, instruction, payload, schema):
        if self.position >= MAX_MODEL_CALLS:
            raise ProtocolError('model_step_limit', 'Professional callback count exceeds the service limit')
        body = {'stage': stage, 'instruction': instruction, 'payload': payload, 'schema': schema}
        request_hash = _digest(body)
        call_id = 'mech:' + _digest({'input_hash': self.input_hash, 'sequence': self.position,
                                    'request_hash': request_hash})
        if self.position == len(self.request.model_receipts):
            raise ModelRequired({'call_id': call_id, 'request_hash': request_hash, **body})
        receipt = self.request.model_receipts[self.position]
        if receipt.call_id != call_id or receipt.request_hash != request_hash:
            raise ProtocolError('model_receipt_mismatch', 'Receipt does not match this input, version and sequential request')
        self.position += 1
        return deepcopy(receipt.output)

    def finish(self):
        if self.position != len(self.request.model_receipts):
            raise ProtocolError('unexpected_model_receipts', 'Method finished with unconsumed model receipts')


def capabilities() -> dict:
    return {'schema_version': SCHEMA_VERSION, 'provider': PROVIDER, 'provider_version': PROVIDER_VERSION,
            'operations': ['evaluate', 'assess_independence'],
            'model_stages': {'evaluate': ['decomposition', 'local_review', 'mechanism_review'],
                             'assess_independence': ['cohort_review']},
            'modes': ['biomedical_research', 'business', 'public_health_metrics'],
            'task_by_operation': {'evaluate': 'mechanism', 'assess_independence': 'independence'},
            'limits': {'request_bytes': MAX_BODY_BYTES, 'model_calls': MAX_MODEL_CALLS},
            'execution': 'stateless_replay_host_owned_models',
            'request_schema': ExecuteRequest.model_json_schema(),
            'context_schema': Context.model_json_schema(),
            'argument_schemas': {'evaluate': EvaluateArguments.model_json_schema(),
                                 'assess_independence': IndependenceArguments.model_json_schema()}}


def _context_and_arguments(request):
    arguments = deepcopy(request.arguments)
    context = Context.model_validate({key: arguments.pop(key) for key in
        ('mode', 'task', 'computation_refs') if key in arguments})
    return context, arguments


def _require_provenance(materials):
    missing = [f'{item.id}.{field}' for item in materials for field in ('source_id', 'record_id', 'version')
               if not getattr(item, field).strip()]
    if missing:
        raise ProtocolError('material_provenance_missing', 'Missing fixed source metadata: ' + ', '.join(missing))


def execute(value: dict) -> dict:
    """Run until one missing model output, or finish using the supplied receipt prefix."""
    response = {'schema_version': SCHEMA_VERSION, 'provider': PROVIDER,
                'provider_version': PROVIDER_VERSION, 'operation_id': None}
    try:
        request = ExecuteRequest.model_validate(value)
        response['operation_id'] = request.operation_id
        if request.provider_version is not None and request.provider_version != PROVIDER_VERSION:
            raise ProtocolError('provider_version_mismatch', 'Resume with the original service version or start a new operation')
        if request.model_receipts and request.provider_version is None:
            raise ProtocolError('provider_version_required', 'Continuations must pin the advertised provider version')
        replay = Replay(request)
        context, arguments = _context_and_arguments(request)
        response['context'] = context.model_dump()
        if context.mode != 'biomedical_research' or context.task == 'metrics':
            replay.finish()
            return {**response, 'status': 'not_applicable', 'error_code': 'professional_task_not_applicable',
                    'reason': 'This service evaluates biomedical mechanisms and study/cohort relationships, not business or public-health metrics'}
        expected_task = {'evaluate': 'mechanism', 'assess_independence': 'independence'}[request.operation]
        if context.task != expected_task:
            raise ProtocolError('operation_task_mismatch', f'{request.operation} requires task={expected_task}')
        if request.operation == 'evaluate':
            if set(arguments) != {'case'}:
                raise ProtocolError('arguments_invalid', 'evaluate requires exactly case plus the common context fields')
            case = Case.model_validate(arguments['case'])
            _require_provenance(case.materials)
            result = evaluate(case, replay.call_json)
        else:
            cohort_case = CohortArguments.model_validate(arguments)
            _require_provenance(cohort_case.materials)
            result = assess_independence(cohort_case, replay.call_json)
        replay.finish()
        return {**response, 'status': 'completed', 'result': result}
    except Exception as error:
        # Existing pure methods wrap callback exceptions. Recover only our exact types.
        cause = error
        while cause is not None:
            if isinstance(cause, ModelRequired):
                return {**response, 'status': 'model_required', 'model_call': cause.call}
            if isinstance(cause, ProtocolError):
                return {**response, 'status': 'needs_input', 'error_code': cause.code, 'message': str(cause)}
            cause = cause.__cause__
        if isinstance(error, IdentityInputError):
            return {**response, **error.as_dict()}
        if isinstance(error, ValidationError):
            return {**response, 'status': 'needs_input', 'error_code': 'input_invalid',
                    'errors': error.errors(include_url=False, include_input=False, include_context=False)}
        if isinstance(error, (ValueError, TypeError)):
            return {**response, 'status': 'needs_input', 'error_code': 'input_invalid', 'message': str(error)[:800]}
        if isinstance(error, EvaluationError):
            return {**response, 'status': 'failed', 'error_code': 'method_failed', 'message': str(error)[:800]}
        return {**response, 'status': 'failed', 'error_code': 'internal_error', 'message': type(error).__name__}


class ProfessionalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, token: str):
        if not token.strip():
            raise ValueError('A private service Bearer token is required')
        self.token = token
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, _format, *args):
        pass  # No request bodies, credentials or model outputs in the HTTP log.

    def reply(self, status, payload):
        payload = {'schema_version': SCHEMA_VERSION, 'provider': PROVIDER,
                   'provider_version': PROVIDER_VERSION, **payload}
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def authorized(self):
        expected = ('Bearer ' + self.server.token).encode()
        received = self.headers.get('Authorization', '').encode()
        if secrets.compare_digest(received, expected):
            return True
        self.reply(401, {'status': 'needs_input', 'error_code': 'unauthorized',
                         'provider': PROVIDER, 'provider_version': PROVIDER_VERSION, 'schema_version': SCHEMA_VERSION})
        return False

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == '/health':
            self.reply(200, {'schema_version': SCHEMA_VERSION, 'provider': PROVIDER,
                             'provider_version': PROVIDER_VERSION, 'status': 'ok'})
        elif self.path == '/v1/capabilities':
            self.reply(200, capabilities())
        else:
            self.reply(404, {'status': 'needs_input', 'error_code': 'not_found'})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path != '/v1/execute':
            self.reply(404, {'status': 'needs_input', 'error_code': 'not_found'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BODY_BYTES:
                self.reply(413, {'status': 'needs_input', 'error_code': 'request_size_invalid'})
                return
            def invalid_constant(value):
                raise ValueError('Invalid JSON number: ' + value)
            value = json.loads(self.rfile.read(size), parse_constant=invalid_constant)
        except (ValueError, UnicodeError, TimeoutError):
            self.reply(400, {'status': 'needs_input', 'error_code': 'invalid_json'})
            return
        self.reply(200, execute(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18561)
    args = parser.parse_args()
    token = os.environ.get('MECHEVIDENCE_SERVICE_TOKEN', '')
    if not token.strip():
        parser.error('Set MECHEVIDENCE_SERVICE_TOKEN in the private service environment')
    server = ProfessionalServer((args.host, args.port), token)
    print(f'MechEvidence {PROVIDER_VERSION} listening on {args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
