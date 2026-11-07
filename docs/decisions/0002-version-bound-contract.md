# 0002 Version-bound evaluation contract (v2)

Status: accepted (2026-11), phase 1 of the service side

## Context

`POST /v1/evaluate` answered 200 for requests that carried fields it did not know (a `version_id`, a binding),
silently dropping them, and its answer said nothing about who evaluated: prompts, rules and model were implicit.
A result could neither be reproduced nor cached nor audited, and a caller that believed a dropped field had been
checked was wrong. The platform requires an evaluation to be tied to the proposition, the question's scope, the
fixed versions of the materials it read, and the evaluator that read them.

[0001](0001-synchronous-two-call-service.md) removed receipts, version hashes and capabilities from the old
protocol. That stays true of v1; it is not rewritten. v2 adds a receipt and hashes on purpose.

## Decision

- **Two contracts side by side.** `/v1/evaluate` keeps the evaluation algorithm and is labelled a compatibility
  result (`binding_state: unbound`, `reusable: false`, evaluator named); unknown fields there are 422
  `field_not_in_contract`. `/v2/evaluate` is the version-bound contract; every request model forbids unknown fields.
  Model outputs (decomposition, observations) stay lenient. Old records are never given an evaluator or binding
  after the fact.
- **Material binding.** A v2 material carries the binding the radar builds (`bioradar.material`): sha256 and length of
  the whole text, literal segments and part segments (source, record, version id, raw sha256, processor/parser/
  renderer version, the part's text sha256, the segment's range and sha256, locators). The service checks that text
  and binding agree and returns the binding hash; it cannot reach Fabric, so it says `self_consistent` and nothing
  more. That the text is that slice of that fixed version is the caller's re-read.
- **Evaluator identity.** `evaluator_id = sha256(canonical({service, contract_version, rules_version, ordering_rule,
  batch_size, prompts.{decompose,review}_sha256, model_config.{provider, model, thinking, temperature, max_tokens,
  endpoint_id}}))`. `code_version` is audit only. `RULES_VERSION` is bumped for anything that changes a conclusion
  from the same observations (scope alias table, comparison and grading rules, admissibility, source grouping).
- **Scope.** Each of species/system/disease is `stated` (with a value), `unknown` (not provided) or `not_required`
  (explicitly not limiting). A blank string means unknown, never not_required; an `unknown` dimension is never
  matched.
- **Reuse key, phase 1 (conservative).** The key covers every input the evaluator uses: proposition, normalised
  scope, the ordered list of materials with their request alias ids, binding hash, title, kinds, level, raw
  identifiers and the derived source group, the batch size and the evaluator id. Aliases are in the key because the
  model sees ids derived from them and the passage numbering depends on request order; the same texts in another
  order or under other aliases are another key. Order-independent reuse needs the evaluator to sort and number its
  input canonically; that is a later rules version, together with a full re-mapping of every cited reference.
- **Evaluator races.** `expected_evaluator_id` that is not the current evaluator gives 409 `evaluator_changed`, before
  anything is evaluated or any model is called.
- **Receipt checks belong to the caller.** `contract.adopt()` is the rule the radar applies: a receipt whose reuse
  key, material binding echo or evaluator id does not match its request is `failed/receipt_mismatch`, the raw
  receipt is kept as an audit row and the conclusion is not usable; a receipt with an incomplete audit is failed
  too; `complete=false` is never cacheable.
- **Permission before every model request (option A).** The caller sends an opaque `authorization_ref`; before
  each actual HTTP request (the format retry included) the service asks a pre-configured trusted endpoint
  (`model_attempt_authorization.v1`) with the ref, the evaluation key (the reuse key), the evaluator id and the attempt (unique
  id, global index, request body hash, model, endpoint id). It sends only on a 200 that says `valid`, names a decision,
  echoes that attempt, key and evaluator, and has not expired. After each counted request it hands the endpoint the
  outcome (`model_attempt_outcome.v1`: delivery, status, raw body and hash, truncation); if that is not recorded, nothing more
  is sent. Unreachable, refusing, bare `valid:true`, wrong echo, expired or unconfigured means nothing is sent. The endpoint
  is never taken from the request. v1 keeps not re-checking inside a request; that is a documented compatibility gap.
- **Raw answers.** `llm_calls[].outputs` has one entry per counted HTTP request: the raw HTTP answer (at most
  256 KiB) with the sha256 of the whole, written down before it is parsed. A request that was sent is never lost: an
  envelope that cannot be read (`choices` not an object) ends the call as `response_malformed` with its output kept, and
  the evaluation is a handleable 502. An answer that does not fit is not archived completely, so the evaluation fails
  closed (502 `output_oversize`, no verdict) with no format retry. A truncated answer's hash is never presented as the
  hash of a stored whole.
- **Transport.** The HTTP side of the call (provider body, send, delivery outcome, request count, envelope) is the shared
  `zk-llm-transport` wheel, pinned by hash; prompts, format retry and error codes stay here.
- **Receipt adoption is strict.** `contract.adopt()` also checks the receipt's structure, verdict domains, that every
  counted request has its archived answer with contiguous numbers, each archived text against its hash, and the
  consistency of `complete` with the material notes; any failure is `failed/receipt_invalid` with the raw receipt kept.

## Consequences

- The radar can cache and replay an evaluation, and refuse a receipt that is not for its request, using the fixtures
  in `examples/contract-v2/` (request, receipt, refusals, key cases, receipt outcomes, service errors).
- The key is deliberately narrower in what it reuses than the final design; the cost is cache misses, not wrong hits.
- The radar side (flag, migration, plan schema with three-state scope, reuse and reading of cached rows) is not part of
  this change.
