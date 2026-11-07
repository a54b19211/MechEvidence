"""Who is evaluating: everything that decides the answer for a given input, as one id."""
from __future__ import annotations

import os

from . import llm
from .assess import BATCH_SIZE, ORDERING_RULE, RULES_VERSION
from .contract import evaluator_id_of
from .models_v2 import CONTRACT_V2, sha256_text
from .prompts import DECOMPOSE, REVIEW


def current_evaluator() -> dict:
    """Read from the live configuration on every call: chat_json reads the model settings per call too, so the
    identity a request is checked against is the one its model calls will use.

    `endpoint_id` is the operator's name for the model endpoint (LLM_ENDPOINT_ID; never a URL or a key).
    `code_version` (MECHEVIDENCE_CODE_VERSION, set at build/deploy) is audit only and not part of evaluator_id."""
    fields = {
        "service": "mechevidence",
        "contract_version": CONTRACT_V2,
        "rules_version": RULES_VERSION,
        "ordering_rule": ORDERING_RULE,
        "batch_size": BATCH_SIZE,
        "prompts": {"decompose_sha256": sha256_text(DECOMPOSE), "review_sha256": sha256_text(REVIEW)},
        "model_config": {
            "provider": os.environ.get("LLM_PROVIDER", llm.DEFAULT_PROVIDER),
            "model": os.environ.get("LLM_MODEL", llm.DEFAULT_MODEL),
            "thinking": llm.STRUCTURED_THINKING,
            "temperature": llm.STRUCTURED_TEMPERATURE,
            "max_tokens": llm.STRUCTURED_MAX_TOKENS,
            "endpoint_id": os.environ.get("LLM_ENDPOINT_ID", "unnamed"),
        },
    }
    return {**fields, "code_version": os.environ.get("MECHEVIDENCE_CODE_VERSION", "unrecorded"),
            "evaluator_id": evaluator_id_of(fields)}
