"""受控的模型替身和最小用例；不是医学证据，也不是真实模型结果。"""
from copy import deepcopy

TEXT = "A changed M in mouse S cells. Blocking M prevented the E response to A."


def material(id="E1", text=TEXT, **fields):
    return {"id": id, "title": "Controlled fixture", "text": text, "source_kind": "primary",
            "document_type": "research_article", "text_level": "abstract", **fields}


def case(**fields):
    return {"question": "Does A cause E through M in mouse system S?", "mechanism": "A changes M; M is necessary for E.",
            "scope": {"species": "mouse", "system": "S"}, "materials": [material()], **fields}


DECOMPOSITION = {"claims": [{"text": "A changes M", "kind": "causal"}, {"text": "M is necessary for E", "kind": "causal"}],
                 "joins": [{"text": "A produces E through M", "from_claim": 1, "to_claim": 2}]}


def obs(targets, passage, quote, relation="supports", **fields):
    return {"targets": targets, "passage": passage, "quote": quote, "relation": relation, "basis": "intervention",
            "causal_test": True, "context": {"species": "mouse", "system": "S", "disease": None},
            "scope_match": {"species": "matched", "system": "matched", "disease": "not_required"}, **fields}


def good_observations():
    return [obs(["C1"], "E1.1", "A changed M"), obs(["C2", "J1"], "E1.2", "Blocking M prevented the E response")]


class Stub:
    """chat_json 的替身：decompose 返回固定拆解；review 按调用次序返回观察（None = 本次调用失败）。"""

    def __init__(self, reviews=None, decomposition=DECOMPOSITION):
        self.reviews = list(reviews if reviews is not None else [good_observations()])
        self.decomposition, self.calls = decomposition, []

    def __call__(self, stage, system, payload, model_cls, **_):
        self.calls.append((stage, deepcopy(payload)))
        if stage == "decompose":
            data = self.decomposition
        else:
            observations = self.reviews.pop(0) if self.reviews else []
            data = None if observations is None else {"observations": observations}
        record = {"stage": stage, "model": "stub", "finish_reason": "stop", "usage": {}, "seconds": 0, "requests": 1,
                  "error": None if data is not None else "output_invalid"}
        return (model_cls.model_validate(deepcopy(data)) if data is not None else None), record
