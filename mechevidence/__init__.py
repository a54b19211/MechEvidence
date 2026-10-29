"""Evidence evaluation functions usable independently of the CLI or model client."""

from .assess import (
    EvaluationError, IdentityInputError, aggregate, direct_baseline, evaluate, prepare_materials,
    prepare_passages, source_identity,
)
from .models import Case, Claim, Join, Material, Scope

__all__ = ["Case", "Claim", "Join", "Material", "Scope", "EvaluationError", "IdentityInputError", "evaluate",
           "direct_baseline", "aggregate", "prepare_materials", "prepare_passages", "source_identity"]
