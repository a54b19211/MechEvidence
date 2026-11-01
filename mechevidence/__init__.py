"""Evidence evaluation functions usable independently of the CLI or model client."""

from .assess import (
    EvaluationError, IdentityInputError, MaterialInputError, aggregate, direct_baseline, evaluate, prepare_materials,
    prepare_passages, source_identity,
)
from .source_notes import generation_view
from .models import Case, Claim, Join, Material, Scope

__all__ = ["Case", "Claim", "Join", "Material", "Scope", "EvaluationError", "IdentityInputError", "MaterialInputError", "evaluate",
           "direct_baseline", "aggregate", "prepare_materials", "prepare_passages", "source_identity", "generation_view"]
