"""The request of POST /v2/evaluate (contract mechevidence.evaluate.v2).

Every model forbids unknown fields. Checks that need two fields at once (the binding against the text, a scope
dimension against its state) raise ValueError("<error_code>: <detail>"); contract.classify() turns that into the
documented error code.
"""
from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, StrictInt, model_validator

from .assess import _known
from .models import (
    FORBID_EXTRA,
    ClaimReq,
    DocumentType,
    JoinReq,
    Material,
    Scope,
    SourceKind,
    Text,
    TextLevel,
)

CONTRACT_V2 = "mechevidence.evaluate.v2"
HEX64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=512)]


class Strict(BaseModel):
    model_config = FORBID_EXTRA


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- scope: every dimension says what it is ----

class ScopeDim(Strict):
    state: Literal["stated", "unknown", "not_required"]
    value: str | None = None

    @model_validator(mode="after")
    def value_matches_state(self):
        if self.state == "stated":
            if self.value is None or not _known(self.value):
                raise ValueError("scope_state_invalid: state=stated needs a non-empty value that is not itself "
                                 "an 'unknown' word (say state=unknown instead)")
        elif self.value is not None:
            raise ValueError(f"scope_state_invalid: state={self.state} must not carry a value")
        return self


def _dimension(raw: Any) -> Any:
    """A bare string is accepted for a dimension: blank or an 'unknown' word means unknown (never not_required),
    anything else is stated. An object must say its state itself."""
    if isinstance(raw, str):
        return {"state": "stated", "value": raw} if _known(raw) else {"state": "unknown"}
    return raw


Dimension = Annotated[ScopeDim, BeforeValidator(_dimension)]


class ScopeV2(Strict):
    species: Dimension
    system: Dimension
    disease: Dimension

    def normal(self) -> dict:
        return {name: getattr(self, name).model_dump(exclude_none=True) for name in ("species", "system", "disease")}

    def to_scope(self) -> Scope:
        """The evaluator's own scope: unknown stays the word `unknown` (an unmatched dimension), not_required is
        None (not asked), so a dimension nobody stated is never read as one that does not matter."""
        pick = {"stated": lambda dim: dim.value, "unknown": lambda dim: "unknown", "not_required": lambda dim: None}
        return Scope(**{name: pick[getattr(self, name).state](getattr(self, name))
                        for name in ("species", "system", "disease")})


# ---- material binding (the same shape as bioradar.material's binding) ----

class Part(Strict):
    kind: Name
    source: Name
    record_id: Name
    version_id: HEX64
    raw_sha256: HEX64
    is_demo: bool = False
    text_sha256: HEX64
    # at least one names the program that produced the text: cleaning/attachment parsing, projection, or the
    # radar's rendering of a registry diff
    processor_version: Name | None = None
    parser_version: Name | None = None
    renderer: Name | None = None
    from_version_id: HEX64 | None = None

    @model_validator(mode="after")
    def names_its_processor(self):
        if not (self.processor_version or self.parser_version or self.renderer):
            raise ValueError("a part needs processor_version, parser_version or renderer")
        return self


class Segment(Strict):
    """Either {literal} (text the radar wrote itself) or {part, start, end, sha256[, locators, tables]}: the slice
    [start, end) of that part's text. Locators and tables are carried back verbatim, not interpreted here."""
    literal: str | None = None
    part: Part | None = None
    start: StrictInt | None = None
    end: StrictInt | None = None
    sha256: HEX64 | None = None
    locators: list[dict[str, Any]] | None = None
    tables: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def one_kind(self):
        extra = (self.part, self.start, self.end, self.sha256, self.locators, self.tables)
        if self.literal is not None:
            if self.literal == "" or any(item is not None for item in extra):
                raise ValueError("a literal segment is a non-empty {literal} and nothing else")
        elif self.part is None or self.start is None or self.end is None or self.sha256 is None:
            raise ValueError("a part segment needs part, start, end and sha256")
        elif not 0 <= self.start < self.end:
            raise ValueError(f"segment range [{self.start}, {self.end}) is not valid")
        return self


class Binding(Strict):
    sha256: HEX64
    chars: StrictInt = Field(ge=1)
    segments: list[Segment] = Field(min_length=1)


class Identifiers(Strict):
    """What groups several records of one publication into one source group (materials.source_identity)."""
    pmid: str | StrictInt | None = None
    doi: str | None = None
    url: str | None = None
    source_id: str = ""
    record_id: str = ""
    source_group: str = ""


class MaterialV2(Strict):
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]
    title: str = ""
    text: Text
    source_kind: SourceKind
    document_type: DocumentType | None = None
    text_level: TextLevel
    identifiers: Identifiers | None = None
    binding: Binding

    @model_validator(mode="after")
    def binding_describes_the_text(self):
        """The service cannot reach Fabric, so it checks what it can: the binding and the text agree. That the
        text is really that slice of that fixed version is the caller's re-read; the receipt says
        binding_checked=self_consistent and nothing more."""
        text, binding = self.text, self.binding
        if not any(segment.part is not None for segment in binding.segments):
            raise ValueError(f"material_binding_missing: {self.id}: the binding has no part segment, so the text "
                             "names no fixed version")
        cursor = 0
        for index, segment in enumerate(binding.segments):
            if segment.literal is not None:
                piece = text[cursor:cursor + len(segment.literal)]
                ok = piece == segment.literal
            else:
                piece = text[cursor:cursor + segment.end - segment.start]
                ok = len(piece) == segment.end - segment.start and sha256_text(piece) == segment.sha256
            if not ok:
                raise ValueError(f"material_binding_mismatch: {self.id}: segment {index} does not match the text "
                                 f"at offset {cursor}")
            cursor += len(piece)
        if cursor != len(text) or binding.chars != len(text):
            raise ValueError(f"material_binding_mismatch: {self.id}: the segments cover {cursor} characters and "
                             f"the binding says {binding.chars}, the text has {len(text)}")
        if sha256_text(text) != binding.sha256:
            raise ValueError(f"material_binding_mismatch: {self.id}: sha256 of the text is not the binding's")
        return self

    def to_material(self) -> Material:
        """The evaluator's own material (what v1 sends), identifiers included."""
        identifiers = self.identifiers.model_dump() if self.identifiers else {}
        return Material(id=self.id, title=self.title, text=self.text, source_kind=self.source_kind,
                        document_type=self.document_type, text_level=self.text_level, **identifiers)


class EvaluateRequestV2(Strict):
    contract_version: Literal["mechevidence.evaluate.v2"]
    question: Text
    mechanism: Text
    scope: ScopeV2
    materials: list[MaterialV2] = Field(default_factory=list)
    claims: list[ClaimReq] | None = None
    joins: list[JoinReq] = Field(default_factory=list)
    # asked for when the caller computed its reuse key against this evaluator; another one now -> 409, no evaluation
    expected_evaluator_id: HEX64 | None = None
    # names the analysis/materials/workspace the caller is entitled to send; checked with the trusted endpoint
    # before every actual model HTTP request (authorization.py)
    authorization_ref: Annotated[str, Field(pattern=r"^[\x21-\x7e]{1,512}$")]
    max_llm_calls: int = Field(20, ge=1, le=100)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.id for item in self.materials]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate material ids")
        return self
