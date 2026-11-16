"""Which records are one publication: the literature identity MechEvidence derives from a material's
identifier fields (PMID / DOI / source group / record id / URL), because it decides how its evaluator groups
sources and so is part of the reuse key. materials.py uses it; examples/identity_contract.json publishes
its cases (tests/test_materials.py)."""

import re
from urllib.parse import unquote, urlsplit


class IdentityError(ValueError):
    def __init__(self, code: str, kind: str, fields: dict):
        self.code, self.kind, self.fields = code, kind, fields
        super().__init__(f"{code}: {kind} {fields}")


_PMID = r"[1-9][0-9]*"
_DOI = r'10\.[0-9]{4,9}/[^\s\x00-\x1f\x7f<>"\\]+'
_ROUTES = {
    "pubmed.ncbi.nlm.nih.gov": rf"/({_PMID})/?",
    "ncbi.nlm.nih.gov": rf"/pubmed/({_PMID})/?",
    "www.ncbi.nlm.nih.gov": rf"/pubmed/({_PMID})/?",
    "europepmc.org": rf"/article/MED/({_PMID})/?",
    "www.europepmc.org": rf"/article/MED/({_PMID})/?",
}


def _identifier_value(value, kind: str, field: str) -> str:
    text = str(value).strip()
    if isinstance(value, bool) or not re.fullmatch(_PMID if kind == "pmid" else _DOI, text):
        raise IdentityError("material_identifier_invalid", kind, {field: value})
    # a DOI's trailing ')' or '.' belongs to the identifier
    return text if kind == "pmid" else text.casefold()


def _url_identifier(value: str) -> tuple[str, str] | None:
    if (
        "\\" in value
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
        or re.search(r"%(?![0-9a-fA-F]{2})", value)
    ):
        return None
    try:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or url.username is not None
            or url.password is not None
            or url.port not in {None, 443 if url.scheme == "https" else 80}
        ):
            return None
        if url.hostname in _ROUTES:
            match = re.fullmatch(_ROUTES[url.hostname], url.path)
            return ("pmid", match.group(1)) if match else None
        if url.hostname in {"doi.org", "dx.doi.org"}:
            doi = unquote(url.path[1:], errors="strict") if url.path.startswith("/") else ""
            return ("doi", doi) if re.fullmatch(_DOI, doi) else None
    except (ValueError, UnicodeError):
        return None
    return None


def _tagged_identifier(value: str) -> tuple[str, str] | None:
    # only a colon or slash declares an identifier; an ordinary group label stays as it is
    for pattern, kind in [
        (r"(?:pmid|pubmed)\s*[:/]\s*(.*)", "pmid"),
        (rf"(?:pmid|pubmed)\s*({_PMID})", "pmid"),
        (r"doi\s*[:/]\s*(.*)", "doi"),
        (rf"doi\s+({_DOI})", "doi"),
    ]:
        match = re.fullmatch(pattern, value, re.I)
        if match:
            return kind, match.group(1)
    if re.fullmatch(_DOI, value):
        return "doi", value
    return _url_identifier(value)


def source_identity(identifiers: dict | None) -> str | None:
    """The group key of the publication a material's identifiers name (pmid: / doi: / group: / record:), or
    None when they name none. A publication identity, not independence of cohorts or experiments."""
    ids = identifiers or {}
    candidates = _candidates(ids)
    # both kinds are checked before one wins, so a PMID cannot hide a contradicting DOI
    for kind, fields in candidates.items():
        if len(set(fields.values())) > 1:
            raise IdentityError("material_identity_conflict", kind, fields)
    for kind in ("pmid", "doi"):
        if candidates[kind]:
            return kind + ":" + next(iter(candidates[kind].values()))
    source_id, record_id = ids.get("source_id") or "", ids.get("record_id") or ""
    group = ids.get("source_group")
    if group:
        return "group:" + " ".join(group.casefold().split())
    if record_id:
        return "record:" + " ".join((source_id + ":" + record_id).casefold().split())
    return None


def _candidates(ids: dict) -> dict[str, dict[str, str]]:
    """The PMID and the DOI each field of the material names, by field."""
    candidates: dict[str, dict[str, str]] = {"pmid": {}, "doi": {}}
    for kind in candidates:
        if ids.get(kind) is not None:
            candidates[kind][kind] = _identifier_value(ids[kind], kind, kind)
    source = (ids.get("source_id") or "").strip().casefold().replace("-", "_")
    for field in ("source_group", "record_id", "url"):
        value = (ids.get(field) or "").strip()
        if not value:
            continue
        parsed = _url_identifier(value) if field == "url" else _tagged_identifier(value)
        if parsed:
            candidates[parsed[0]][field] = _identifier_value(parsed[1], parsed[0], field)
        elif field == "record_id" and source == "pubmed":
            candidates["pmid"][field] = _identifier_value(value, "pmid", field)
        elif field == "record_id" and source in {"europe_pmc", "europepmc"} and value[:4].upper() == "MED:":
            candidates["pmid"][field] = _identifier_value(value[4:], "pmid", field)
    return candidates
