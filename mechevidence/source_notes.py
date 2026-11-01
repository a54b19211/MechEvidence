"""Check source locations and apply the existing reviews' information-state decisions."""
from copy import deepcopy


def _targets(local, mechanism, passages):
    targets, findings = {}, {}
    for prefix, review in [("local", local), ("mechanism", mechanism)]:
        if review is None:
            continue
        for observation in review["observations"]:
            base = f'{prefix}.observations.{observation["id"]}'
            cited = [passages[pid] for pid in observation["passage_ids"]]
            material_ids = {p["material_id"] for p in cited}
            groups = {p["source_group"] for p in cited if p["source_group"].startswith(("pmid:", "doi:"))}
            allowed = {pid for pid, passage in passages.items()
                       if passage["material_id"] in material_ids or passage["source_group"] in groups}
            findings[base + ".finding"] = observation
            targets[base + ".rationale"] = (observation, "rationale", allowed)
            for index in range(len(observation["limitations"])):
                targets[f"{base}.limitations.{index}"] = (observation["limitations"], index, allowed)
        for index, gap in enumerate(review["gaps"]):
            for field in ("reason", "missing_experiment"):
                targets[f"{prefix}.gaps.{index}.{field}"] = (gap, field, None)
        if prefix == "mechanism":
            targets["mechanism.rationale"] = (review, "rationale", None)
            for field in ("comparisons", "combinations"):
                for index, item in enumerate(review[field]):
                    targets[f"mechanism.{field}.{index}.rationale"] = (item, "rationale", None)
    return targets, findings


def review_source_notes(local, mechanism, passages, materials):
    """Only declared text spans change; relations, obligations and evidence IDs stay fixed.

    Exact anchors prove where text occurs, not whether it entails the model's statement.
    Both semantic stages still have to read the source and can make errors.
    """
    raw_local = local.model_dump()
    raw_mechanism = mechanism.model_dump() if mechanism is not None else None
    if local.source_notes is None or mechanism is not None and mechanism.source_notes is None:
        raise ValueError("source_notes are required; old receipts need explicit legacy_replay=True")
    reviewed_local, reviewed_mechanism = deepcopy(raw_local), deepcopy(raw_mechanism)
    targets, findings = _targets(reviewed_local, reviewed_mechanism, passages)
    notes = local.source_notes if mechanism is None else mechanism.source_notes
    required = {(note.target, note.original) for note in local.source_notes}
    actual = [(note.target, note.original) for note in notes]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate source-note target/span")
    if mechanism is not None and not required <= set(actual):
        raise ValueError("mechanism review must recheck every local source note against the original passages")
    finding_notes, edits, checked = {}, {}, []
    for note in notes:
        finding = findings.get(note.target)
        if finding is not None:
            if note.original:
                raise ValueError("a finding annotation has no text replacement span")
            allowed = set(finding["passage_ids"])
            finding_notes[finding["id"]] = {"status": note.status, "negative_kind": note.negative_kind}
        else:
            if note.target not in targets:
                raise ValueError(f"unknown or protected source-note target: {note.target}")
            container, key, allowed = targets[note.target]
            text = container[key]
            start = text.find(note.original)
            if not note.original or start < 0 or text.find(note.original, start + 1) >= 0:
                raise ValueError(f"source-note original span must occur exactly once: {note.target}")
        anchors = []
        for anchor in note.anchors:
            passage = passages.get(anchor.passage_id)
            if passage is None or allowed is not None and anchor.passage_id not in allowed:
                raise ValueError(f"source note cites an unavailable or unrelated passage: {note.target}")
            start, end = anchor.start - passage["start"], anchor.end - passage["start"]
            if not (0 <= start < end <= len(passage["text"])) or passage["text"][start:end] != anchor.quote:
                raise ValueError(f"source-note quote/position does not match actual text: {note.target}")
            material = materials[passage["material_id"]]
            anchors.append({**anchor.model_dump(), "material_id": material.id,
                            "source_id": material.source_id, "record_id": material.record_id,
                            "version": material.version, "url": material.url,
                            "source_group": passage["source_group"]})
        statement = (f"本次允许材料未报告或未充分说明：{note.statement}；相关事实仍未知。"
                     if note.status == "not_reported" else note.statement)
        item = {"target": note.target, "status": note.status, "statement": statement,
                "negative_kind": note.negative_kind, "anchors": anchors,
                "scope_passage_ids": sorted(allowed if allowed is not None else passages)}
        if finding is None:
            start = text.index(note.original)
            end = start + len(note.original)
            edits.setdefault(note.target, []).append((start, end, statement))
            item.update(original_start=start, original_end=end)
        checked.append(item)
    if set(finding_notes) != {item["id"] for item in findings.values()}:
        raise ValueError("each observation needs its own finding source note")
    for target, spans in edits.items():
        ordered = sorted(spans)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"overlapping source-note revisions: {target}")
        container, key, _ = targets[target]
        for start, end, replacement in reversed(ordered):
            container[key] = container[key][:start] + replacement + container[key][end:]
    for item in checked:
        if item["target"] in findings and (
                item["status"] == "not_reported" or item["negative_kind"] in {"not_disclosed", "method_not_performed"}):
            # A known information gap must not leave a contradictory factual assertion active.
            findings[item["target"]]["rationale"] = item["statement"]
    reviewed_local.pop("source_notes", None)
    if reviewed_mechanism is not None:
        reviewed_mechanism.pop("source_notes", None)
    return reviewed_local, reviewed_mechanism, {
        "status": "reviewed", "method": "source-reporting-v1", "notes": checked, "findings": finding_notes,
        "raw_reviews": {"local_review": raw_local, "mechanism_review": raw_mechanism},
    }


def generation_view(result):
    """Return reviewed result text for downstream generation; keep raw reviews in storage."""
    reporting = result.get("source_reporting", {})
    if reporting.get("status") != "reviewed" or reporting.get("method") != "source-reporting-v1":
        raise ValueError("generation requires a current reviewed source-reporting result")
    view = deepcopy(result)
    view["source_reporting"].pop("raw_reviews", None)
    return view
