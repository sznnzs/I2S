from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

PROTOCOL = "source_reference_repair"
REPAIRABLE = frozenset(
    {
        "unsupported_localized_quote",
        "proposal_evidence_anchor_mismatch",
        "missing_or_ineligible_anchor_reference",
        "uncited_anchor_evidence",
    }
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def repair_reference_fields(proposal, anchors):
    source = {}
    for anchor in anchors:
        aid = anchor.get("anchor_id")
        if not isinstance(aid, str) or aid in source:
            raise ValueError("admission_repair_ambiguous_source_anchor")
        source[aid] = anchor
    repaired = copy.deepcopy(proposal)
    evidence = repaired.get("evidence_interpretation")
    if (
        not isinstance(evidence, dict)
        or not isinstance(evidence.get("localized_facts"), list)
        or not evidence["localized_facts"]
    ):
        raise ValueError("admission_repair_missing_localized_facts")
    derived, edits = [], []
    for index, fact in enumerate(evidence["localized_facts"]):
        if not isinstance(fact, dict):
            raise ValueError("admission_repair_invalid_fact")
        aid, tid = fact.get("anchor_id"), fact.get("transition_id")
        anchor = source.get(aid) if isinstance(aid, str) else None
        if anchor is None or anchor.get("eligibility") is not True:
            raise ValueError("admission_repair_unresolvable_or_ineligible_anchor")
        matches = [t for t in anchor.get("localized_evidence", []) if t.get("transition_id") == tid]
        if not isinstance(tid, str) or len(matches) != 1:
            raise ValueError("admission_repair_unresolvable_or_ambiguous_transition")
        if aid not in derived:
            derived.append(aid)
        for field in ("observation_before", "observation_after"):
            original = matches[0].get(field)
            if not isinstance(original, str) or len(original.strip()) < 8:
                raise ValueError("admission_repair_source_quote_not_eligible")
            quote_field = field + "_quote"
            quote = fact.get(quote_field)

            if isinstance(quote, str) and len(quote.strip()) >= 8 and quote in original:
                continue
            edits.append(
                {
                    "path": f"evidence_interpretation.localized_facts[{index}].{quote_field}",
                    "before": quote,
                    "after": original,
                    "anchor_id": aid,
                    "transition_id": tid,
                }
            )
            fact[quote_field] = original
    for name, section in (("evidence_interpretation", evidence), ("proposal", repaired)):
        ids = section.get("supporting_anchor_ids")

        if (
            isinstance(ids, list)
            and all(isinstance(x, str) for x in ids)
            and set(ids) == set(derived)
        ):
            continue
        edits.append(
            {"path": name + ".supporting_anchor_ids", "before": ids, "after": list(derived)}
        )
        section["supporting_anchor_ids"] = list(derived)
    return repaired, edits


def recover_proposals(
    bank, proposals, anchors, transitions, task_id, episode_index, task_instruction
):
    if not proposals:
        return proposals, []
    if len(proposals) != 1:
        return proposals, [
            {
                "event": "induction_admission_repair",
                "protocol": PROTOCOL,
                "status": "unsupported_multi_proposal",
                "raw_output_preserved": True,
            }
        ]
    original = copy.deepcopy(proposals[0])

    def evaluate(proposal):
        probe = copy.deepcopy(bank)
        return probe.apply(
            [proposal], anchors, transitions, task_id, episode_index, task_instruction
        )

    original_events = evaluate(original)
    rejections = [e for e in original_events if e.get("event") == "proposal_rejected"]
    if len(rejections) != 1 or rejections[0].get("reason") not in REPAIRABLE:
        return proposals, []
    annotation = {
        "event": "induction_admission_repair",
        "protocol": PROTOCOL,
        "original_rejection": rejections[0]["reason"],
        "original_proposal": original,
        "original_proposal_sha256": digest(original),
        "raw_output_preserved": True,
        "semantic_fields_changed": False,
        "semantic_validator_changed": False,
    }
    try:
        repaired, edits = repair_reference_fields(original, anchors)
    except (ValueError, TypeError, KeyError) as exc:
        return proposals, [
            {**annotation, "status": "unresolvable_source_reference", "reason": str(exc)}
        ]
    if not edits:
        return proposals, [{**annotation, "status": "no_reference_edit_available"}]
    repaired_events = evaluate(repaired)
    remaining = [e.get("reason") for e in repaired_events if e.get("event") == "proposal_rejected"]
    annotation.update(
        edits=edits,
        repaired_proposal_sha256=digest(repaired),
        repaired_proposal=repaired,
        remaining_rejections=remaining,
    )
    if remaining:
        return proposals, [{**annotation, "status": "still_semantically_rejected"}]
    if not any(
        e.get("event") == "skill_lifecycle" and e.get("reason") != "no_supported_content_change"
        for e in repaired_events
    ):
        return proposals, [{**annotation, "status": "no_admitted_operation"}]
    return [repaired], [{**annotation, "status": "recovered_by_unchanged_validator"}]
