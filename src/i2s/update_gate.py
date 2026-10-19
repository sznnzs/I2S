from __future__ import annotations

import copy
import re

from . import update_eligibility as old
from .semantic_checks import task_roles

POLICY = "task_scoped_relations"


def source_witness(relation, transitions, instruction):
    target = task_roles(instruction).get("target")
    rec = relation["receptacle"]
    if not target:
        return None
    locations = {}
    witnesses = {}
    current = relation["transition"]
    step = current.get("step_index", -1)
    for t in sorted(transitions, key=lambda t: t.get("step_index", -1)):
        if t.get("step_index", -1) >= step:
            continue
        if t.get("action_valid") and not t.get("execution_error"):
            act = t.get("action", "").lower()
            moved = re.fullmatch(
                r"(?:take|move) (" + re.escape(target) + r" \d+) (?:from|to) .+", act
            )
            if moved:
                locations.pop(moved[1], None)
                witnesses.pop(moved[1], None)
        text = t.get("observation_after", "").lower()
        for found, contents in re.findall(r"(?:on|in) the ([a-z]+ \d+), you see ([^.]+)", text):
            for obj in [o for o, r in locations.items() if r == found]:
                locations.pop(obj)
                witnesses.pop(obj, None)
            for obj in re.findall(r"\b" + re.escape(target) + r" \d+\b", contents):
                locations[obj] = found
                witnesses[obj] = t["transition_id"]
    text = current.get("observation_before", "").lower()
    for found, contents in re.findall(r"(?:on|in) the ([a-z]+ \d+), you see ([^.]+)", text):
        for obj in [o for o, r in locations.items() if r == found]:
            locations.pop(obj)
            witnesses.pop(obj, None)
        for obj in re.findall(r"\b" + re.escape(target) + r" \d+\b", contents):
            locations[obj] = found
            witnesses[obj] = current["transition_id"] + ":before"
    eligible = sorted(o for o, r in locations.items() if r == rec)
    return (
        {"target": eligible[0], "source": rec, "transition_id": witnesses[eligible[0]]}
        if eligible
        else None
    )


def _rename(value, from_role, to_role):
    if isinstance(value, dict):
        return {k: _rename(v, from_role, to_role) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v, from_role, to_role) for v in value]
    if isinstance(value, str):
        return re.sub(r"\b" + from_role + r"\b", to_role, value)
    return value


def classify_update(
    anchors, skills, transitions=(), *, task_instruction="", initial_observation=""
):
    decision = old.classify_update(
        anchors,
        skills,
        transitions,
        task_instruction=task_instruction,
        initial_observation=initial_observation,
    )
    by = {a["anchor_id"]: a for a in anchors}
    active = [s for s in skills if s.get("status") == "active"]

    roles = old._task_roles(task_instruction)
    for check in decision["anchor_checks"]:
        if check["covered"]:
            continue
        relation, _ = old._relation(by[check["anchor_id"]], roles)
        if not relation or not roles:
            continue
        witness = source_witness(relation, transitions, task_instruction)
        if not witness:
            continue
        state = old._prefix_state(relation["transition"], transitions, initial_observation)
        for record in sorted(active, key=lambda r: r["creation_order"]):
            fields = {k: record[k] for k in old.CONTENT_FIELDS}
            if re.search(r"\bcandidate\b", str(fields)):
                continue
            normalized = {**copy.deepcopy(record), **_rename(fields, "source", "candidate")}
            proof, _ = old._cover(normalized, relation, state, roles)
            if proof:
                proof = _rename(proof, "candidate", "source")
                proof["source_identity_witness"] = witness
                known = set(record.get("supporting_anchor_ids", [])) | {
                    x["anchor_id"] for x in record.get("evidence_accumulation", [])
                }
                check.update(
                    covered=True,
                    reason="exact_relation_plus_prior_native_source_identity",
                    certificate=proof,
                    already_accumulated=check["anchor_id"] in known,
                )
                break
    covered = sum(c["covered"] for c in decision["anchor_checks"])
    eligible = len(decision["anchor_checks"])
    known = {h.get("rule") for r in active for h in r.get("residual_history", [])}
    new = [
        c["anchor_id"]
        for c in decision["anchor_checks"]
        if not c["covered"] and c.get("relation_signature", {}).get("rule") not in known
    ]
    refinement = [
        c["anchor_id"]
        for c in decision["anchor_checks"]
        if not c["covered"] and c["anchor_id"] not in new
    ]
    category = (
        "NO_ELIGIBLE_ANCHOR"
        if not eligible
        else (
            "NO_CHANGE_NEEDED"
            if all(c.get("already_accumulated") for c in decision["anchor_checks"])
            else "REDUNDANT_EVIDENCE"
        )
        if covered == eligible
        else "NEW_PROCEDURE"
        if new or not active
        else "POSSIBLE_REFINEMENT"
    )
    decision.update(
        policy=POLICY,
        classification=category,
        call_required=category in {"NEW_PROCEDURE", "POSSIBLE_REFINEMENT"},
        covered_anchor_count=covered,
        covered_anchor_fraction=covered / eligible if eligible else None,
        eligible_anchor_ids=[c["anchor_id"] for c in decision["anchor_checks"]],
        covered_anchor_ids=[c["anchor_id"] for c in decision["anchor_checks"] if c["covered"]],
        new_relation_candidates=new,
        possible_refinement_candidates=refinement,
        generation_calls_avoided=int(bool(eligible) and covered == eligible),
        tokens_avoided=None,
        token_avoidance_note="Counterfactual completion length is unobserved; no invented token saving. Duplicate local-evidence input removal is separately reported as a deterministic token proxy.",
    )
    return decision
