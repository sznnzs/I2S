from __future__ import annotations

import copy
import hashlib
import json
import re

from .applicability_filter import _matches, observable_state, task_roles

VERSION = "trajectory_projection"
CONTENT_FIELDS = ("applicability", "procedure", "outcome_check", "fallback")
TRANSITION_FIELDS = (
    "transition_id",
    "step_index",
    "action",
    "observation_before",
    "observation_after",
    "action_valid",
    "execution_error",
    "explicit_failure",
    "source_role",
    "observable_effect",
)
ANCHOR_FIELDS = (
    "anchor_id",
    "small_proposed_action",
    "proposal_executed",
    "correction",
    "support_type",
    "eligibility",
    "transition_id",
    "supporting_transition_ids",
    "localized_evidence",
    "effect",
    "withheld_proposal_counterfactual",
)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _transition(row):
    return {k: copy.deepcopy(row[k]) for k in TRANSITION_FIELDS if k in row}


def prerequisite_evidence(anchor, transitions, instruction):
    first = min(anchor["localized_evidence"], key=lambda t: t["step_index"])
    state = observable_state(
        transitions,
        current_observation=first["observation_before"],
        current_step=first["step_index"],
    )
    roles = task_roles(instruction)
    conditions = []
    if state["holding"] is not None and roles.get("target"):
        holds = any(_matches(entity, roles["target"]) for entity in state["holding"])
        conditions.append("holding(target)" if holds else "not_holding(target)")
    if state["location"] is not None and roles.get("destination"):
        at = _matches(state["location"], roles["destination"])
        conditions.append("at(destination)" if at else "not_at(destination)")
    return {
        "anchor_id": anchor["anchor_id"],
        "observable_entry_conditions": conditions,
        "inventory": state["holding"],
        "inventory_witness": state["holding_evidence"],
        "location": state["location"],
        "location_witness": state["location_evidence"],
    }


def project_payload(payload):
    trajectory = payload["completed_trajectory"]
    transitions = sorted(payload["transitions"], key=lambda t: t["step_index"])
    anchors = payload["eligible_anchors"]
    positions = {t["transition_id"]: i for i, t in enumerate(transitions)}
    reasons = {}

    def keep(index, reason):
        if 0 <= index < len(transitions):
            reasons.setdefault(index, set()).add(reason)

    anchor_positions = []
    for anchor in anchors:
        for witness in anchor["localized_evidence"]:
            if witness["transition_id"] in positions:
                index = positions[witness["transition_id"]]
                anchor_positions.append(index)
                keep(index, "eligible_local_evidence")
                keep(index - 1, "immediate_prerequisite")

                for prior in range(index - 1, -1, -1):
                    if re.match(
                        r"(?:take |move |put |inventory$)",
                        str(transitions[prior].get("action", "")),
                    ):
                        keep(prior, "inventory_prerequisite")
                        break
    first_anchor = min(anchor_positions, default=len(transitions))
    seen = set()
    for index, row in enumerate(transitions):
        action = str(row.get("action", ""))
        after = str(row.get("observation_after", ""))
        key = (action, after)
        novel = key not in seen
        seen.add(key)
        valid = row.get("action_valid") is not False and not row.get("execution_error")
        changed = (
            valid
            and after != str(row.get("observation_before", ""))
            and not row.get("explicit_failure")
        )
        native_mutation = bool(
            re.match(r"(?:take |move |put |open |close |clean |heat |cool |use |toggle )", action)
        )
        if native_mutation and changed and novel:
            keep(
                index,
                "state_changing_prerequisite"
                if index < first_anchor
                else "downstream_state_change",
            )
        elif index > first_anchor and changed and novel:
            keep(index, "downstream_observable_transition")
    if transitions:
        keep(len(transitions) - 1, "terminal_native_feedback")

    selected = [_transition(transitions[i]) for i in sorted(reasons)]
    local = []
    for anchor in anchors:
        record = {k: copy.deepcopy(anchor[k]) for k in ANCHOR_FIELDS if k in anchor}

        record["localized_evidence"] = copy.deepcopy(anchor["localized_evidence"])
        local.append(record)
    skills = [
        {k: copy.deepcopy(s[k]) for k in ("skill_id", "version", *CONTENT_FIELDS) if k in s}
        for s in payload["current_skill_state"]
    ]
    result = {
        "serialization": VERSION,
        "reading_rule": (
            "Only eligible_anchors provide new operator/effect support. The projected "
            "trajectory supplies entry-state prerequisites and downstream context. Missing "
            "steps are omitted context, never evidence of failure or success. Unexecuted "
            "proposals are not failed actions. Terminal feedback cannot certify a local "
            "operation."
        ),
        "task_goal": trajectory["instruction"],
        "initial_observation": trajectory["initial_observation"],
        "anchor_prerequisite_evidence": [
            prerequisite_evidence(a, transitions, trajectory["instruction"]) for a in anchors
        ],
        "eligible_anchors": local,
        "projected_trajectory": selected,
        "terminal_task_feedback": {
            "native_observation": transitions[-1]["observation_after"]
            if transitions
            else trajectory["initial_observation"],
            "step_count": len(transitions),
        },
        "current_skill_state": skills,
    }
    if "task_feedback" in payload:
        result["terminal_task_feedback"].update(copy.deepcopy(payload["task_feedback"]))
    report = {
        "projection_version": VERSION,
        "source_payload_sha256": hashlib.sha256(dumps(payload).encode()).hexdigest(),
        "projected_payload_sha256": hashlib.sha256(dumps(result).encode()).hexdigest(),
        "eligible_anchor_ids": [a["anchor_id"] for a in anchors],
        "local_evidence_exact": all(
            a["localized_evidence"] == b["localized_evidence"]
            for a, b in zip(anchors, local, strict=True)
        ),
        "source_transition_count": len(transitions),
        "projected_transition_count": len(selected),
        "selected_transitions": [
            {"transition_id": transitions[i]["transition_id"], "reasons": sorted(reasons[i])}
            for i in sorted(reasons)
        ],
        "omitted_transition_ids": [
            t["transition_id"] for i, t in enumerate(transitions) if i not in reasons
        ],
        "source_skill_count": len(payload["current_skill_state"]),
        "projected_skill_count": len(skills),
        "skill_semantic_content_exact": all(
            all(a[k] == b[k] for k in CONTENT_FIELDS)
            for a, b in zip(payload["current_skill_state"], skills, strict=True)
        ),
        "update_skipped": False,
    }
    return result, report


def projected_system(system):
    return system.replace(
        "Read the complete trajectory for task context",
        "Read the projected trajectory for task context",
    )


def project_compatible_payload(payload):
    from .induction_compaction import compact_payload

    _, report = project_payload(payload)
    selected = {r["transition_id"] for r in report["selected_transitions"]}
    source = copy.deepcopy(payload)
    source["transitions"] = [t for t in source["transitions"] if t["transition_id"] in selected]
    steps = {t["step_index"] for t in source["transitions"]}
    source["completed_trajectory"]["steps"] = [
        {
            k: copy.deepcopy(v)
            for k, v in step.items()
            if k in {"step_index", "action", "environment_response"}
        }
        for step in payload["completed_trajectory"]["steps"]
        if step["step_index"] in steps
    ]
    allowed = {"skill_id", "version", "status", "representation", *CONTENT_FIELDS}
    source["current_skill_state"] = [
        {k: copy.deepcopy(v) for k, v in card.items() if k in allowed}
        for card in payload["current_skill_state"]
    ]
    compact, _ = compact_payload(source, {})
    compact.pop("raw_reference", None)
    compact.pop("raw_utf8_bytes", None)
    compact["local_evidence"] = copy.deepcopy(source["eligible_anchors"])
    compact["current_skill_state"] = copy.deepcopy(source["current_skill_state"])
    report["projection_version"] = "trajectory_projection_compatible"
    report["projected_payload_sha256"] = hashlib.sha256(dumps(compact).encode()).hexdigest()
    report["input_shape"] = "original_reference_pool_and_literal_local_evidence"
    return compact, report


def project_role_bound_payload(payload):
    compact, report = project_compatible_payload(payload)
    instruction = payload["completed_trajectory"]["instruction"]
    roles = task_roles(instruction)
    contexts = []
    for anchor in payload["eligible_anchors"]:
        prior = prerequisite_evidence(anchor, payload["transitions"], instruction)
        correction = str(anchor["correction"])
        match = re.fullmatch(r"go to ([a-z]+ \d+)", correction)
        binding = {}
        conditions = [p for p in prior["observable_entry_conditions"] if "holding(" in p]
        if match:
            entity = match[1]
            role = (
                "destination"
                if roles.get("destination") and _matches(entity, roles["destination"])
                else "candidate"
            )
            binding[role] = entity
            if prior["location"] is not None:
                conditions.append(
                    ("at" if prior["location"] == entity else "not_at") + "(" + role + ")"
                )
        contexts.append(
            {
                "anchor_id": anchor["anchor_id"],
                "literal_navigation_argument_binding": binding,
                "observable_entry_conditions": conditions,
                "inventory_witness": prior["inventory_witness"],
                "location_witness": prior["location_witness"],
            }
        )
    result = {
        "task_goal": instruction,
        "instruction_role_bindings": roles,
        "anchor_prerequisite_evidence": contexts,
        **compact,
    }
    report["projection_version"] = "trajectory_projection_role_bound"
    report["projected_payload_sha256"] = hashlib.sha256(dumps(result).encode()).hexdigest()
    return result, report


def full_context_reasons(payload):
    anchors = payload["eligible_anchors"]
    reasons = []
    if any(
        "you are not carrying anything." in t["observation_before"].lower()
        for a in anchors
        for t in a["localized_evidence"]
    ):
        reasons.append("explicit_empty_inventory_entry_scope")
    if not payload["current_skill_state"] and len(anchors) > 1:
        reasons.append("multiple_anchors_at_first_acquisition")
    return reasons


def project_guarded_payload(payload):
    reasons = full_context_reasons(payload)
    projected, report = project_payload(payload)

    projected.pop("anchor_prerequisite_evidence", None)
    if reasons:
        projected = copy.deepcopy(payload)
        report["projected_transition_count"] = len(payload["transitions"])
        report["omitted_transition_ids"] = []
    report.update(
        projection_version="trajectory_projection_guarded",
        full_context_required=bool(reasons),
        full_context_reasons=reasons,
        projected_payload_sha256=hashlib.sha256(dumps(projected).encode()).hexdigest(),
    )
    return projected, report
