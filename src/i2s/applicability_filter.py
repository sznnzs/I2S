from __future__ import annotations

import re

POLICY = "observable_applicability"


def _entity_type(value):
    return re.sub(r"\s+\d+$", "", value.lower().strip())


def task_roles(instruction):
    match = re.fullmatch(
        r"\s*put (?:a |an |some |the )?([a-z]+(?: \d+)?) (?:in|on|at) ([a-z]+(?: \d+)?)\.?\s*",
        instruction.lower(),
    )
    return {"target": match[1], "destination": match[2]} if match else {}


def _matches(observed, instructed):
    return (
        observed == instructed
        if re.search(r" \d+$", instructed)
        else _entity_type(observed) == instructed
    )


def observable_state(transitions, *, current_observation, current_step):
    state = {"holding": None, "location": None, "holding_evidence": None, "location_evidence": None}

    def observe(text, source):
        text = str(text).lower()
        location = re.search(r"you (?:arrive at |are facing (?:the )?)([a-z]+ \d+)\.", text)
        if location:
            state["location"] = location[1]
            state["location_evidence"] = {**source, "observable_statement": location[0]}
        if "you are not carrying anything." in text:
            state["holding"] = []
            state["holding_evidence"] = {
                **source,
                "observable_statement": "you are not carrying anything.",
            }
        inventory = re.search(r"you are carrying: (.+?)\.", text)
        if inventory:
            entities = re.findall(r"[a-z]+ \d+", inventory[1])
            if entities:
                state["holding"] = entities
                state["holding_evidence"] = {**source, "observable_statement": inventory[0]}

    for transition in sorted(transitions, key=lambda t: t.get("step_index", current_step)):
        step = transition.get("step_index")
        if step is None or step >= current_step:
            continue
        source = {
            "source": "executed_prefix",
            "step_index": step,
            "transition_id": transition.get("transition_id"),
        }
        observe(transition.get("observation_before", ""), {**source, "field": "observation_before"})
        if transition.get("action_valid") is True and not transition.get("execution_error"):
            action = str(transition.get("action", "")).lower()
            after = str(transition.get("observation_after", "")).lower()
            take = re.fullmatch(r"take (.+) from (.+)", action)
            move = re.fullmatch(r"move (.+) to (.+)", action)
            if take:
                expected = f"you pick up the {take[1]} from the {take[2]}."
                if after.startswith(expected):
                    state["holding"] = [take[1]]
                    state["holding_evidence"] = {
                        **source,
                        "action": action,
                        "observable_statement": expected,
                    }
            if move:
                expected = f"you move the {move[1]} to the {move[2]}."
                if after.startswith(expected):
                    state["holding"] = []
                    state["holding_evidence"] = {
                        **source,
                        "action": action,
                        "observable_statement": expected,
                    }
        observe(transition.get("observation_after", ""), {**source, "field": "observation_after"})
    observe(current_observation, {"source": "current_observation", "step_index": current_step})
    return state


def filter_skills(
    records,
    transitions,
    *,
    task_instruction,
    current_observation,
    current_step,
    require_true: bool = False,
):
    roles = task_roles(task_instruction)
    state = observable_state(
        transitions, current_observation=current_observation, current_step=current_step
    )
    retained, excluded = [], []
    status_by_skill = []
    for record in records:
        false_conditions = []
        unknown_conditions = []

        if record.get("representation") != "semantic_operators":
            if require_true:
                unknown_conditions.append(
                    {"predicate": None, "reason": "unsupported_applicability_representation"}
                )
            else:
                retained.append(record)
                continue
        conditions = list(record.get("applicability", []))
        if require_true and not conditions:
            unknown_conditions.append({"predicate": None, "reason": "empty_applicability"})
        for condition in conditions:
            truth, evidence = None, None
            if (
                condition in {"holding(target)", "not_holding(target)"}
                and "target" in roles
                and state["holding"] is not None
            ):
                holding = any(_matches(entity, roles["target"]) for entity in state["holding"])
                truth = holding if condition == "holding(target)" else not holding
                evidence = state["holding_evidence"]
            if (
                condition in {"at(destination)", "not_at(destination)"}
                and "destination" in roles
                and state["location"] is not None
            ):
                at = _matches(state["location"], roles["destination"])
                truth = at if condition == "at(destination)" else not at
                evidence = state["location_evidence"]
            if truth is False:
                false_conditions.append(
                    {
                        "predicate": condition,
                        "observed_truth": False,
                        "task_role_bindings": dict(roles),
                        "evidence": evidence,
                    }
                )
            elif truth is None and require_true:
                unknown_conditions.append(
                    {
                        "predicate": condition,
                        "observed_truth": None,
                        "task_role_bindings": dict(roles),
                        "evidence": evidence,
                        "reason": "applicability_not_established",
                    }
                )
        if false_conditions or (require_true and unknown_conditions):
            status = "FALSE" if false_conditions else "UNKNOWN"
            excluded.append(
                {
                    "skill_id": record["skill_id"],
                    "version": record["version"],
                    "false_conditions": false_conditions,
                    "unknown_conditions": unknown_conditions,
                    "applicability_status": status,
                }
            )
            status_by_skill.append(
                {"skill_id": record["skill_id"], "version": record["version"], "status": status}
            )
        else:
            retained.append(record)
            status_by_skill.append(
                {
                    "skill_id": record["skill_id"],
                    "version": record["version"],
                    "status": "TRUE" if require_true else "UNFILTERED",
                }
            )
    return retained, {
        "event": "skill_applicability_filter",
        "policy": POLICY,
        "step_index": current_step,
        "retained_skill_ids": [r["skill_id"] for r in retained],
        "status_by_skill": status_by_skill,
        "excluded": excluded,
        "content_changed": False,
        "lifecycle_changed": False,
        "entry_gate": "all_true" if require_true else "hide_false_only",
        "unknown_predicates_retained": not require_true,
        "role_boundary": "Exact instruction entity names; numberless ALFWorld goal names denote the observed matching object/receptacle type. No hidden evaluator identity.",
    }
