from __future__ import annotations

import copy
import re

POLICY = "covered_relations"
CONTENT_FIELDS = ("applicability", "procedure", "outcome_check", "fallback")


def _noun(entity):
    return re.sub(r"\s+\d+$", "", entity.lower().strip())


def _task_roles(instruction):
    match = re.fullmatch(
        r"\s*put (?:a |an |some |the )?([a-z]+) (?:in|on|at) ([a-z]+)\.?\s*", instruction.lower()
    )
    return {"target": match[1], "destination": match[2].rstrip(".")} if match else {}


def _observe(state, text):
    text = str(text).lower()
    location = re.search(r"you (?:arrive at |are facing (?:the )?)([a-z]+ \d+)\.", text)
    if location:
        state["location"] = location[1]
    if "you are not carrying anything." in text:
        state["holding"] = set()
    carrying = re.search(r"you are carrying: (.+?)\.", text)
    if carrying:
        state["holding"] = set(re.findall(r"([a-z]+ \d+)", carrying[1]))
    for receptacle, status in re.findall(r"(?:the )?([a-z]+ \d+) is (closed|open)\.", text):
        state["container"][receptacle] = status


def _advance(state, transition):
    _observe(state, transition.get("observation_before", ""))
    if transition.get("action_valid") is True and not transition.get("execution_error"):
        action = transition.get("action", "").lower()
        after = transition.get("observation_after", "").lower()
        take = re.fullmatch(r"take (.+) from (.+)", action)
        place = re.fullmatch(r"move (.+) to (.+)", action)
        if take and after.startswith(f"you pick up the {take[1]} from the {take[2]}."):
            state["holding"] = {take[1]}
        if place and after.startswith(f"you move the {place[1]} to the {place[2]}."):
            if state["holding"] is not None:
                state["holding"].discard(place[1])
    _observe(state, transition.get("observation_after", ""))


def _prefix_state(transition, transitions, initial_observation):
    state = {"location": None, "holding": None, "container": {}}
    _observe(state, initial_observation)
    step = transition.get("step_index")
    for prior in sorted(transitions, key=lambda x: x.get("step_index", -1)):
        if step is None or prior.get("step_index", -1) >= step:
            continue
        _advance(state, prior)
    _observe(state, transition.get("observation_before", ""))
    return state


def _relation(anchor, task_roles):
    local = anchor.get("localized_evidence", [])
    if any(
        t.get("explicit_failure") or t.get("action_valid") is False or t.get("execution_error")
        for t in local
    ):
        return None, "explicit_local_failure_requires_induction"
    correction = anchor.get("correction", "").lower()
    chosen = [
        t
        for t in local
        if t.get("action", "").lower() == correction
        and t.get("transition_id") == anchor.get("transition_id")
    ]
    if len(chosen) != 1:
        return None, "missing_unique_correction_transition"
    transition = chosen[0]
    from .agent import observable_effect

    effect, _ = observable_effect(
        correction,
        transition.get("observation_before", ""),
        transition.get("observation_after", ""),
        transition.get("action_valid"),
        transition.get("execution_error"),
    )
    if not effect or effect != transition.get("observable_effect"):
        return None, "unverified_native_effect"
    after = transition.get("observation_after", "").lower()
    navigation = re.fullmatch(r"go to (.+)", correction)
    opening = re.fullmatch(r"open (.+)", correction)
    acquire = re.fullmatch(r"take (.+) from (.+)", correction)
    place = re.fullmatch(r"move (.+) to (.+)", correction)
    if navigation or opening:
        receptacle = (navigation or opening)[1]
        contents = "you see" in after and receptacle in after
        if opening and not contents:
            return None, "opening_contents_not_observed"
        return {
            "rule": "arrival_reveals_state" if navigation else "opening_reveals_contents",
            "op": "NAVIGATE" if navigation else "OPEN_IF_NEEDED",
            "receptacle": receptacle,
            "postcondition": "contents_observed" if contents else "at",
            "transition": transition,
        }, None
    if acquire:
        if _noun(acquire[1]) != task_roles.get("target"):
            return None, "acquired_object_not_bound_to_exact_task_target"
        return {
            "rule": "taking_establishes_possession",
            "op": "ACQUIRE",
            "receptacle": acquire[2],
            "target": acquire[1],
            "postcondition": "holding",
            "transition": transition,
        }, None
    if place:
        if _noun(place[1]) != task_roles.get("target") or _noun(place[2]) != task_roles.get(
            "destination"
        ):
            return None, "placement_roles_not_bound_to_instruction"
        return {
            "rule": "placement_confirmation",
            "op": "PLACE",
            "receptacle": place[2],
            "target": place[1],
            "postcondition": "placed",
            "transition": transition,
        }, None
    return None, "effect_outside_coverage_vocabulary"


def _predicate_observed(predicate, bindings, state, relation, task_roles):
    match = re.fullmatch(r"([a-z_]+)\((.*?)\)", predicate)
    if not match:
        return False
    name, args = match[1], match[2].split(",") if match[2] else []
    entity = bindings.get(args[0]) if args else None
    location = state["location"]
    if name == "transfer_task":
        return args == ["target", "destination"] and bool(task_roles)
    if name in {"holding", "not_holding"} and args == ["target"]:
        held = state["holding"]
        if held is None or not task_roles.get("target"):
            return False
        present = any(_noun(x) == task_roles["target"] for x in held)
        return present if name == "holding" else not present
    if name in {"at", "not_at"}:
        if not location:
            return False
        if entity:
            at = location == entity
        elif args == ["destination"] and task_roles.get("destination"):
            return name == "not_at" and _noun(location) != task_roles["destination"]
        else:
            return False
        return at if name == "at" else not at
    if name == "reachable":
        return entity == relation["receptacle"]
    if name in {"closed", "open"}:
        return bool(entity and state["container"].get(entity) == name)
    if name == "visible" and args and args[0] == "target" and len(args) == 2:
        target, receptacle = bindings.get("target"), bindings.get(args[1])
        before = relation["transition"].get("observation_before", "").lower()
        return bool(
            target
            and receptacle
            and target in before
            and receptacle in before
            and "you see" in before
        )

    return False


def _cover(record, relation, state, task_roles):
    if (
        record.get("representation") != "semantic_operators"
        or record.get("semantic_validation") != "source_bound_certificate_pass"
    ):
        return None, "skill_not_semantically_admitted"
    if relation["rule"] not in {r.get("rule") for r in record.get("residual_history", [])}:
        return None, "relation_not_in_supported_history"
    steps = record.get("procedure", [])
    candidates = [s for s in steps if s.get("op") == relation["op"]]
    for step in candidates:
        args = step.get("args", [])
        if not args:
            continue
        rec_role = args[-1]
        if rec_role not in {"candidate", "source", "destination"}:
            continue
        if rec_role == "destination" and _noun(relation["receptacle"]) != task_roles.get(
            "destination"
        ):
            continue
        if rec_role == "source" and relation["op"] != "ACQUIRE":
            continue
        bindings = {rec_role: relation["receptacle"]}
        if relation.get("target"):
            bindings["target"] = relation["target"]
        if not all(
            _predicate_observed(p, bindings, state, relation, task_roles)
            for p in record.get("applicability", []) + step.get("when", [])
        ):
            continue

        post = relation["postcondition"]
        expected = (
            f"{post}({rec_role})"
            if post in {"at", "contents_observed"}
            else "holding(target)"
            if post == "holding"
            else "placed(target,destination)"
        )
        outcomes = set(record.get("outcome_check", []))
        outcomes.update(s.get("args", [None])[0] for s in steps if s.get("op") == "VERIFY")
        if expected not in outcomes:
            continue
        if post == "contents_observed" and relation["op"] == "NAVIGATE":
            matching_inspect = [
                s
                for s in steps[steps.index(step) + 1 :]
                if s.get("op") == "INSPECT" and s.get("args") == [rec_role]
            ]
            if not matching_inspect:
                continue
            after_state = copy.deepcopy(state)
            _advance(after_state, relation["transition"])
            if not any(
                all(
                    _predicate_observed(p, bindings, after_state, relation, task_roles)
                    for p in s.get("when", [])
                )
                for s in matching_inspect
            ):
                continue
        return {
            "skill_id": record["skill_id"],
            "version": record["version"],
            "rule": relation["rule"],
            "operator": {"op": step["op"], "args": args},
            "observable_postcondition": expected,
            "role_bindings": bindings,
            "applicability_witness": list(record["applicability"]),
            "prior_supporting_anchor_ids": sorted(
                {
                    a
                    for h in record.get("residual_history", [])
                    if h.get("rule") == relation["rule"]
                    for a in h.get("anchor_ids", [])
                }
            ),
            "transition_id": relation["transition"]["transition_id"],
            "certificate_scope": "structured_relation_only_not_residual_prose",
        }, None
    return None, "operator_outcome_or_observable_applicability_not_covered"


def classify_update(
    anchors, skills, transitions=(), *, task_instruction="", initial_observation=""
):
    eligible = [a for a in anchors if a.get("eligibility")]
    active = [r for r in skills if r.get("status") == "active"]
    checks = []
    task_roles = _task_roles(task_instruction)
    for anchor in eligible:
        relation, reason = _relation(anchor, task_roles)
        result = {
            "anchor_id": anchor["anchor_id"],
            "covered": False,
            "reason": reason or ("no_active_skill" if not active else None),
        }
        if relation:
            result["relation_signature"] = {k: relation[k] for k in ("rule", "op", "postcondition")}
            state = _prefix_state(relation["transition"], transitions, initial_observation)
            for record in sorted(active, key=lambda r: r["creation_order"]):
                proof, reason = _cover(record, relation, state, task_roles)
                if proof:
                    result.update(
                        covered=True,
                        reason="exact_structured_relation_and_observable_applicability",
                        certificate=proof,
                    )
                    known = set(record.get("supporting_anchor_ids", []))
                    known.update(x["anchor_id"] for x in record.get("evidence_accumulation", []))
                    result["already_accumulated"] = anchor["anchor_id"] in known
                    break
                result["reason"] = reason
        checks.append(result)
    covered = sum(c["covered"] for c in checks)
    if not eligible:
        category = "NO_ELIGIBLE_ANCHOR"
    elif covered == len(eligible):
        category = (
            "NO_CHANGE_NEEDED"
            if all(c["already_accumulated"] for c in checks)
            else "REDUNDANT_EVIDENCE"
        )
    else:
        known_rules = {h.get("rule") for r in active for h in r.get("residual_history", [])}
        unseen_rules = [
            c.get("relation_signature", {}).get("rule") for c in checks if not c["covered"]
        ]
        category = (
            "NEW_PROCEDURE"
            if not active or any(r and r not in known_rules for r in unseen_rules)
            else "POSSIBLE_REFINEMENT"
        )
    return {
        "event": "update_eligibility",
        "policy": POLICY,
        "classification": category,
        "call_required": category in {"NEW_PROCEDURE", "POSSIBLE_REFINEMENT"},
        "eligible_anchor_count": len(eligible),
        "covered_anchor_count": covered,
        "covered_anchor_fraction": covered / len(eligible) if eligible else None,
        "anchor_checks": checks,
    }


def accumulate_evidence(bank, decision, task_id, episode_index):
    events = []
    for check in decision["anchor_checks"]:
        if not check["covered"] or check.get("already_accumulated"):
            continue
        proof = check["certificate"]
        record = bank.latest().get(proof["skill_id"])
        if not record or record["version"] != proof["version"] or record["status"] != "active":
            raise ValueError("coverage_state_changed_before_accumulation")
        entry = {
            "anchor_id": check["anchor_id"],
            "task_id": task_id,
            "episode_index": episode_index,
            "policy": decision.get("policy", POLICY),
            **copy.deepcopy(proof),
        }
        accumulated = record.setdefault("evidence_accumulation", [])
        if check["anchor_id"] not in {e["anchor_id"] for e in accumulated}:
            accumulated.append(entry)
            events.append(
                {
                    "event": "skill_evidence_accumulated",
                    "skill_id": proof["skill_id"],
                    "version": proof["version"],
                    "content_changed": False,
                    "state_version_changed": False,
                    "evidence": copy.deepcopy(entry),
                }
            )
    return events
