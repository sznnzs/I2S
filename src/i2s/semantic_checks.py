from __future__ import annotations

import copy
import re


def task_roles(instruction):
    text = instruction.strip().lower().rstrip(".")
    patterns = [
        r"find two ([a-z]+) and put them (?:in|on|at) (?:a |an |the )?([a-z]+)",
        r"put (?:a |an |some |the |two )?(?:clean |hot |cool )?([a-z]+) (?:in|on|at) (?:a |an |the )?([a-z]+)",
        r"(?:clean|heat|cool) (?:a |an |some |the )?([a-z]+) and put it (?:in|on|at) (?:a |an |the )?([a-z]+)",
    ]
    for p in patterns:
        m = re.fullmatch(p, text)
        if m:
            return {"target": m[1], "destination": m[2]}
    m = re.fullmatch(r"examine (?:a |an |the )?([a-z]+) with (?:a |an |the )?([a-z]+)", text)
    if m:
        return {"target": m[1], "station": m[2]}

    m = re.fullmatch(r"look at ([a-z]+) under (?:the )?(desklamp|floorlamp)", text)
    return {"target": m[1], "station": m[2]} if m else {}


def validate_roles(evidence, anchors, instruction):
    roles = task_roles(instruction)
    by = {a["anchor_id"]: a for a in anchors}
    for fact in evidence.get("localized_facts", []):
        bindings = fact.get("role_bindings", {})
        for role in ("target", "destination"):
            if role in bindings:
                noun = re.sub(r"\s+\d+$", "", bindings[role].lower())
                if role not in roles or noun != roles[role]:
                    raise ValueError("exact_task_role_mismatch:" + role)
        source = bindings.get("source")
        if source:
            a = by[fact["anchor_id"]]
            target = roles.get("target")

            supported = False
            for t in a["localized_evidence"]:
                before = t.get("observation_before", "").lower()
                act = t.get("action", "").lower()
                after = t.get("observation_after", "").lower()
                seen = re.search(
                    r"(?:on|in) the " + re.escape(source.lower()) + r", you see (.+?)(?:\.|$)",
                    before,
                )
                supported |= bool(
                    target and seen and re.search(r"\b" + re.escape(target) + r" \d+\b", seen[1])
                )
                taken = re.fullmatch(
                    r"take ("
                    + re.escape(target or "__unknown__")
                    + r" \d+) from "
                    + re.escape(source.lower()),
                    act,
                )
                supported |= bool(
                    taken
                    and t.get("action_valid")
                    and not t.get("execution_error")
                    and after.startswith(f"you pick up the {taken[1]} from the {source.lower()}.")
                )
            if not supported:
                raise ValueError("source_role_without_exact_target_observation")


def validate_recovery(content, anchors):
    fb = content["fallback"]
    ops = fb["procedure"]
    out = set(fb["outcome_check"])
    inverse = {
        "not_at": "at",
        "closed": "open",
        "contents_unobserved": "contents_observed",
        "not_holding": "holding",
        "placement_unconfirmed": "placed",
        "cleaning_unconfirmed": "cleaned",
        "heating_unconfirmed": "heated",
        "cooling_unconfirmed": "cooled",
        "activation_unconfirmed": "activated",
    }
    establishes = {
        "at": "NAVIGATE",
        "open": "OPEN_IF_NEEDED",
        "contents_observed": "INSPECT",
        "holding": "ACQUIRE",
        "placed": "PLACE",
        "cleaned": "CLEAN",
        "heated": "HEAT",
        "cooled": "COOL",
        "activated": "ACTIVATE",
    }
    from .semantic_benchmarks import SPECS

    for _, op, _role, pred in SPECS.values():
        inverse[pred + "_unconfirmed"] = pred
        establishes[pred] = op
    from .science_objects import RECOVERY_INVERSE, RECOVERY_OPS

    inverse.update(RECOVERY_INVERSE)
    establishes.update(RECOVERY_OPS)
    linked = False
    for cond in fb["when"]:
        name, args = cond.split("(", 1)
        args = args[:-1].split(",") if args[:-1] else []
        if name == "action_rejected":
            if not any(
                t.get("explicit_failure") for a in anchors for t in a.get("localized_evidence", [])
            ):
                raise ValueError("rejection_fallback_without_observed_failure")
            continue
        if name not in inverse:
            continue
        post = inverse[name] + "(" + ",".join(args) + ")"
        if post not in out:
            continue
        for op in ops:
            if op["op"] != establishes[inverse[name]]:
                continue
            required = op["args"][: len(args)]
            if required == args:
                linked = True
    if not linked:
        raise ValueError("fallback_condition_action_verification_unlinked")


RELATIONS = {
    "arrival_reveals_state": "A native arrival supplies the visible state of the bound receptacle; target presence requires a direct observation.",
    "opening_reveals_contents": "Opening the bound closed receptacle exposes its currently visible contents; target presence requires a separate exact observation.",
    "taking_establishes_possession": "The exact task target is acquired from its observed source and native feedback establishes possession.",
    "placement_confirmation": "The exact held target is moved to the instruction-bound destination and native feedback confirms this local placement.",
    "placement_requires_arrival": "A locally rejected placement is followed by arrival at its destination; placement requires a subsequent native placement acknowledgement.",
    "recover_from_rejected_action": "A locally rejected action is followed by the certified recovery operation and its observable local check; task completion requires separate verification.",
    "cleaning_confirmation": "The native cleaning transition establishes cleaning of the bound task target at the observed station.",
    "heating_confirmation": "The native heating transition establishes heating of the bound task target at the observed station.",
    "cooling_confirmation": "The native cooling transition establishes cooling of the bound task target at the observed station.",
    "activation_confirmation": "The native activation transition establishes activation of the selected light; examination and task completion remain separate.",
}


def narrow_residual(proof):
    from .semantic_benchmarks import RELATIONS as benchmark_relations

    result = copy.deepcopy(proof)
    res = result["residual"]
    text = {**RELATIONS, **benchmark_relations}.get(res["rule"])
    if text:
        res["original_relation"] = res["relation"]
        res["relation"] = text
        res["narrowing_policy"] = "local_effect_description"
    return result


def preserve_delivery_phase(content, proof, anchors, instruction):
    audit = {
        "policy": "source_observed_delivery",
        "applied": False,
        "old_applicability": list(content["applicability"]),
        "evidence": [],
    }
    audit["reason"] = "not_a_destination_navigation_card"
    operations = [(s["op"], s["args"]) for s in content["procedure"] if s["op"] != "VERIFY"]
    if operations != [("NAVIGATE", ["destination"])]:
        return audit
    if "transfer_task(target,destination)" not in content["applicability"]:
        return audit
    roles = task_roles(instruction)
    if not {"target", "destination"} <= roles.keys():
        return audit
    by_id = {a["anchor_id"]: a for a in anchors if a.get("eligibility")}
    audit["reason"] = "cited_anchor_does_not_establish_delivery_phase"
    for aid in proof["anchor_ids"]:
        anchor = by_id.get(aid, {})
        witnesses = []
        for t in anchor.get("localized_evidence", []):
            destination = re.fullmatch(
                r"go to (" + re.escape(roles["destination"]) + r" \d+)", t.get("action", "").lower()
            )
            pickup = re.fullmatch(
                r"you pick up the (" + re.escape(roles["target"]) + r" \d+) from the (.+)\.",
                t.get("observation_before", "").strip().lower(),
            )
            if (
                destination
                and pickup
                and pickup[2] != destination[1]
                and t.get("action_valid")
                and not t.get("execution_error")
                and (t.get("observable_effect") or {}).get("effect") == "navigation_arrival"
                and t.get("observation_after", "")
                .lower()
                .startswith("you arrive at " + destination[1] + ".")
            ):
                witnesses.append(
                    {
                        "anchor_id": aid,
                        "transition_id": t["transition_id"],
                        "observation_before": t["observation_before"],
                        "action": t["action"],
                        "observation_after": t["observation_after"],
                    }
                )
        if not witnesses:
            return audit
        audit["evidence"].extend(witnesses)
    if not audit["evidence"]:
        return audit
    content["applicability"] = list(
        dict.fromkeys(content["applicability"] + ["holding(target)", "not_at(destination)"])
    )
    audit.update(
        applied=content["applicability"] != audit["old_applicability"],
        new_applicability=list(content["applicability"]),
        reason="source_phase_preserved",
    )
    return audit
