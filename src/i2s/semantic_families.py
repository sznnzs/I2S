from __future__ import annotations

import re

ROLES = {"station"}
OPS = {"CLEAN": 2, "HEAT": 2, "COOL": 2, "ACTIVATE": 1}
PREDICATES = {
    "cleaned": 1,
    "heated": 1,
    "cooled": 1,
    "activated": 1,
    "cleaning_unconfirmed": 1,
    "heating_unconfirmed": 1,
    "cooling_unconfirmed": 1,
    "activation_unconfirmed": 1,
}
RULE_EFFECTS = {
    "cleaning_confirmation": "object_cleaned",
    "heating_confirmation": "object_heated",
    "cooling_confirmation": "object_cooled",
    "activation_confirmation": "tool_activated",
}
RECOVERY_PREFIXES = tuple(p + "(" for p in PREDICATES if p.endswith("unconfirmed"))


def contains_family_content(content):
    if isinstance(content, dict):
        return content.get("op") in OPS or any(contains_family_content(v) for v in content.values())
    if isinstance(content, list):
        return any(contains_family_content(v) for v in content)
    if isinstance(content, str):
        return (
            content == "station"
            or content.split("(", 1)[0] in PREDICATES
            or bool(re.search(r"(?:\(|,)station(?:,|\))", content))
        )
    return False


def observable_family_effect(action, before, after, action_valid, execution_error=None):
    if not action_valid or execution_error or str(before).strip() == str(after).strip():
        return None
    action, after = str(action).lower().strip(), str(after).lower()
    match = re.fullmatch(r"(clean|heat|cool) (.+) with (.+)", action)
    if match:
        prefix = f"you {match[1]} the {match[2]} using the {match[3]}."
        if after.startswith(prefix):
            return {
                "effect": {
                    "clean": "object_cleaned",
                    "heat": "object_heated",
                    "cool": "object_cooled",
                }[match[1]],
                "expected_observation_prefix": prefix,
            }
    match = re.fullmatch(r"use ((?:desklamp|floorlamp) \d+)", action)
    if match:
        prefix = f"you turn on the {match[1]}."
        if after.startswith(prefix):
            return {"effect": "tool_activated", "expected_observation_prefix": prefix}
    return None


def predicate_role_check(name, args):
    if name in PREDICATES:
        expected = ["station"] if name in {"activated", "activation_unconfirmed"} else ["target"]
        if args != expected:
            raise ValueError("family_predicate_role_mismatch")


def operator_role_check(name, args):
    if name in OPS:
        expected = ["station"] if name == "ACTIVATE" else ["target", "station"]
        if args != expected:
            raise ValueError("family_operator_role_mismatch")


def support(evidence, anchors):
    by_id = {a["anchor_id"]: a for a in anchors if a.get("eligibility")}
    ops, predicates, effects = set(), set(), set()
    for fact in evidence["localized_facts"]:
        anchor = by_id[fact["anchor_id"]]
        transition = next(
            t for t in anchor["localized_evidence"] if t["transition_id"] == fact["transition_id"]
        )
        proof = observable_family_effect(
            transition["action"],
            transition["observation_before"],
            transition["observation_after"],
            transition.get("action_valid"),
            transition.get("execution_error"),
        )
        if not proof or proof != transition.get("observable_effect"):
            continue
        bindings = fact["role_bindings"]
        target, station = (
            str(bindings.get("target", "")).lower(),
            str(bindings.get("station", "")).lower(),
        )
        action = transition["action"].lower()
        for verb, code, predicate in (
            ("clean", "CLEAN", "cleaned"),
            ("heat", "HEAT", "heated"),
            ("cool", "COOL", "cooled"),
        ):
            if target and station and action == f"{verb} {target} with {station}":
                ops.add((code, ("target", "station")))
                predicates.update(
                    {f"{predicate}(target)", "holding(target)", "at(station)", "reachable(station)"}
                )
                effects.add(proof["effect"])
        if station and action == "use " + station and proof["effect"] == "tool_activated":
            ops.add(("ACTIVATE", ("station",)))
            predicates.update({"activated(station)", "reachable(station)"})
            effects.add(proof["effect"])
    rule = evidence["procedural_residual"]["rule"]
    if rule in RULE_EFFECTS and RULE_EFFECTS[rule] not in effects:
        raise ValueError("family_residual_without_exact_native_effect_and_roles")
    return ops, predicates, effects


def predicate_text(name):
    return {
        "cleaned": "visible native feedback confirms cleaning the exact task target",
        "heated": "visible native feedback confirms heating the exact task target",
        "cooled": "visible native feedback confirms cooling the exact task target",
        "activated": "visible native feedback confirms that the selected light is on",
        "cleaning_unconfirmed": "cleaning the exact task target is not yet confirmed by visible feedback",
        "heating_unconfirmed": "heating the exact task target is not yet confirmed by visible feedback",
        "cooling_unconfirmed": "cooling the exact task target is not yet confirmed by visible feedback",
        "activation_unconfirmed": "activation of the selected light is not yet confirmed by visible feedback",
    }[name]


def operator_text(name):
    return {
        "CLEAN": "While holding the exact task target at an accessible cleaning station, select its currently admissible cleaning interaction.",
        "HEAT": "While holding the exact task target at an accessible heating station, select its currently admissible heating interaction.",
        "COOL": "While holding the exact task target at an accessible cooling station, select its currently admissible cooling interaction.",
        "ACTIVATE": "Activate the selected currently accessible light using an admissible interaction; activation alone does not verify task completion.",
    }[name]
