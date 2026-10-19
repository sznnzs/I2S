from __future__ import annotations

import re

from . import science_objects

ROLES = {
    "query",
    "item",
    "option",
    "room",
    "device",
}
SPECS = {
    "catalog_search_results": ("webshop", "SEARCH_CATALOG", "query", "catalog_results"),
    "item_inspection_details": ("webshop", "INSPECT_ITEM", "item", "item_details"),
    "option_selection_acknowledged": ("webshop", "SELECT_OPTION", "option", "option_selected"),
    "room_arrival_acknowledged": ("scienceworld", "APPROACH_ROOM", "room", "room_reached"),
    "container_open_acknowledged": (
        "scienceworld",
        "OPEN_CONTAINER",
        "candidate",
        "container_open",
    ),
    "container_contents_revealed": (
        "scienceworld",
        "INSPECT_CONTAINER",
        "candidate",
        "container_contents",
    ),
    "device_activation_acknowledged": (
        "scienceworld",
        "ACTIVATE_DEVICE",
        "device",
        "device_active",
    ),
}
OPS = {s[1]: 1 for s in SPECS.values()}
PREDICATES = {p: 1 for s in SPECS.values() for p in [s[3], s[3] + "_unconfirmed"]}
PREDICATES.update(
    {
        "visible_item": 1,
        "available_option": 1,
        "accessible_room": 1,
        "accessible_container": 1,
        "accessible_device": 1,
        "search_context": 1,
    }
)
RULES = set(SPECS)
ROLES.update(science_objects.ROLES)
OPS.update(science_objects.OPS)
PREDICATES.update(science_objects.PREDICATES)
RULES.update(science_objects.RULES)
RELATIONS = {
    "catalog_search_results": "The submitted query reveals a visible catalog results page. Relevance and task success require separate checks.",
    "item_inspection_details": "Opening a currently visible product identifier reveals its detail page. Task requirement matching remains a separate check.",
    "option_selection_acknowledged": "Selecting a currently available option produces the matching selected-option marker. Confirm selection from that marker and check purchase requirements separately.",
    "room_arrival_acknowledged": "The native transition acknowledges arrival in the bound room. Target presence requires a direct observation.",
    "container_open_acknowledged": "The native acknowledgement establishes that the bound container was opened. Its contents require separate inspection.",
    "container_contents_revealed": "Inspection reveals the bound container contents. Confirm a requested item's presence from its explicit appearance in that observation.",
    "device_activation_acknowledged": "The native acknowledgement establishes activation of the bound device. Downstream transformations and task success require separate verification.",
}
RELATIONS.update(science_objects.RELATIONS)


def native_effect(benchmark, action, before, after, valid, error=None, *, version="v1"):
    if not valid or error or str(before).strip() == str(after).strip():
        return None
    a, b, o = map(lambda x: str(x).lower().strip(), [action, before, after])
    rule = literal = None
    if version == "native_public_feedback" and benchmark == "scienceworld":
        if o.startswith("environment response:\n"):
            o = o[len("environment response:\n") :]
        o = o.splitlines()[0] if o else o
    if benchmark == "webshop":
        search = re.fullmatch(r"search\[(.+)\]", a)
        click = re.fullmatch(r"click\[(.+)\]", a)
        if (
            search
            and re.search(r"page \d+ \(total results: [1-9]\d*\)", o)
            and re.search(r"\[button\] [a-z0-9]{10} \[button_\]", o)
        ):
            rule, literal = "catalog_search_results", search[1]
        elif click:
            value = click[1]
            if (
                re.fullmatch("[a-z0-9]{10}", value)
                and f"[button] {value} [button_]" in b
                and all(
                    s in o
                    for s in [
                        "price:",
                        "[button] description [button_]",
                        "[button] buy now [button_]",
                    ]
                )
            ):
                rule, literal = "item_inspection_details", value
            elif (
                f"[button] {value} [button_]" in b
                and f"[clicked button] {value} [clicked button_]" in o
                and f"[clicked button] {value} [clicked button_]" not in b
            ):
                rule, literal = "option_selection_acknowledged", value
    elif benchmark == "scienceworld":
        move = re.fullmatch(r"teleport to (.+)", a)
        opened = re.fullmatch(r"open (.+)", a)
        inspect = re.fullmatch(r"look in (.+)", a)
        activate = re.fullmatch(r"activate (.+)", a)
        if move and o == f"you teleport to the {move[1]}.":
            rule, literal = "room_arrival_acknowledged", move[1]
        elif opened:
            head = opened[1].split(" in ", 1)[0]
            if o == f"the {head} is now open.":
                rule, literal = "container_open_acknowledged", opened[1]
        elif inspect:
            head = inspect[1].split(" in ", 1)[0]
            if o.startswith(f"inside the {head} is:"):
                rule, literal = "container_contents_revealed", inspect[1]
        elif activate and o == f"the {activate[1]} is now activated.":
            rule, literal = "device_activation_acknowledged", activate[1]
        if version == "native_public_feedback":
            door = re.fullmatch(r"go to door to (.+)", a)
            go = re.fullmatch(r"go to (.+)", a)
            if door and o == f"you move through the door to the {door[1]}.":
                rule, literal = "room_arrival_acknowledged", door[1]
            elif go and o == f"you move to the {go[1]}.":
                rule, literal = "room_arrival_acknowledged", go[1]
    if rule:
        return {
            "effect": rule,
            "benchmark": benchmark,
            "native_role": SPECS[rule][2],
            "native_literal": literal,
            "scope": "local_public_acknowledgement_only",
        }
    return science_objects.native_effect(
        benchmark, action, before, after, valid, error, version=version
    )


def contains_content(value):
    if isinstance(value, dict):
        return value.get("op") in OPS or any(contains_content(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_content(v) for v in value)
    return isinstance(value, str) and (value in ROLES or value.split("(", 1)[0] in PREDICATES)


def role_check(name, args, operator=False):
    if science_objects.role_check(name, args, operator=operator):
        return
    for _, op, role, pred in SPECS.values():
        if (operator and name == op) or (not operator and name in {pred, pred + "_unconfirmed"}):
            if args != [role]:
                raise ValueError("benchmark_semantic_role_mismatch:" + name)
    expected = {
        "visible_item": "item",
        "available_option": "option",
        "accessible_room": "room",
        "accessible_container": "candidate",
        "accessible_device": "device",
        "search_context": "query",
    }
    if not operator and name in expected and args != [expected[name]]:
        raise ValueError("benchmark_predicate_role_mismatch")


def support(evidence, anchors, benchmark, *, version="v1"):
    by = {a["anchor_id"]: a for a in anchors if a.get("eligibility")}
    ops, predicates, effects = set(), set(), set()
    for fact in evidence["localized_facts"]:
        t = next(
            t
            for t in by[fact["anchor_id"]]["localized_evidence"]
            if t["transition_id"] == fact["transition_id"]
        )
        extra = science_objects.support_fact(
            fact, by[fact["anchor_id"]], benchmark, version=version
        )
        if extra is not None:
            new_ops, new_predicates, new_effects = extra
            ops.update(new_ops)
            predicates.update(new_predicates)
            effects.update(new_effects)
            continue
        proof = native_effect(
            benchmark,
            t["action"],
            t["observation_before"],
            t["observation_after"],
            t.get("action_valid"),
            t.get("execution_error"),
            version=version,
        )
        if not proof or proof != t.get("observable_effect"):
            continue
        rule = proof["effect"]
        _, op, role, pred = SPECS[rule]
        if fact["role_bindings"].get(role, "").lower() != proof["native_literal"]:
            raise ValueError("benchmark_native_role_identity_mismatch")
        ops.add((op, (role,)))
        predicates.add(pred + "(" + role + ")")
        effects.add(rule)
        guard = {
            "SEARCH_CATALOG": "search_context",
            "INSPECT_ITEM": "visible_item",
            "SELECT_OPTION": "available_option",
            "APPROACH_ROOM": "accessible_room",
            "OPEN_CONTAINER": "accessible_container",
            "INSPECT_CONTAINER": "accessible_container",
            "ACTIVATE_DEVICE": "accessible_device",
        }.get(op)
        if guard:
            predicates.add(guard + "(" + role + ")")
    rule = evidence["procedural_residual"]["rule"]
    if rule in science_objects.RULES:
        if benchmark != "scienceworld" or version != science_objects.VERSION or rule not in effects:
            raise ValueError("science_object_residual_without_exact_native_effect")
    elif rule not in SPECS or SPECS[rule][0] != benchmark or rule not in effects:
        raise ValueError("benchmark_residual_without_exact_native_effect")
    return ops, predicates, effects


def operator_text(name):
    if name in science_objects.OPS:
        return science_objects.operator_text(name)
    return {
        "SEARCH_CATALOG": "Form a concise query from the current task requirements and search the catalog; inspect the returned candidates without assuming relevance.",
        "INSPECT_ITEM": "Open the exact currently visible product identifier and inspect its current details before judging task requirements.",
        "SELECT_OPTION": "Select the exact requested option value when currently available; require its matching selected-option marker before proceeding.",
        "APPROACH_ROOM": "Navigate to the currently identified accessible room with the native navigation interaction; verify the arrival acknowledgement.",
        "OPEN_CONTAINER": "Open the exact currently accessible container and check its native open acknowledgement.",
        "INSPECT_CONTAINER": "Inspect the exact currently accessible open container and read its revealed contents without inventing an unseen target.",
        "ACTIVATE_DEVICE": "Activate the exact currently accessible device and verify its native activation acknowledgement; do not infer a downstream transformation.",
    }[name]


def predicate_text(name):
    if name in science_objects.PREDICATES:
        return science_objects.predicate_text(name)
    names = {
        "catalog_results": "the query has produced a visible catalog results page",
        "item_details": "the selected visible product has an open detail page",
        "option_selected": "the exact option has a matching selected-option marker",
        "room_reached": "native feedback acknowledges arrival in the selected room",
        "container_open": "native feedback acknowledges opening the selected container",
        "container_contents": "native feedback reveals the selected container contents",
        "device_active": "native feedback acknowledges activation of the selected device",
        "visible_item": "the selected product identifier is visible as an available click target",
        "available_option": "the task-requested option value is currently available",
        "accessible_room": "the selected room is currently accessible",
        "accessible_container": "the selected container is currently accessible",
        "accessible_device": "the selected device is currently accessible",
        "search_context": "the current task supplies visible catalog search requirements",
    }
    if name.endswith("_unconfirmed"):
        return "visible history has not yet confirmed that " + names[name[:-12]]
    return names[name]
