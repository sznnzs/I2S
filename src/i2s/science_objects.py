from __future__ import annotations

import re

VERSION = "native_public_feedback"
ROLES = {"science_object", "science_container"}
SIGNATURES = {
    "object_move_acknowledged": (
        "MOVE_OBJECT",
        ("science_object", "science_container"),
        "object_at_container",
    ),
    "object_pickup_acknowledged": ("PICK_UP_OBJECT", ("science_object",), "object_in_inventory"),
}
RULES = set(SIGNATURES)
OPS = {op: len(roles) for op, roles, pred in SIGNATURES.values()}
PREDICATES = {
    pred + suffix: len(roles)
    for op, roles, pred in SIGNATURES.values()
    for suffix in ("", "_unconfirmed")
}
RECOVERY_INVERSE = {pred + "_unconfirmed": pred for op, roles, pred in SIGNATURES.values()}
RECOVERY_OPS = {pred: op for op, roles, pred in SIGNATURES.values()}
RELATIONS = {
    "object_move_acknowledged": "The exact native reply acknowledges moving the bound object to the bound container. Check subsequent state and task requirements from current observations.",
    "object_pickup_acknowledged": "The exact native reply acknowledges moving the bound object to inventory. Check current possession and task requirements from current observations.",
}


def first_response(after):
    if not isinstance(after, str):
        return ""
    value = after.strip()
    if value.lower().startswith("environment response:\n"):
        value = value[len("environment response:\n") :]
    return value.splitlines()[0].strip() if value else ""


def _literal(value):
    return bool(re.fullmatch(r"[a-z0-9]+(?:[ -][a-z0-9]+)*", value))


def native_effect(benchmark, action, before, after, valid, error=None, *, version=None):
    if benchmark != "scienceworld" or version != VERSION:
        return None
    if valid is not True or error or not all(isinstance(x, str) for x in (action, before, after)):
        return None
    if before.strip() == after.strip() or "\n" in action or "\r" in action:
        return None
    action, head = action.lower().strip(), first_response(after).lower()
    move = re.fullmatch(r"move (.+) to (.+)", action)
    pickup = re.fullmatch(r"pick up (.+)", action)
    if move and all(_literal(x) for x in move.groups()) and move[1] != move[2]:
        if head != f"you move the {move[1]} to the {move[2]}.":
            return None
        rule = "object_move_acknowledged"
        bindings = {"science_object": move[1], "science_container": move[2]}
    elif pickup and _literal(pickup[1]):
        if head != f"you move the {pickup[1]} to the inventory.":
            return None
        rule = "object_pickup_acknowledged"
        bindings = {"science_object": pickup[1]}
    else:
        return None
    return {
        "effect": rule,
        "benchmark": "scienceworld",
        "native_bindings": bindings,
        "native_effect_version": VERSION,
        "scope": "local_public_object_receipt_only",
    }


def role_check(name, args, *, operator=False):
    for op, roles, pred in SIGNATURES.values():
        if (operator and name == op) or (not operator and name in {pred, pred + "_unconfirmed"}):
            if args != list(roles):
                raise ValueError("science_object_semantic_role_mismatch:" + name)
            return True
    return False


def support_fact(fact, anchor, benchmark, *, version):
    if benchmark != "scienceworld" or version != VERSION:
        return None
    local = anchor.get("localized_evidence", [])
    chosen = [t for t in local if t.get("transition_id") == fact.get("transition_id")]
    if len(chosen) != 1:
        raise ValueError("science_object_nonunique_local_transition")
    t = chosen[0]
    proof = native_effect(
        benchmark,
        t.get("action"),
        t.get("observation_before"),
        t.get("observation_after"),
        t.get("action_valid"),
        t.get("execution_error"),
        version=version,
    )
    recorded = t.get("observable_effect") or {}
    if proof is None:
        if recorded.get("effect") in RULES:
            raise ValueError("science_object_receipt_not_reproducible")
        return None
    if (
        anchor.get("eligibility") is not True
        or fact.get("anchor_id") != anchor.get("anchor_id")
        or t.get("transition_id") != anchor.get("transition_id")
        or t.get("source_role") != "large"
        or anchor.get("correction") != t.get("action")
    ):
        raise ValueError("science_object_not_exact_large_correction")
    if proof != recorded or anchor.get("effect") != proof:
        raise ValueError("science_object_logged_effect_mismatch")
    for field in ("observation_before", "observation_after"):
        quote = fact.get(field + "_quote")
        if not isinstance(quote, str) or len(quote.strip()) < 8 or quote not in t[field]:
            raise ValueError("science_object_exact_public_quote_required")
    if first_response(t["observation_after"]) not in fact["observation_after_quote"]:
        raise ValueError("science_object_quote_must_include_native_receipt")
    bindings = fact.get("role_bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(proof["native_bindings"]):
        raise ValueError("science_object_exact_role_set_required")
    if any(
        not isinstance(bindings[k], str) or bindings[k].lower() != v
        for k, v in proof["native_bindings"].items()
    ):
        raise ValueError("science_object_native_role_identity_mismatch")
    op, roles, pred = SIGNATURES[proof["effect"]]
    return {(op, roles)}, {pred + "(" + ",".join(roles) + ")"}, {proof["effect"]}


def operator_text(name):
    return {
        "MOVE_OBJECT": "Bind science_object and science_container from the current public state and plan. Move that object to that container when needed and verify the matching native relocation acknowledgement.",
        "PICK_UP_OBJECT": "Bind science_object from the current public state and plan. Pick up that object when needed and verify its matching move-to-inventory acknowledgement before proceeding.",
    }[name]


def predicate_text(name):
    inverse = name.endswith("_unconfirmed")
    base = name[: -len("_unconfirmed")] if inverse else name
    text = {
        "object_at_container": "the native reply has acknowledged relocation of the bound science_object to the bound science_container",
        "object_in_inventory": "the native reply has acknowledged moving the bound science_object to inventory",
    }[base]
    return "the current public history has not yet confirmed that " + text if inverse else text
