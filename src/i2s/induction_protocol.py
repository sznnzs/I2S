from __future__ import annotations

import json

VERSION = "i2s_skills_likelihood"


def induction_generation(generation, learning):
    result = dict(generation)
    if (
        learning.get("protocol_repair_version") == VERSION
        and "induction_max_new_tokens" in learning
    ):
        count = learning["induction_max_new_tokens"]
        if type(count) is not int or not 0 < count < int(result["max_model_len"]):
            raise ValueError("invalid induction output budget")
        result["max_new_tokens"] = count
        result["max_input_tokens"] = min(
            int(result["max_input_tokens"]), int(result["max_model_len"]) - count
        )
    return result


def induction_uses_schema(learning, refinement_phase):
    if learning.get("protocol_repair_version") == VERSION:
        return False
    return bool(learning.get("schema_constrained_induction") or refinement_phase)


def final_json_text(raw):
    text = str(raw).strip()
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise ValueError("unclosed_reasoning_envelope")
        text = text[end + len("</think>") :].strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n") : -3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("final_response_must_be_json_object")
    return text


def induction_system(base, benchmark):
    if benchmark == "alfworld":
        return base + (
            "\nAdditional ALFWorld family vocabulary: role station; CLEAN(target,station), "
            "HEAT(target,station), COOL(target,station), ACTIVATE(station). Predicates "
            "cleaned(target), heated(target), cooled(target), activated(station), with "
            "cleaning_unconfirmed(target), heating_unconfirmed(target), "
            "cooling_unconfirmed(target), activation_unconfirmed(station) for recovery. "
            "Certificates cleaning_confirmation, heating_confirmation, cooling_confirmation, "
            "activation_confirmation require their exact native acknowledgement. Activation "
            "does not prove examination or task success."
        )
    from .semantic_benchmarks import RELATIONS, SPECS, operator_text, predicate_text

    prefix = base.split("Canonical roles:", 1)[0]
    rows = []
    for rule, (bench, op, role, pred) in SPECS.items():
        if bench != benchmark:
            continue
        rows.append(
            f"Role {role}: bind to the exact native_literal from an eligible effect witness. "
            f"Operator {op}({role}): {operator_text(op)} "
            f"Observable predicates {pred}({role}) and {pred}_unconfirmed({role}): {predicate_text(pred)}. "
            f"Certificate {rule}: {RELATIONS[rule]}"
        )
    if not rows:
        raise ValueError("missing benchmark induction vocabulary")
    predicates = [
        f"{pred}{suffix}({role})"
        for b, op, role, pred in SPECS.values()
        if b == benchmark
        for suffix in ("", "_unconfirmed")
    ]
    roles = sorted({role for b, op, role, pred in SPECS.values() if b == benchmark})
    operations = sorted({op for b, op, role, pred in SPECS.values() if b == benchmark} | {"VERIFY"})
    grammar = {
        "condition_strings": predicates,
        "operator_names": operations,
        "role_strings": roles,
        "operation_object": {
            "op": "one operator_names string",
            "args": [
                "one role_strings string; for VERIFY use one condition_strings string instead"
            ],
            "when": ["optional condition_strings only"],
        },
        "applicability": ["condition_strings only"],
        "outcome_check": ["supported positive condition_strings only"],
        "fallback": {
            "when": ["condition_strings ending in _unconfirmed(role)"],
            "procedure": ["operation objects with args as arrays"],
            "outcome_check": ["supported positive condition_strings only"],
        },
    }
    binding_instruction = "Read effect.native_role/native_literal to bind the cited action exactly."
    if benchmark == "scienceworld":
        from .science_objects import ROLES, SIGNATURES

        signatures = {op: [role] for bench, op, role, pred in SPECS.values() if bench == benchmark}
        signatures["VERIFY"] = ["one supported condition_strings predicate"]
        for rule, (op, arguments, pred) in SIGNATURES.items():
            signatures[op] = list(arguments)
            grammar["operator_names"].append(op)
            grammar["condition_strings"].extend(
                f"{pred}{suffix}({','.join(arguments)})" for suffix in ("", "_unconfirmed")
            )
            rows.append(
                f"Operator {op}({','.join(arguments)}): {operator_text(op)} "
                f"Certificate {rule}: {RELATIONS[rule]}"
            )
        grammar["role_strings"] = sorted(set(grammar["role_strings"]) | ROLES)
        grammar["operator_names"] = sorted(set(grammar["operator_names"]))
        grammar["operator_signatures"] = signatures
        grammar["operation_object"]["args"] = [
            "exact ordered role arguments from operator_signatures; VERIFY takes one supported predicate string"
        ]
        grammar["fallback"]["when"] = [
            "condition_strings with _unconfirmed and the exact one-role or two-role signature"
        ]
        binding_instruction = (
            "For single-role effects read effect.native_role/native_literal. "
            "For object effects bind every role to its exact value in effect.native_bindings "
            "from the same cited large correction transition. Preserve exact before/after "
            "quotes, including the native acknowledgement and complete object names."
        )
    return (
        prefix
        + "\nNative benchmark: "
        + benchmark
        + ". Do not use ALFWorld roles or operations.\n"
        + "\n".join(rows)
        + "\nUse only these native signatures, plus VERIFY with one supported predicate argument. "
        + binding_instruction
        + (
            " A short conditional, one-operation partial skill is allowed, with a corresponding"
            " unconfirmed-condition recovery and positive recheck. Only one proposal; return "
            "null if unsupported. Do not hard-code the source instance or claim strategic "
            "value/task success from a local state change.\nSTRICT SERIALIZATION CONTRACT:\n"
        )
        + json.dumps(grammar)
        + (
            "\nAll conditions, including applicability and optional when, MUST be exact strings "
            "from condition_strings, including parentheses. Never use free-text conditions or "
            "invent names such as visible_hand. args MUST be a JSON array of strings, NEVER an "
            "object/mapping; role variables have no dollar-sign prefix. Express additional "
            "discussion only in procedural_residual.relation, not condition fields. A positive "
            "predicate is the corresponding observable outcome; an _unconfirmed predicate is "
            "its verification condition. Every operator and positive outcome still requires "
            "exact cited support. Return the specified two-section JSON only."
        )
    )
