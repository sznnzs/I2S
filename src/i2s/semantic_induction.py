from __future__ import annotations

import copy
import json
import re

from . import semantic_benchmarks as benchmark_semantics
from . import semantic_families as family_semantics
from .utils import estimate_tokens

FIELDS = ("applicability", "procedure", "outcome_check", "fallback")
ROLES = (
    {"target", "source", "destination", "candidate"}
    | family_semantics.ROLES
    | benchmark_semantics.ROLES
)
ALIASES = {
    "GO_TO": "NAVIGATE",
    "VISIT": "NAVIGATE",
    "LOCATE": "NAVIGATE",
    "EXAMINE": "INSPECT",
    "OBSERVE": "INSPECT",
    "OPEN": "OPEN_IF_NEEDED",
    "TAKE": "ACQUIRE",
    "PICK_UP": "ACQUIRE",
    "MOVE": "PLACE",
    "PUT": "PLACE",
    "CHECK": "VERIFY",
}
ROLE_ALIASES = {
    "target_object": "target",
    "object": "target",
    "source_receptacle": "source",
    "receptacle": "candidate",
    "container": "candidate",
    "receptacle_role": "candidate",
    "destination_receptacle": "destination",
}
PREDICATES = {
    "transfer_task": 2,
    "reachable": 1,
    "at": 1,
    "not_at": 1,
    "closed": 1,
    "open": 1,
    "contents_observed": 1,
    "contents_unobserved": 1,
    "visible": 2,
    "holding": 1,
    "not_holding": 1,
    "placed": 2,
    "placement_unconfirmed": 2,
    "action_rejected": 0,
}
PRED_ALIASES = {
    "target_held": "holding",
    "in_inventory": "holding",
    "container_closed": "closed",
    "arrived": "at",
    "target_placed": "placed",
    "task_requires": "transfer_task",
}
OPS = {"NAVIGATE": 1, "INSPECT": 1, "OPEN_IF_NEEDED": 1, "ACQUIRE": 2, "PLACE": 2, "VERIFY": 1}
RULES = {
    "arrival_reveals_state",
    "opening_reveals_contents",
    "taking_establishes_possession",
    "placement_requires_arrival",
    "placement_confirmation",
    "recover_from_rejected_action",
}
PREDICATES.update(family_semantics.PREDICATES)
OPS.update(family_semantics.OPS)
RULES.update(family_semantics.RULE_EFFECTS)
PREDICATES.update(benchmark_semantics.PREDICATES)
OPS.update(benchmark_semantics.OPS)
RULES.update(benchmark_semantics.RULES)


def visible_payload(payload):
    result = copy.deepcopy(payload)
    result.pop("task_feedback", None)
    for step in result["completed_trajectory"]["steps"]:
        step.pop("model_response", None)
        step.pop("reasoning", None)
    return result


def role(value):
    s = str(value).strip().strip("<>").lower().replace(" ", "_")
    s = ROLE_ALIASES.get(s, s)
    if s not in ROLES:
        raise ValueError("unknown_semantic_role:" + s)
    return s


def atom(value):
    if not isinstance(value, str):
        raise ValueError("predicate_not_string")
    match = re.fullmatch(r"\s*([A-Za-z_]+)\s*\((.*?)\)\s*", value)
    if not match:
        raise ValueError("invalid_observable_predicate")
    name = PRED_ALIASES.get(match[1].lower(), match[1].lower())
    args = [role(x) for x in match[2].split(",") if x.strip()]
    if name not in PREDICATES or len(args) != PREDICATES[name]:
        raise ValueError("unknown_or_hidden_predicate:" + name)
    if name in {"holding", "not_holding"} and args != ["target"]:
        raise ValueError("predicate_role_mismatch")
    if name in {"placed", "placement_unconfirmed", "transfer_task"} and args != [
        "target",
        "destination",
    ]:
        raise ValueError("predicate_role_mismatch")
    family_semantics.predicate_role_check(name, args)
    benchmark_semantics.role_check(name, args)
    return name + "(" + ",".join(args) + ")"


def atoms(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError("missing_observable_conditions")
    return list(dict.fromkeys(atom(x) for x in values))


def operation(value):
    if isinstance(value, str):
        m = re.fullmatch(r"\s*([A-Za-z_]+)\((.*)\)\s*", value)
        if not m:
            raise ValueError("invalid_semantic_operation")
        value = {
            "op": m[1],
            "args": [m[2]] if m[1].upper() in {"VERIFY", "CHECK"} else m[2].split(","),
        }
    if not isinstance(value, dict) or not set(value) <= {"op", "args", "when"}:
        raise ValueError("invalid_semantic_operation_fields")
    name = str(value.get("op", "")).upper().replace(" ", "_")
    name = ALIASES.get(name, name)
    args = value.get("args")
    if name not in OPS or not isinstance(args, list) or len(args) != OPS[name]:
        raise ValueError("unknown_semantic_operator_or_arity")
    args = [atom(args[0])] if name == "VERIFY" else [role(x) for x in args]
    if name in {"NAVIGATE", "INSPECT", "OPEN_IF_NEEDED"} and args[0] == "target":
        raise ValueError("receptacle_role_required")
    if name == "ACQUIRE" and (args[0] != "target" or args[1] not in {"source", "candidate"}):
        raise ValueError("acquire_role_mismatch")
    if name == "PLACE" and args != ["target", "destination"]:
        raise ValueError("place_role_mismatch")
    family_semantics.operator_role_check(name, args)
    benchmark_semantics.role_check(name, args, operator=True)
    result = {"op": name, "args": args}
    if value.get("when"):
        result["when"] = atoms(value["when"])
    return result


def procedure(values):
    if not isinstance(values, list) or not values:
        raise ValueError("missing_semantic_procedure")
    result = []
    for v in values:
        s = operation(v)
        if not result or s != result[-1]:
            result.append(s)
    return result


def normalize(content, bindings=None, task_instruction=""):
    if bindings is not None:
        names = {}
        for r, literals in bindings.items():
            for literal in literals:
                for name in (literal.lower(), re.sub(r"\s+\d+$", "", literal.lower())):
                    names.setdefault(name, set()).add(r)
        task = re.fullmatch(
            r"\s*put (?:a |an |some |the )?(.+?) (?:in|on|at) (.+?)\.?\s*", task_instruction.lower()
        )
        if task:
            for name, r in ((task[1], "target"), (task[2].rstrip("."), "destination")):
                names.setdefault(name, set()).add(r)
        aliases = {k: next(iter(v)) for k, v in names.items() if len(v) == 1}

        def project(value):
            if isinstance(value, dict):
                return {k: project(v) for k, v in value.items()}
            if isinstance(value, list):
                return [project(x) for x in value]
            if isinstance(value, str):
                if value.strip().lower() in aliases:
                    return aliases[value.strip().lower()]
                return re.sub(
                    r"(?<=[(,])\s*([^(),]+?)\s*(?=[,)])",
                    lambda m: aliases.get(m[1].strip().lower(), m[1].strip()),
                    value,
                )
            return value

        content = project(content)
    fb = content.get("fallback")
    if not isinstance(fb, dict) or set(fb) != {"when", "procedure", "outcome_check"}:
        raise ValueError("fallback_requires_condition_recovery_recheck")
    return {
        "applicability": atoms(content.get("applicability")),
        "procedure": procedure(content.get("procedure")),
        "outcome_check": atoms(content.get("outcome_check")),
        "fallback": {
            "when": atoms(fb["when"]),
            "procedure": procedure(fb["procedure"]),
            "outcome_check": atoms(fb["outcome_check"]),
        },
    }


def parse_output(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"evidence_interpretation", "skill_proposal"}:
        raise ValueError("invalid_two_section_schema")
    p, e = value["skill_proposal"], value["evidence_interpretation"]
    if p is None:
        return []
    if not isinstance(p, dict) or not isinstance(e, dict):
        raise ValueError("invalid_two_section_content")
    return [{**p, "evidence_interpretation": e}]


def render_card(record):
    names = {
        "target": "the exact target object named in this task",
        "destination": "the destination named in this task",
        "source": "the receptacle where the exact target was observed",
        "candidate": "the currently chosen visible reachable receptacle",
        "station": "the currently selected observable processing station or light",
    }

    def predicate(value):
        name, rest = value.split("(", 1)
        if name in benchmark_semantics.PREDICATES:
            return benchmark_semantics.predicate_text(name)
        if name in family_semantics.PREDICATES:
            return family_semantics.predicate_text(name)
        args = rest[:-1].split(",") if rest[:-1] else []
        r = names.get(args[0], "") if args else ""
        forms = {
            "reachable": f"{r} is currently reachable",
            "at": f"you are currently at {r}",
            "not_at": f"you are not currently at {r}",
            "closed": f"{r} is visibly closed",
            "open": f"{r} is visibly open",
            "contents_observed": f"the contents of {r} have been visibly revealed",
            "contents_unobserved": f"the contents of {r} have not yet been revealed",
            "holding": "you are holding the exact task target",
            "not_holding": "you are not holding the exact task target",
            "placed": "visible feedback confirms the exact target was placed at the task destination",
            "placement_unconfirmed": "placement of the exact target at the task destination is not confirmed",
            "transfer_task": "the instruction requires transferring the exact target to the named destination",
            "visible": "the exact target is visible at "
            + (
                names.get(args[1], "the observed source")
                if len(args) > 1
                else "the observed source"
            ),
            "action_rejected": "the environment explicitly rejected the preceding action",
        }
        return forms[name]

    def op(step):
        code, args = step["op"], step["args"]
        r = names.get(args[0], "")
        forms = {
            "NAVIGATE": f"Approach {r} using a currently admissible navigation action.",
            "INSPECT": f"Read the already revealed contents of {r} before choosing the next interaction.",
            "OPEN_IF_NEEDED": f"If {r} is visibly closed and accessible, open it to reveal its contents.",
            "ACQUIRE": "Acquire the visible exact task target from its observed source.",
            "PLACE": "Place the held exact task target at the currently reached task destination.",
            "VERIFY": "Check whether " + (predicate(args[0]) if code == "VERIFY" else "") + ".",
        }
        forms.update({name: family_semantics.operator_text(name) for name in family_semantics.OPS})
        forms.update(
            {name: benchmark_semantics.operator_text(name) for name in benchmark_semantics.OPS}
        )
        text = forms[code]
        if step.get("when"):
            text = "If " + " and ".join(predicate(x) for x in step["when"]) + ": " + text
        return text

    lines = [
        f"Skill {record['skill_id']}, version {record['version']}. Semantic guidance; bind roles from this task.",
        "Applicability: " + " and ".join(predicate(p) for p in record["applicability"]) + ".",
        "Procedure:",
    ]
    lines += [f"{i}. {op(step)}" for i, step in enumerate(record["procedure"], 1)]
    lines += [
        "Outcome check: " + " and ".join(predicate(p) for p in record["outcome_check"]) + ".",
        "Recovery: if "
        + " and ".join(predicate(p) for p in record["fallback"]["when"])
        + ": "
        + " ".join(op(s) for s in record["fallback"]["procedure"])
        + " Then re-check whether "
        + " and ".join(predicate(p) for p in record["fallback"]["outcome_check"])
        + ".",
    ]
    return "\n".join(lines)


def _key(op):
    return (op["op"], tuple(op["args"]))


def certificate(
    evidence,
    anchors,
    task_instruction="",
    family_support=False,
    benchmark_support=None,
    benchmark_version="v1",
):
    eligible = {a["anchor_id"]: a for a in anchors if a.get("eligibility")}
    aids = evidence.get("supporting_anchor_ids")
    if not isinstance(aids, list) or not aids or any(a not in eligible for a in aids):
        raise ValueError("missing_or_ineligible_anchor_reference")
    facts = evidence.get("localized_facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("missing_localized_facts")
    covered, supported_ops, supported_predicates, effects, bound = set(), set(), set(), set(), {}
    for fact in facts:
        aid = fact.get("anchor_id")
        if aid not in aids:
            raise ValueError("fact_anchor_outside_cited_support")
        a = eligible[aid]
        t = next(
            (t for t in a["localized_evidence"] if t["transition_id"] == fact.get("transition_id")),
            None,
        )
        if not t:
            raise ValueError("nonlocalized_transition_reference")
        for field in ("observation_before", "observation_after"):
            q = fact.get(field + "_quote")
            if not isinstance(q, str) or len(q.strip()) < 8 or q not in t[field]:
                raise ValueError("unsupported_localized_quote")
        bindings = fact.get("role_bindings")
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError("missing_observable_role_bindings")
        bindings = {role(k): v for k, v in bindings.items()}
        if not family_support and set(bindings) & family_semantics.ROLES:
            raise ValueError("family_role_requires_opt_in")
        if not benchmark_support and set(bindings) & benchmark_semantics.ROLES:
            raise ValueError("benchmark_role_requires_opt_in")
        local = t["observation_before"] + " " + t["action"] + " " + t["observation_after"]
        anchor_local = " ".join(
            x["observation_before"] + " " + x["action"] + " " + x["observation_after"]
            for x in a["localized_evidence"]
        )
        for r, literal in bindings.items():
            if (
                not isinstance(literal, str)
                or not literal
                or literal.lower() not in anchor_local.lower()
            ):
                raise ValueError("ungrounded_role_binding")
            if r in {"target", "destination"} and task_instruction:
                noun = re.sub(r"\s+\d+$", "", literal.lower())
                if not re.search(r"\b" + re.escape(noun) + r"\b", task_instruction.lower()):
                    raise ValueError("role_not_in_task_instruction")

            bound.setdefault(r, [])
            if literal not in bound[r]:
                bound[r].append(literal)
        covered.add(aid)
        effect = (t.get("observable_effect") or {}).get("effect")
        if not effect:
            continue
        effects.add(effect)
        if benchmark_support:
            continue
        act = t["action"].lower()
        obs = t["observation_after"].lower()
        for r, literal in bindings.items():
            literal = literal.lower()
            if act == "go to " + literal:
                supported_ops.add(("NAVIGATE", (r,)))
                supported_predicates.update({f"at({r})", f"reachable({r})"})
            if act == "open " + literal:
                supported_ops.add(("OPEN_IF_NEEDED", (r,)))
                supported_predicates.update({f"open({r})", f"at({r})", f"reachable({r})"})
            if r != "target" and literal in obs and ("you see" in obs or " is open" in obs):
                supported_ops.add(("INSPECT", (r,)))
                supported_predicates.add(f"contents_observed({r})")
            if r != "target" and literal + " is closed" in local.lower():
                supported_predicates.add(f"closed({r})")
        obj = bindings.get("target", "").lower()
        for r in ("source", "candidate"):
            rec = bindings.get(r, "").lower()
            if obj and rec and act == f"take {obj} from {rec}":
                supported_ops.add(("ACQUIRE", ("target", r)))
                supported_predicates.update({"holding(target)", f"visible(target,{r})"})
        dest = bindings.get("destination", "").lower()
        if obj and dest and act == f"move {obj} to {dest}":
            supported_ops.add(("PLACE", ("target", "destination")))
            supported_predicates.update(
                {"placed(target,destination)", "holding(target)", "at(destination)"}
            )
    if covered != set(aids):
        raise ValueError("uncited_anchor_evidence")
    residual = evidence.get("procedural_residual")
    if not isinstance(residual, dict) or residual.get("type") not in {
        "prerequisite",
        "ordering_constraint",
        "verification_step",
        "recovery_rule",
        "conditional_branch",
    }:
        raise ValueError("missing_explicit_procedural_residual")
    rule = residual.get("rule")
    relation = residual.get("relation")
    if rule not in RULES or not isinstance(relation, str) or len(relation.split()) < 7:
        raise ValueError("invalid_or_generic_residual")
    if rule in family_semantics.RULE_EFFECTS and not family_support:
        raise ValueError("family_rule_requires_opt_in")
    if family_support:
        family_ops, family_predicates, family_effects = family_semantics.support(evidence, anchors)
        supported_ops.update(family_ops)
        supported_predicates.update(family_predicates)
        effects.update(family_effects)
    if benchmark_support:
        new_ops, new_predicates, new_effects = benchmark_semantics.support(
            evidence, anchors, benchmark_support, version=benchmark_version
        )
        supported_ops.update(new_ops)
        supported_predicates.update(new_predicates)
        effects.update(new_effects)
    elif rule in benchmark_semantics.RULES:
        raise ValueError("benchmark_rule_requires_opt_in")
    required = {
        "arrival_reveals_state": "navigation_arrival",
        "opening_reveals_contents": "container_opened",
        "taking_establishes_possession": "object_taken",
        "placement_confirmation": "object_placed",
    }
    if rule in required and required[rule] not in effects:
        raise ValueError("residual_relation_unsupported_by_local_effect")
    if rule == "placement_requires_arrival":
        okay = False
        for aid in aids:
            a = eligible[aid]
            failures = [t for t in a["localized_evidence"] if t.get("explicit_failure")]
            for t in failures:
                m = re.fullmatch(r"move .+ to (.+)", t["action"])
                okay |= bool(m and a["correction"] == "go to " + m[1])
        if not okay:
            raise ValueError("ordering_requires_local_failed_placement_and_arrival")
    if rule == "recover_from_rejected_action" and not any(
        any(t.get("explicit_failure") for t in eligible[a]["localized_evidence"]) for a in aids
    ):
        raise ValueError("recovery_requires_local_failure")
    if re.search(
        r"\b(always succeeds|guarantees|hidden|oracle|future task|ground truth|evaluation score)\b",
        relation,
        re.I,
    ):
        raise ValueError("hidden_future_or_unbounded_claim")
    return {
        "anchor_ids": aids,
        "ops": supported_ops,
        "predicates": supported_predicates,
        "effects": effects,
        "residual": residual,
        "bindings": bound,
    }


def validate(p, anchors, old=None, config=None, task_instruction=""):
    config = config or {}
    evidence = p.get("evidence_interpretation")
    if not isinstance(evidence, dict):
        raise ValueError("missing_evidence_interpretation")
    if config.get("ablation") == "no_evidence":
        from .ablations import unsupported_certificate

        proof = unsupported_certificate(evidence, anchors, p, task_instruction)
    else:
        proof = certificate(
            evidence,
            anchors,
            task_instruction,
            config.get("semantic_family_support") == "alfworld",
            config.get("semantic_benchmark_support"),
            config.get("native_evidence_version", "v1"),
        )
    if config.get("semantic_checks") == "local_role_recovery":
        from .semantic_checks import narrow_residual, validate_roles

        if not config.get("semantic_benchmark_support"):
            validate_roles(evidence, anchors, task_instruction)
        proof = narrow_residual(proof)
    aids = p.get("supporting_anchor_ids")
    if not isinstance(aids, list) or set(aids) != set(proof["anchor_ids"]):
        raise ValueError("proposal_evidence_anchor_mismatch")
    content = normalize(p, proof["bindings"], task_instruction)
    if config.get("phase_applicability") == "source_observed_delivery":
        from .semantic_checks import preserve_delivery_phase

        proof["phase_applicability_audit"] = preserve_delivery_phase(
            content, proof, anchors, task_instruction
        )
    if benchmark_semantics.contains_content(content) and not config.get(
        "semantic_benchmark_support"
    ):
        raise ValueError("benchmark_semantics_not_enabled")
    if config.get("semantic_family_support") != "alfworld":
        from .semantic_families import contains_family_content

        if contains_family_content(content):
            raise ValueError("family_semantics_not_enabled")
    eligible_by_id = {a["anchor_id"]: a for a in anchors if a.get("eligibility")}
    for condition in content["applicability"]:
        if old and condition in old["applicability"]:
            continue
        m = re.fullmatch(r"not_at\((\w+)\)", condition)
        if m and m[1] in proof["bindings"]:
            states = [
                eligible_by_id[aid]["localized_evidence"][0]["observation_before"].lower()
                for aid in proof["anchor_ids"]
            ]
            literals = proof["bindings"][m[1]]
            if states and all(
                any(
                    re.search(
                        r"you (?:are facing (?:the )?|arrive at )" + re.escape(x.lower()) + r"[.,]",
                        state,
                    )
                    for x in literals
                )
                for state in states
            ):
                raise ValueError("applicability_contradicted_by_local_state")
    raw_content = json.dumps(content)
    if re.search(
        r"trial_T|alfworld:|gamefile|/home/|\b[a-z]+\s+\d+\b|example|placeholder|schema",
        raw_content,
        re.I,
    ):
        raise ValueError("placeholder_or_source_instance_leakage")
    if len(content["procedure"]) > int(config.get("max_skill_steps", 8)):
        raise ValueError("semantic_procedure_over_budget")
    if estimate_tokens(json.dumps(content)) > int(config.get("card_tokens", 900)):
        raise ValueError("semantic_card_over_budget")
    supported = set(proof["ops"])
    predicates = set(proof["predicates"])
    if old:
        supported.update(
            _key(x) for x in old["procedure"] + old["fallback"]["procedure"] if x["op"] != "VERIFY"
        )
        predicates.update(old["outcome_check"] + old["fallback"]["outcome_check"])
        for s in old["procedure"] + old["fallback"]["procedure"]:
            if s["op"] == "VERIFY":
                predicates.add(s["args"][0])
        if not set(old["applicability"]) <= set(content["applicability"]):
            raise ValueError("unsupported_applicability_broadening")
        pos = 0
        for step in content["procedure"]:
            if pos < len(old["procedure"]) and step == old["procedure"][pos]:
                pos += 1
        if pos != len(old["procedure"]) or not set(old["outcome_check"]) <= set(
            content["outcome_check"]
        ):
            raise ValueError("supported_content_not_preserved")
        changed = {k for k in FIELDS if content[k] != old[k]}
        declared = evidence.get("implicated_skill_fields", [])
        if not changed <= set(declared):
            raise ValueError("unlocalized_revision_fields")
    for s in content["procedure"] + content["fallback"]["procedure"]:
        if s["op"] == "VERIFY":
            if s["args"][0] not in predicates:
                raise ValueError("unsupported_verification_predicate:" + s["args"][0])
        elif _key(s) not in supported:
            raise ValueError("operator_without_local_support:" + str(_key(s)))
    for pred in content["outcome_check"] + content["fallback"]["outcome_check"]:
        if pred not in predicates:
            raise ValueError("outcome_not_locally_observed:" + pred)
    for checks in (content["outcome_check"], content["fallback"]["outcome_check"]):
        at_roles = [
            re.fullmatch(r"at\((\w+)\)", p)[1] for p in checks if re.fullmatch(r"at\((\w+)\)", p)
        ]
        if len(at_roles) > 1:
            entities = [set(proof["bindings"].get(r, [])) for r in at_roles]
            if all(entities) and not set.intersection(*entities):
                raise ValueError("contradictory_simultaneous_locations")
    if not any(s["op"] != "VERIFY" for s in content["procedure"]):
        raise ValueError("procedure_has_no_operation")
    if not any(s["op"] != "VERIFY" for s in content["fallback"]["procedure"]):
        raise ValueError("nonactionable_semantic_fallback")
    conditions = content["fallback"]["when"]
    recovery_markers = (
        "not_at(",
        "closed(",
        "contents_unobserved(",
        "not_holding(",
        "placement_unconfirmed(",
        "action_rejected(",
    )
    if config.get("semantic_family_support") == "alfworld":
        recovery_markers += family_semantics.RECOVERY_PREFIXES
    if config.get("semantic_benchmark_support"):
        recovery_markers += tuple(
            p + "(" for p in benchmark_semantics.PREDICATES if p.endswith("_unconfirmed")
        )
    if not any(x.startswith(recovery_markers) for x in conditions):
        raise ValueError("fallback_lacks_observable_failure_condition")
    if config.get("semantic_checks") == "local_role_recovery":
        from .semantic_checks import validate_recovery

        validate_recovery(content, anchors)

    return content, proof


def apply_semantic(
    bank, proposals, anchors, transitions, task_id, episode_index, task_instruction=""
):
    del transitions
    events, affected = [], set()
    for i, p in enumerate(proposals):
        try:
            op, sid = p.get("operation"), p.get("skill_id")
            if bank.config.get("ablation") == "no_evolution" and op in {"REVISE", "RETIRE"}:
                raise ValueError("content_evolution_disabled")
            old = bank.latest().get(sid)
            if op not in {"ADD", "REVISE", "RETAIN"}:
                raise ValueError("unknown_semantic_update")
            if op == "ADD" and sid is not None:
                raise ValueError("add_requires_new_identity")
            if op != "ADD" and (old is None or old["status"] != "active"):
                raise ValueError("unknown_or_inactive_target")
            if old and old.get("representation") != "semantic_operators":
                raise ValueError("skill_representation_mismatch")
            if op == "RETAIN":
                if bank.config.get("ablation") == "no_evidence":
                    from .ablations import unsupported_certificate

                    proof = unsupported_certificate(
                        p["evidence_interpretation"], anchors, old, task_instruction
                    )
                else:
                    proof = certificate(
                        p["evidence_interpretation"],
                        anchors,
                        task_instruction,
                        bank.config.get("semantic_family_support") == "alfworld",
                        bank.config.get("semantic_benchmark_support"),
                        bank.config.get("native_evidence_version", "v1"),
                    )
                if set(p.get("supporting_anchor_ids", [])) != set(proof["anchor_ids"]):
                    raise ValueError("proposal_evidence_anchor_mismatch")
                if bank.config.get("semantic_checks") == "local_role_recovery":
                    from .semantic_checks import narrow_residual, validate_roles

                    if not bank.config.get("semantic_benchmark_support"):
                        validate_roles(p["evidence_interpretation"], anchors, task_instruction)
                    proof = narrow_residual(proof)
                content = {k: copy.deepcopy(old[k]) for k in FIELDS}
            else:
                content, proof = validate(p, anchors, old, bank.config, task_instruction)
            if old and sid in affected:
                raise ValueError("multiple_versions_same_task")
            if op == "ADD":
                duplicate = next(
                    (x for x in bank.active() if all(x[k] == content[k] for k in FIELDS)), None
                )
                if duplicate:
                    old, sid = duplicate, duplicate["skill_id"]
            if old and all(old[k] == content[k] for k in FIELDS):
                actual, record = "RETAIN", old
            else:
                actual = "REVISE" if old else "ADD"
                record = {
                    **(copy.deepcopy(old) if old else {}),
                    **content,
                    "skill_id": sid if old else f"skill-{bank.next_order:04d}",
                    "version": old["version"] + 1 if old else 1,
                    "creation_order": old["creation_order"] if old else bank.next_order,
                    "created_episode": old["created_episode"] if old else episode_index,
                    "version_created_episode": episode_index,
                    "status": "active",
                    "representation": "semantic_operators",
                    "semantic_validation": "evidence_gate_ablated"
                    if bank.config.get("ablation") == "no_evidence"
                    else "source_bound_certificate_pass",
                    "supporting_anchor_ids": list(old["supporting_anchor_ids"]) if old else [],
                    "source_task_ids": list(old["source_task_ids"]) if old else [],
                    "residual_history": list(old.get("residual_history", [])) if old else [],
                }
                bank.records.append(record)
                if not old:
                    bank.next_order += 1
            record["supporting_anchor_ids"] = list(
                dict.fromkeys(record["supporting_anchor_ids"] + proof["anchor_ids"])
            )
            record["source_task_ids"] = list(dict.fromkeys(record["source_task_ids"] + [task_id]))
            record.setdefault("residual_history", []).append(
                {"task_id": task_id, "anchor_ids": proof["anchor_ids"], **proof["residual"]}
            )
            affected.add(record["skill_id"])
            bank.version += 1
            events.append(
                {
                    "event": "skill_lifecycle",
                    "operation": actual,
                    "skill_id": record["skill_id"],
                    "version": record["version"],
                    "creation_order": record["creation_order"],
                    "supporting_anchor_ids": proof["anchor_ids"],
                    "source_task_ids": [task_id],
                    "normalization_changed_surface": any(p.get(k) != content[k] for k in FIELDS)
                    if op != "RETAIN"
                    else False,
                    "evidence_interpretation": p["evidence_interpretation"],
                    "record": copy.deepcopy(record),
                }
            )
            if "phase_applicability_audit" in proof:
                events.append(
                    {
                        "event": "phase_applicability_audit",
                        "skill_id": record["skill_id"],
                        "version": record["version"],
                        **proof["phase_applicability_audit"],
                    }
                )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            events.append(
                {
                    "event": "proposal_rejected",
                    "proposal_index": i,
                    "reason": str(exc),
                    "proposal": p,
                }
            )
    for r in bank.active():
        if r["skill_id"] not in affected:
            events.append(
                {
                    "event": "skill_lifecycle",
                    "operation": "RETAIN",
                    "skill_id": r["skill_id"],
                    "version": r["version"],
                    "reason": "no_supported_content_change",
                }
            )
    events.extend(bank.capacity())
    return events
