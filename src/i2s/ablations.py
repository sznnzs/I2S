import copy
import json
import re

VARIANTS = {"full", "no_skills", "no_anchors", "no_trajectory", "no_evidence", "no_evolution"}


def update_anchors(variant, anchors, transitions, task_id):
    if variant == "no_anchors":
        return [
            {
                "anchor_id": t["transition_id"],
                "task_id": task_id,
                "eligibility": bool(t.get("observable_effect")),
                "correction": t["action"],
                "effect": t.get("observable_effect"),
                "localized_evidence": [copy.deepcopy(t)],
            }
            for t in transitions
        ]
    if variant == "no_evidence":
        return [{**copy.deepcopy(a), "eligibility": True} for a in anchors]
    return anchors


def induction_payload(variant, *, serialized, transitions, anchors, task, outcome, skills):
    if variant == "no_anchors":
        return json.dumps(
            {
                "task_instruction": task.instruction,
                "executed_trajectory": [
                    {
                        "transition_id": t["transition_id"],
                        "action": t["action"],
                        "observation_before": t["observation_before"],
                        "observation_after": t["observation_after"],
                    }
                    for t in transitions
                ],
                "task_feedback": {"success": bool(outcome.get("success"))},
                "current_skill_state": [
                    {
                        k: v
                        for k, v in r.items()
                        if k not in {"supporting_anchor_ids", "residual_history", "source_task_ids"}
                    }
                    for r in skills
                ],
            },
            ensure_ascii=False,
        )
    if variant == "no_trajectory":
        return json.dumps(
            {
                "task_instruction": task.instruction,
                "local_evidence": anchors,
                "current_skill_state": skills,
            },
            ensure_ascii=False,
        )
    return serialized


def unsupported_certificate(evidence, anchors, proposal, task_instruction):
    from .semantic_induction import RULES, _key, normalize

    by_id = {a["anchor_id"]: a for a in anchors}
    aids = evidence.get("supporting_anchor_ids")
    if not isinstance(aids, list) or not aids or any(a not in by_id for a in aids):
        raise ValueError("nonexistent_source_reference")
    bindings, covered = {}, set()
    for fact in evidence.get("localized_facts", []):
        aid = fact.get("anchor_id")
        if aid not in aids:
            raise ValueError("source_reference_mismatch")
        transition = next(
            (
                t
                for t in by_id[aid]["localized_evidence"]
                if t["transition_id"] == fact.get("transition_id")
            ),
            None,
        )
        if transition is None:
            raise ValueError("nonexistent_transition")
        for field in ("observation_before", "observation_after"):
            quote = fact.get(field + "_quote")
            if (
                not isinstance(quote, str)
                or len(quote.strip()) < 8
                or quote not in transition[field]
            ):
                raise ValueError("nonexistent_source_quote")
        local = (
            transition["observation_before"]
            + " "
            + transition["action"]
            + " "
            + transition["observation_after"]
        )
        for role, literal in fact.get("role_bindings", {}).items():
            if not isinstance(literal, str) or literal.lower() not in local.lower():
                raise ValueError("unobservable_role_binding")
            bindings.setdefault(role, []).append(literal)
        covered.add(aid)
    if covered != set(aids):
        raise ValueError("uncited_source")
    residual = evidence.get("procedural_residual", {})
    relation = residual.get("relation", "")
    if (
        residual.get("rule") not in RULES
        or residual.get("type")
        not in {
            "prerequisite",
            "ordering_constraint",
            "verification_step",
            "recovery_rule",
            "conditional_branch",
        }
        or not isinstance(relation, str)
        or len(relation.split()) < 7
    ):
        raise ValueError("invalid_residual_schema")
    if re.search(
        r"\b(hidden|oracle|future task|ground truth|evaluation score|guarantees)\b", relation, re.I
    ):
        raise ValueError("hidden_future_or_unbounded_claim")
    content = normalize(proposal, bindings, task_instruction)
    operations = content["procedure"] + content["fallback"]["procedure"]
    predicates = set(content["outcome_check"] + content["fallback"]["outcome_check"])
    predicates.update(s["args"][0] for s in operations if s["op"] == "VERIFY")
    return {
        "anchor_ids": aids,
        "bindings": bindings,
        "ops": {_key(s) for s in operations if s["op"] != "VERIFY"},
        "predicates": predicates,
        "residual": residual,
        "evidence_support_ablated": True,
    }
