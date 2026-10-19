from collections import defaultdict

from .agent import opportunity_id, transition_id

UPDATE = "skill_induction_refinement"


def annotate_opportunities(calls, steps, events):
    by_step = defaultdict(list)
    for c in calls:
        by_step[(c["episode_index"], c["step_index"])].append(c)
    props = {e["proposal_call_id"]: e for e in events if e.get("event") == "small_proposal"}
    step_lookup = {s["step_index"]: s for s in steps}
    current = None
    for (ep, si), rows in sorted(by_step.items()):
        proposals = [
            c for c in rows if c["purpose"] == "environment_action" and c.get("role") == "small"
        ]
        if len(proposals) > 1:
            raise ValueError("multiple_small_proposals_at_one_step")
        if proposals:
            current = (
                None
                if proposals[0].get("request_status") == "request_failed"
                else opportunity_id(ep, si)
            )
        recovery = not proposals
        for c in rows:
            old = c["purpose"]
            is_update = old == UPDATE
            c["request_purpose"] = {
                "environment_action": "small_action",
                "confidence_assessment": "reliability_assessment",
                "repair_action": "large_recovery" if recovery else "large_intervention",
                UPDATE: UPDATE,
            }.get(old, old)
            c["decision_opportunity_id"] = None if is_update else current
            c["accounting_phase"] = "post_task_update" if is_update else "online"
            if c["call_id"] in props:
                c["declared_skill_refs"] = props[c["call_id"]]["declared_skill_refs"]
        if si in step_lookup and any(
            c["purpose"] in {"environment_action", "confidence_assessment", "repair_action"}
            for c in rows
        ):
            s = step_lookup[si]
            s["decision_opportunity_id"] = current
            s["transition_id"] = transition_id(ep, si)
            s["observation_id"] = s["transition_id"] + ":after"
            s["small_proposal_call_id"] = proposals[0]["call_id"] if proposals else None
            s["declared_skill_refs"] = props.get(s["small_proposal_call_id"], {}).get(
                "declared_skill_refs", []
            )
