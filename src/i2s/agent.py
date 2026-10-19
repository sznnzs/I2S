from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict

from .models import ModelInvocationError
from .policy import InterventionPolicy
from .policy_support import allocate_request_generation, parse_action_output
from .schemas import ModelCallPosition, PolicyEpisodeUpdate, PositionedModelCall
from .utils import estimate_tokens

CONTENT_FIELDS = ("applicability", "procedure", "outcome_check", "fallback")


def build_induction_payload(trajectory, eligible, transitions, outcome, skills):
    visible = asdict(trajectory)
    for step in visible["steps"]:
        try:
            response = json.loads(step["model_response"])
            response.pop("skill_refs", None)
            step["model_response"] = json.dumps(response, ensure_ascii=False)
        except (ValueError, AttributeError):
            pass
    return {
        "completed_trajectory": visible,
        "eligible_anchors": eligible,
        "transitions": transitions,
        "task_feedback": {k: outcome.get(k) for k in ("status", "success", "step_count")},
        "current_skill_state": skills,
    }


def opportunity_id(episode_index, proposal_step):
    return f"op-{episode_index:05d}-{proposal_step:03d}"


def transition_id(episode_index, step):
    return f"tr-{episode_index:05d}-{step:03d}"


def declared_references(raw, available):
    try:
        value = json.loads(raw)
        refs = value.get("skill_refs")
        if not isinstance(refs, list):
            return [], [], "missing_or_invalid_skill_refs"
        valid, excluded = [], []
        for ref in refs:
            if (
                isinstance(ref, dict)
                and set(ref) == {"skill_id", "version"}
                and isinstance(ref["version"], int)
                and not isinstance(ref["version"], bool)
                and available.get(str(ref["skill_id"])) == ref["version"]
            ):
                if ref not in valid:
                    valid.append(ref)
            else:
                excluded.append(ref)
        return valid, excluded, None
    except (ValueError, AttributeError, TypeError):
        return [], [], "invalid_proposal_json"


def observable_effect(action, before, after, action_valid, execution_error=None):
    if not action_valid or execution_error:
        return None, "explicit_execution_failure"
    a, b, o = str(action or "").lower().strip(), str(before).lower(), str(after).lower()
    if o.strip() == b.strip():
        return None, "unchanged_observation"
    patterns = [
        (r"go to (.+)", lambda m: f"you arrive at {m[1]}.", "navigation_arrival"),
        (r"open (.+)", lambda m: f"you open the {m[1]}.", "container_opened"),
        (r"close (.+)", lambda m: f"you close the {m[1]}.", "container_closed"),
        (
            r"take (.+) from (.+)",
            lambda m: f"you pick up the {m[1]} from the {m[2]}.",
            "object_taken",
        ),
        (r"move (.+) to (.+)", lambda m: f"you move the {m[1]} to the {m[2]}.", "object_placed"),
    ]
    for pattern, expected, kind in patterns:
        match = re.fullmatch(pattern, a)
        if match and o.startswith(expected(match)):
            return {"effect": kind, "expected_observation_prefix": expected(match)}, None

    from .semantic_families import observable_family_effect

    family_effect = observable_family_effect(a, b, o, action_valid, execution_error)
    if family_effect:
        return family_effect, None
    return None, "no_whitelisted_command_effect_pair"


class ProceduralState:
    def __init__(self, config):
        self.config = dict(config)
        self.records: list[dict] = []
        self.version = 0
        self.next_order = 1
        self.contradictions: dict[str, list[dict]] = {}

    def latest(self):
        return {r["skill_id"]: r for r in self.records}

    def active(self):
        return sorted(
            (r for r in self.latest().values() if r["status"] == "active"),
            key=lambda r: r["creation_order"],
        )

    @staticmethod
    def render(r):
        if r.get("representation") == "semantic_operators":
            from .semantic_induction import render_card

            return render_card(r)
        return json.dumps(
            {k: r[k] for k in ("skill_id", "version", *CONTENT_FIELDS)}, ensure_ascii=False
        )

    def capacity(self):
        events = []
        while self.active() and (
            len(self.active()) > int(self.config.get("max_skills", 6))
            or estimate_tokens("\n\n".join(self.render(r) for r in self.active()))
            > int(self.config.get("context_tokens", 2400))
        ):
            oldest = self.active()[0]
            oldest["status"] = "capacity_excluded"
            events.append(
                {
                    "event": "skill_lifecycle",
                    "operation": "CAPACITY_EXCLUDE",
                    "skill_id": oldest["skill_id"],
                    "version": oldest["version"],
                    "creation_order": oldest["creation_order"],
                    "reason": "oldest_first_fifo",
                }
            )
        return events

    def apply(self, proposals, anchors, failures, task_id, episode_index, task_instruction=""):
        from .semantic_induction import apply_semantic

        if self.config.get("revision_mode", "semantic") != "semantic":
            raise ValueError("I2S requires semantic skill admission")
        return apply_semantic(
            self, proposals, anchors, failures, task_id, episode_index, task_instruction
        )

    def state_dict(self):
        return copy.deepcopy(
            {
                "records": self.records,
                "state_version": self.version,
                "next_order": self.next_order,
                "contradictions": self.contradictions,
            }
        )

    def load(self, value):
        self.records = copy.deepcopy(value.get("records", []))
        self.version = int(value.get("state_version", 0))
        self.next_order = int(value.get("next_order", 1))
        self.contradictions = copy.deepcopy(value.get("contradictions", {}))


class I2SPolicy(InterventionPolicy):
    records_decision_opportunities = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.learning = dict(config.get("learning") or {})
        self.enabled = bool(self.learning.get("skills_enabled", False))

        self.applicability_gated_activation = bool(
            self.learning.get("applicability_gated_activation", False)
        )
        self.bank = ProceduralState(self.learning)

    def state_dict(self):
        return {
            "schema": "learning-procedural-state-v1",
            "skills_enabled": self.enabled,
            **self.bank.state_dict(),
        }

    def load_state_dict(self, value):
        self.bank.load(value)

    def skill_records(self):
        return tuple(copy.deepcopy(self.bank.records))

    def start_episode(self, task, *, episode_index, initial_observation, context_manager):
        super().start_episode(
            task,
            episode_index=episode_index,
            initial_observation=initial_observation,
            context_manager=context_manager,
        )
        self.task = task
        self.finished = False
        self.anchors, self.proposals, self.transitions = [], [], []
        self.current_op = None
        self.current_proposal = None
        self.active_at_start = copy.deepcopy(self.bank.active()) if self.enabled else []
        presented_at_start = self.active_at_start
        if self.enabled and self.active_at_start and self.applicability_gated_activation:
            from .applicability_filter import filter_skills

            presented_at_start, filter_event = filter_skills(
                self.active_at_start,
                self.transitions,
                task_instruction=self.task.instruction,
                current_observation=initial_observation,
                current_step=1,
                require_true=True,
            )
            self.episode_events.append({**filter_event, "task_id": task.task_id})
        context_manager.set_skills(
            [(r["skill_id"], self.bank.render(r)) for r in presented_at_start],
            state_version=self.bank.version,
        )
        self.episode_events.append(
            {
                "event": "learning_task_start",
                "task_id": task.task_id,
                "split": task.split,
                "evidence_class": "I2S",
                "active_skills": [
                    {"skill_id": r["skill_id"], "version": r["version"]}
                    for r in self.active_at_start
                ],
            }
        )

    def act(self, state, *, context_manager, generation, call_position=None):
        presented_at_action = self.active_at_start
        if (
            self.enabled
            and self.active_at_start
            and (
                self.learning.get("observable_applicability_filter", False)
                or self.applicability_gated_activation
            )
        ):
            from .applicability_filter import filter_skills

            presented_at_action, filter_event = filter_skills(
                self.active_at_start,
                self.transitions,
                task_instruction=self.task.instruction,
                current_observation=state.observation,
                current_step=state.step_index,
                require_true=self.applicability_gated_activation,
            )
            context_manager.set_skills(
                [(r["skill_id"], self.bank.render(r)) for r in presented_at_action],
                state_version=self.bank.version,
            )

            self.episode_events.append({**filter_event, "task_id": state.task_id})
        decision, prompt = super().act(
            state,
            context_manager=context_manager,
            generation=generation,
            call_position=call_position,
        )
        small = next((c for c in decision.model_calls if c.purpose == "environment_action"), None)
        if small is not None:
            self.current_op = opportunity_id(self.current_episode_index, state.step_index)
            proposal_prompt = prompt.proposal or prompt
            shown = set(proposal_prompt.context.selected_external_item_ids)
            available = {
                r["skill_id"]: r["version"] for r in self.active_at_start if r["skill_id"] in shown
            }
            reference_output = small.raw_output or ""
            if self.learning.get("json_envelope_protocol") == "whole_json_fence":
                from .json_envelope import normalize_json_envelope

                reference_output = normalize_json_envelope(reference_output)[0]
            if self.learning.get("protocol_repair_version") == "i2s_skills_likelihood":
                from .induction_protocol import final_json_text

                try:
                    reference_output = final_json_text(reference_output)
                except ValueError:
                    pass
            refs, excluded, error = declared_references(reference_output, available)
            _, action, _, parse_error = parse_action_output(
                small.raw_output or "",
                benchmark=state.benchmark,
                parser=self.parser,
                max_action_chars=self.max_action_chars,
            )
            self.current_proposal = {
                "event": "small_proposal",
                "decision_opportunity_id": self.current_op,
                "proposal_call_id": small.call_id,
                "step_index": state.step_index,
                "task_id": state.task_id,
                "split": self.task.split,
                "proposed_action": action,
                "parse_error": parse_error,
                "proposal_executed": decision.intervention == "none" and action is not None,
                "declared_skill_refs": refs,
                "excluded_skill_refs": excluded,
                "declaration_error": error,
                "available_skill_versions": available,
                "context_call_id": small.call_id,
            }
            omitted = [r["skill_id"] for r in presented_at_action if r["skill_id"] not in shown]
            if omitted:
                self.episode_events.append(
                    {
                        "event": "context_capacity_exclusion",
                        "decision_opportunity_id": self.current_op,
                        "skill_ids": omitted,
                        "reason": "oldest_first_fifo_prompt_budget",
                        "retirement": False,
                    }
                )
            self.proposals.append(self.current_proposal)
            self.episode_events.append(self.current_proposal)
        if self.current_op is None:
            raise ValueError("recovery_without_small_proposal")
        return decision, prompt

    def observe(self, *, state, decision, transition):
        after = transition.policy_observation or transition.observation
        tid = transition_id(self.current_episode_index, state.step_index)
        effect, ambiguity = observable_effect(
            decision.action,
            state.observation,
            after,
            transition.action_valid,
            transition.execution_error,
        )
        if self.learning.get("semantic_benchmark_support"):
            from .semantic_benchmarks import native_effect

            effect = native_effect(
                self.learning["semantic_benchmark_support"],
                decision.action,
                state.observation,
                after,
                transition.action_valid,
                transition.execution_error,
                version=self.learning.get("native_evidence_version", "v1"),
            )
            ambiguity = None if effect else "no_exact_benchmark_public_acknowledgement"
        record = {
            "transition_id": tid,
            "observation_id": tid + ":after",
            "decision_opportunity_id": self.current_op,
            "step_index": state.step_index,
            "action": decision.action,
            "observation_before": state.observation,
            "observation_after": after,
            "action_valid": transition.action_valid,
            "execution_error": transition.execution_error,
            "explicit_failure": not transition.action_valid or bool(transition.execution_error),
            "source_role": "large" if decision.intervention == "repair" else "small",
            "observable_effect": effect,
        }
        self.transitions.append(record)
        if decision.intervention == "repair":
            failed = [
                r
                for r in self.transitions[:-1]
                if r["decision_opportunity_id"] == self.current_op
                and r["source_role"] == "small"
                and r["explicit_failure"]
            ]
            anchor = {
                "event": "intervention_anchor",
                "anchor_id": f"anchor-{self.current_episode_index:05d}-{state.step_index:03d}",
                "task_id": self.task.task_id,
                "decision_opportunity_id": self.current_op,
                "proposal_call_id": self.current_proposal["proposal_call_id"],
                "small_proposed_action": self.current_proposal["proposed_action"],
                "proposal_executed": self.current_proposal["proposal_executed"],
                "correction": decision.action,
                "support_type": (
                    "B_explicit_failure_recovery" if failed else "A_immediate_transition"
                )
                if effect
                else None,
                "eligibility": effect is not None,
                "ambiguity_reason": ambiguity,
                "transition_id": tid,
                "supporting_transition_ids": [r["transition_id"] for r in failed] + [tid],
                "supporting_observation_ids": [r["observation_id"] for r in failed]
                + [tid + ":after"],
                "localized_evidence": [*failed, record],
                "effect": effect,
                "withheld_proposal_counterfactual": "unknown"
                if not self.current_proposal["proposal_executed"]
                else "executed",
            }
            self.anchors.append(anchor)
            self.episode_events.append(anchor)
        super().observe(state=state, decision=decision, transition=transition)

    def finish_episode(
        self, *, episode_index, outcome, trajectory, context_manager, generation, max_steps
    ):
        del context_manager
        if self.finished:
            raise ValueError("task_update_already_finished")
        self.finished = True
        for p in self.proposals:
            own = [
                t
                for t in self.transitions
                if t["decision_opportunity_id"] == p["decision_opportunity_id"]
                and t["source_role"] == "small"
            ]
            negative = any(t["explicit_failure"] for t in own)
            positive = bool(outcome.get("success")) and any(t["observable_effect"] for t in own)
            label = "UNCERTAIN" if negative else "CONFIDENT" if positive else None
            self.episode_events.append(
                {
                    "event": "reliability_candidate",
                    "task_id": self.task.task_id,
                    "split": self.task.split,
                    "evidence_class": "I2S_RELIABILITY_CANDIDATE",
                    "decision_opportunity_id": p["decision_opportunity_id"],
                    "context_call_id": p["context_call_id"],
                    "proposed_action": p["proposed_action"],
                    "candidate_label": label,
                    "label_reason": "explicit_small_failure"
                    if negative
                    else "local_effect_in_successful_task"
                    if positive
                    else "withheld_or_ambiguous",
                    "transition_refs": [t["transition_id"] for t in own],
                    "localized_evidence": own,
                }
            )
        if self.learning.get("ablation"):
            from .ablations import update_anchors

            self.anchors = update_anchors(
                self.learning["ablation"], self.anchors, self.transitions, self.task.task_id
            )
        eligible = [a for a in self.anchors if a["eligibility"]]
        calls = []
        proposals = []
        update_eligibility = None
        if self.enabled and self.learning.get("update_policy") in {
            "covered_relations",
            "task_scoped_relations",
        }:
            if self.learning.get("revision_mode") != "semantic":
                raise ValueError("covered_relations_v1_requires_semantic")
            from .update_eligibility import accumulate_evidence, classify_update

            if self.learning.get("update_policy") == "task_scoped_relations":
                from .update_gate import classify_update
            update_eligibility = classify_update(
                eligible,
                self.active_at_start,
                self.transitions,
                task_instruction=self.task.instruction,
                initial_observation=trajectory.initial_observation,
            )
            self.episode_events.append(update_eligibility)

            self.episode_events.extend(
                accumulate_evidence(self.bank, update_eligibility, self.task.task_id, episode_index)
            )
        if not self.enabled or not eligible:
            self.episode_events.append(
                {
                    "event": "induction_skipped",
                    "reason": "matched_no_skills" if not self.enabled else "zero_eligible_anchors",
                }
            )
        elif update_eligibility is not None and not update_eligibility["call_required"]:
            self.episode_events.append(
                {
                    "event": "induction_skipped",
                    "reason": self.learning["update_policy"],
                    "classification": update_eligibility["classification"],
                    "eligible_anchor_ids": [a["anchor_id"] for a in eligible],
                }
            )
        else:
            refinement_phase = bool(
                self.learning.get("phase_selected_induction") and self.active_at_start
            )

            induction_skills = [
                {k: v for k, v in r.items() if k != "evidence_accumulation"}
                for r in self.active_at_start
            ]
            induction_anchors = eligible
            if (
                update_eligibility is not None
                and self.learning.get("update_policy") == "task_scoped_relations"
            ):
                covered_ids = set(update_eligibility["covered_anchor_ids"])
                induction_anchors = [a for a in eligible if a["anchor_id"] not in covered_ids]
                omitted = [a for a in eligible if a["anchor_id"] in covered_ids]
                self.episode_events.append(
                    {
                        "event": "covered_evidence_input_omitted",
                        "anchor_ids": sorted(covered_ids),
                        "estimated_duplicate_local_evidence_tokens": estimate_tokens(
                            json.dumps(omitted, ensure_ascii=False)
                        )
                        if omitted
                        else 0,
                        "all_trajectory_transitions_retained": True,
                        "actual_token_savings_claim": False,
                    }
                )
            payload = build_induction_payload(
                trajectory, induction_anchors, self.transitions, outcome, induction_skills
            )
            if self.learning.get("revision_mode") == "semantic":
                payload.pop("task_feedback", None)
                if self.learning.get("semantic_visible_trajectory") or refinement_phase:
                    from .semantic_induction import visible_payload

                    payload = visible_payload(payload)
            if self.learning.get("induction_task_role_hints") is not None:
                from .induction_role_hints import augment_induction_payload

                payload, public_role_hint_event = augment_induction_payload(
                    payload, self.learning, self.task.benchmark
                )
                if public_role_hint_event is not None:
                    self.episode_events.append(public_role_hint_event)
            if self.learning.get("grounding_validator") == "alfworld":
                from .skill_grounding import grounding_issues

                payload["admission_defects"] = [
                    {"skill_id": r["skill_id"], "issues": grounding_issues(r)}
                    for r in self.active_at_start
                ]
            projection_passthrough = False
            if self.learning.get("trajectory_projection") == "trajectory_projection_guarded":
                from .trajectory_projection import full_context_reasons

                reasons = full_context_reasons(payload)
                projection_passthrough = bool(reasons)
                if projection_passthrough:
                    self.episode_events.append(
                        {
                            "event": "trajectory_projection",
                            "projection_version": "trajectory_projection_guarded",
                            "full_context_required": True,
                            "full_context_reasons": reasons,
                            "update_skipped": False,
                            "original_full_serialization_preserved": True,
                        }
                    )
            if (
                self.learning.get("trajectory_projection")
                in {
                    "trajectory_projection",
                    "trajectory_projection_compatible",
                    "trajectory_projection_role_bound",
                    "trajectory_projection_guarded",
                }
                and not projection_passthrough
            ):
                from .trajectory_projection import (
                    dumps,
                    project_compatible_payload,
                    project_guarded_payload,
                    project_payload,
                    project_role_bound_payload,
                )

                projector = {
                    "trajectory_projection": project_payload,
                    "trajectory_projection_compatible": project_compatible_payload,
                    "trajectory_projection_role_bound": project_role_bound_payload,
                    "trajectory_projection_guarded": project_guarded_payload,
                }[self.learning["trajectory_projection"]]
                payload, projection = projector(payload)
                self.episode_events.append(
                    {
                        "event": "trajectory_projection",
                        **projection,
                        "archive_records": [
                            "tasks.jsonl",
                            "steps.jsonl",
                            "model_calls.jsonl",
                            "method_events.jsonl",
                        ],
                    }
                )
                serialized = dumps(payload)
            elif self.learning.get("trajectory_serialization") == "lossless_reference_pool":
                from .induction_compaction import compact_payload, dumps, literal_evidence_index

                raw_payload = payload
                payload, compaction = compact_payload(
                    payload,
                    {
                        "task_id": self.task.task_id,
                        "episode_index": episode_index,
                        "records": [
                            "tasks.jsonl",
                            "steps.jsonl",
                            "model_calls.jsonl",
                            "method_events.jsonl",
                        ],
                        "join": "episode_index or task_id; call IDs and transition IDs identify records",
                    },
                )
                self.episode_events.append({"event": "trajectory_compaction", **compaction})
                if self.learning.get("literal_evidence_index"):
                    if "admission_defects" in raw_payload:
                        payload["admission_defects"] = raw_payload["admission_defects"]
                    if self.learning.get("revision_mode") == "semantic":
                        payload["local_evidence"] = raw_payload["eligible_anchors"]
                        payload["current_skill_state"] = raw_payload["current_skill_state"]
                    else:
                        payload["citation_index"] = literal_evidence_index(raw_payload)
                serialized = dumps(payload)
            else:
                serialized = json.dumps(payload, ensure_ascii=False)
            if self.learning.get("ablation"):
                from .ablations import induction_payload

                serialized = induction_payload(
                    self.learning["ablation"],
                    serialized=serialized,
                    transitions=self.transitions,
                    anchors=eligible,
                    task=self.task,
                    outcome=outcome,
                    skills=self.active_at_start,
                )
            induction_system = (
                self.prompt_builder.refinement_system
                if refinement_phase
                else self.prompt_builder.induction_system
            )
            if not induction_system:
                raise ValueError("missing_frozen_induction_prompt")
            if self.learning.get("protocol_repair_version") == "i2s_skills_likelihood":
                from .induction_protocol import induction_system as native_induction_system

                induction_system = native_induction_system(
                    self.prompt_builder.induction_system, self.task.benchmark
                )
            if (
                self.learning.get("trajectory_projection")
                in {
                    "trajectory_projection",
                    "trajectory_projection_compatible",
                    "trajectory_projection_role_bound",
                    "trajectory_projection_guarded",
                }
                and not projection_passthrough
            ):
                from .trajectory_projection import projected_system

                induction_system = projected_system(induction_system)
            if self.learning.get("induction_task_role_hints") is not None:
                from .induction_role_hints import clarify_induction_system

                induction_system = clarify_induction_system(
                    induction_system, self.learning, self.task.benchmark
                )
            messages = (
                {"role": "system", "content": induction_system},
                {"role": "user", "content": serialized},
            )
            effective = self.model_pool.generation_for(self.large_model_id, generation)
            from .induction_protocol import induction_generation, induction_uses_schema

            effective = induction_generation(effective, self.learning)
            tokens = estimate_tokens("\n".join(m["content"] for m in messages))
            if tokens > int(effective["max_input_tokens"]):
                self.episode_events.append(
                    {
                        "event": "induction_skipped",
                        "reason": "complete_trajectory_exceeds_input_budget",
                        "estimated_input_tokens": tokens,
                    }
                )
            else:
                position = ModelCallPosition(episode_index, max_steps, 7)
                request = allocate_request_generation(effective, estimated_input_tokens=tokens)
                if induction_uses_schema(self.learning, refinement_phase):
                    schema = (
                        self.prompt_builder.refinement_schema
                        if refinement_phase
                        else self.prompt_builder.induction_schema
                    )
                    if not schema:
                        raise ValueError("missing_frozen_induction_schema")

                    def transport_schema(x):
                        if isinstance(x, dict):
                            return {
                                k: transport_schema(v) for k, v in x.items() if k != "uniqueItems"
                            }
                        return [transport_schema(v) for v in x] if isinstance(x, list) else x

                    request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "grounded_induction",
                            "strict": True,
                            "schema": transport_schema(schema),
                        },
                    }
                try:
                    call = self.model_pool.generate(
                        self.large_model_id,
                        messages,
                        "skill_induction_refinement",
                        request,
                        position=position,
                    )
                except ModelInvocationError as exc:
                    call = exc.call
                    outcome["method_update_status"] = "failed"
                    outcome["method_update_error"] = str(exc)
                calls.append(PositionedModelCall(call, position))
                event = {
                    "event": "joint_induction",
                    "model_call_ids": [call.call_id],
                    "eligible_anchor_ids": [a["anchor_id"] for a in eligible],
                    "generation_calls": 1,
                }
                if self.learning.get("phase_selected_induction"):
                    event["induction_prompt_phase"] = (
                        "D3_existing_state" if refinement_phase else "D1_empty_state"
                    )
                event["request_status"] = call.request_status
                try:
                    if self.learning.get("revision_mode") == "semantic":
                        from .semantic_induction import parse_output

                        induction_output = call.raw_output or ""
                        if self.learning.get("protocol_repair_version") == "i2s_skills_likelihood":
                            from .induction_protocol import final_json_text

                            induction_output = final_json_text(induction_output)
                        proposals = parse_output(induction_output)
                    else:
                        value = json.loads(call.raw_output or "")
                        if (
                            not isinstance(value, dict)
                            or set(value) != {"proposals"}
                            or not isinstance(value["proposals"], list)
                        ):
                            raise ValueError("invalid_proposals_schema")
                        proposals = value["proposals"]
                    event.update(parse_status="ok", proposal_count=len(proposals))
                except ValueError as exc:
                    event.update(parse_status="failed", parse_error=str(exc), proposal_count=0)
                self.episode_events.append(event)
        if self.enabled:
            if self.learning.get("admission_reference_repair") is not None:
                from .induction_admission_repair import (
                    PROTOCOL as reference_repair_protocol,
                )
                from .induction_admission_repair import (
                    recover_proposals,
                )

                if self.learning["admission_reference_repair"] != reference_repair_protocol:
                    raise ValueError("unsupported_admission_reference_repair")
                proposals, reference_repair_events = recover_proposals(
                    self.bank,
                    proposals,
                    eligible,
                    self.transitions,
                    self.task.task_id,
                    episode_index,
                    self.task.instruction,
                )
                for reference_repair_event in reference_repair_events:
                    reference_repair_event["source_model_call_ids"] = [
                        item.call.call_id for item in calls
                    ]
                self.episode_events.extend(reference_repair_events)
            self.episode_events.extend(
                self.bank.apply(
                    proposals,
                    eligible,
                    self.transitions,
                    self.task.task_id,
                    episode_index,
                    self.task.instruction,
                )
            )
        self.episode_events.append({"event": "learning_task_end", "state": self.state_dict()})
        return PolicyEpisodeUpdate(model_calls=tuple(calls), events=tuple(self.episode_events))
