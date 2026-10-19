from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .models import ModelPool
from .policy_support import (
    allocate_request_generation,
    constrain_current_action,
    parse_action_output,
)
from .prompts import PromptBuild, PromptBuilder
from .schemas import ActionDecision, ModelCallPosition, PolicyState, StepResult


@dataclass(frozen=True)
class RepairRequest:
    trigger: str
    previous_action: str | None = None
    previous_response: str | None = None
    environment_feedback: str | None = None
    confidence_reason: str | None = None
    origin_trigger: str | None = None
    intervention_id: str | None = None
    intervention_action_index: int = 1


class RepairController:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        large_model_id: str,
        model_pool: ModelPool,
        prompt_builder: PromptBuilder,
        parser: str,
        max_action_chars: int,
    ) -> None:
        self.config = dict(config)
        self.large_model_id = str(large_model_id)
        self.model_pool = model_pool
        self.prompt_builder = prompt_builder
        self.parser = str(parser)
        self.max_action_chars = int(max_action_chars)
        raw_external_context = self.config.get("include_external_context", False)
        if not isinstance(raw_external_context, bool):
            raise ValueError("repair.include_external_context must be boolean")
        self.include_external_context = raw_external_context
        raw_parse_retry = self.config.get("parse_retry_once", False)
        if not isinstance(raw_parse_retry, bool):
            raise ValueError("repair.parse_retry_once must be boolean")
        self.parse_retry_once = raw_parse_retry
        raw_bootstrap_samples = self.config.get("bootstrap_consensus_samples", 1)
        if (
            isinstance(raw_bootstrap_samples, bool)
            or not isinstance(raw_bootstrap_samples, int)
            or raw_bootstrap_samples not in {1, 3}
        ):
            raise ValueError("repair.bootstrap_consensus_samples must be 1 or 3")
        self.bootstrap_consensus_samples = raw_bootstrap_samples
        raw_triggers = dict(self.config.get("triggers") or {})
        self.parse_failure_enabled = bool(raw_triggers.get("parse_failure", True))
        self.invalid_action_enabled = bool(raw_triggers.get("invalid_action", True))
        self.no_progress_enabled = bool(raw_triggers.get("no_progress", False))
        self.visible_failure_markers = tuple(
            str(item).strip().lower()
            for item in raw_triggers.get("visible_failure_markers", [])
            if str(item).strip()
        )
        raw_budget = raw_triggers.get("small_step_budget")
        self.small_step_budget = None if raw_budget is None else int(raw_budget)
        self.max_actions_per_episode = int(self.config.get("max_actions_per_episode", 3))
        if self.max_actions_per_episode <= 0:
            raise ValueError("repair max_actions_per_episode must be positive")
        self.max_actions_per_intervention = int(self.config.get("max_actions_per_intervention", 1))
        if self.max_actions_per_intervention <= 0:
            raise ValueError("repair max_actions_per_intervention must be positive")
        if self.max_actions_per_intervention > self.max_actions_per_episode:
            raise ValueError(
                "repair max_actions_per_intervention cannot exceed max_actions_per_episode"
            )
        raw_continue_after_rejection = self.config.get("continue_after_rejected_action", False)
        if not isinstance(raw_continue_after_rejection, bool):
            raise ValueError("repair.continue_after_rejected_action must be boolean")
        self.continue_after_rejected_action = raw_continue_after_rejection

    def enabled(self, trigger: str) -> bool:
        return {
            "parse_failure": self.parse_failure_enabled,
            "invalid_action": self.invalid_action_enabled,
            "no_progress": self.no_progress_enabled,
            "visible_failure": bool(self.visible_failure_markers),
            "small_step_budget": self.small_step_budget is not None,
        }.get(trigger, False)

    @staticmethod
    def _instruction(request: RepairRequest) -> str:
        proposal = (
            f" Proposed action: {request.previous_action}." if request.previous_action else ""
        )
        concern = (
            f" The small model's concern: {request.confidence_reason}"
            if request.confidence_reason
            else ""
        )
        if request.trigger == "parse_failure":
            return (
                "The previous response did not provide an executable action."
                " Choose a valid next action for the current state."
            )
        if request.trigger == "invalid_action":
            return (
                "The previous action was rejected by the environment. "
                "Choose a different valid next action for the current state."
            )
        if request.trigger == "no_progress":
            return (
                "The previous action did not change the visible state. "
                "Choose a next action that makes progress."
            )
        if request.trigger == "visible_failure":
            return (
                "The latest environment response indicates that the previous "
                "attempt failed. Choose a next action that recovers from it."
            )
        if request.trigger == "low_confidence":
            return (
                "The small model reported low confidence in its proposed action. "
                "Choose the most reliable valid next action for the current state."
                + proposal
                + concern
            )
        if request.trigger == "model_requested_large":
            return (
                "The small model explicitly requested help with this decision. "
                "Choose the most reliable valid next action for the current state."
                + proposal
                + concern
            )
        if request.trigger == "confidence_assessment_error":
            return (
                "The small model did not provide a usable confidence assessment. "
                "Choose the most reliable valid next action for the current state."
            )
        if request.trigger == "skill_bootstrap":
            return (
                "No reusable procedure has yet been acquired for this task family. "
                "Demonstrate the most reliable valid next action from the current "
                "task instruction and visible state."
            )
        if request.trigger == "repair_followup":
            return (
                "Continue resolving the current task from the latest visible "
                "result. Choose the next valid action."
            )
        return "Review the current state and choose the most useful next action."

    def classify_transition(
        self,
        *,
        state: PolicyState,
        decision: ActionDecision,
        transition: StepResult,
        consecutive_small_steps: int,
    ) -> RepairRequest | None:
        if bool(transition.success) or bool(transition.done):
            return None
        visible_feedback = str(transition.policy_observation or transition.observation)
        if not transition.action_valid and self.invalid_action_enabled:
            return RepairRequest(
                trigger="invalid_action",
                previous_action=decision.action,
                environment_feedback=visible_feedback,
            )
        lowered = visible_feedback.lower()
        if self.visible_failure_markers and any(
            marker in lowered for marker in self.visible_failure_markers
        ):
            return RepairRequest(
                trigger="visible_failure",
                previous_action=decision.action,
                environment_feedback=visible_feedback,
            )
        if self.no_progress_enabled and visible_feedback.strip() == state.observation.strip():
            return RepairRequest(
                trigger="no_progress",
                previous_action=decision.action,
                environment_feedback=visible_feedback,
            )
        if self.small_step_budget is not None and consecutive_small_steps >= self.small_step_budget:
            return RepairRequest(
                trigger="small_step_budget",
                previous_action=decision.action,
                environment_feedback=visible_feedback,
            )
        return None

    def repair(
        self,
        state: PolicyState,
        request: RepairRequest,
        *,
        context_manager: Any,
        generation: Mapping[str, Any],
        position: ModelCallPosition,
    ) -> tuple[ActionDecision, PromptBuild]:
        effective_generation = self.model_pool.generation_for(self.large_model_id, generation)
        prompt = self.prompt_builder.build(
            state,
            context_manager,
            max_input_tokens=int(effective_generation["max_input_tokens"]),
            additional_instruction=(
                self._instruction(request)
                + (
                    " Original small-model proposal: "
                    + str(request.previous_action or request.previous_response)
                    if self.config.get("learning_include_proposal")
                    else ""
                )
            ),
            include_external_context=self.include_external_context,
        )
        request_generation = allocate_request_generation(
            effective_generation,
            estimated_input_tokens=prompt.estimated_input_tokens,
        )
        request_generation = constrain_current_action(
            request_generation, state, self.prompt_builder.prompt_contract
        )
        calls = []
        positions = []
        parsed: list[tuple[str | None, str | None, str | None, str | None]] = []
        sample_count = (
            self.bootstrap_consensus_samples if request.trigger == "skill_bootstrap" else 1
        )
        for sample_index in range(sample_count):
            sample_position = replace(position, call_slot=position.call_slot + sample_index)
            try:
                sample_call = self.model_pool.generate(
                    self.large_model_id,
                    prompt.messages,
                    "repair_action",
                    request_generation,
                    position=sample_position,
                )
            except Exception as exc:
                if hasattr(exc, "call"):
                    exc.call_slot = sample_position.call_slot
                    exc.prior_model_calls = tuple(calls)
                    exc.prior_model_call_positions = tuple(positions)
                raise
            calls.append(sample_call)
            positions.append(sample_position)
            parsed.append(
                parse_action_output(
                    sample_call.raw_output or "",
                    benchmark=state.benchmark,
                    parser=self.parser,
                    max_action_chars=self.max_action_chars,
                )
            )

        selected_action_call_id: str | None = None
        if sample_count == 3:
            counts = Counter(
                action
                for _raw, action, _mode, error in parsed
                if error is None and action is not None
            )
            winner = next(
                (action for action, count in counts.items() if count >= 2),
                None,
            )
            if winner is None:
                raw_action = None
                action = None
                parse_mode = None
                parse_error = "bootstrap_consensus_failed"
            else:
                selected_index = max(
                    index
                    for index, (_raw, candidate, _mode, error) in enumerate(parsed)
                    if error is None and candidate == winner
                )
                raw_action, action, parse_mode, parse_error = parsed[selected_index]
                selected_action_call_id = calls[selected_index].call_id
        else:
            raw_action, action, parse_mode, parse_error = parsed[0]

        if sample_count == 1 and parse_error is not None and self.parse_retry_once:
            retry_position = replace(position, call_slot=4)
            try:
                retry_call = self.model_pool.generate(
                    self.large_model_id,
                    prompt.messages,
                    "repair_action",
                    request_generation,
                    position=retry_position,
                )
            except Exception as exc:
                if hasattr(exc, "call"):
                    exc.call_slot = retry_position.call_slot
                    exc.prior_model_calls = tuple(calls)
                    exc.prior_model_call_positions = tuple(positions)
                raise
            calls.append(retry_call)
            positions.append(retry_position)
            raw_action, action, parse_mode, parse_error = parse_action_output(
                retry_call.raw_output or "",
                benchmark=state.benchmark,
                parser=self.parser,
                max_action_chars=self.max_action_chars,
            )
        return (
            ActionDecision(
                raw_action=raw_action,
                action=action,
                parse_mode=parse_mode,
                parse_error=parse_error,
                source_model=self.large_model_id,
                model_calls=tuple(calls),
                intervention="repair",
                model_call_positions=tuple(positions),
                trigger=request.trigger,
                repair_intervention_id=request.intervention_id,
                repair_origin_trigger=(request.origin_trigger or request.trigger),
                repair_intervention_action_index=(request.intervention_action_index),
                selected_action_call_id=selected_action_call_id,
            ),
            prompt,
        )
