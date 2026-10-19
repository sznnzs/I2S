from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .confidence import ConfidenceEvidence, ConfidenceRouter
from .models import ModelPool
from .policy_support import (
    allocate_request_generation,
    constrain_current_action,
    parse_action_output,
)
from .prompts import PromptBuild, PromptBuilder
from .schemas import (
    ActionDecision,
    CompletedTrajectory,
    ModelCall,
    ModelCallPosition,
    PolicyEpisodeUpdate,
    PolicyState,
    StepResult,
    TaskSpec,
)


class AgentPolicy:
    @property
    def model_ids(self) -> tuple[str, ...]:
        raise NotImplementedError

    def start_episode(
        self,
        task: TaskSpec,
        *,
        episode_index: int,
        initial_observation: str,
        context_manager: Any,
    ) -> None:
        del task, episode_index, initial_observation, context_manager

    def observe(
        self,
        *,
        state: PolicyState,
        decision: ActionDecision,
        transition: StepResult,
    ) -> None:
        del state, decision, transition

    def continue_after_transition(
        self,
        *,
        state: PolicyState,
        decision: ActionDecision,
        transition: StepResult,
    ) -> bool:
        del state, decision, transition
        return False

    def finish_episode(
        self,
        *,
        episode_index: int,
        outcome: Mapping[str, Any],
        trajectory: CompletedTrajectory,
        context_manager: Any,
        generation: Mapping[str, Any],
        max_steps: int,
    ) -> PolicyEpisodeUpdate:
        del (
            episode_index,
            outcome,
            trajectory,
            context_manager,
            generation,
            max_steps,
        )
        return PolicyEpisodeUpdate()

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def skill_records(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value:
            raise ValueError("this policy has no persistent state")


class InterventionPolicy(AgentPolicy):
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        repair_config: Mapping[str, Any],
        router_config: Mapping[str, Any] | None = None,
        model_pool: ModelPool,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.config = dict(config)
        self.model_id = str(config["model_id"])
        self.parser = str(config.get("action_parser", "json_only"))
        self.max_action_chars = int(config.get("max_action_chars", 256))
        self.model_pool = model_pool
        self.prompt_builder = prompt_builder
        self.confidence_router = ConfidenceRouter(dict(router_config or {}))

        from .repair import RepairController

        self.reliability_model_id = str(
            dict(router_config or {}).get("reliability_model_id") or self.model_id
        )
        self.large_model_id = str(config["large_model_id"])
        self.repair_controller = RepairController(
            repair_config,
            large_model_id=self.large_model_id,
            model_pool=model_pool,
            prompt_builder=prompt_builder,
            parser=self.parser,
            max_action_chars=self.max_action_chars,
        )
        self.pending_repair: Any = None
        self.consecutive_small_steps = 0
        self.repair_count = 0
        self.repair_intervention_count = 0
        self.current_episode_index = 0
        self.last_applied_repair_request: Any = None
        self.episode_events: list[Mapping[str, Any]] = []

    def _propose_action(
        self,
        state: PolicyState,
        *,
        context_manager: Any,
        generation: Mapping[str, Any],
        call_position: ModelCallPosition | None = None,
    ) -> tuple[ActionDecision, PromptBuild]:
        effective_generation = self.model_pool.generation_for(self.model_id, generation)
        prompt = self.prompt_builder.build(
            state,
            context_manager,
            max_input_tokens=int(effective_generation["max_input_tokens"]),
        )
        request_generation = allocate_request_generation(
            effective_generation,
            estimated_input_tokens=prompt.estimated_input_tokens,
        )
        request_generation = constrain_current_action(
            request_generation, state, self.prompt_builder.prompt_contract
        )
        call = self.model_pool.generate(
            self.model_id,
            prompt.messages,
            "environment_action",
            request_generation,
            position=call_position,
        )
        raw_action, action, parse_mode, parse_error = parse_action_output(
            call.raw_output or "",
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
                source_model=self.model_id,
                model_calls=(call,),
                intervention="none",
                model_call_positions=((call_position,) if call_position is not None else ()),
                confidence_route="not_assessed",
            ),
            prompt,
        )

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.model_id,
                    self.large_model_id,
                    self.reliability_model_id,
                )
            )
        )

    def start_episode(
        self,
        task: TaskSpec,
        *,
        episode_index: int,
        initial_observation: str,
        context_manager: Any,
    ) -> None:
        del task, initial_observation, context_manager
        self.pending_repair = None
        self.consecutive_small_steps = 0
        self.repair_count = 0
        self.repair_intervention_count = 0
        self.current_episode_index = int(episode_index)
        self.last_applied_repair_request = None
        self.episode_events = []

    def _start_repair_intervention(self, request: Any) -> Any:
        self.repair_intervention_count += 1
        return replace(
            request,
            origin_trigger=(request.origin_trigger or request.trigger),
            intervention_id=(
                f"repair-{self.current_episode_index:05d}-{self.repair_intervention_count:03d}"
            ),
            intervention_action_index=1,
        )

    @staticmethod
    def _repair_event_fields(request: Any) -> dict[str, Any]:
        return {
            "trigger": request.trigger,
            "origin_trigger": request.origin_trigger or request.trigger,
            "repair_intervention_id": request.intervention_id,
            "repair_intervention_action_index": (request.intervention_action_index),
        }

    @staticmethod
    def _repair_position(position: ModelCallPosition | None) -> ModelCallPosition:
        if position is None:
            raise ValueError("repair policy requires a logical call position")
        return ModelCallPosition(
            episode_index=position.episode_index,
            step_index=position.step_index,
            call_slot=3,
        )

    @staticmethod
    def _confidence_position(
        position: ModelCallPosition | None,
    ) -> ModelCallPosition:
        if position is None:
            raise ValueError("confidence routing requires a logical call position")
        return ModelCallPosition(
            episode_index=position.episode_index,
            step_index=position.step_index,
            call_slot=2,
        )

    def _confidence_threshold(self) -> float:
        return self.confidence_router.threshold

    def _call_confidence(
        self,
        state: PolicyState,
        decision: ActionDecision,
        *,
        context_manager: Any,
        threshold: float,
        generation: Mapping[str, Any],
        call_position: ModelCallPosition | None,
    ) -> tuple[ConfidenceEvidence, ModelCall]:
        if decision.action is None or decision.parse_error is not None:
            raise ValueError("confidence assessment requires a parsed action")
        effective_generation = self.model_pool.generation_for(self.reliability_model_id, generation)
        prompt = self.prompt_builder.build_confidence(
            state,
            proposed_action=str(decision.raw_action or decision.action),
            threshold=threshold,
            max_input_tokens=int(effective_generation["max_input_tokens"]),
            context_manager=context_manager,
            include_external_context=(self.confidence_router.include_external_context),
            include_episode_history=(self.confidence_router.include_episode_history),
        )
        request_generation = allocate_request_generation(
            effective_generation,
            estimated_input_tokens=prompt.estimated_input_tokens,
        )
        position = self._confidence_position(call_position)
        try:
            call = self.model_pool.generate(
                self.reliability_model_id,
                prompt.messages,
                "confidence_assessment",
                request_generation,
                position=position,
            )
        except Exception as exc:
            if hasattr(exc, "call"):
                exc.call_slot = position.call_slot
                exc.prior_model_calls = decision.model_calls
                exc.prior_model_call_positions = decision.model_call_positions
            raise
        return self.confidence_router.evidence(call.raw_output or ""), call

    def _call_repair(
        self,
        state: PolicyState,
        request: Any,
        *,
        context_manager: Any,
        generation: Mapping[str, Any],
        call_position: ModelCallPosition | None,
    ) -> tuple[ActionDecision, PromptBuild]:
        position = self._repair_position(call_position)
        try:
            teacher_controller = self.repair_controller
            if (getattr(self, "learning", None) or {}).get(
                "teacher_observation_dedup"
            ) == "teacher_observation":
                from .teacher_observation import teacher_repair_controller

                teacher_controller = teacher_repair_controller(
                    self.repair_controller,
                    receipt_sink=self.episode_events.append,
                    large_model_id=self.large_model_id,
                    episode_index=position.episode_index,
                    step_index=position.step_index,
                    call_slot=position.call_slot,
                )
            return teacher_controller.repair(
                state,
                request,
                context_manager=context_manager,
                generation=generation,
                position=position,
            )
        except Exception as exc:
            if hasattr(exc, "call"):
                if not hasattr(exc, "call_slot"):
                    exc.call_slot = position.call_slot
            raise

    def act(
        self,
        state: PolicyState,
        *,
        context_manager: Any,
        generation: Mapping[str, Any],
        call_position: ModelCallPosition | None = None,
    ) -> tuple[ActionDecision, PromptBuild]:
        confidence_threshold = self._confidence_threshold()
        if self.pending_repair is not None:
            request = self.pending_repair
            self.pending_repair = None
            self.last_applied_repair_request = request
            decision, prompt = self._call_repair(
                state,
                request,
                context_manager=context_manager,
                generation=generation,
                call_position=call_position,
            )
            self.consecutive_small_steps = 0
            self.repair_count += 1
            self.episode_events.append(
                {
                    "event": "repair_applied",
                    "step_index": state.step_index,
                    "source_model": self.large_model_id,
                    **self._repair_event_fields(request),
                }
            )
            return replace(
                decision,
                confidence_threshold=confidence_threshold,
                confidence_route=(
                    "large_followup" if request.trigger == "repair_followup" else "large_passive"
                ),
            ), prompt

        small_decision, small_prompt = self._propose_action(
            state,
            context_manager=context_manager,
            generation=generation,
            call_position=call_position,
        )
        self.consecutive_small_steps += 1
        self.last_applied_repair_request = None
        active_trigger: str | None
        if small_decision.parse_error and self.repair_controller.parse_failure_enabled:
            active_trigger = "parse_failure"
            assessed_decision = replace(
                small_decision,
                confidence_threshold=confidence_threshold,
                confidence_route="large",
            )
        elif small_decision.parse_error is None and small_decision.action is not None:
            evidence, confidence_call = self._call_confidence(
                state,
                small_decision,
                context_manager=context_manager,
                threshold=confidence_threshold,
                generation=generation,
                call_position=call_position,
            )
            active_trigger = self.confidence_router.trigger(
                evidence, threshold=confidence_threshold
            )
            assessed_decision = replace(
                small_decision,
                model_calls=(*small_decision.model_calls, confidence_call),
                model_call_positions=(
                    *small_decision.model_call_positions,
                    self._confidence_position(call_position),
                ),
                reported_confidence=evidence.reported,
                confidence_error=evidence.parse_error,
                confidence_threshold=confidence_threshold,
                confidence_route=("large" if active_trigger else "small"),
                request_large_model=evidence.request_large_model,
                confidence_reason=evidence.reason,
            )
        else:
            active_trigger = None
            assessed_decision = replace(
                small_decision,
                confidence_threshold=confidence_threshold,
                confidence_route="small",
            )
        if small_decision.parse_error is None and small_decision.action is not None:
            can_apply_active_repair = bool(
                active_trigger is not None
                and self.repair_count < self.repair_controller.max_actions_per_episode
            )
            self.episode_events.append(
                {
                    "event": "router_decision",
                    "step_index": state.step_index,
                    "reported_confidence": assessed_decision.reported_confidence,
                    "confidence_error": assessed_decision.confidence_error,
                    "threshold": confidence_threshold,
                    "request_large_model": assessed_decision.request_large_model,
                    "reason": assessed_decision.confidence_reason,
                    "route": (
                        "large"
                        if can_apply_active_repair
                        else "small_repair_limit"
                        if active_trigger
                        else "small"
                    ),
                    "trigger": active_trigger,
                }
            )
        if (
            active_trigger is not None
            and self.repair_count < self.repair_controller.max_actions_per_episode
        ):
            from .repair import RepairRequest

            request = self._start_repair_intervention(
                RepairRequest(
                    trigger=active_trigger,
                    previous_action=assessed_decision.action,
                    previous_response=(
                        small_decision.model_calls[-1].raw_output
                        if active_trigger == "parse_failure"
                        else None
                    ),
                    confidence_reason=assessed_decision.confidence_reason,
                )
            )
            self.last_applied_repair_request = request
            try:
                repair_decision, repair_prompt = self._call_repair(
                    state,
                    request,
                    context_manager=context_manager,
                    generation=generation,
                    call_position=call_position,
                )
            except Exception as exc:
                if hasattr(exc, "call"):
                    repair_prior_calls = tuple(getattr(exc, "prior_model_calls", ()))
                    repair_prior_positions = tuple(getattr(exc, "prior_model_call_positions", ()))
                    exc.prior_model_calls = *assessed_decision.model_calls, *repair_prior_calls
                    exc.prior_model_call_positions = (
                        *assessed_decision.model_call_positions,
                        *repair_prior_positions,
                    )
                raise
            self.consecutive_small_steps = 0
            self.repair_count += 1
            self.episode_events.append(
                {
                    "event": "repair_applied",
                    "step_index": state.step_index,
                    "source_model": self.large_model_id,
                    **self._repair_event_fields(request),
                }
            )
            combined = ActionDecision(
                raw_action=repair_decision.raw_action,
                action=repair_decision.action,
                parse_mode=repair_decision.parse_mode,
                parse_error=repair_decision.parse_error,
                source_model=repair_decision.source_model,
                model_calls=(
                    *assessed_decision.model_calls,
                    *repair_decision.model_calls,
                ),
                intervention="repair",
                model_call_positions=(
                    *assessed_decision.model_call_positions,
                    *repair_decision.model_call_positions,
                ),
                trigger=active_trigger,
                reported_confidence=assessed_decision.reported_confidence,
                confidence_error=assessed_decision.confidence_error,
                confidence_threshold=confidence_threshold,
                confidence_route="large",
                request_large_model=assessed_decision.request_large_model,
                confidence_reason=assessed_decision.confidence_reason,
                repair_intervention_id=(repair_decision.repair_intervention_id),
                repair_origin_trigger=repair_decision.repair_origin_trigger,
                repair_intervention_action_index=(repair_decision.repair_intervention_action_index),
            )
            return combined, replace(repair_prompt, proposal=small_prompt)
        if active_trigger is not None:
            self.episode_events.append(
                {
                    "event": "repair_limit_reached",
                    "step_index": state.step_index,
                    "trigger": active_trigger,
                }
            )
            assessed_decision = replace(
                assessed_decision,
                confidence_route="small_repair_limit",
                trigger=active_trigger,
            )
        return assessed_decision, small_prompt

    def observe(
        self,
        *,
        state: PolicyState,
        decision: ActionDecision,
        transition: StepResult,
    ) -> None:
        if decision.intervention == "repair":
            self.pending_repair = None
            request = self.last_applied_repair_request
            can_continue = bool(
                request is not None
                and (
                    transition.action_valid or self.repair_controller.continue_after_rejected_action
                )
                and not transition.success
                and not transition.done
                and request.intervention_action_index
                < self.repair_controller.max_actions_per_intervention
                and self.repair_count < self.repair_controller.max_actions_per_episode
            )
            if can_continue:
                from .repair import RepairRequest

                assert request is not None
                self.pending_repair = RepairRequest(
                    trigger="repair_followup",
                    previous_action=decision.action,
                    environment_feedback=str(
                        transition.policy_observation or transition.observation
                    ),
                    origin_trigger=(request.origin_trigger or request.trigger),
                    intervention_id=request.intervention_id,
                    intervention_action_index=(request.intervention_action_index + 1),
                )
                self.episode_events.append(
                    {
                        "event": "repair_burst_continued",
                        "step_index": state.step_index,
                        **self._repair_event_fields(self.pending_repair),
                    }
                )
                return
            if transition.success or transition.done:
                handoff_reason = "task_terminal"
            elif not transition.action_valid:
                handoff_reason = "repair_action_rejected"
            elif (
                request is not None
                and request.intervention_action_index
                >= self.repair_controller.max_actions_per_intervention
            ):
                handoff_reason = "intervention_limit"
            else:
                handoff_reason = "episode_limit"
            self.episode_events.append(
                {
                    "event": "repair_handoff_to_small",
                    "step_index": state.step_index,
                    "reason": handoff_reason,
                    **(self._repair_event_fields(request) if request is not None else {}),
                }
            )
            return
        request = self.repair_controller.classify_transition(
            state=state,
            decision=decision,
            transition=transition,
            consecutive_small_steps=self.consecutive_small_steps,
        )
        if request is None:
            return
        if self.repair_count >= self.repair_controller.max_actions_per_episode:
            self.episode_events.append(
                {
                    "event": "repair_limit_reached",
                    "step_index": state.step_index,
                    "trigger": request.trigger,
                }
            )
            return
        self.pending_repair = self._start_repair_intervention(request)
        self.episode_events.append(
            {
                "event": "repair_scheduled",
                "step_index": state.step_index,
                "previous_action": self.pending_repair.previous_action,
                **self._repair_event_fields(self.pending_repair),
            }
        )

    def continue_after_transition(
        self,
        *,
        state: PolicyState,
        decision: ActionDecision,
        transition: StepResult,
    ) -> bool:
        del state

        rejected_repair_handoff = bool(
            decision.intervention == "repair"
            and not transition.action_valid
            and not transition.done
        )
        return self.pending_repair is not None or rejected_repair_handoff

    def finish_episode(
        self,
        *,
        episode_index: int,
        outcome: Mapping[str, Any],
        trajectory: CompletedTrajectory,
        context_manager: Any,
        generation: Mapping[str, Any],
        max_steps: int,
    ) -> PolicyEpisodeUpdate:
        del (
            episode_index,
            outcome,
            trajectory,
            context_manager,
            generation,
            max_steps,
        )
        return PolicyEpisodeUpdate(events=tuple(self.episode_events))


def build_policy(
    config: Mapping[str, Any],
    *,
    repair_config: Mapping[str, Any] | None,
    router_config: Mapping[str, Any] | None = None,
    model_pool: ModelPool,
    prompt_builder: PromptBuilder,
) -> AgentPolicy:
    if config.get("type", "i2s") != "i2s":
        raise ValueError("policy.type must be i2s")
    from .agent import I2SPolicy

    return I2SPolicy(
        config,
        repair_config=dict(repair_config or {}),
        router_config=dict(router_config or {}),
        model_pool=model_pool,
        prompt_builder=prompt_builder,
    )
