from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Mapping

from .artifact_schemas import (
    EPISODE_RECORD,
    METHOD_EVENT,
    MODEL_CALL,
    STEP_RECORD,
    TASK_RECORD,
)
from .benchmarks.client import BenchmarkWorkerClient
from .context import ContextManager, build_context_manager
from .models import ModelInvocationError
from .policy import AgentPolicy
from .prompts import PromptBuild
from .recording import now_iso
from .schemas import (
    CompletedTrajectory,
    ModelCallPosition,
    PolicyState,
    StepResult,
    TaskSpec,
    TrajectoryStep,
    VisibleTransition,
)


def _model_visible_environment_response(transition: StepResult) -> str:
    return str(transition.policy_observation or transition.observation)


def _prompt_context_record(prompt: PromptBuild, *, model_call_id: str) -> dict[str, Any]:
    context = prompt.context
    return {
        "model_call_id": str(model_call_id),
        "estimated_input_tokens": int(prompt.estimated_input_tokens),
        "admissible_action_count": int(prompt.admissible_action_count),
        "shown_action_count": int(prompt.shown_action_count),
        "dropped_history_steps": int(context.dropped_history_steps),
        "external_context_estimated_tokens": int(context.external_context_tokens),
        "external_context_type": str(context.external_context_type),
        "external_available_item_count": int(context.available_external_item_count),
        "external_selected_item_count": int(context.selected_external_item_count),
        "external_selected_item_ids": list(context.selected_external_item_ids),
        "context_state_version": int(context.state_version),
        "observation_truncated": bool(prompt.observation_truncated),
        "actions_truncated": bool(prompt.actions_truncated),
    }


@dataclass(frozen=True)
class EpisodeRequest:
    sequence_index: int
    episode_index: int
    episode_id: str
    selected_task: TaskSpec
    environment_seed: int


@dataclass(frozen=True)
class EpisodeBundle:
    request: EpisodeRequest
    task_row: Mapping[str, Any]
    episode_row: Mapping[str, Any]
    step_rows: tuple[Mapping[str, Any], ...]
    model_call_rows: tuple[Mapping[str, Any], ...]
    method_event_rows: tuple[Mapping[str, Any], ...]
    error_rows: tuple[Mapping[str, Any], ...]
    visible_trajectory: CompletedTrajectory

    @property
    def infrastructure_failure(self) -> bool:
        return (
            str(self.episode_row.get("status") or "")
            in {"reset_error", "environment_error", "model_error"}
            or self.episode_row.get("method_update_status") == "failed"
        )


def _task_row(
    task: TaskSpec,
    *,
    episode_id: str,
    environment_seed: int,
    initial_observation: str | None = None,
    initial_admissible_actions: tuple[str, ...] = (),
    initial_evaluation_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema": TASK_RECORD,
        "episode_id": episode_id,
        "environment_seed": int(environment_seed),
        **task.to_dict(),
    }
    if initial_observation is not None:
        row.update(
            {
                "initial_observation": initial_observation,
                "initial_admissible_actions": list(initial_admissible_actions),
                "initial_evaluation_info": dict(initial_evaluation_info or {}),
            }
        )
    return row


def _model_call_row(
    call: Any,
    *,
    request: EpisodeRequest,
    step_index: int,
    call_slot: int,
) -> dict[str, Any]:
    return {
        "schema": MODEL_CALL,
        "episode_id": request.episode_id,
        "episode_index": request.episode_index,
        "step_index": step_index,
        "call_slot": call_slot,
        **call.to_dict(),
    }


class EpisodeExecutor:
    def __init__(
        self,
        *,
        worker: BenchmarkWorkerClient,
        policy: AgentPolicy,
        benchmark_config: Mapping[str, Any],
        context_config: Mapping[str, Any],
        generation: Mapping[str, Any],
        seeds: Mapping[str, Any],
    ) -> None:
        self.worker = worker
        self.policy = policy
        self.benchmark_config = dict(benchmark_config)
        self.context_config = dict(context_config)
        self.generation = dict(generation)
        self.seeds = dict(seeds)

    def execute(
        self,
        request: EpisodeRequest,
        *,
        context_manager: ContextManager | None = None,
    ) -> EpisodeBundle:
        context = context_manager or build_context_manager(self.context_config)
        started = now_iso()
        status = "reset_error"
        success = False
        step_count = 0
        total_reward = 0.0
        native_metrics: dict[str, Any] = {}
        error: str | None = None
        task = request.selected_task
        task_recorded = False
        initial_observation: str | None = None
        task_row: dict[str, Any] | None = None
        step_rows: list[dict[str, Any]] = []
        model_call_rows: list[dict[str, Any]] = []
        method_event_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        trajectory_steps: list[TrajectoryStep] = []
        policy_started = False
        try:
            reset = self.worker.reset(
                request.selected_task,
                environment_seed=request.environment_seed,
            )
            task = reset.task
            initial_observation = reset.observation
            native_metrics = dict(reset.evaluation_info)
            task_row = _task_row(
                task,
                episode_id=request.episode_id,
                environment_seed=request.environment_seed,
                initial_observation=reset.observation,
                initial_admissible_actions=reset.admissible_actions,
                initial_evaluation_info=reset.evaluation_info,
            )
            task_recorded = True
            context.reset(task)
            self.policy.start_episode(
                task,
                episode_index=request.episode_index,
                initial_observation=reset.observation,
                context_manager=context,
            )
            policy_started = True
            observation = reset.observation
            actions = reset.admissible_actions
            status = "max_steps"
            for step_index in range(1, int(self.benchmark_config["max_steps"]) + 1):
                state = PolicyState(
                    task_id=task.task_id,
                    benchmark=task.benchmark,
                    family=task.family,
                    instruction=task.instruction,
                    step_index=step_index,
                    observation=observation,
                    admissible_actions=actions,
                )
                try:
                    decision, prompt = self.policy.act(
                        state,
                        context_manager=context,
                        generation=self.generation,
                        call_position=ModelCallPosition(
                            episode_index=request.episode_index,
                            step_index=step_index,
                            call_slot=1,
                        ),
                    )
                except ModelInvocationError as exc:
                    prior_calls = tuple(getattr(exc, "prior_model_calls", ()))
                    prior_positions = tuple(getattr(exc, "prior_model_call_positions", ()))
                    for prior_index, prior_call in enumerate(prior_calls, start=1):
                        prior_position = (
                            prior_positions[prior_index - 1]
                            if prior_positions
                            else ModelCallPosition(
                                episode_index=request.episode_index,
                                step_index=step_index,
                                call_slot=prior_index,
                            )
                        )
                        model_call_rows.append(
                            _model_call_row(
                                prior_call,
                                request=request,
                                step_index=prior_position.step_index,
                                call_slot=prior_position.call_slot,
                            )
                        )
                    failed_slot = int(getattr(exc, "call_slot", 1))
                    model_call_rows.append(
                        _model_call_row(
                            exc.call,
                            request=request,
                            step_index=step_index,
                            call_slot=failed_slot,
                        )
                    )
                    status = "model_error"
                    error = str(exc)
                    error_rows.append(
                        {
                            "time": now_iso(),
                            "episode_id": request.episode_id,
                            "stage": "model",
                            "error": error,
                        }
                    )
                    break
                positions = decision.model_call_positions
                if positions and len(positions) != len(decision.model_calls):
                    raise ValueError("policy returned inconsistent model call positions")
                for call_index, call in enumerate(decision.model_calls, start=1):
                    position = (
                        positions[call_index - 1]
                        if positions
                        else ModelCallPosition(
                            episode_index=request.episode_index,
                            step_index=step_index,
                            call_slot=call_index,
                        )
                    )
                    model_call_rows.append(
                        _model_call_row(
                            call,
                            request=request,
                            step_index=position.step_index,
                            call_slot=position.call_slot,
                        )
                    )
                if decision.selected_action_call_id is not None:
                    primary_call = next(
                        item
                        for item in decision.model_calls
                        if item.call_id == decision.selected_action_call_id
                        and item.purpose in {"environment_action", "repair_action"}
                    )
                else:
                    primary_call = next(
                        item
                        for item in reversed(decision.model_calls)
                        if item.purpose in {"environment_action", "repair_action"}
                    )
                proposal_call = next(
                    (item for item in decision.model_calls if item.purpose == "environment_action"),
                    None,
                )
                proposal_prompt = (
                    prompt.proposal
                    if prompt.proposal is not None
                    else prompt
                    if proposal_call is not None
                    else None
                )
                context_prompt = proposal_prompt or prompt
                proposal_prompt_record = (
                    _prompt_context_record(
                        proposal_prompt,
                        model_call_id=proposal_call.call_id,
                    )
                    if proposal_prompt is not None and proposal_call is not None
                    else None
                )
                executed_prompt_record = _prompt_context_record(
                    prompt, model_call_id=primary_call.call_id
                )
                base_step = {
                    "schema": STEP_RECORD,
                    "episode_id": request.episode_id,
                    "task_id": task.task_id,
                    "step_index": step_index,
                    "admissible_actions_before": list(actions),
                    "admissible_action_count": context_prompt.admissible_action_count,
                    "shown_action_count": context_prompt.shown_action_count,
                    "estimated_input_tokens": prompt.estimated_input_tokens,
                    "prompt_contexts": {
                        "proposal": proposal_prompt_record,
                        "executed_action": executed_prompt_record,
                    },
                    "dropped_history_steps": (context_prompt.context.dropped_history_steps),
                    "external_context_estimated_tokens": (
                        context_prompt.context.external_context_tokens
                    ),
                    "external_context_type": (context_prompt.context.external_context_type),
                    "external_available_item_count": (
                        context_prompt.context.available_external_item_count
                    ),
                    "external_selected_item_count": (
                        context_prompt.context.selected_external_item_count
                    ),
                    "external_selected_item_ids": list(
                        context_prompt.context.selected_external_item_ids
                    ),
                    "memory_available_trajectory_count": (
                        context_prompt.context.available_external_item_count
                    ),
                    "memory_selected_trajectory_count": (
                        context_prompt.context.selected_external_item_count
                    ),
                    "memory_selected_trajectory_ids": list(
                        context_prompt.context.selected_external_item_ids
                    ),
                    "context_state_version": context_prompt.context.state_version,
                    "observation_truncated": context_prompt.observation_truncated,
                    "actions_truncated": context_prompt.actions_truncated,
                    "source_model": decision.source_model,
                    "model_call_ids": [item.call_id for item in decision.model_calls],
                    "intervention": decision.intervention,
                    "repair_trigger": decision.trigger,
                    "reported_confidence": decision.reported_confidence,
                    "confidence_error": decision.confidence_error,
                    "confidence_threshold": decision.confidence_threshold,
                    "confidence_route": decision.confidence_route,
                    "request_large_model": decision.request_large_model,
                    "confidence_reason": decision.confidence_reason,
                    "repair_intervention_id": (decision.repair_intervention_id),
                    "repair_origin_trigger": decision.repair_origin_trigger,
                    "repair_intervention_action_index": (decision.repair_intervention_action_index),
                    "selected_action_call_id": (decision.selected_action_call_id),
                    "raw_action": decision.raw_action,
                    "action": decision.action,
                    "parse_mode": decision.parse_mode,
                    "parse_error": decision.parse_error,
                }
                if decision.parse_error or decision.action is None:
                    row = {
                        **base_step,
                        "decision_status": "parse_failed",
                        "environment_status": "not_called",
                        "action_valid": False,
                        "execution_error": "action_parse_failed",
                        "observation_after": observation,
                        "reward": None,
                        "success": False,
                        "done": False,
                        "online_feedback": {},
                        "evaluation_info": {},
                    }
                    step_rows.append(row)
                    trajectory_steps.append(
                        TrajectoryStep(
                            step_index=step_index,
                            reasoning=primary_call.reasoning_output,
                            model_response=primary_call.raw_output or "",
                            action=decision.action,
                            environment_response=None,
                        )
                    )
                    step_count = step_index
                    status = "parse_error"
                    break
                transition = self.worker.step(decision.action)
                row = {
                    **base_step,
                    "decision_status": "parsed",
                    "environment_status": ("executed" if transition.action_valid else "rejected"),
                    "action_valid": transition.action_valid,
                    "execution_error": transition.execution_error,
                    "observation_after": transition.observation,
                    "policy_observation_after": transition.policy_observation,
                    "reward": transition.reward,
                    "success": transition.success,
                    "done": transition.done,
                    "online_feedback": dict(transition.online_feedback),
                    "evaluation_info": dict(transition.evaluation_info),
                }
                step_rows.append(row)
                trajectory_steps.append(
                    TrajectoryStep(
                        step_index=step_index,
                        reasoning=primary_call.reasoning_output,
                        model_response=primary_call.raw_output or "",
                        action=decision.action,
                        environment_response=(_model_visible_environment_response(transition)),
                    )
                )
                step_count = step_index
                if transition.reward is not None:
                    total_reward += transition.reward
                native_metrics = dict(transition.evaluation_info)
                should_terminate_invalid = (
                    not transition.action_valid
                    and self.benchmark_config.get("invalid_action_policy", "terminate")
                    == "terminate"
                )
                self.policy.observe(
                    state=state,
                    decision=decision,
                    transition=transition,
                )
                continue_after_transition = self.policy.continue_after_transition(
                    state=state,
                    decision=decision,
                    transition=transition,
                )
                if should_terminate_invalid and not continue_after_transition:
                    status = "invalid_action"
                    break
                context.observe(
                    VisibleTransition(
                        step_index=step_index,
                        action=decision.action,
                        observation=_model_visible_environment_response(transition),
                        action_valid=transition.action_valid,
                        execution_error=transition.execution_error,
                    )
                )
                observation = transition.policy_observation or transition.observation
                actions = transition.admissible_actions
                if bool(transition.success):
                    success = True
                    status = "success"
                    break
                if transition.done:
                    status = "environment_done"
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "environment_error"
            error_rows.append(
                {
                    "time": now_iso(),
                    "episode_id": request.episode_id,
                    "stage": "environment",
                    "error": error,
                    "traceback": traceback.format_exc(),
                }
            )
            if not task_recorded:
                task_row = _task_row(
                    task,
                    episode_id=request.episode_id,
                    environment_seed=request.environment_seed,
                )
        finally:
            try:
                self.worker.close_episode()
            except Exception as close_exc:
                error_rows.append(
                    {
                        "time": now_iso(),
                        "episode_id": request.episode_id,
                        "stage": "close_episode",
                        "error": (f"{type(close_exc).__name__}: {close_exc}"),
                    }
                )

        assert task_row is not None
        episode_row = {
            "schema": EPISODE_RECORD,
            "episode_id": request.episode_id,
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "family": task.family,
            "status": status,
            "success": success,
            "step_count": step_count,
            "total_reward": total_reward,
            "native_metrics": native_metrics,
            "sampling_seed": int(self.seeds["sampling"]),
            "environment_seed": request.environment_seed,
            "generation_seed_base": int(self.seeds["generation"]),
            "started_at": started,
            "completed_at": now_iso(),
            "error": error,
        }
        visible_trajectory = CompletedTrajectory(
            trajectory_id=f"trajectory-{request.episode_index:05d}",
            task_id=task.task_id,
            benchmark=task.benchmark,
            family=task.family,
            instruction=task.instruction,
            initial_observation=initial_observation,
            steps=tuple(trajectory_steps),
        )
        try:
            if not policy_started:
                return EpisodeBundle(
                    request=request,
                    task_row=task_row,
                    episode_row=episode_row,
                    step_rows=tuple(step_rows),
                    model_call_rows=tuple(model_call_rows),
                    method_event_rows=tuple(method_event_rows),
                    error_rows=tuple(error_rows),
                    visible_trajectory=visible_trajectory,
                )
            update = self.policy.finish_episode(
                episode_index=request.episode_index,
                outcome=episode_row,
                trajectory=visible_trajectory,
                context_manager=context,
                generation=self.generation,
                max_steps=int(self.benchmark_config["max_steps"]),
            )
            for positioned in update.model_calls:
                model_call_rows.append(
                    _model_call_row(
                        positioned.call,
                        request=request,
                        step_index=positioned.position.step_index,
                        call_slot=positioned.position.call_slot,
                    )
                )
            for event in update.events:
                method_event_rows.append(
                    {
                        "schema": METHOD_EVENT,
                        "time": now_iso(),
                        "episode_id": request.episode_id,
                        "episode_index": request.episode_index,
                        **dict(event),
                    }
                )
        except ModelInvocationError as exc:
            failed_slot = int(getattr(exc, "call_slot", 7))
            model_call_rows.append(
                _model_call_row(
                    exc.call,
                    request=request,
                    step_index=int(self.benchmark_config["max_steps"]),
                    call_slot=failed_slot,
                )
            )
            episode_row["environment_success_before_method_error"] = bool(
                episode_row.get("success")
            )
            episode_row["method_update_status"] = "failed"
            episode_row["method_update_error"] = str(exc)
            error_rows.append(
                {
                    "time": now_iso(),
                    "episode_id": request.episode_id,
                    "stage": "method_update",
                    "error": str(exc),
                }
            )
        except Exception as exc:
            method_error = f"{type(exc).__name__}: {exc}"
            episode_row["environment_success_before_method_error"] = bool(
                episode_row.get("success")
            )
            episode_row["method_update_status"] = "failed"
            episode_row["method_update_error"] = method_error
            error_rows.append(
                {
                    "time": now_iso(),
                    "episode_id": request.episode_id,
                    "stage": "method_update",
                    "error": method_error,
                    "traceback": traceback.format_exc(),
                }
            )
        if getattr(self.policy, "records_decision_opportunities", False):
            from .stream_metrics import annotate_opportunities

            annotate_opportunities(model_call_rows, step_rows, method_event_rows)
        return EpisodeBundle(
            request=request,
            task_row=task_row,
            episode_row=episode_row,
            step_rows=tuple(step_rows),
            model_call_rows=tuple(model_call_rows),
            method_event_rows=tuple(method_event_rows),
            error_rows=tuple(error_rows),
            visible_trajectory=visible_trajectory,
        )
