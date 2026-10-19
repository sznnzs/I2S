from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    benchmark: str
    family: str
    split: str
    instruction: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskSpec:
        return cls(
            task_id=str(value["task_id"]),
            benchmark=str(value["benchmark"]),
            family=str(value["family"]),
            split=str(value["split"]),
            instruction=str(value.get("instruction") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PolicyState:
    task_id: str
    benchmark: str
    family: str
    instruction: str
    step_index: int
    observation: str
    admissible_actions: tuple[str, ...]


@dataclass(frozen=True)
class ResetResult:
    task: TaskSpec
    observation: str
    admissible_actions: tuple[str, ...]
    evaluation_info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "observation": self.observation,
            "admissible_actions": list(self.admissible_actions),
            "evaluation_info": dict(self.evaluation_info),
        }


@dataclass(frozen=True)
class StepResult:
    observation: str
    admissible_actions: tuple[str, ...]
    done: bool
    action_valid: bool
    execution_error: str | None
    policy_observation: str | None = None
    online_feedback: Mapping[str, Any] = field(default_factory=dict)
    reward: float | None = None
    success: bool | None = None
    evaluation_info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "admissible_actions": list(self.admissible_actions),
            "done": self.done,
            "action_valid": self.action_valid,
            "execution_error": self.execution_error,
            "policy_observation": self.policy_observation,
            "online_feedback": dict(self.online_feedback),
            "reward": self.reward,
            "success": self.success,
            "evaluation_info": dict(self.evaluation_info),
        }


@dataclass(frozen=True)
class VisibleTransition:
    step_index: int
    action: str
    observation: str
    action_valid: bool
    execution_error: str | None


@dataclass(frozen=True)
class TrajectoryStep:
    step_index: int
    reasoning: str | None
    model_response: str
    action: str | None
    environment_response: str | None


@dataclass(frozen=True)
class CompletedTrajectory:
    trajectory_id: str
    task_id: str
    benchmark: str
    family: str
    instruction: str
    initial_observation: str | None
    steps: tuple[TrajectoryStep, ...]


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ModelCallPosition:
    episode_index: int
    step_index: int
    call_slot: int = 1


@dataclass(frozen=True)
class ModelCall:
    call_id: str
    model_id: str
    model_name: str
    role: str
    purpose: str
    messages: tuple[Mapping[str, str], ...]
    generation: Mapping[str, Any]
    raw_output: str | None
    reasoning_output: str | None
    usage: ModelUsage
    latency_seconds: float
    retry_count: int
    request_status: str
    error: str | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None
    raw_output_source: str | None = None
    generation_attempts: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [dict(item) for item in self.messages]
        if data["raw_output_source"] is None:
            data.pop("raw_output_source")
        if not data["generation_attempts"]:
            data.pop("generation_attempts")
        return data


@dataclass(frozen=True)
class ActionDecision:
    raw_action: str | None
    action: str | None
    parse_mode: str | None
    parse_error: str | None
    source_model: str
    model_calls: tuple[ModelCall, ...]
    intervention: str = "none"
    model_call_positions: tuple[ModelCallPosition, ...] = ()
    trigger: str | None = None
    reported_confidence: float | None = None
    confidence_error: str | None = None
    confidence_threshold: float | None = None
    confidence_route: str | None = None
    request_large_model: bool | None = None
    confidence_reason: str | None = None
    repair_intervention_id: str | None = None
    repair_origin_trigger: str | None = None
    repair_intervention_action_index: int | None = None
    selected_action_call_id: str | None = None


@dataclass(frozen=True)
class PositionedModelCall:
    call: ModelCall
    position: ModelCallPosition


@dataclass(frozen=True)
class PolicyEpisodeUpdate:
    model_calls: tuple[PositionedModelCall, ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ContextBuild:
    history_text: str
    external_context: str
    estimated_tokens: int
    dropped_history_steps: int
    truncated: bool
    external_context_tokens: int = 0
    selected_external_item_count: int = 0
    available_external_item_count: int = 0
    selected_external_item_ids: tuple[str, ...] = ()
    state_version: int = 0
    external_context_heading: str = "Previous completed task trajectories:"
    external_context_type: str = "none"
