from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .context import ContextManager
from .schemas import ContextBuild, PolicyState
from .utils import estimate_tokens


@dataclass(frozen=True)
class PromptBuild:
    messages: tuple[Mapping[str, str], ...]
    context: ContextBuild
    estimated_input_tokens: int
    admissible_action_count: int
    shown_action_count: int
    observation_truncated: bool
    actions_truncated: bool
    proposal: PromptBuild | None = None


class PromptBuilder:
    def __init__(
        self,
        *,
        prompt_root: str | Path,
        prompt_id: str,
        prompt_contract: str = "semantic_state",
    ) -> None:
        directory = Path(prompt_root).resolve() / prompt_id
        self.prompt_id = prompt_id
        induction_path = directory / "induction_system.txt"
        self.induction_system = (
            induction_path.read_text(encoding="utf-8").strip() if induction_path.is_file() else None
        )
        schema_path = directory / "induction_schema.json"
        self.induction_schema = (
            json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else None
        )
        refinement_path = directory / "refinement_system.txt"
        self.refinement_system = (
            refinement_path.read_text(encoding="utf-8").strip()
            if refinement_path.is_file()
            else None
        )
        refinement_schema_path = directory / "refinement_schema.json"
        self.refinement_schema = (
            json.loads(refinement_schema_path.read_text(encoding="utf-8"))
            if refinement_schema_path.is_file()
            else None
        )
        self.prompt_contract = str(prompt_contract)
        if self.prompt_contract not in {
            "explicit_task_state",
            "current_action_state",
            "constrained_action_state",
            "semantic_state",
        }:
            raise ValueError("unsupported prompt_contract")
        self.common_system = (directory / "common_system.txt").read_text(encoding="utf-8").strip()
        self.confidence_system = (
            (directory / "confidence_system.txt").read_text(encoding="utf-8").strip()
        )
        self.benchmark_system = {
            name: (directory / f"{name}_system.txt").read_text(encoding="utf-8").strip()
            for name in ("alfworld", "scienceworld", "webshop")
        }

    @staticmethod
    def _awaiting_numbered_choice(state: PolicyState) -> bool:
        if state.benchmark != "scienceworld":
            return False
        actions = tuple(str(item) for item in state.admissible_actions)
        return (
            bool(actions)
            and all(item.isdigit() for item in actions)
            and (
                "please enter the number for the action you intended"
                in str(state.observation).lower()
            )
        )

    @classmethod
    def _action_heading(cls, state: PolicyState, count: int) -> str:
        if state.benchmark != "scienceworld":
            return f"Admissible actions ({count})"
        actions = tuple(str(item) for item in state.admissible_actions)
        if cls._awaiting_numbered_choice(state):
            return f"Admissible disambiguation choices ({count})"
        has_pattern = any("OBJ" in item for item in actions)
        has_exact = any("OBJ" not in item for item in actions)
        if has_pattern and has_exact:
            return f"Available actions and patterns ({count})"
        if actions and not has_pattern:
            return f"Admissible actions ({count})"
        return f"Available action patterns ({count})"

    @classmethod
    def _turn_control_instructions(
        cls,
        state: PolicyState,
        context_manager: ContextManager,
        *,
        prompt_contract: str,
    ) -> tuple[str, ...]:
        if prompt_contract not in {
            "explicit_task_state",
            "current_action_state",
            "constrained_action_state",
            "semantic_state",
        }:
            return ()
        instructions: list[str] = []
        if (
            tuple(getattr(context_manager, "selected_skills", ()))
            and prompt_contract == "semantic_state"
        ):
            instructions.append(
                "Read the applicable semantic skill as guidance for this current state. "
                "Execute a step only when its visible preconditions hold and its result is not already established. "
                "Skip completed results; defer steps whose preconditions do not hold. "
                "Choose one exact CURRENT admissible action to realize the next useful step. "
                "If the skill's relation informed this choice, cite its exact skill_id and version in skill_refs."
            )
        elif tuple(getattr(context_manager, "selected_skills", ())):
            instructions.append(
                "Treat each reusable procedure as state-conditioned guidance, "
                "not a command to restart its steps. Compare it with the "
                "current observation and inventory, skip every procedure step "
                "whose visible precondition or result is already satisfied, "
                "and continue from the earliest unsatisfied step."
            )
        if cls._awaiting_numbered_choice(state):
            instructions.append(
                "The environment is currently waiting for a numbered disambiguation choice. For this turn, the action must be exactly one displayed number; do not repeat, expand, or replace the underlying command."
            )
        return tuple(instructions)

    @staticmethod
    def _explicit_task_constraints(state: PolicyState) -> str:
        if state.benchmark != "scienceworld":
            return ""
        match = re.search(
            r"\bfirst,\s*focus\s+on\s+(?:the\s+)?(.+?)(?:\.|$)",
            str(state.instruction),
            flags=re.IGNORECASE,
        )
        if match is None:
            return ""
        target = re.sub(r"\s+", " ", match.group(1)).strip()
        if not target:
            return ""
        return (
            "Explicit task constraint copied from the instruction:\n"
            f"- required focus target (exact full name): {target}\n\n"
        )

    def _uses_explicit_task_constraints(self) -> bool:
        return self.prompt_contract == "explicit_task_state"

    def build(
        self,
        state: PolicyState,
        context_manager: ContextManager,
        *,
        max_input_tokens: int,
        additional_instruction: str = "",
        include_external_context: bool = True,
    ) -> PromptBuild:
        if state.benchmark not in self.benchmark_system:
            raise ValueError(f"prompt bundle has no {state.benchmark} prompt")
        system = self.common_system + "\n\n" + self.benchmark_system[state.benchmark]
        actions = list(state.admissible_actions)
        actions_text = "\n".join(f"- {item}" for item in actions) or "(none)"
        action_heading = self._action_heading(state, len(actions))
        task_constraints = (
            self._explicit_task_constraints(state) if self._uses_explicit_task_constraints() else ""
        )
        fixed = f"Task instruction:\n{state.instruction}\n\n{task_constraints}Current observation:\n{state.observation}\n\n{action_heading}:\n{actions_text}\n\n"
        history_heading = "Previous interactions:\n"
        final_instructions = []
        if str(additional_instruction).strip():
            final_instructions.append(str(additional_instruction).strip())
        final_instructions.extend(
            self._turn_control_instructions(
                state,
                context_manager,
                prompt_contract=self.prompt_contract,
            )
        )
        final_instructions.append("Return the next action now as the required JSON object.")
        final_instruction = "\n\n" + "\n\n".join(final_instructions)
        default_external_heading = "Previous completed task trajectories:\n"
        safety_margin = max(
            48,
            int(getattr(context_manager, "input_safety_margin_tokens", 48)),
            math.ceil(
                int(max_input_tokens)
                * float(getattr(context_manager, "input_safety_margin_ratio", 0.0))
            ),
        )
        structural = default_external_heading + fixed + history_heading + final_instruction
        available_context = max(
            0,
            int(max_input_tokens) - estimate_tokens(system + structural) - safety_margin,
        )
        context = context_manager.build_context(
            available_tokens=available_context,
            include_external_context=include_external_context,
        )
        target_input = int(max_input_tokens) - safety_margin
        for _ in range(8):
            external_heading = context.external_context_heading.strip() + "\n"
            external = (
                external_heading + context.external_context + "\n\n"
                if context.external_context
                else ""
            )
            if self.prompt_contract in {"constrained_action_state", "semantic_state"}:
                user = (
                    external
                    + history_heading
                    + context.history_text
                    + "\n\n"
                    + fixed
                    + final_instruction
                )
            else:
                user = external + fixed + history_heading + context.history_text + final_instruction
            estimated = estimate_tokens(system + user)
            if estimated <= target_input:
                break
            overflow = estimated - target_input
            available_context = max(0, available_context - overflow - 32)
            context = context_manager.build_context(
                available_tokens=available_context,
                include_external_context=include_external_context,
            )
        external_heading = context.external_context_heading.strip() + "\n"
        external = (
            external_heading + context.external_context + "\n\n" if context.external_context else ""
        )
        if self.prompt_contract in {"constrained_action_state", "semantic_state"}:
            user = (
                external
                + history_heading
                + context.history_text
                + "\n\n"
                + fixed
                + final_instruction
            )
        else:
            user = external + fixed + history_heading + context.history_text + final_instruction
        estimated = estimate_tokens(system + user)
        if estimated > int(max_input_tokens):
            raise ValueError(
                f"complete visible context requires about {estimated} tokens but model input budget is {max_input_tokens}"
            )
        return PromptBuild(
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
            context=context,
            estimated_input_tokens=estimated,
            admissible_action_count=len(actions),
            shown_action_count=len(actions),
            observation_truncated=False,
            actions_truncated=False,
        )

    def build_confidence(
        self,
        state: PolicyState,
        *,
        proposed_action: str,
        threshold: float,
        max_input_tokens: int,
        context_manager: ContextManager | None = None,
        include_external_context: bool = False,
        include_episode_history: bool = False,
    ) -> PromptBuild:
        system = self.confidence_system + "\n\n" + self.benchmark_system[state.benchmark]
        actions = list(state.admissible_actions)
        action_heading = self._action_heading(state, len(actions))
        actions_text = "\n".join(f"- {item}" for item in actions) or "(none)"
        task_constraints = (
            self._explicit_task_constraints(state) if self._uses_explicit_task_constraints() else ""
        )
        fixed_prefix = f"Task instruction:\n{state.instruction}\n\n{task_constraints}Current observation:\n{state.observation}\n\n{action_heading}:\n{actions_text}\n\n"
        decision_suffix = f"Proposed action:\n{proposed_action}\n\nReference confidence threshold: {float(threshold):.6g}\n\nDecide whether this proposal should be executed by the small model or handed to the large model."
        empty_context = ContextBuild(
            history_text="",
            external_context="",
            estimated_tokens=0,
            dropped_history_steps=0,
            truncated=False,
        )
        context = empty_context
        external = ""
        history = ""
        if include_external_context or include_episode_history:
            if context_manager is None:
                raise ValueError("confidence context requires a context manager")
            safety_margin = max(
                48,
                int(getattr(context_manager, "input_safety_margin_tokens", 48)),
                math.ceil(
                    int(max_input_tokens)
                    * float(
                        getattr(
                            context_manager,
                            "input_safety_margin_ratio",
                            0.0,
                        )
                    )
                ),
            )
            available_context = max(
                0,
                int(max_input_tokens)
                - estimate_tokens(
                    system
                    + fixed_prefix
                    + decision_suffix
                    + ("Previous interactions:\n" if include_episode_history else "")
                )
                - safety_margin,
            )
            target_input = int(max_input_tokens) - safety_margin
            for _ in range(8):
                context = context_manager.build_context(
                    available_tokens=available_context,
                    include_external_context=include_external_context,
                )
                external = (
                    context.external_context_heading.strip()
                    + "\n"
                    + context.external_context
                    + "\n\n"
                    if include_external_context and context.external_context
                    else ""
                )
                history = (
                    "Previous interactions:\n" + context.history_text + "\n\n"
                    if include_episode_history
                    else ""
                )
                if self.prompt_contract == "constrained_action_state":
                    user = external + history + fixed_prefix + decision_suffix
                else:
                    user = external + fixed_prefix + history + decision_suffix
                estimated = estimate_tokens(system + user)
                if estimated <= target_input:
                    break
                overflow = estimated - target_input
                available_context = max(0, available_context - overflow - 32)
        if self.prompt_contract == "constrained_action_state":
            user = external + history + fixed_prefix + decision_suffix
        else:
            user = external + fixed_prefix + history + decision_suffix
        estimated = estimate_tokens(system + user)
        if estimated > int(max_input_tokens):
            raise ValueError("complete confidence assessment input exceeds the model input budget")
        return PromptBuild(
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
            context=context,
            estimated_input_tokens=estimated,
            admissible_action_count=len(actions),
            shown_action_count=len(actions),
            observation_truncated=False,
            actions_truncated=False,
        )
