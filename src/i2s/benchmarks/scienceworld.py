from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any, Mapping

from i2s.schemas import ResetResult, StepResult, TaskSpec

from .base import BenchmarkAdapter
from .common import normalize_action

COMPACT_VALID_ACTION_GROUP_LIMITS = {
    "compact_actions_32": 32,
    "compact_actions_16": 16,
}
SCENARIO_IDS = {
    "conductivity": "3-3",
    "conductivity-unknown": "3-4",
}


class ScienceWorldAdapter(BenchmarkAdapter):
    name = "scienceworld"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.scenario = str(config.get("scenario", "conductivity"))
        self.split = str(config.get("split", "evaluation"))
        if self.split not in {"train", "dev", "evaluation"}:
            raise ValueError("ScienceWorld split must be train, dev, or evaluation")
        self.simplification = str(config.get("simplification", "easy"))
        self.max_steps = int(config.get("max_steps", 50))
        self.action_space_presentation = str(
            config.get("action_space_presentation") or "official_templates"
        )
        if self.action_space_presentation not in {
            "official_templates",
            "official_valid_actions",
            "compact_actions_32",
            "compact_actions_16",
        }:
            raise ValueError(
                "ScienceWorld action_space_presentation must be "
                "official_templates, official_valid_actions, or "
                "compact_actions_32/v2"
            )
        raw_targets = config.get("target_entities")
        self.target_entities = (
            tuple(str(item).strip().casefold() for item in raw_targets)
            if raw_targets is not None
            else ()
        )
        self.task_family_label = str(config.get("task_family_label") or "").strip()
        if bool(self.target_entities) != bool(self.task_family_label):
            raise ValueError(
                "ScienceWorld target_entities and task_family_label must be configured together"
            )
        self.family = (
            f"{self.scenario}::{self.task_family_label}"
            if self.task_family_label
            else self.scenario
        )
        self.environment: Any = None
        self.action_templates: list[str] = []
        self.last_evaluation: dict[str, Any] = {}

    def _environment(self) -> Any:
        if self.environment is None:
            from scienceworld import ScienceWorldEnv

            self.environment = ScienceWorldEnv(envStepLimit=self.max_steps)
        return self.environment

    @staticmethod
    def _native_scenario(scenario: str) -> str:
        from scienceworld.utils import infer_task

        return infer_task(SCENARIO_IDS.get(scenario, scenario))

    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        environment = self._environment()
        task_names = list(environment.get_task_names())
        native_scenario = self._native_scenario(self.scenario)
        if native_scenario not in task_names:
            raise ValueError(f"unknown ScienceWorld scenario: {self.scenario}")
        environment.load(native_scenario, 0, self.simplification)
        if self.split == "train":
            variation_ids = environment.get_variations_train()
        elif self.split == "dev":
            variation_ids = environment.get_variations_dev()
        else:
            variation_ids = (
                set(range(environment.get_max_variations(native_scenario)))
                - set(environment.get_variations_train())
                - set(environment.get_variations_dev())
            )
        variations = sorted(int(item) for item in variation_ids)
        tasks: list[TaskSpec] = []
        target_counts: Counter[str] = Counter()
        for variation in variations:
            instruction = ""
            target_entity: str | None = None
            if self.target_entities:
                environment.load(native_scenario, variation, self.simplification)
                instruction = str(environment.get_task_description()).strip()
                target_entity = self._target_entity(instruction)
                target_counts[target_entity] += 1
                if target_entity not in self.target_entities:
                    continue
            metadata = {
                "task_name": self.scenario,
                "variation_idx": variation,
                "simplification": self.simplification,
            }
            if target_entity is not None:
                metadata["target_entity"] = target_entity
            tasks.append(
                TaskSpec(
                    task_id=(f"scienceworld:{self.scenario}:{self.split}:{variation}"),
                    benchmark=self.name,
                    family=self.family,
                    split=self.split,
                    instruction=instruction,
                    metadata=metadata,
                )
            )
        scenario_names = {self._native_scenario(name): name for name in SCENARIO_IDS}
        inventory = {
            "all_scenario_count": len(task_names),
            "eligible_task_count": len(tasks),
            "scenario": self.scenario,
            "split": self.split,
            "available_scenarios": [scenario_names.get(name, name) for name in task_names],
        }
        if self.target_entities:
            inventory.update(
                {
                    "target_entities": list(self.target_entities),
                    "task_family_label": self.task_family_label,
                    "task_family": self.family,
                    "all_target_entity_counts": dict(sorted(target_counts.items())),
                }
            )
        return tasks, inventory

    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        random.seed(int(environment_seed))
        try:
            import numpy as np

            np.random.seed(int(environment_seed))
        except ImportError:
            pass
        environment = self._environment()
        task_name = str(task.metadata["task_name"])
        variation = int(task.metadata["variation_idx"])
        environment.load(
            self._native_scenario(task_name),
            variation,
            str(task.metadata.get("simplification", self.simplification)),
        )
        observation, info = environment.reset()
        instruction = task.instruction or str(environment.get_task_description()).strip()
        if self.target_entities:
            target_entity = self._target_entity(instruction)
            if target_entity not in self.target_entities:
                raise ValueError("ScienceWorld task target is outside the configured task family")
        info = dict(info or {})
        self.action_templates = self._actions_for_observation(environment, str(observation), info)
        score = int(info.get("score", 0))
        self.last_evaluation = {
            "task_score": score,
            "success": score >= 100,
            "task_name": task_name,
            "variation_idx": variation,
        }
        resolved = TaskSpec(
            task_id=task.task_id,
            benchmark=task.benchmark,
            family=task.family,
            split=task.split,
            instruction=instruction,
            metadata=dict(task.metadata),
        )
        return ResetResult(
            task=resolved,
            observation=self._policy_observation(str(observation), info),
            admissible_actions=tuple(self.action_templates),
            evaluation_info=dict(self.last_evaluation),
        )

    def step(self, action: str) -> StepResult:
        environment = self._environment()
        normalized = normalize_action(action, self.name)
        observation, reward, done, info = environment.step(normalized)
        info = dict(info or {})
        self.action_templates = self._actions_for_observation(environment, str(observation), info)
        score = int(info.get("score", 0))
        success = score >= 100
        parser_rejected = self._parser_rejected(str(observation))
        self.last_evaluation = {
            "task_score": score,
            "success": success,
            "task_name": self.scenario,
            "variation_idx": int(getattr(environment, "variationIdx", -1)),
        }
        return StepResult(
            observation=str(observation),
            admissible_actions=tuple(self.action_templates),
            done=bool(done),
            action_valid=not parser_rejected,
            execution_error="action_not_parsed" if parser_rejected else None,
            policy_observation=self._policy_observation(str(observation), info),
            online_feedback={
                "moves": int(info.get("moves", 0)),
                "action_space_presentation": (
                    "official_disambiguation_choices"
                    if self._disambiguation_choices(str(observation))
                    else self.action_space_presentation
                ),
                "action_template_count": len(self.action_templates),
                "action_count": len(self.action_templates),
            },
            reward=float(reward),
            success=success,
            evaluation_info=dict(self.last_evaluation),
        )

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()
            self.environment = None
        self.action_templates = []

    @staticmethod
    def _ordered_actions(actions: list[str]) -> list[str]:
        return sorted((str(item) for item in actions), key=str.lower)

    @staticmethod
    def _disambiguation_choices(observation: str) -> list[str]:
        text = str(observation)
        if "please enter the number for the action you intended" not in text.lower():
            return []
        choices = {match.group(1) for match in re.finditer(r"(?m)^\s*(\d+)\s*:\s+\S", text)}
        return sorted(choices, key=int)

    def _actions_for_observation(
        self,
        environment: Any,
        observation: str,
        info: Mapping[str, Any] | None = None,
    ) -> list[str]:
        choices = self._disambiguation_choices(observation)
        if choices:
            return choices
        if self.action_space_presentation == "official_valid_actions":
            registered = (info or {}).get("valid")
            if registered is None:
                registered = environment.get_valid_action_object_combinations()
            if isinstance(registered, (str, bytes)):
                registered = [registered]
            return self._ordered_actions(list(dict.fromkeys(str(item) for item in registered)))
        if self.action_space_presentation in COMPACT_VALID_ACTION_GROUP_LIMITS:
            return self._compact_official_actions(
                environment,
                group_limit=COMPACT_VALID_ACTION_GROUP_LIMITS[self.action_space_presentation],
            )
        return self._ordered_actions(list(dict.fromkeys(environment.get_possible_actions())))

    def _compact_official_actions(self, environment: Any, *, group_limit: int) -> list[str]:
        combinations_getter = getattr(
            environment,
            "get_valid_action_object_combinations_with_templates",
            None,
        )
        templates_getter = getattr(environment, "get_possible_actions_with_IDs", None)
        if not callable(combinations_getter) or not callable(templates_getter):
            return self._ordered_actions(
                list(
                    dict.fromkeys(
                        str(item) for item in (environment.get_valid_action_object_combinations())
                    )
                )
            )

        grouped: dict[int, list[str]] = {}
        for record in combinations_getter():
            if not isinstance(record, Mapping):
                continue
            action = str(record.get("action") or "").strip()
            try:
                template_id = int(record["template_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if action:
                grouped.setdefault(template_id, []).append(action)

        patterns: dict[int, list[str]] = {}
        for record in templates_getter():
            if not isinstance(record, Mapping):
                continue
            pattern = str(record.get("action_example") or "").strip()
            try:
                template_id = int(record["template_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if pattern:
                patterns.setdefault(template_id, []).append(pattern)

        compact: list[str] = []
        for template_id in sorted(grouped):
            exact = list(dict.fromkeys(grouped[template_id]))
            if len(exact) <= int(group_limit):
                compact.extend(exact)
                continue
            official_patterns = list(dict.fromkeys(patterns.get(template_id, ())))
            compact.extend(official_patterns or exact)
        return self._ordered_actions(list(dict.fromkeys(compact)))

    @staticmethod
    def _parser_rejected(observation: str) -> bool:
        normalized = str(observation).strip().lower()
        return normalized.startswith(
            (
                "no known action matches that input",
                "unknown action.",
            )
        )

    def _target_entity(self, instruction: str) -> str:
        if self._native_scenario(self.scenario) != self._native_scenario("conductivity"):
            raise ValueError("ScienceWorld target_entities requires scenario=conductivity")
        match = re.search(
            r"determine\s+if\s+(.+?)\s+is\s+electrically\s+conductive\.",
            str(instruction),
            flags=re.IGNORECASE,
        )
        if match is None or not match.group(1).strip():
            raise ValueError(
                "cannot derive the ScienceWorld target entity from the task instruction"
            )
        return match.group(1).strip().casefold()

    @staticmethod
    def _policy_observation(response: str, info: Mapping[str, Any]) -> str:
        response = str(response).strip()
        room = str(info.get("look") or "").strip()
        inventory = str(info.get("inv") or "").strip()
        sections: list[str] = []
        if response and response != room and response != inventory:
            sections.append(f"Environment response:\n{response}")
        if room:
            sections.append(f"Current room:\n{room}")
        if inventory:
            sections.append(f"Current inventory:\n{inventory}")
        return "\n\n".join(sections) or response
