from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

from i2s.schemas import ResetResult, StepResult, TaskSpec
from i2s.utils import project_root

from .base import BenchmarkAdapter
from .common import (
    alfworld_family,
    extract_alfworld_instruction,
    first_observation,
    flatten_info,
    normalize_action,
    scalar,
)


class AlfWorldAdapter(BenchmarkAdapter):
    name = "alfworld"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.config_path = (
            project_root()
            / Path(
                str(config.get("config_path", "configs/environments/alfworld.yaml"))
            ).expanduser()
        ).resolve()
        self.data_dir = (
            project_root() / Path(str(config.get("data_dir", "data/alfworld"))).expanduser()
        ).resolve()
        self.split = str(config.get("split", "eval_in_distribution"))
        if self.split not in {"train", "eval_in_distribution", "eval_out_of_distribution"}:
            raise ValueError(
                "ALFWorld split must be train, eval_in_distribution, or eval_out_of_distribution"
            )
        self.scenario = str(config.get("scenario", "pick_clean_then_place_in_recep"))
        self.max_steps = int(config.get("max_steps", 50))
        if self.max_steps <= 0:
            raise ValueError("ALFWorld max_steps must be positive")
        self.environment: Any = None
        self.current_actions: list[str] = []
        self.last_evaluation: dict[str, Any] = {}

    def _load_settings(self) -> dict[str, Any]:
        import yaml

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"ALFWorld configuration is missing: {self.config_path}. "
                "Run python scripts/setup_environments.py alfworld --download."
            )
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"ALFWorld data_dir does not exist: {self.data_dir}")
        settings = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("ALFWorld configuration must be a YAML mapping")
        if settings.get("env", {}).get("type") != "AlfredTWEnv":
            raise ValueError("The ALFWorld text benchmark requires env.type=AlfredTWEnv")

        for section, fields in (
            ("dataset", ("data_path", "eval_id_data_path", "eval_ood_data_path")),
            ("logic", ("domain", "grammar")),
        ):
            for field in fields:
                value = settings.get(section, {}).get(field)
                if value is None:
                    continue
                value = str(value).replace("${ALFWORLD_DATA}", str(self.data_dir))
                value = value.replace("$ALFWORLD_DATA", str(self.data_dir))
                path = Path(os.path.expandvars(value)).expanduser()
                settings[section][field] = str((self.config_path.parent / path).resolve())
        self._set_environment_horizon(settings, self.max_steps)
        data_field = {
            "train": "data_path",
            "eval_in_distribution": "eval_id_data_path",
            "eval_out_of_distribution": "eval_ood_data_path",
        }[self.split]
        split_path = settings.get("dataset", {}).get(data_field)
        if not split_path or not Path(split_path).is_dir():
            raise FileNotFoundError(f"ALFWorld split data is missing: {split_path}")
        return settings

    def _base_environment(self) -> Any:
        settings = self._load_settings()

        os.environ["ALFWORLD_DATA"] = str(self.data_dir)
        from alfworld.agents.environment import get_environment

        return get_environment(settings["env"]["type"])(settings, train_eval=self.split)

    @staticmethod
    def _set_environment_horizon(settings: dict[str, Any], max_steps: int) -> None:
        training_method = str(settings.get("general", {}).get("training_method") or "")
        training = settings.get(training_method, {}).get("training")
        if not isinstance(training, dict):
            raise ValueError(
                "ALFWorld upstream config has no training horizon for "
                f"training_method={training_method!r}"
            )
        training["max_nb_steps_per_episode"] = int(max_steps)

    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        base = self._base_environment()
        all_gamefiles = sorted(str(path) for path in base.game_files)
        available_scenarios = sorted({alfworld_family(path) for path in all_gamefiles})
        if self.scenario not in available_scenarios:
            raise ValueError(
                f"ALFWorld scenario {self.scenario!r} is not present in split {self.split!r}"
            )
        selected = [path for path in all_gamefiles if alfworld_family(path) == self.scenario]
        tasks = [
            TaskSpec(
                task_id=(f"alfworld:{self.split}:{self.scenario}:{Path(gamefile).parent.name}"),
                benchmark=self.name,
                family=self.scenario,
                split=self.split,
                metadata={"gamefile": gamefile, "dataset": self.split},
            )
            for gamefile in selected
        ]
        return tasks, {
            "all_task_count": len(all_gamefiles),
            "eligible_task_count": len(tasks),
            "scenario": self.scenario,
            "split": self.split,
            "available_scenarios": available_scenarios,
            "max_episode_steps": self.max_steps,
        }

    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        self.close_episode()
        random.seed(int(environment_seed))
        try:
            import numpy as np

            np.random.seed(int(environment_seed))
        except ImportError:
            pass
        gamefile = str(task.metadata["gamefile"])
        base = self._base_environment()
        base.game_files = [gamefile]
        base.num_games = 1
        self.environment = base.init_env(batch_size=1)
        observations, raw_info = self.environment.reset()
        info = flatten_info(raw_info)
        observation = first_observation(observations)
        actions = self._format_actions(info.get("admissible_commands"))
        self.current_actions = actions
        game_score = scalar(info.get("score", 0.0))
        self.last_evaluation = {
            "won": bool(scalar(info.get("won", False))),
            "game_score": float(game_score or 0.0),
            "gamefile": gamefile,
        }
        resolved_task = TaskSpec(
            task_id=task.task_id,
            benchmark=task.benchmark,
            family=task.family,
            split=task.split,
            instruction=task.instruction or extract_alfworld_instruction(observation),
            metadata=dict(task.metadata),
        )
        return ResetResult(
            task=resolved_task,
            observation=observation,
            admissible_actions=tuple(actions),
            evaluation_info=dict(self.last_evaluation),
        )

    def step(self, action: str) -> StepResult:
        if self.environment is None:
            raise RuntimeError("ALFWorld episode has not been reset")
        normalized = normalize_action(action, self.name)
        available = {normalize_action(item, self.name) for item in self.current_actions}
        if normalized not in available:
            return StepResult(
                observation="The action was not admissible; the environment state did not change.",
                admissible_actions=tuple(self.current_actions),
                done=False,
                action_valid=False,
                execution_error="action_not_admissible",
                evaluation_info=dict(self.last_evaluation),
            )
        observations, scores, dones, raw_info = self.environment.step([normalized])
        info = flatten_info(raw_info)
        observation = first_observation(observations)
        self.current_actions = self._format_actions(info.get("admissible_commands"))
        won = bool(scalar(info.get("won", False)))
        score = scalar(scores)
        game_score = scalar(score)
        self.last_evaluation = {
            "won": won,
            "game_score": float(game_score or 0.0),
        }
        hidden = {
            "won",
            "goal_condition_success_rate",
            "reward",
            "score",
            "admissible_commands",
            "extra.gamefile",
        }
        online = {key: value for key, value in info.items() if key not in hidden}
        return StepResult(
            observation=observation,
            admissible_actions=tuple(self.current_actions),
            done=bool(scalar(dones)),
            action_valid=True,
            execution_error=None,
            online_feedback=online,
            reward=float(scalar(score) or 0.0),
            success=won,
            evaluation_info=dict(self.last_evaluation),
        )

    @staticmethod
    def _format_actions(value: Any) -> list[str]:
        if value is None:
            return ["look"]
        current = value
        while (
            isinstance(current, (list, tuple))
            and len(current) == 1
            and isinstance(current[0], (list, tuple))
        ):
            current = current[0]
        result = [str(item).lower() for item in list(current)]
        return list(dict.fromkeys(result)) or ["look"]

    def close_episode(self) -> None:
        if self.environment is not None:
            try:
                self.environment.close()
            except Exception:
                pass
            self.environment = None
        self.current_actions = []
