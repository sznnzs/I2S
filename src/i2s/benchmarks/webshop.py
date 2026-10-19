from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from i2s.schemas import ResetResult, StepResult, TaskSpec
from i2s.utils import project_root

from .base import BenchmarkAdapter
from .common import extract_webshop_instruction, format_webshop_observation, normalize_action
from .webshop_compat import (
    PROFILE_ID,
    RETRIEVAL_BACKEND_ID,
    REWARD_NLP_BACKEND_ID,
    SEARCH_PROTOCOL_ID,
    UPSTREAM_COMMIT,
    bind_native_data_paths,
    native_search_document,
    prepare_native_import,
    validate_full_index,
    validate_profile,
)


class WebShopAdapter(BenchmarkAdapter):
    name = "webshop"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        validate_profile(config)
        self.source_dir = (
            project_root() / Path(str(config.get("source_dir", "third_party/WebShop"))).expanduser()
        ).resolve()
        self.data_dir = (
            project_root() / Path(str(config.get("data_dir", "data/webshop"))).expanduser()
        ).resolve()
        self.index_dir = (
            project_root()
            / Path(str(config.get("index_dir", self.data_dir / "search_index"))).expanduser()
        ).resolve()
        self.catalog = "full"
        self.inventory_seed = int(config.get("inventory_seed", 1))
        self.max_actions = 0
        self.observation_mode = str(config.get("observation_mode", "text_rich"))
        self.environment_profile = PROFILE_ID
        self.retrieval_backend = RETRIEVAL_BACKEND_ID
        self.reward_nlp_backend = REWARD_NLP_BACKEND_ID
        self.split = str(config.get("split", "evaluation"))
        scenario = config.get("scenario") or {}
        self.scenario = {"category": scenario} if isinstance(scenario, str) else dict(scenario)
        self.environment: Any = None
        self.current_task: TaskSpec | None = None
        self.current_observation = ""
        self.current_actions: list[str] = []

        self._environment_action_by_public_action: dict[str, str] = {}
        self.current_environment_seed: int | None = None
        self.last_evaluation: dict[str, Any] = {
            "task_score": 0.0,
            "success": False,
            **self._protocol_fields(),
        }

    def _protocol_fields(self) -> dict[str, str]:
        return {
            "environment_profile": self.environment_profile,
            "observation_mode": self.observation_mode,
            "retrieval_backend": self.retrieval_backend,
            "reward_nlp_backend": self.reward_nlp_backend,
            "search_protocol": SEARCH_PROTOCOL_ID,
        }

    def _ensure_environment(self) -> Any:
        if self.environment is not None:
            return self.environment
        prepare_native_import(self.source_dir)
        if not self.index_dir.is_dir() or not any(self.index_dir.glob("segments_*")):
            raise FileNotFoundError(
                f"WebShop Lucene index is missing: {self.index_dir}. "
                "Run python scripts/setup_environments.py webshop --build-index."
            )
        from importlib.metadata import version

        if version("en-core-web-sm") != "3.7.1":
            raise RuntimeError("The official reward profile requires en_core_web_sm==3.7.1")
        self._seed_episode(self.inventory_seed)
        import torch

        torch.manual_seed(self.inventory_seed)
        bind_native_data_paths(self.data_dir)
        from pyserini.search.lucene import LuceneSearcher
        from web_agent_site.envs import web_agent_text_env

        searcher = LuceneSearcher(str(self.index_dir))
        original_search_initializer = web_agent_text_env.init_search_engine

        def configured_search_engine(num_products=None):
            del num_products
            return searcher

        web_agent_text_env.init_search_engine = configured_search_engine
        try:
            environment = web_agent_text_env.WebAgentTextEnv(
                observation_mode=self.observation_mode,
                file_path=str(self.data_dir / "items_shuffle.json"),
                num_products=None,
                human_goals=True,
                num_prev_obs=0,
                num_prev_actions=0,
                get_image=0,
            )
            products = environment.server.all_products
            empty_count = sum(not native_search_document(p)["contents"].strip() for p in products)
            validate_full_index(
                searcher,
                product_count=len(products),
                empty_document_count=empty_count,
            )
            target_asins = {str(goal["asin"]) for goal in environment.server.goals}
            if any(searcher.doc(asin) is None for asin in target_asins):
                raise ValueError("The WebShop index is missing a human-goal target product")
        except Exception:
            close = getattr(searcher, "close", None)
            if callable(close):
                close()
            raise
        finally:
            web_agent_text_env.init_search_engine = original_search_initializer
        self.environment = environment
        return environment

    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        goals = self._ensure_environment().server.goals
        start, stop = {
            "evaluation": (0, 500),
            "dev": (500, 1500),
            "train": (1500, len(goals)),
        }[self.split]
        if stop > len(goals):
            raise ValueError("The official human-goal inventory is incomplete")
        category = self.scenario.get("category")
        tasks: list[TaskSpec] = []
        for index in range(start, stop):
            goal = goals[index]
            if category and str(goal["category"]) != str(category):
                continue
            instruction = str(goal["instruction_text"])
            tasks.append(
                TaskSpec(
                    task_id=f"webshop:full:human:seed{self.inventory_seed}:{self.split}:{index}",
                    benchmark=self.name,
                    family=str(goal["category"]),
                    split=self.split,
                    instruction=instruction,
                    metadata={
                        "session_index": index,
                        "catalog": "full",
                        "inventory_seed": self.inventory_seed,
                        "human_goals": True,
                        "category": goal["category"],
                        "product_category": goal["product_category"],
                        "official_goal_shuffle_seed": 233,
                        "upstream_commit": UPSTREAM_COMMIT,
                        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                        **self._protocol_fields(),
                    },
                )
            )
        return tasks, {
            "all_task_count": len(goals),
            "eligible_task_count": len(tasks),
            "split": self.split,
            "goal_index_start": start,
            "goal_index_stop": stop,
            "human_goals": True,
            "catalog": "full",
            "inventory_seed": self.inventory_seed,
            "upstream_commit": UPSTREAM_COMMIT,
            "category_counts": dict(Counter(task.metadata["category"] for task in tasks)),
            **self._protocol_fields(),
        }

    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        environment = self._ensure_environment()
        self._seed_episode(environment_seed)
        self.current_environment_seed = int(environment_seed)
        raw_observation, _ = environment.reset(session=int(task.metadata["session_index"]))
        instruction = extract_webshop_instruction(str(raw_observation)) or task.instruction
        self.current_task = TaskSpec(
            task_id=task.task_id,
            benchmark=task.benchmark,
            family=task.family,
            split=task.split,
            instruction=instruction,
            metadata={**dict(task.metadata), "environment_seed": int(environment_seed)},
        )
        self.current_observation = self._format_observation(str(raw_observation), instruction)
        self.current_actions = self._available_actions(environment.get_available_actions())
        self.last_evaluation = {
            "task_score": 0.0,
            "success": False,
            "environment_seed": self.current_environment_seed,
            **self._protocol_fields(),
        }
        return ResetResult(
            task=self.current_task,
            observation=self.current_observation,
            admissible_actions=tuple(self.current_actions),
            evaluation_info=dict(self.last_evaluation),
        )

    def step(self, action: str) -> StepResult:
        if self.current_task is None:
            raise RuntimeError("WebShop episode has not been reset")
        environment = self._ensure_environment()
        normalized = normalize_action(action, self.name)
        if not self._is_available(normalized):
            return StepResult(
                observation="The action was not admissible; the environment state did not change.",
                admissible_actions=tuple(self.current_actions),
                done=False,
                action_valid=False,
                execution_error="action_not_admissible",
                evaluation_info=dict(self.last_evaluation),
            )
        environment_action = self._environment_action_by_public_action.get(normalized, normalized)
        raw_observation, raw_reward, done, _ = environment.step(environment_action)
        reward = float(raw_reward)
        success = bool(done and math.isclose(reward, 1.0, abs_tol=1e-12))
        self.last_evaluation = {
            "task_score": reward,
            "success": success,
            "environment_seed": self.current_environment_seed,
            **self._protocol_fields(),
        }
        self.current_observation = self._format_observation(
            str(raw_observation),
            self.current_task.instruction,
        )
        self.current_actions = (
            [] if done else self._available_actions(environment.get_available_actions())
        )
        return StepResult(
            observation=self.current_observation,
            admissible_actions=tuple(self.current_actions),
            done=bool(done),
            action_valid=True,
            execution_error=None,
            online_feedback={"admissible_action_count": len(self.current_actions)},
            reward=reward,
            success=success,
            evaluation_info=dict(self.last_evaluation),
        )

    def _available_actions(self, available: Mapping[str, Any]) -> list[str]:
        actions = ["search[<your query>]"]
        environment_actions: dict[str, str] = {}
        for item in available.get("clickables", []):
            environment_action = f"click[{str(item).lower()}]"
            public_action = normalize_action(environment_action, self.name)
            actions.append(public_action)
            environment_actions.setdefault(public_action, environment_action)
        selected = sorted(dict.fromkeys(actions), key=self._action_rank)
        self._environment_action_by_public_action = environment_actions
        return selected

    @staticmethod
    def _format_observation(raw_observation: str, instruction: str) -> str:
        return format_webshop_observation(raw_observation, instruction)

    @staticmethod
    def _action_rank(action: str) -> tuple[int, str]:
        if action.startswith("search["):
            return (0, action)
        if action == "click[buy now]":
            return (1, action)
        if action in {"click[description]", "click[features]", "click[reviews]"}:
            return (2, action)
        if re.fullmatch(r"click\[[a-z0-9]{8,10}\]", action):
            return (3, action)
        return (4, action)

    def _is_available(self, action: str) -> bool:
        if action == "search[<your query>]":
            return False
        if action.startswith("search["):
            query = action[len("search[") : -1].strip() if action.endswith("]") else ""
            if re.match(r"^<[racq]>(?:\s|$)", query, flags=re.IGNORECASE):
                return False
            return bool(query)
        return action in self.current_actions

    @staticmethod
    def _seed_episode(seed: int) -> None:
        random.seed(int(seed))
        try:
            import numpy as np

            np.random.seed(int(seed) % (2**32))
        except ImportError:
            pass

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()
            close = getattr(self.environment.server.search_engine, "close", None)
            if callable(close):
                close()
            self.environment = None
        self.current_task = None
        self.current_actions = []
        self.current_environment_seed = None
