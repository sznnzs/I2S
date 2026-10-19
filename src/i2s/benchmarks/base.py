from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from i2s.schemas import ResetResult, StepResult, TaskSpec


class BenchmarkAdapter(ABC):
    name: str

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @abstractmethod
    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        pass

    @abstractmethod
    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        pass

    @abstractmethod
    def step(self, action: str) -> StepResult:
        pass

    def close_episode(self) -> None:
        return None

    def close(self) -> None:
        self.close_episode()
