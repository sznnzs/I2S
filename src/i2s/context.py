from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Mapping

from .schemas import (
    ContextBuild,
    TaskSpec,
    VisibleTransition,
)
from .utils import estimate_tokens


class ContextManager(ABC):
    @abstractmethod
    def reset(self, task: TaskSpec) -> None:
        pass

    @abstractmethod
    def build_context(
        self, *, available_tokens: int, include_external_context: bool = True
    ) -> ContextBuild:
        pass

    @abstractmethod
    def observe(self, transition: VisibleTransition) -> None:
        pass


class HistoryContextManager(ContextManager):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.max_steps = int(config.get("history_max_steps", 8))
        self.max_tokens = int(config.get("history_max_tokens", 2048))
        self.input_safety_margin_tokens = int(config.get("input_safety_margin_tokens", 48))
        self.input_safety_margin_ratio = float(config.get("input_safety_margin_ratio", 0.0))
        if self.input_safety_margin_tokens < 0:
            raise ValueError("input_safety_margin_tokens must be non-negative")
        if not 0.0 <= self.input_safety_margin_ratio < 1.0:
            raise ValueError("input_safety_margin_ratio must be in [0, 1)")
        self.external_context_max_tokens = int(config.get("external_context_max_tokens", 0))
        if self.external_context_max_tokens != 0:
            raise ValueError("HistoryContextManager does not accept external memory or skills")
        self.history: list[VisibleTransition] = []

    def reset(self, task: TaskSpec) -> None:
        del task
        self.history = []

    @staticmethod
    def _format(items: list[VisibleTransition]) -> str:
        if not items:
            return "(none)"
        chunks = []
        for item in items:
            chunks.append(
                f"Step {item.step_index}:\n"
                f"Action: {item.action}\n"
                f"Environment response:\n{item.observation}"
            )
        return "\n\n".join(chunks)

    def build_context(
        self, *, available_tokens: int, include_external_context: bool = True
    ) -> ContextBuild:
        del include_external_context
        budget = max(0, int(available_tokens))
        if self.max_tokens > 0:
            budget = min(budget, self.max_tokens)
        candidates = self.history[-self.max_steps :] if self.max_steps > 0 else list(self.history)
        dropped = max(0, len(self.history) - len(candidates))
        text = self._format(candidates)
        tokens = estimate_tokens(text) if candidates else 0
        while candidates and tokens > budget:
            candidates = candidates[1:]
            dropped += 1
            text = self._format(candidates)
            tokens = estimate_tokens(text) if candidates else 0
        return ContextBuild(
            history_text=text,
            external_context="",
            estimated_tokens=tokens,
            dropped_history_steps=dropped,
            truncated=dropped > 0,
        )

    def observe(self, transition: VisibleTransition) -> None:
        self.history.append(transition)


class SkillContextManager(HistoryContextManager):
    def __init__(self, config: Mapping[str, Any]) -> None:
        history_config = dict(config)
        history_config["external_context_max_tokens"] = 0
        super().__init__(history_config)
        self.external_context_max_tokens = int(config.get("external_context_max_tokens", 256))
        if self.external_context_max_tokens <= 0:
            raise ValueError("skill context external_context_max_tokens must be positive")
        self.fifo_capacity = config.get("capacity_policy") == "oldest_first_fifo"
        self.selected_skills: tuple[tuple[str, str], ...] = ()

        self.state_version = 0

    def reset(self, task: TaskSpec) -> None:
        super().reset(task)
        self.selected_skills = ()

    def set_skills(
        self,
        skills: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        *,
        state_version: int,
    ) -> None:
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_skill_id, raw_text in skills:
            skill_id = str(raw_skill_id).strip()
            text = str(raw_text).strip()
            if not skill_id or not text:
                raise ValueError("skill ID and text must be non-empty")
            if skill_id in seen:
                raise ValueError(f"duplicate selected skill ID: {skill_id}")
            seen.add(skill_id)
            normalized.append((skill_id, text))
        self.selected_skills = tuple(normalized)
        self.state_version = int(state_version)

    @staticmethod
    def _format_skills(items: list[tuple[str, str]]) -> str:
        return "\n\n".join(text for _skill_id, text in items)

    def build_context(
        self, *, available_tokens: int, include_external_context: bool = True
    ) -> ContextBuild:
        total_budget = max(0, int(available_tokens))
        if not include_external_context or not self.selected_skills:
            history = super().build_context(available_tokens=total_budget)
            return ContextBuild(
                **{
                    **asdict(history),
                    "state_version": self.state_version,
                    "available_external_item_count": len(self.selected_skills),
                    "external_context_heading": (
                        "Reusable procedure:"
                        if len(self.selected_skills) == 1
                        else "Reusable procedures:"
                    ),
                    "external_context_type": "skill",
                }
            )
        external_budget = min(total_budget, self.external_context_max_tokens)
        selected: list[tuple[str, str]] = []
        external_text = ""
        external_tokens = 0
        candidates = list(self.selected_skills)
        if self.fifo_capacity:
            while candidates and estimate_tokens(self._format_skills(candidates)) > external_budget:
                candidates.pop(0)
        for item in candidates:
            trial = self._format_skills([*selected, item])
            trial_tokens = estimate_tokens(trial)
            if trial_tokens > external_budget:
                continue
            selected.append(item)
            external_text = trial
            external_tokens = trial_tokens
        history = super().build_context(available_tokens=total_budget - external_tokens)
        return ContextBuild(
            history_text=history.history_text,
            external_context=external_text,
            estimated_tokens=history.estimated_tokens + external_tokens,
            dropped_history_steps=history.dropped_history_steps,
            truncated=(history.truncated or len(selected) < len(self.selected_skills)),
            external_context_tokens=external_tokens,
            selected_external_item_count=len(selected),
            available_external_item_count=len(self.selected_skills),
            selected_external_item_ids=tuple(item[0] for item in selected),
            state_version=self.state_version,
            external_context_heading=(
                "Reusable procedure:" if len(selected) == 1 else "Reusable procedures:"
            ),
            external_context_type="skill",
        )


def build_context_manager(config: Mapping[str, Any]) -> ContextManager:
    if config.get("type", "skill") != "skill":
        raise ValueError("context.type must be skill")
    return SkillContextManager(config)
