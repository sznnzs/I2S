from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_schemas import (
    FROZEN_TASK_SELECTION,
    TASK_SELECTION,
    schema_name,
)
from .schemas import TaskSpec

_FROZEN_SELECTION_SCHEMAS = {FROZEN_TASK_SELECTION, TASK_SELECTION}
_CORE_IDENTITY_FIELDS = ("benchmark", "scenario", "split", "partition_role")
_OPTIONAL_IDENTITY_FIELDS = (
    "catalog",
    "inventory_seed",
    "simplification",
    "environment_profile",
    "retrieval_backend",
    "reward_nlp_backend",
    "sampling_group_key",
    "target_entities",
    "task_family_label",
)


def canonical_scenario(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if any(type(key) is not str for key in value):
        raise TypeError("scenario identity keys must be strings")
    return {key: child for key, child in value.items() if child is not None}


def strict_identity_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if any(type(key) is not str for key in (*left.keys(), *right.keys())):
            return False
        if set(left) != set(right):
            return False
        return all(strict_identity_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        return len(left) == len(right) and all(
            strict_identity_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _task_ids_from_value(value: Any, *, source: Path) -> list[str]:
    if isinstance(value, Mapping):
        if "selected_task_ids" in value:
            rows = value["selected_task_ids"]
        elif "tasks" in value:
            rows = value["tasks"]
        else:
            raise ValueError(
                f"task selection JSON object must contain selected_task_ids or tasks: {source}"
            )
    elif isinstance(value, list):
        rows = value
    else:
        raise TypeError(f"unsupported task selection payload in {source}")
    if not isinstance(rows, list):
        raise TypeError(f"task selection IDs must be a list: {source}")
    result: list[str] = []
    for index, row in enumerate(rows, start=1):
        task_id = row.get("task_id") if isinstance(row, Mapping) else row
        if not isinstance(task_id, (str, int)) or not str(task_id).strip():
            raise ValueError(f"invalid task ID at {source}:{index}")
        result.append(str(task_id))
    return result


def _resolve_reference(
    config: Mapping[str, Any],
    source_value: Any,
    *,
    label: str,
) -> Path:
    path = Path(str(source_value)).expanduser()
    if not path.is_absolute():
        config_path = config.get("config_path")
        if not config_path:
            raise ValueError(f"relative {label} requires config_path")
        path = Path(str(config_path)).parent / path
    return path.resolve(strict=True)


def _duplicate_ids(task_ids: Iterable[str]) -> list[str]:
    counts = Counter(task_ids)
    return sorted(task_id for task_id, count in counts.items() if count > 1)


def selection_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = dict(config["benchmark"])

    scenario = canonical_scenario(benchmark.get("scenario"))
    identity: dict[str, Any] = {
        "benchmark": benchmark["name"],
        "scenario": scenario,
        "split": benchmark.get("split"),
        "partition_role": benchmark.get("partition_role"),
    }
    for key in _OPTIONAL_IDENTITY_FIELDS:
        if key in benchmark and benchmark.get(key) is not None:
            identity[key] = benchmark.get(key)
    return identity


def build_selection_record(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    schema: str,
) -> dict[str, Any]:
    return {
        "schema": schema,
        **selection_identity(config),
        "inventory": dict(inventory),
        **dict(selection),
    }


def load_frozen_task_ids(
    config: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]] | None:
    benchmark = dict(config["benchmark"])
    source_value = benchmark.get("selection_file")
    if not source_value:
        return None
    path = _resolve_reference(config, source_value, label="selection file")
    if path.suffix.lower() != ".json":
        raise ValueError("frozen selection file must be .json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"frozen selection must be a JSON object: {path}")
    payload = dict(value)
    selection_schema = schema_name(payload)
    if selection_schema not in _FROZEN_SELECTION_SCHEMAS:
        raise ValueError(f"unsupported frozen selection schema {selection_schema!r}: {path}")
    task_ids = _task_ids_from_value(payload, source=path)
    if not task_ids:
        raise ValueError(f"frozen selection contains no task IDs: {path}")
    duplicates = _duplicate_ids(task_ids)
    if duplicates:
        raise ValueError(f"frozen selection contains duplicate task IDs: {duplicates[:5]}")
    declared_count = payload.get("selected_task_count")
    if declared_count is None or int(declared_count) != len(task_ids):
        raise ValueError("frozen selection selected_task_count does not match selected_task_ids")
    source_sampling_seed = payload.get("sampling_seed")
    if selection_schema == FROZEN_TASK_SELECTION and (
        source_sampling_seed is None or isinstance(source_sampling_seed, bool)
    ):
        raise ValueError(f"frozen selection is missing a valid sampling_seed: {path}")
    if source_sampling_seed is not None:
        source_sampling_seed = int(source_sampling_seed)

    expected = selection_identity(config)
    inventory = dict(payload.get("inventory") or {})
    metadata: dict[str, Any] = {}
    for key in (*_CORE_IDENTITY_FIELDS, *_OPTIONAL_IDENTITY_FIELDS):
        if key not in expected:
            continue
        present = key in payload or key in inventory
        source_value_for_key = payload.get(key) if key in payload else inventory.get(key)
        if key == "scenario":
            source_value_for_key = canonical_scenario(source_value_for_key)
        if selection_schema == FROZEN_TASK_SELECTION and not present:
            raise ValueError(f"frozen selection is missing identity field {key}: {path}")
        if key in _CORE_IDENTITY_FIELDS and not present:
            raise ValueError(f"selection is missing identity field {key}: {path}")
        if present and not strict_identity_equal(source_value_for_key, expected[key]):
            raise ValueError(
                f"frozen selection {key} does not match current config: "
                f"{source_value_for_key!r} != {expected[key]!r}"
            )
        if present:
            metadata[key] = source_value_for_key
    metadata.update(
        {
            "schema": selection_schema,
            "sampling_seed": source_sampling_seed,
            "selected_task_count": len(task_ids),
        }
    )
    return task_ids, {
        "source_path": str(path),
        "source_metadata": metadata,
    }


def load_exclusion_ids(config: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    benchmark = dict(config["benchmark"])
    inline = [str(item) for item in benchmark.get("exclude_task_ids", [])]
    source_value = benchmark.get("exclude_task_ids_file")
    source_ids: list[str] = []
    source_path: str | None = None
    source_metadata: dict[str, Any] = {}
    if source_value:
        path = _resolve_reference(config, source_value, label="exclusion file")
        source_path = str(path)
        if path.suffix == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            source_ids = _task_ids_from_value(values, source=path)
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            source_ids = _task_ids_from_value(value, source=path)
            if isinstance(value, Mapping):
                source_metadata = {
                    key: value.get(key)
                    for key in (
                        "benchmark",
                        "scenario",
                        "split",
                        "sampling_seed",
                        "selected_task_count",
                    )
                    if key in value
                }
                source_schema = schema_name(value)
                if source_schema:
                    source_metadata["schema"] = source_schema
        else:
            raise ValueError("exclusion file must be .json or .jsonl")
    requested = sorted(set(inline) | set(source_ids))
    return set(requested), {
        "source_path": source_path,
        "source_metadata": source_metadata,
        "inline_task_ids": sorted(set(inline)),
        "source_task_ids": sorted(set(source_ids)),
        "requested_task_ids": requested,
    }


def enforce_partition_exclusion(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    benchmark = config["benchmark"]
    exclusions_configured = bool(
        benchmark.get("exclude_task_ids") or benchmark.get("exclude_task_ids_file")
    )
    if (
        benchmark.get("partition_role") == "evaluation"
        and exclusions_configured
        and int(selection.get("excluded_task_count") or 0) <= 0
    ):
        raise ValueError(
            "evaluation exclusion matched zero tasks; check benchmark family, split, and source"
        )


def _ordered_inventory(tasks: Iterable[TaskSpec]) -> list[TaskSpec]:
    ordered = sorted(tasks, key=lambda task: task.task_id)
    duplicates = _duplicate_ids(task.task_id for task in ordered)
    if duplicates:
        raise ValueError(f"benchmark inventory contains duplicate task IDs: {duplicates[:5]}")
    return ordered


def select_tasks(
    tasks: Iterable[TaskSpec],
    *,
    sample_count: int,
    seed: int,
    excluded_task_ids: Iterable[str] = (),
    sampling_group_key: str | None = None,
) -> tuple[list[TaskSpec], dict[str, object]]:
    excluded = set(excluded_task_ids)
    ordered = _ordered_inventory(tasks)
    matched_exclusions = {task.task_id for task in ordered if task.task_id in excluded}
    candidates = [task for task in ordered if task.task_id not in excluded]
    if sample_count > len(candidates):
        raise ValueError(
            f"requested {sample_count} samples but only {len(candidates)} eligible tasks exist"
        )
    rng = random.Random(int(seed))
    group_metadata: dict[str, object] = {}
    if sampling_group_key is None:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected = shuffled[:sample_count]
        method = "sorted_then_seeded_shuffle"
    else:
        groups: dict[str, list[TaskSpec]] = {}
        for task in candidates:
            raw_group = task.metadata.get(sampling_group_key)
            if not isinstance(raw_group, (str, int)) or not str(raw_group).strip():
                raise ValueError(
                    "balanced sampling requires every task to contain non-empty "
                    f"metadata[{sampling_group_key!r}]"
                )
            groups.setdefault(str(raw_group), []).append(task)
        group_order = sorted(groups)
        rng.shuffle(group_order)
        for group in group_order:
            rng.shuffle(groups[group])
        selected = []
        round_index = 0
        while len(selected) < sample_count:
            added = False
            for group in group_order:
                bucket = groups[group]
                if round_index < len(bucket):
                    selected.append(bucket[round_index])
                    added = True
                    if len(selected) == sample_count:
                        break
            if not added:
                raise RuntimeError("balanced sampling exhausted candidates unexpectedly")
            round_index += 1
        method = "sorted_group_round_robin_seeded_shuffle"
        eligible_group_counts = Counter(
            str(task.metadata[sampling_group_key]) for task in candidates
        )
        selected_group_counts = Counter(str(task.metadata[sampling_group_key]) for task in selected)
        group_metadata = {
            "sampling_group_key": sampling_group_key,
            "eligible_group_count": len(eligible_group_counts),
            "selected_group_count": len(selected_group_counts),
            "eligible_group_counts": dict(sorted(eligible_group_counts.items())),
            "selected_group_counts": dict(sorted(selected_group_counts.items())),
        }
    metadata: dict[str, object] = {
        "method": method,
        "sampling_seed": int(seed),
        "eligible_task_count": len(candidates),
        "requested_excluded_task_id_count": len(excluded),
        "excluded_task_count": len(matched_exclusions),
        "matched_excluded_task_ids": sorted(matched_exclusions),
        "selected_task_count": len(selected),
        "eligible_family_counts": dict(sorted(Counter(task.family for task in candidates).items())),
        "selected_family_counts": dict(sorted(Counter(task.family for task in selected).items())),
        "selected_task_ids": [task.task_id for task in selected],
        **group_metadata,
    }
    return selected, metadata


def resolve_task_selection(
    tasks: Iterable[TaskSpec],
    config: Mapping[str, Any],
) -> tuple[list[TaskSpec], dict[str, Any]]:
    excluded_ids, exclusion = load_exclusion_ids(config)
    frozen = load_frozen_task_ids(config)
    benchmark = dict(config["benchmark"])
    if frozen is None:
        selected, raw_selection = select_tasks(
            tasks,
            sample_count=int(benchmark["sample_count"]),
            seed=int(dict(config["seeds"])["sampling"]),
            excluded_task_ids=excluded_ids,
            sampling_group_key=benchmark.get("sampling_group_key"),
        )
        selection = dict(raw_selection)
    else:
        ordered = _ordered_inventory(tasks)
        inventory_by_id = {task.task_id: task for task in ordered}
        selected_ids, source = frozen
        configured_count = int(benchmark["sample_count"])
        if configured_count != len(selected_ids):
            raise ValueError(
                "benchmark.sample_count must equal the frozen selection size: "
                f"{configured_count} != {len(selected_ids)}"
            )
        missing = [task_id for task_id in selected_ids if task_id not in inventory_by_id]
        if missing:
            raise ValueError(
                "frozen selection contains task IDs outside the current eligible inventory: "
                f"{missing[:5]}"
            )
        overlap = sorted(set(selected_ids) & excluded_ids)
        if overlap:
            raise ValueError(f"frozen selection overlaps configured exclusions: {overlap[:5]}")
        matched_exclusions = sorted(set(inventory_by_id) & excluded_ids)
        selected = [inventory_by_id[task_id] for task_id in selected_ids]
        eligible = [task for task in ordered if task.task_id not in excluded_ids]
        selection = {
            "method": "frozen_task_ids_file",
            "sampling_seed": source["source_metadata"].get("sampling_seed"),
            "eligible_task_count": len(eligible),
            "requested_excluded_task_id_count": len(excluded_ids),
            "excluded_task_count": len(matched_exclusions),
            "matched_excluded_task_ids": matched_exclusions,
            "selected_task_count": len(selected),
            "eligible_family_counts": dict(
                sorted(Counter(task.family for task in eligible).items())
            ),
            "selected_family_counts": dict(
                sorted(Counter(task.family for task in selected).items())
            ),
            "selected_task_ids": selected_ids,
            "selection_source": source,
        }
    selection["exclusion"] = exclusion
    enforce_partition_exclusion(config, selection)
    return selected, selection
