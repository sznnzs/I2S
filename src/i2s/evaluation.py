from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import write_json


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _physical_calls(call: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    attempts = call.get("generation_attempts")
    if attempts:
        for attempt in attempts:
            yield from _physical_calls(attempt)
    else:
        yield call


def summarize_usage(
    calls: Iterable[Mapping[str, Any]], *, small_weight: float = 0.25
) -> dict[str, Any]:
    role_rows: dict[str, list[Mapping[str, Any]]] = {"small": [], "large": []}
    seen: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    logical_count = 0
    logical_roles: Counter[str] = Counter()
    for call in calls:
        identity = (call.get("ledger_source", ""), call.get("attempt_id", ""), call.get("call_id"))
        if call.get("call_id") and identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        logical_count += 1
        logical_roles[str(call.get("role", "unknown"))] += 1
        for physical in _physical_calls(call):
            role = str(physical.get("role", "unknown"))
            role_rows.setdefault(role, []).append(physical)

            for _ in range(int(physical.get("retry_count", 0))):
                role_rows[role].append(
                    {"role": role, "usage": {}, "request_status": "unreported_retry"}
                )
    by_role = {}
    for role, rows in sorted(role_rows.items()):
        totals: dict[str, Any] = {
            "physical_requests": len(rows),
            "logical_calls": logical_roles[role],
        }
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            values = [dict(row.get("usage") or {}).get(field) for row in rows]
            known = [
                value
                for value in values
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            ]
            totals[field] = sum(known) if len(known) == len(values) else None
            totals[f"known_{field}"] = sum(known)
            totals[f"missing_{field}_requests"] = len(values) - len(known)
        totals["failed_requests"] = sum(
            row.get("request_status") != "response_received" for row in rows
        )
        by_role[role] = totals
    small = by_role["small"]["output_tokens"]
    large = by_role["large"]["output_tokens"]
    unknown_role = any(
        role not in {"small", "large"} and row["physical_requests"] for role, row in by_role.items()
    )
    complete = small is not None and large is not None and not unknown_role
    return {
        "logical_calls": logical_count,
        "physical_requests": sum(row["physical_requests"] for row in by_role.values()),
        "duplicate_logical_rows_ignored": duplicate_count,
        "by_role": by_role,
        "output_usage_complete": complete,
        "small_output_weight": small_weight,
        "weighted_output_tokens": large + small_weight * small if complete else None,
        "weighted_known_output_tokens": by_role["large"]["known_output_tokens"]
        + small_weight * by_role["small"]["known_output_tokens"],
        "formula": "large_output_tokens + 0.25 * small_output_tokens",
        "reasoning_accounting": "reported output tokens include reasoning; reasoning tokens are metadata only",
    }


def summarize_native_scores(
    benchmark: str, episodes: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = list(episodes)
    if benchmark == "alfworld":
        values = [float(bool(row.get("success"))) for row in rows]
        rule = "100 * native success proportion"
        reported = [100.0 * value for value in values]
        native_field = "success"
    else:
        native_field = "native_metrics.task_score"
        values = [dict(row.get("native_metrics") or {}).get("task_score") for row in rows]
        complete = all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in values
        )
        if benchmark == "webshop":
            rule = "100 * mean native task_score"
            reported = [100.0 * value for value in values] if complete else []
        elif benchmark == "scienceworld":
            rule = "mean max(0, native task_score)"
            reported = [max(0.0, value) for value in values] if complete else []
        else:
            raise ValueError(f"unsupported benchmark score: {benchmark}")
        if not complete:
            return {
                "raw_score": None,
                "reported_score": None,
                "score_complete": False,
                "native_score_field": native_field,
                "reported_score_rule": rule,
            }
    return {
        "raw_score": sum(values) / len(values) if values else None,
        "reported_score": sum(reported) / len(reported) if reported else None,
        "score_complete": bool(values),
        "native_score_field": native_field,
        "reported_score_rule": rule,
    }


def evaluate_run(run_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(root / "episodes.jsonl")
    attempt_starts = read_jsonl(root / "attempt_starts.jsonl")
    committed = [episode for episode in episodes if episode.get("committed", True)]
    main_calls = read_jsonl(root / "model_calls.jsonl")
    committed_steps = [
        step for step in read_jsonl(root / "steps.jsonl") if step.get("committed", True)
    ]
    indices = [episode.get("episode_index", episode["episode_id"]) for episode in committed]
    usage = summarize_usage(main_calls)
    if not (root / "model_calls.jsonl").is_file():
        usage["output_usage_complete"] = False
        usage["weighted_output_tokens"] = None
    count = len(committed)
    expected = int(manifest.get("registered_task_count", 0))
    success_count = sum(bool(episode.get("success")) for episode in committed)
    errors: list[str] = []
    if manifest.get("status") != "completed":
        errors.append("run_not_completed")
    if count != expected or not expected:
        errors.append("committed_task_count_mismatch")
    if len(set(indices)) != len(indices):
        errors.append("duplicate_committed_episode")
    selection_path = root / "task_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected_ids = selection.get("selected_task_ids", [])
        if len(expected_ids) != expected:
            errors.append("selection_manifest_task_count_mismatch")
        if [episode["task_id"] for episode in committed] != expected_ids[:count]:
            errors.append("committed_task_identity_or_order_mismatch")
    if any(
        episode.get("status") in {"reset_error", "model_error", "environment_error"}
        or episode.get("method_update_status") == "failed"
        for episode in committed
    ):
        errors.append("infrastructure_failure_was_committed")
    call_ids = {call.get("call_id") for call in main_calls}
    for step in read_jsonl(root / "steps.jsonl"):
        if any(call_id not in call_ids for call_id in step.get("model_call_ids", [])):
            errors.append("step_references_missing_model_call")
            break
    for filename, digest in manifest.get("evidence_sha256", {}).items():
        path = root / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append("evidence_hash_mismatch:" + filename)
    for filename in (
        "task_selection.json",
        "config.resolved.yaml",
        "tasks.jsonl",
        "steps.jsonl",
        "model_calls.jsonl",
        "method_events.jsonl",
        "method_state.json",
    ):
        if not (root / filename).is_file():
            errors.append("missing_artifact:" + filename)
    errors = list(dict.fromkeys(errors))
    roles = {call["model_id"]: call.get("role") for call in main_calls if call.get("model_id")}
    decision_steps = len(committed_steps)
    large_steps = sum(roles.get(step.get("source_model")) == "large" for step in committed_steps)
    weighted = usage["weighted_output_tokens"]
    usage["per_task_denominator"] = (
        "committed tasks; numerator includes all attempts and skill updates"
    )
    usage["logical_calls_per_task"] = usage["logical_calls"] / count if count else None
    usage["physical_requests_per_task"] = usage["physical_requests"] / count if count else None
    usage["weighted_output_tokens_per_task"] = (
        weighted / count if count and weighted is not None else None
    )
    for role in usage["by_role"].values():
        role["logical_calls_per_task"] = role["logical_calls"] / count if count else None
        for field in ("input_tokens", "output_tokens"):
            role[f"{field}_per_task"] = (
                role[field] / count if count and role[field] is not None else None
            )
    metrics = {
        "schema": "i2s-metrics",
        "registered_task_count": expected,
        "completed_task_count": count,
        "success_count": success_count,
        "success_proportion": success_count / count if count else None,
        "coverage": count / expected if expected else None,
        "attempt_count": len(attempt_starts) if attempt_starts else len(episodes),
        "status_counts": dict(Counter(str(episode.get("status")) for episode in committed)),
        **summarize_native_scores(
            str(
                manifest.get("benchmark")
                or (committed[0]["benchmark"] if committed else "alfworld")
            ),
            committed,
        ),
        "decision_steps": {
            "scope": "committed main-stream episodes",
            "total": decision_steps,
            "large_model": large_steps,
            "large_model_step_rate": large_steps / decision_steps if decision_steps else None,
            "per_task": decision_steps / count if count else None,
        },
        "native_results": [
            {
                key: episode.get(key)
                for key in ("episode_id", "task_id", "success", "status", "native_metrics")
            }
            for episode in committed
        ],
        "usage": usage,
    }
    validation = {
        "schema": "i2s-run-validation",
        "valid": not errors,
        "errors": errors,
        "task_outcomes_complete": count == expected and expected > 0,
        "cost_complete": usage["output_usage_complete"],
        "scope": "artifact completeness and native episode outcomes",
    }
    if write:
        write_json(root / "metrics.json", metrics)
        write_json(root / "validation.json", validation)
    return {"metrics": metrics, "validation": validation}
