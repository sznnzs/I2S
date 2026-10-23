from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .benchmarks.client import BenchmarkWorkerClient
from .config import active_model_ids, deep_merge, validate_config
from .evaluation import evaluate_run, read_jsonl
from .execution import EpisodeExecutor, EpisodeRequest
from .io import append_jsonl, write_json, write_jsonl, write_yaml
from .models import ModelInvocationError, ModelPool, resolve_credentials
from .policy import build_policy
from .prompts import PromptBuilder
from .sampling import build_selection_record, resolve_task_selection
from .schemas import TaskSpec
from .utils import project_root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _semantic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.pop("config_path", None)
    result.pop("run_name", None)
    result.get("execution", {}).pop("family_workers", None)
    return result


def runtime_config(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root = project_root()
    result = copy.deepcopy(dict(config))
    result["output_root"] = str(_path(result.get("output_root", "runs"), root))
    result["prompt_root"] = str(_path(result.get("prompt_root", "prompts"), root))
    benchmark = result["benchmark"]
    for key in (
        "config_path",
        "source_dir",
        "data_dir",
        "index_dir",
        "selection_file",
        "exclude_task_ids_file",
        "jar_path",
    ):
        if benchmark.get(key):
            benchmark[key] = str(_path(benchmark[key], root))
    runtime = benchmark.get("runtime", {})
    for key in ("python", "java_home"):
        if runtime.get(key) and ("/" in str(runtime[key]) or key == "java_home"):
            runtime[key] = str(_path(runtime[key], root))
    for model in result["model_pool"]["models"].values():
        if model.get("adapter_path"):
            model["adapter_path"] = str(_path(model["adapter_path"], root))
        tokenizer = str(model.get("tokenizer_path") or "")
        if tokenizer and (tokenizer.startswith(("/", ".", "~")) or (root / tokenizer).exists()):
            model["tokenizer_path"] = str(_path(tokenizer, root))
        if model.get("scoring_mode"):
            model["scoring_audit_path"] = str(run_dir / "reliability_scoring.jsonl")
    return result


def check_runtime(config: Mapping[str, Any]) -> None:
    validate_config(config)
    root = project_root()
    for model_id in active_model_ids(config):
        model = config["model_pool"]["models"][model_id]
        resolve_credentials(model)
        if model.get("scoring_mode") != "vllm_full_label_likelihood":
            continue
        if not model.get("tokenizer_path"):
            raise ValueError("the confidence scorer requires tokenizer_path")
        if not model.get("adapter_path"):
            continue
        adapter = _path(model["adapter_path"], root)
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            if not (adapter / filename).is_file():
                raise FileNotFoundError(
                    f"required reliability adapter file is missing: {adapter / filename}"
                )
        actual = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        if model.get("adapter_sha256") and actual != model["adapter_sha256"]:
            raise ValueError("reliability adapter checksum mismatch")


def _snapshot_prompts(config: Mapping[str, Any], run_dir: Path, *, resume: bool) -> PromptBuilder:
    policy = config["policy"]
    source = _path(config.get("prompt_root", "prompts"), project_root()) / policy["prompt_id"]
    target = run_dir / "prompt_snapshot"
    if not resume:
        target.mkdir(exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.is_file() and item.suffix in {".txt", ".json"}:
                shutil.copyfile(item, target / item.name)
        hashes = {
            item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted(target.iterdir())
        }
        write_json(run_dir / "prompt_manifest.json", hashes)
    else:
        for name, expected in _json(run_dir / "prompt_manifest.json").items():
            if hashlib.sha256((target / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"recorded prompt snapshot changed: {name}")
    return PromptBuilder(
        prompt_root=run_dir, prompt_id="prompt_snapshot", prompt_contract=policy["prompt_contract"]
    )


def _attach_call_journal(pool: ModelPool, run_dir: Path, position: dict[str, Any]) -> None:
    original = pool.generate
    request_sequence = len(read_jsonl(run_dir / "requests.jsonl"))

    def generate(model_id, messages, purpose, generation, *, position=None):
        nonlocal request_sequence
        current = journal_position.copy()
        request_sequence += 1
        request_id = f"request-{request_sequence:08d}"
        model = pool.config["models"][model_id]
        request = {
            **current,
            "request_id": request_id,
            "time": _now(),
            "model_id": model_id,
            "role": model["role"],
            "purpose": purpose,
            "position": asdict(position) if position is not None else None,
            "messages": [dict(message) for message in messages],
            "generation": dict(generation),
        }
        append_jsonl(run_dir / "requests.jsonl", request)
        try:
            call = original(model_id, messages, purpose, generation, position=position)
        except ModelInvocationError as exc:
            append_jsonl(
                run_dir / "responses.jsonl",
                {**current, "request_id": request_id, **exc.call.to_dict()},
            )
            raise
        append_jsonl(
            run_dir / "responses.jsonl", {**current, "request_id": request_id, **call.to_dict()}
        )
        return call

    journal_position = position
    pool.generate = generate


def _attach_environment_journal(worker, run_dir: Path, position: dict[str, Any]) -> None:
    reset, step = worker.reset, worker.step

    def recorded_reset(task, *, environment_seed):
        result = reset(task, environment_seed=environment_seed)
        append_jsonl(
            run_dir / "environment_transitions.jsonl",
            {
                **position,
                "event": "reset",
                "environment_seed": environment_seed,
                **result.to_dict(),
            },
        )
        return result

    def recorded_step(action):
        result = step(action)
        append_jsonl(
            run_dir / "environment_transitions.jsonl",
            {
                **position,
                "event": "step",
                "action": action,
                **result.to_dict(),
            },
        )
        return result

    worker.reset, worker.step = recorded_reset, recorded_step


def _attempts(run_dir: Path) -> list[dict[str, Any]]:
    return [_json(path) for path in sorted((run_dir / "attempts").glob("*.json"))]


def _materialize_logs(run_dir: Path) -> list[dict[str, Any]]:
    attempts = _attempts(run_dir)
    rows: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ("tasks", "episodes", "steps", "model_calls", "method_events", "errors")
    }
    annotated_calls = {}
    for attempt in attempts:
        tags = {key: attempt[key] for key in ("attempt_id", "episode_index", "committed")}
        for key in ("tasks", "episodes", "steps", "method_events", "errors"):
            rows[key].extend({**row, **tags} for row in attempt[key])
        for call in attempt["model_calls"]:
            annotated_calls[(attempt["attempt_id"], call["call_id"])] = {**call, **tags}
    responses = read_jsonl(run_dir / "responses.jsonl")
    returned = {response["request_id"] for response in responses}
    for response in responses:
        tags = annotated_calls.get((response["attempt_id"], response["call_id"]), {})
        rows["model_calls"].append({**response, **tags})

    for request in read_jsonl(run_dir / "requests.jsonl"):
        if request["request_id"] not in returned:
            rows["model_calls"].append(
                {
                    **request,
                    "call_id": request["request_id"],
                    "request_status": "interrupted_unknown",
                    "usage": {
                        "input_tokens": None,
                        "output_tokens": None,
                        "reasoning_tokens": None,
                    },
                    "error": "request was submitted without a recorded response",
                }
            )
    for identity, call in annotated_calls.items():
        if not any(
            (row.get("attempt_id"), row.get("call_id")) == identity for row in rows["model_calls"]
        ):
            rows["model_calls"].append(call)
    rows["errors"].extend(read_jsonl(run_dir / "runtime_errors.jsonl"))
    for key, values in rows.items():
        write_jsonl(run_dir / f"{key}.jsonl", values)
    return attempts


def _restore_state(policy, run_dir: Path, attempts: list[dict[str, Any]]) -> int:
    committed = [attempt for attempt in attempts if attempt["committed"]]
    latest = committed[-1] if committed else None
    state = latest["method_state"] if latest else _json(run_dir / "initial_method_state.json")
    policy.load_state_dict(state)
    write_json(run_dir / "method_state.json", state)
    write_jsonl(run_dir / "skill_bank.jsonl", policy.skill_records())
    return len(committed)


def run_experiment(
    config: Mapping[str, Any],
    source_config: str | Path,
    run_id: str | None = None,
    *,
    resume: bool = False,
) -> Path:
    validate_config(config)
    label = (
        run_id
        or f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{config['benchmark']['name']}_{config.get('method', 'i2s')}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError(
            "run_id must be a simple filename containing letters, digits, dots, underscores, or hyphens"
        )
    run_dir = _path(config.get("output_root", "runs"), project_root()) / label
    effective = runtime_config(config, run_dir)
    effective["config_path"] = str(Path(source_config).expanduser().resolve())
    fingerprint = _digest(_semantic_config(effective))
    if resume:
        manifest = _json(run_dir / "manifest.json")
        if manifest["config_sha256"] != fingerprint:
            raise ValueError(
                "resume requires the original method, task selection, budgets, endpoints, and seeds"
            )
        if manifest["status"] == "completed":
            return run_dir
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "attempts").mkdir()
        manifest = {
            "schema": "i2s-run-manifest",
            "run_id": label,
            "status": "preparing",
            "started_at": _now(),
            "registered_task_count": 0,
            "completed_task_count": 0,
            "benchmark": effective["benchmark"]["name"],
            "method": effective.get("method", "i2s"),
            "config_sha256": fingerprint,
            "source_config": Path(source_config).name,
            "resumable": True,
        }
        write_yaml(run_dir / "config.resolved.yaml", effective)
        write_json(run_dir / "method_state.json", {})

    lock_handle = (run_dir / ".writer.lock").open("a+")
    import fcntl

    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        raise RuntimeError("this run already has an active writer") from None
    write_json(run_dir / "manifest.json", manifest)
    worker = None
    pool = None
    policy = None
    interrupted = False
    try:
        check_runtime(effective)
        prompts = _snapshot_prompts(
            effective, run_dir, resume=resume and (run_dir / "prompt_manifest.json").is_file()
        )
        worker = BenchmarkWorkerClient(
            benchmark=effective["benchmark"]["name"],
            config=effective["benchmark"],
            package_root=project_root(),
            log_path=run_dir / "benchmark_worker.log",
            cache_root=run_dir / "cache",
            timeout_seconds=float(effective["benchmark"].get("worker_timeout_seconds", 300)),
        )
        inventory, details = worker.inventory()
        if resume and (run_dir / "task_selection.json").is_file():
            selection_record = _json(run_dir / "task_selection.json")
            by_id = {task.task_id: task for task in inventory}
            selected = []
            for task_data in selection_record["tasks"]:
                task = TaskSpec.from_dict(task_data)
                if task.task_id not in by_id or by_id[task.task_id].to_dict() != task.to_dict():
                    raise ValueError(
                        f"task inventory changed since the original run: {task.task_id}"
                    )
                selected.append(task)
        else:
            selected, selection = resolve_task_selection(inventory, effective)
            selection_record = build_selection_record(
                effective, selection, details, schema="task-selection"
            )
            selection_record["tasks"] = [task.to_dict() for task in selected]
            write_json(run_dir / "task_selection.json", selection_record)
        if len({task.family for task in selected}) > 1:
            raise ValueError(
                "one run must contain one independent family; select a scenario or use --all-families"
            )
        manifest["registered_task_count"] = len(selected)
        write_json(
            run_dir / "environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "benchmark_worker": worker.describe(),
                "inventory": details,
            },
        )
        pool = ModelPool(
            effective["model_pool"],
            run_id=label,
            generation_seed=effective["seeds"]["generation"],
            max_steps=effective["benchmark"]["max_steps"],
            max_calls_per_step=int(effective["execution"].get("max_calls_per_step", 8)),
        )
        journal_position: dict[str, Any] = {}
        _attach_call_journal(pool, run_dir, journal_position)
        _attach_environment_journal(worker, run_dir, journal_position)
        policy = build_policy(
            effective["policy"],
            repair_config=effective.get("repair", {}),
            router_config=effective["router"],
            model_pool=pool,
            prompt_builder=prompts,
        )
        if not (run_dir / "initial_method_state.json").is_file():
            write_json(run_dir / "initial_method_state.json", policy.state_dict())
        attempts = _materialize_logs(run_dir)
        completed = _restore_state(policy, run_dir, attempts)
        manifest.pop("stop_reason", None)
        manifest.update(
            status="running", completed_task_count=completed, resumed_at=_now() if resume else None
        )
        write_json(run_dir / "manifest.json", manifest)
        write_json(
            run_dir / "models.json",
            {
                model_id: pool.public_registry(effective["generation"])[model_id]
                for model_id in policy.model_ids
            },
        )
        executor = EpisodeExecutor(
            worker=worker,
            policy=policy,
            benchmark_config=effective["benchmark"],
            context_config=effective["context"],
            generation=effective["generation"],
            seeds=effective["seeds"],
        )
        offset = int(effective["execution"].get("episode_index_offset", 0))
        for sequence_index, task in enumerate(selected, start=1):
            if sequence_index <= completed:
                continue
            episode_index = sequence_index + offset
            prior = sum(attempt["episode_index"] == episode_index for attempt in attempts)

            starts = read_jsonl(run_dir / "attempt_starts.jsonl")
            prior = max(prior, sum(start["episode_index"] == episode_index for start in starts))
            attempt_id = f"e{episode_index:05d}-a{prior + 1:03d}"
            pool.run_id = label if prior == 0 else f"{label}-retry{prior + 1}"
            request = EpisodeRequest(
                sequence_index,
                episode_index,
                f"{label}:e{episode_index:05d}",
                task,
                effective["seeds"]["environment"] + episode_index - 1,
            )
            journal_position.update(
                attempt_id=attempt_id, episode_index=episode_index, episode_id=request.episode_id
            )
            append_jsonl(
                run_dir / "attempt_starts.jsonl",
                {**journal_position, "time": _now(), "task_id": task.task_id},
            )
            before = copy.deepcopy(policy.state_dict())
            bundle = executor.execute(request)
            committed = not bundle.infrastructure_failure
            state = policy.state_dict() if committed else before
            attempt = {
                "attempt_id": attempt_id,
                "episode_index": episode_index,
                "committed": committed,
                "request": asdict(request),
                "tasks": [dict(bundle.task_row)],
                "episodes": [dict(bundle.episode_row)],
                "steps": list(bundle.step_rows),
                "model_calls": list(bundle.model_call_rows),
                "method_events": list(bundle.method_event_rows),
                "errors": list(bundle.error_rows),
                "trajectory": asdict(bundle.visible_trajectory),
                "method_state": state,
            }
            write_json(run_dir / "attempts" / f"{attempt_id}.json", attempt)
            attempts = _materialize_logs(run_dir)
            completed = _restore_state(policy, run_dir, attempts)
            manifest["completed_task_count"] = completed
            manifest["last_attempt"] = attempt_id
            if not committed:
                manifest.update(
                    status="stopped",
                    stop_reason=bundle.episode_row.get("method_update_error")
                    or bundle.episode_row.get("error")
                    or bundle.episode_row["status"],
                )
                break
            write_json(run_dir / "manifest.json", manifest)
        else:
            manifest.update(status="completed", resumable=False)
    except KeyboardInterrupt:
        interrupted = True
        manifest.update(status="interrupted", stop_reason="user interruption")
        append_jsonl(
            run_dir / "runtime_errors.jsonl",
            {"time": _now(), "stage": "run", "error": "KeyboardInterrupt"},
        )
    except Exception as exc:
        manifest.update(status="stopped", stop_reason=f"{type(exc).__name__}: {exc}")
        append_jsonl(
            run_dir / "runtime_errors.jsonl",
            {
                "time": _now(),
                "stage": "run",
                "error": manifest["stop_reason"],
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        for resource in (pool, worker):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    append_jsonl(
                        run_dir / "runtime_errors.jsonl",
                        {"time": _now(), "stage": "close", "error": f"{type(exc).__name__}: {exc}"},
                    )
        attempts = _materialize_logs(run_dir)
        if policy is not None and (run_dir / "initial_method_state.json").is_file():
            _restore_state(policy, run_dir, attempts)
        manifest["updated_at"] = _now()
        evidence = [
            *run_dir.glob("*.jsonl"),
            run_dir / "config.resolved.yaml",
            run_dir / "task_selection.json",
            run_dir / "method_state.json",
            *sorted((run_dir / "attempts").glob("*.json")),
            *sorted((run_dir / "prompt_snapshot").glob("*")),
        ]
        manifest["evidence_sha256"] = {
            str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in evidence
            if path.is_file()
        }
        write_json(run_dir / "manifest.json", manifest)
        evaluation = evaluate_run(run_dir)
        manifest["artifact_validation_passed"] = evaluation["validation"]["valid"]
        write_json(run_dir / "manifest.json", manifest)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
    if interrupted:
        raise KeyboardInterrupt
    return run_dir


def family_configs(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    declared = config["benchmark"].get("families")
    if not isinstance(declared, list) or not declared:
        raise ValueError("--all-families requires benchmark.families in environments.yaml")
    result = []
    names = set()
    for family in declared:
        override = {"scenario": family} if isinstance(family, str) else copy.deepcopy(dict(family))
        name = str(override.pop("name", override.get("scenario", "")))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("family names must be unique, nonempty filename components")
        names.add(name)
        child = copy.deepcopy(dict(config))
        child["benchmark"] = deep_merge(child["benchmark"], override)
        child["benchmark"].pop("families", None)
        result.append((name, child))
    return result


def run_families(
    config: Mapping[str, Any],
    source_config: str | Path,
    run_id: str | None = None,
    *,
    resume: bool = False,
) -> list[Path]:
    families = family_configs(config)
    prefix = (
        run_id
        or f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{config['benchmark']['name']}_{config.get('method', 'i2s')}"
    )
    workers = min(int(config["execution"].get("family_workers", 1)), len(families))

    def run(item):
        name, child = item
        return run_experiment(child, source_config, f"{prefix}_{name}", resume=resume)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, families))
