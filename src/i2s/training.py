from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .confidence_scoring import LABELS, assessment_messages, encode_pair, label_log_likelihoods

DEFAULT_MODEL = "Qwen/Qwen3-8B"
THRESHOLDS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
NATIVE_FILES = (
    "manifest.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "steps.jsonl",
    "model_calls.jsonl",
    "method_events.jsonl",
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation: int = 4
    gradient_clip: float = 1.0
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    seed: int = 7311
    max_input_tokens: int = 8192
    dtype: str = "bfloat16"

    def validate(self) -> None:
        for field in ("epochs", "gradient_accumulation", "rank", "alpha", "max_input_tokens"):
            if not isinstance(getattr(self, field), int) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL file is missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fresh_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def public_model_id(model: str, model_id: str | None = None) -> str:
    local = Path(model).exists() or Path(model).is_absolute() or model.startswith((".", "~"))
    reference = model_id if model_id is not None else None if local else model
    if reference is None or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", reference
    ):
        raise ValueError(
            "Use a public model ID, or supply --model-id owner/model when --model is a local path"
        )
    return reference


def task_key(row: Mapping[str, Any]) -> str:
    return _hash([row["benchmark"], row["task_id"]])


def _input_id(messages: Sequence[Mapping[str, str]], action: str) -> str:
    return _hash({"messages": assessment_messages(list(messages)), "action": action})


def _check_messages(messages, action):
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user"}
            or not isinstance(message.get("content"), str)
            for message in messages
        )
        or not any(message["role"] == "user" for message in messages)
    ):
        raise ValueError("Messages must contain only pre-action system/user text")
    proposals = re.findall(r"Proposed action:\n([^\n]+)", "\n".join(m["content"] for m in messages))
    if not proposals or proposals[-1].strip() != action:
        raise ValueError("The assessment prompt must contain the exact proposed action")


def read_records(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No labelled records in {path}")
    seen: set[str] = set()
    for row in rows:
        for field in ("record_id", "task_id", "benchmark", "proposed_action", "label_rule"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Every record needs a nonempty {field}")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(
                "Labels must be 0 (UNCERTAIN) or 1 (CONFIDENT); unknown labels must be excluded"
            )
        if not isinstance(row.get("label_evidence"), dict) or not row["label_evidence"]:
            raise ValueError("Every record needs localized label_evidence")
        messages = row.get("messages")
        _check_messages(messages, row["proposed_action"])
        if row["record_id"] != _input_id(messages, row["proposed_action"]):
            raise ValueError("record_id does not match the visible input digest")
        if row["record_id"] in seen:
            raise ValueError("Duplicate visible inputs must be resolved before training")
        seen.add(row["record_id"])
    return rows


def validate_splits(
    train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]
) -> None:
    if not train or not validation:
        raise ValueError("Both training and validation data are required")
    if {task_key(row) for row in train} & {task_key(row) for row in validation}:
        raise ValueError("Training and validation tasks overlap")
    if {row["record_id"] for row in train} & {row["record_id"] for row in validation}:
        raise ValueError("Training and validation visible inputs overlap")
    for name, rows in (("training", train), ("validation", validation)):
        if {row["label"] for row in rows} != {0, 1}:
            raise ValueError(
                f"Both labels are required in {name}; collect more supported task groups"
            )


def _kind(entity: str) -> str:
    return re.sub(r" \d+$", "", entity.strip())


def _after(step: Mapping[str, Any]) -> str:
    return str(step.get("policy_observation_after") or step.get("observation_after") or "").strip()


def _visible_state(
    prior: Sequence[Mapping[str, Any]],
) -> tuple[str | None, dict[str, str], set[str]]:
    held, known, visited = None, {}, set()
    for step in prior:
        action, observation = str(step["action"]), _after(step)
        match = re.fullmatch(r"take (.+) from (.+)", action)
        if match and f"You pick up the {match[1]} from the {match[2]}." in observation:
            held = match[1]
            known.pop(match[1], None)
        match = re.fullmatch(r"move (.+) to (.+)", action)
        if match and f"You move the {match[1]} to the {match[2]}." in observation:
            held = None
            known[match[1]] = match[2]
        match = re.match(r"You arrive at (.+?)\.", observation)
        if match:
            visited.add(match[1])
        match = re.search(r"(?:On|In) the (.+?), you see (.+)\.", observation)
        if match:
            for entity in re.findall(r"([a-z]+ \d+)", match[2]):
                known[entity] = match[1]
    return held, known, visited


def _recovery(prior, future, anchors, horizon, target, destination):
    seen = {_after(step) for step in prior}
    for step in future[:horizon]:
        if not 0 < step["step_index"] - prior[-1]["step_index"] <= horizon:
            continue
        action = str(step["action"])
        for anchor in anchors.get((step["episode_id"], step["step_index"]), []):
            if anchor.get("eligibility") is not True or anchor.get("correction") != action:
                continue
            if step.get("action_valid") is not True or step.get("execution_error"):
                continue
            take = re.fullmatch(r"take (.+) from (.+)", action)
            place = re.fullmatch(r"move (.+) to (.+)", action)
            if (
                (action.startswith(("go to ", "open ")) and _after(step) not in seen)
                or (take and _kind(take[1]) == target)
                or (place and _kind(place[1]) == target and _kind(place[2]) == destination)
            ):
                return {
                    "step_index": step["step_index"],
                    "action": action,
                    "observation_after": _after(step),
                    "bounded_step_distance": step["step_index"] - prior[-1]["step_index"],
                }
    return None


def classify_local_evidence(proposal, messages, step, prior, future, anchors, source_success):
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    instruction_match = re.search(r"Task instruction:\n(.*?)\n\n", user, re.S)
    instruction = instruction_match[1].strip() if instruction_match else ""
    roles = re.fullmatch(r"put (?:some|a|an) ([a-z]+) (?:on|in) (?:a |an )?([a-z]+)\.", instruction)
    proof: dict[str, Any] = {"instruction": instruction}
    action = proposal.get("proposed_action")
    if proposal.get("proposal_executed") is not True:
        return None, "WITHHELD_COUNTERFACTUAL_UNKNOWN", proof
    if (
        step is None
        or step.get("action") != action
        or step.get("intervention") == "repair"
        or step.get("source_role") == "large"
    ):
        return None, "NO_IDENTICAL_SMALL_TRANSITION", proof
    if not roles:
        return None, "UNSUPPORTED_TASK_INSTRUCTION", proof
    target, destination = roles.groups()
    before_match = re.search(r"Current observation:\n(.*?)\n\nAdmissible actions", user, re.S)
    before, after = before_match[1].strip() if before_match else "", _after(step)
    proof.update(
        observation_before=before,
        observation_after=after,
        step_index=step["step_index"],
        source_success=source_success,
    )
    if step.get("execution_error") or step.get("action_valid") is False:
        return 0, "EXPLICIT_EXECUTION_FAILURE", proof
    if step.get("action_valid") is not True or not before or not after:
        return None, "MISSING_PUBLIC_TRANSITION_EVIDENCE", proof
    visible_prior = [
        s
        for s in prior
        if s.get("action") and str(s["action"]) in user and _after(s) and _after(s) in user
    ]
    held, known, visited = _visible_state(visible_prior)
    placement = re.fullmatch(r"move (.+) to (.+)", action)
    if placement and _kind(placement[1]) == target and _kind(placement[2]) != destination:
        recovery = _recovery([*prior, step], future, anchors, 8, target, destination)
        if recovery:
            return 0, "WRONG_DESTINATION_WITH_SUPPORTED_RECOVERY", {**proof, "recovery": recovery}
    if before == after and any(s["action"] == action and _after(s) == after for s in visible_prior):
        recovery = _recovery([*prior, step], future, anchors, 3, target, destination)
        if recovery:
            return 0, "OBSERVABLE_REPEAT_NO_EFFECT_WITH_RECOVERY", {**proof, "recovery": recovery}
    if source_success is not True:
        return None, "NO_SUPPORTED_LOCAL_NEGATIVE", proof
    take = re.fullmatch(r"take (.+) from (.+)", action)
    if (
        take
        and _kind(take[1]) == target
        and f"You pick up the {take[1]} from the {take[2]}." in after
    ):
        return 1, "EXACT_TARGET_ACQUISITION", proof
    if (
        placement
        and _kind(placement[1]) == target
        and _kind(placement[2]) == destination
        and f"You move the {placement[1]} to the {placement[2]}." in after
    ):
        return 1, "EXACT_GOAL_PLACEMENT", proof
    navigation = re.fullmatch(r"go to (.+)", action)
    if navigation and after.startswith(f"You arrive at {navigation[1]}.") and before != after:
        if held and _kind(held) == target and _kind(navigation[1]) == destination:
            return 1, "HELD_TARGET_DESTINATION_ARRIVAL", proof
        if not held and any(
            _kind(obj) == target and location == navigation[1] for obj, location in known.items()
        ):
            return 1, "VISIBLE_TARGET_SOURCE_ARRIVAL", proof
        if (
            not held
            and navigation[1] not in visited
            and not any(_kind(obj) == target for obj in known)
            and "you see" in after
        ):
            return 1, "FIRST_EXPLORATION_VISIBLE_STATE", proof
    opening = re.fullmatch(r"open (.+)", action)
    if (
        opening
        and not held
        and "you open" in after.lower()
        and "you see" in after.lower()
        and not any(s["action"] == action for s in visible_prior)
    ):
        return 1, "FIRST_OPENING_VISIBLE_STATE", proof
    return None, "AMBIGUOUS_LOCAL_PROGRESS", proof


def collect_native_records(run: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    for name in NATIVE_FILES:
        if not (run / name).is_file():
            raise FileNotFoundError(f"Native run is missing {name}: {run}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("artifact_validation_passed") is not True
    ):
        raise ValueError(
            "Calibration data preparation requires completed, artifact-validated benchmark runs"
        )

    def committed(name):
        return [row for row in read_jsonl(run / name) if row.get("committed") is not False]

    tasks = {row["task_id"]: row for row in committed("tasks.jsonl")}
    if not tasks or any(row.get("split") != "train" for row in tasks.values()):
        raise ValueError("Confidence calibration requires benchmark training-split tasks")
    episodes = {
        (row["episode_id"], row.get("attempt_id")): row for row in committed("episodes.jsonl")
    }
    calls = read_jsonl(run / "model_calls.jsonl")
    call_by_id = {row["call_id"]: row for row in calls}
    assessments: dict[tuple, list] = defaultdict(list)
    for row in calls:
        if row.get("purpose") == "confidence_assessment" and row.get("committed") is not False:
            assessments[(row["episode_id"], row["step_index"], row.get("attempt_id"))].append(row)
    steps: dict[tuple, list] = defaultdict(list)
    for row in committed("steps.jsonl"):
        steps[(row["episode_id"], row.get("attempt_id"))].append(row)
    for group in steps.values():
        group.sort(key=lambda row: row["step_index"])
    events = committed("method_events.jsonl")
    anchors: dict[Any, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.get("event") == "intervention_anchor":
            for transition in event.get("localized_evidence", []):
                if transition.get("action") == event.get("correction"):
                    anchors[event.get("attempt_id")][
                        (event["episode_id"], transition["step_index"])
                    ].append(event)
    hashes = {name: file_hash(run / name) for name in NATIVE_FILES}
    source_id = _hash(hashes)
    records = []
    for proposal in events:
        if proposal.get("event") != "small_proposal":
            continue
        task = tasks.get(proposal.get("task_id"))
        source_call = call_by_id.get(proposal.get("context_call_id"))
        if task is None or source_call is None:
            raise ValueError("A small proposal has no matching task or source model call")
        episode_id, index = source_call["episode_id"], source_call["step_index"]
        attempt = source_call.get("attempt_id")
        if proposal.get("attempt_id") != attempt:
            raise ValueError("Proposal and source model call belong to different attempts")
        matches = assessments.get((episode_id, index, attempt), [])
        action = proposal.get("proposed_action")
        base = {
            "task_id": task["task_id"],
            "benchmark": task["benchmark"],
            "source_id": source_id,
            "source_example": _hash([source_id, proposal.get("context_call_id")]),
            "proposed_action": action,
        }
        if len(matches) != 1 or not isinstance(action, str) or not action:
            records.append({**base, "label": None, "label_rule": "MISSING_OR_AMBIGUOUS_ASSESSMENT"})
            continue
        call = matches[0]
        messages = call["messages"]
        _check_messages(messages, action)
        episode = episodes.get((episode_id, attempt))
        if episode is None or episode.get("task_id") != task["task_id"]:
            raise ValueError("A proposal has no matching completed task episode")
        group = steps.get((episode_id, attempt), [])
        current = [row for row in group if row["step_index"] == index]
        if len(current) > 1:
            raise ValueError("Multiple executed transitions share an episode step")
        if task["benchmark"] != "alfworld":
            label, rule, proof = None, "UNSUPPORTED_BENCHMARK", {}
        else:
            label, rule, proof = classify_local_evidence(
                proposal,
                messages,
                current[0] if current else None,
                [row for row in group if row["step_index"] < index],
                [row for row in group if row["step_index"] > index],
                anchors[attempt],
                episode.get("success"),
            )
        records.append(
            {
                **base,
                "record_id": _input_id(messages, action),
                "messages": assessment_messages(messages),
                "label": label,
                "label_rule": rule,
                "label_evidence": proof,
            }
        )
    if not records:
        raise ValueError(
            "No small proposals found; collect runs with confidence assessments enabled"
        )
    return records, hashes


def split_records(records, *, validation_fraction=0.3, seed=10120, train_cap=12):
    if not 0 < validation_fraction < 1 or train_cap <= 0:
        raise ValueError("Require 0 < validation_fraction < 1 and a positive per-task class cap")
    supported = [row for row in records if row.get("label") in (0, 1)]
    unique = {}
    for row in sorted(supported, key=lambda r: (r["record_id"], task_key(r))):
        previous = unique.get(row["record_id"])
        if previous is not None:
            if previous["label"] != row["label"]:
                raise ValueError("Conflicting labels for identical visible inputs")
            if task_key(previous) != task_key(row):
                raise ValueError(
                    "Identical inputs occur under different tasks; resolve task identity before splitting"
                )
        else:
            unique[row["record_id"]] = row
    groups: dict[str, list] = defaultdict(list)
    for row in unique.values():
        groups[task_key(row)].append(row)

    def rank(key):
        return _hash([seed, key])

    negative = sorted(
        (key for key, rows in groups.items() if any(row["label"] == 0 for row in rows)), key=rank
    )
    positive = sorted(set(groups) - set(negative), key=rank)
    validation_tasks = set()
    for stratum in (negative, positive):
        if len(stratum) > 1:
            count = max(1, min(len(stratum) - 1, round(len(stratum) * validation_fraction)))
            validation_tasks.update(stratum[:count])
    train, validation = [], []
    counts: Counter = Counter()
    for row in sorted(unique.values(), key=lambda r: r["record_id"]):
        key = task_key(row)
        if key in validation_tasks:
            validation.append(row)
        elif counts[(key, row["label"])] < train_cap:
            train.append(row)
            counts[(key, row["label"])] += 1
    validate_splits(train, validation)
    return train, validation


def prepare_data(
    runs: Sequence[Path],
    output: Path,
    *,
    validation_fraction=0.3,
    seed=10120,
    train_cap=12,
    exclude_tasks: Sequence[Path] = (),
) -> dict[str, Any]:
    if not runs:
        raise ValueError("Supply at least one freshly collected native run")
    excluded_tasks = {task_key(row) for path in exclude_tasks for row in read_jsonl(path)}
    records, sources = [], []
    for run in runs:
        rows, hashes = collect_native_records(run)
        if any(task_key(row) in excluded_tasks for row in rows):
            raise ValueError("A source run overlaps the supplied held-out task inventory")
        records.extend(rows)
        sources.append(hashes)
    train, validation = split_records(
        records, validation_fraction=validation_fraction, seed=seed, train_cap=train_cap
    )
    report = {
        "labels": list(LABELS),
        "label_scope": "Rule-derived local evidence; ALFWorld simple placement tasks",
        "input_boundary": "Pre-action visible assessment only; label evidence is never a model input",
        "split_rule": "Task-group split stratified by negative support, before model score inspection",
        "validation_fraction": validation_fraction,
        "seed": seed,
        "train_cap_per_task_and_label": train_cap,
        "source_file_hashes": sources,
        "source_records": len(records),
        "label_rules": dict(Counter(row["label_rule"] for row in records)),
        "train": _dataset_summary(train),
        "validation": _dataset_summary(validation),
        "task_overlap": 0,
        "excluded_task_inventory_hashes": [file_hash(path) for path in exclude_tasks],
    }
    fresh_output(output)
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "validation.jsonl", validation)
    retained = {(row["source_example"], row["record_id"]) for row in train + validation}
    _write_jsonl(
        output / "excluded.jsonl",
        [row for row in records if (row["source_example"], row.get("record_id")) not in retained],
    )
    _write_json(output / "data_manifest.json", report)
    return report


def _dataset_summary(rows) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "tasks": len({task_key(row) for row in rows}),
        "labels": {LABELS[label]: sum(row["label"] == label for row in rows) for label in (0, 1)},
    }


def metrics(labels: Sequence[int], scores: Sequence[float], threshold=0.99) -> dict[str, Any]:
    if (
        not labels
        or len(labels) != len(scores)
        or any(type(label) is not int or label not in (0, 1) for label in labels)
    ):
        raise ValueError("Metrics require equally sized, nonempty binary labels and scores")
    if not 0 <= threshold <= 1 or any(
        not math.isfinite(score) or not 0 <= score <= 1 for score in scores
    ):
        raise ValueError("Scores and threshold must be finite probabilities")
    predicted = [int(score >= threshold) for score in scores]
    tn, fp, fn, tp = (
        sum(y == a and p == b for y, p in zip(labels, predicted, strict=True))
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    positive, negative = sum(labels), len(labels) - sum(labels)

    by_score: dict[float, Counter] = defaultdict(Counter)
    for label, score in zip(labels, scores, strict=True):
        by_score[score][label] += 1
    wins, preceding_negatives = 0.0, 0
    for score in sorted(by_score):
        group = by_score[score]
        wins += group[1] * (preceding_negatives + 0.5 * group[0])
        preceding_negatives += group[0]
    return {
        "threshold": threshold,
        "positive": positive,
        "negative": negative,
        "auroc": wins / (positive * negative) if positive and negative else None,
        "balanced_accuracy": (tp / positive + tn / negative) / 2 if positive and negative else None,
        "accuracy": (tp + tn) / len(labels),
        "coverage": (tp + fp) / len(labels),
        "selective_risk": fp / (tp + fp) if tp + fp else None,
        "brier": sum((score - label) ** 2 for label, score in zip(labels, scores, strict=True))
        / len(labels),
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_true_columns_predicted": [[tn, fp], [fn, tp]],
        },
    }


def lora_parameters(model) -> list[Any]:
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Training requires trainable calibration adapters and a frozen backbone")
    return [parameter for _, parameter in trainable]


def accumulation_size(index: int, total: int, accumulation: int) -> int:
    return min(accumulation, total - (index // accumulation) * accumulation)


def check_adapter(adapter: Path, model_id: str) -> dict[str, Any]:
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(
            "A trained calibration adapter with configuration and weights is required"
        )
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != model_id:
        raise ValueError(
            "Adapter type or public base-model ID does not match the requested backbone"
        )
    provenance = adapter / "reliability_config.json"
    if not provenance.is_file():
        raise FileNotFoundError(
            "Adapter is missing reliability_config.json for the training-task overlap check"
        )
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    for field in ("training_task_hashes", "training_input_hashes"):
        if not isinstance(metadata.get(field), list) or not metadata[field]:
            raise ValueError(f"Adapter {field} provenance is missing")
    return metadata


def save_adapter(model, adapter: Path, metadata: Mapping[str, Any]) -> None:
    reference = metadata["base_model"]
    for adapter_config in model.peft_config.values():
        adapter_config.base_model_name_or_path = reference
    model.get_base_model().config._name_or_path = reference
    model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    _write_json(adapter / "reliability_config.json", metadata)


def _dependencies():
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            'Install the training dependencies with pip install -e ".[training]"'
        ) from exc
    return torch, LoraConfig, PeftModel, get_peft_model, AutoModelForCausalLM, AutoTokenizer


def _evaluate(model, tokenizer, rows, encoded, threshold):
    import torch

    model.eval()
    predictions, usage = [], {"candidate_input_tokens": 0, "candidate_label_tokens": 0}
    with torch.inference_mode():
        for row in rows:
            prompt, labels = encoded[row["record_id"]]
            scores = label_log_likelihoods(model, tokenizer, prompt, labels)
            predictions.append(
                {
                    "record_id": row["record_id"],
                    "task_hash": task_key(row),
                    "label": row["label"],
                    "confidence": scores.softmax(0)[1].item(),
                    "label_log_likelihoods": dict(zip(LABELS, scores.tolist(), strict=True)),
                }
            )
            usage["candidate_input_tokens"] += len(prompt) * len(LABELS)
            usage["candidate_label_tokens"] += sum(map(len, labels))
    result = metrics(
        [row["label"] for row in predictions], [row["confidence"] for row in predictions], threshold
    )
    return predictions, result, usage


def run_training(
    *,
    model: str,
    train_path: Path,
    validation_path: Path,
    output: Path,
    model_id: str | None = None,
    config: TrainingConfig | None = None,
    device: str = "cuda:0",
    threshold: float = 0.99,
) -> dict[str, Any]:
    if config is None:
        config = TrainingConfig()
    config.validate()
    reference = public_model_id(model, model_id)
    train, validation = read_records(train_path), read_records(validation_path)
    validate_splits(train, validation)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    torch, LoraConfig, _, get_peft_model, AutoModelForCausalLM, AutoTokenizer = _dependencies()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an available device explicitly")
    fresh_output(output)
    metadata = {
        **asdict(config),
        "base_model": reference,
        "labels": list(LABELS),
        "target_modules": ["q_proj", "v_proj"],
        "objective": "Inverse-class-frequency weighted cross entropy of both complete label log likelihoods",
        "only_adapter_trainable": True,
        "epoch_selection": "Final epoch",
        "threshold": threshold,
        "training_task_hashes": sorted({task_key(row) for row in train}),
        "training_input_hashes": sorted(row["record_id"] for row in train),
        "validation_task_hashes": sorted({task_key(row) for row in validation}),
        "data_sha256": {"train": file_hash(train_path), "validation": file_hash(validation_path)},
        "data": {"train": _dataset_summary(train), "validation": _dataset_summary(validation)},
    }
    _write_json(output / "training_config.json", metadata)
    _write_json(output / "status.json", {"status": "running"})
    try:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(model)
        encoded = {
            row["record_id"]: encode_pair(
                tokenizer, row["messages"], max_input_tokens=config.max_input_tokens
            )
            for row in train + validation
        }
        base = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=getattr(torch, config.dtype),
            attn_implementation="sdpa",
            device_map={"": device},
        )
        base.config.use_cache = False
        network = get_peft_model(
            base,
            LoraConfig(
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        network.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        network.enable_input_require_grads()
        parameters = lora_parameters(network)
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        counts = Counter(row["label"] for row in train)
        usage = {"candidate_input_tokens": 0, "candidate_label_tokens": 0}
        with (output / "training_progress.jsonl").open("w", encoding="utf-8") as progress:
            for epoch in range(config.epochs):
                order = list(train)
                random.Random(config.seed + epoch).shuffle(order)
                network.train()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index, row in enumerate(order):
                    prompt, labels = encoded[row["record_id"]]
                    scores = label_log_likelihoods(network, tokenizer, prompt, labels)
                    gold = row["label"]
                    loss = torch.nn.functional.cross_entropy(
                        scores[None], torch.tensor([gold], device=scores.device)
                    )
                    weight = len(train) / (len(LABELS) * counts[gold])
                    (
                        loss
                        * weight
                        / accumulation_size(index, len(order), config.gradient_accumulation)
                    ).backward()
                    losses.append(float(loss.detach()))
                    usage["candidate_input_tokens"] += len(prompt) * len(LABELS)
                    usage["candidate_label_tokens"] += sum(map(len, labels))
                    if (index + 1) % config.gradient_accumulation == 0 or index + 1 == len(order):
                        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    if (index + 1) % 10 == 0 or index + 1 == len(order):
                        event = {
                            "epoch": epoch + 1,
                            "decision": index + 1,
                            "total": len(order),
                            "mean_loss": sum(losses) / len(losses),
                        }
                        line = json.dumps(event, allow_nan=False)
                        progress.write(line + "\n")
                        progress.flush()
                        print(line, flush=True)
        save_adapter(network, output, metadata)
        predictions, result, validation_usage = _evaluate(
            network, tokenizer, validation, encoded, threshold
        )
        grid = [
            metrics(
                [row["label"] for row in predictions],
                [row["confidence"] for row in predictions],
                value,
            )
            for value in THRESHOLDS
        ]
        selected = max(
            grid, key=lambda item: (item["balanced_accuracy"], item["coverage"], -item["threshold"])
        )
        report = {
            "validation": result,
            "threshold_grid": grid,
            "validation_selected_threshold": selected["threshold"],
            "threshold_scope": "Action-level validation threshold selection; deployment uses the configured routing threshold",
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "training_usage": usage,
            "validation_usage": validation_usage,
        }
        _write_jsonl(output / "validation_predictions.jsonl", predictions)
        _write_json(output / "validation_metrics.json", report)
        _write_json(output / "status.json", {"status": "completed"})
        return report
    except BaseException as exc:
        _write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise


def run_evaluation(
    *,
    model: str,
    adapter: Path | None = None,
    validation_path: Path,
    output: Path,
    model_id: str | None = None,
    device: str = "cuda:0",
    threshold: float = 0.99,
) -> dict[str, Any]:
    if adapter is not None:
        reference = public_model_id(model, model_id)
        metadata = check_adapter(adapter, reference)
    else:
        from huggingface_hub import hf_hub_download

        reference = model_id or model
        metadata_path = (
            Path(model) / "reliability_config.json"
            if Path(model).is_dir()
            else Path(hf_hub_download(model, "reliability_config.json"))
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = read_records(validation_path)
    if set(metadata["training_task_hashes"]) & {task_key(row) for row in rows}:
        raise ValueError("Evaluation tasks overlap the confidence model's training tasks")
    if set(metadata["training_input_hashes"]) & {row["record_id"] for row in rows}:
        raise ValueError("Evaluation visible inputs overlap the confidence model's training inputs")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    torch, _, PeftModel, _, AutoModelForCausalLM, AutoTokenizer = _dependencies()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an available device explicitly")
    fresh_output(output)
    _write_json(output / "status.json", {"status": "running"})
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        encoded = {
            row["record_id"]: encode_pair(
                tokenizer, row["messages"], max_input_tokens=metadata["max_input_tokens"]
            )
            for row in rows
        }
        base = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=getattr(torch, metadata.get("weight_dtype", metadata["dtype"])),
            attn_implementation="sdpa",
            device_map={"": device},
        )
        network = (
            PeftModel.from_pretrained(base, adapter, is_trainable=False)
            if adapter is not None else base
        ).eval()
        predictions, result, usage = _evaluate(network, tokenizer, rows, encoded, threshold)
        report = {
            "metrics": result,
            "usage": usage,
            "model": reference,
            "base_model": metadata["base_model"],
            "data_sha256": file_hash(validation_path),
            "reuses_training_validation": bool(
                set(metadata.get("validation_task_hashes", [])) & {task_key(row) for row in rows}
            ),
            "scope": "Action-conditioned confidence evaluation",
        }
        _write_jsonl(output / "predictions.jsonl", predictions)
        _write_json(output / "metrics.json", report)
        _write_json(output / "status.json", {"status": "completed"})
        return report
    except BaseException as exc:
        _write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
