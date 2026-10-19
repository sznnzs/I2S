from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Callable, Mapping

from . import semantic_checks

HINT_PROTOCOL = "public_task_role_types"
PARSER_COMPATIBILITY = "alfworld_official_look_under"
CONFIG_FIELD = "induction_task_role_hints"
PAYLOAD_FIELD = "public_task_role_hints"
_SYSTEM_CLARIFICATION = (
    "\nPUBLIC TASK ROLE TYPES: public_task_role_hints contains only constraints "
    "parsed from the public task instruction. These are TYPE names, not native "
    "instance bindings, observation/effect evidence, or proof of success. Use "
    "only the exact localized evidence to bind actual instances and support "
    "every proposed relation, operation, and outcome; all existing validation "
    "requirements remain unchanged.\n"
)


def build_task_role_hints(
    instruction: str,
    *,
    benchmark: str = "alfworld",
    task_role_parser: Callable[[str], Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol": HINT_PROTOCOL,
        "benchmark": benchmark,
        "source": "public_task_instruction",
        "status": "unsupported_benchmark",
        "task_role_types": {},
        "literal_instance_bindings_supplied": False,
        "environment_evidence_supplied": False,
        "generated_proposal_modified": False,
        "semantic_validator_modified_by_hints": False,
    }
    if benchmark != "alfworld":
        return result
    if not isinstance(instruction, str) or not instruction.strip():
        result["status"] = "invalid_instruction"
        return result
    parser = task_role_parser if task_role_parser is not None else semantic_checks.task_roles
    parsed = parser(instruction)
    if not parsed:
        result["status"] = "unsupported_instruction_syntax"
        return result
    if not isinstance(parsed, Mapping) or set(parsed) not in (
        {"target", "destination"},
        {"target", "station"},
    ):
        result["status"] = "ambiguous_parser_result"
        return result
    types = dict(parsed)
    for value in types.values():
        if not isinstance(value, str) or not re.fullmatch(r"[a-z]+", value):
            result["status"] = "non_type_parser_result"
            return result
        if re.search(r"\b" + re.escape(value) + r"\b", instruction.lower()) is None:
            result["status"] = "parser_role_not_in_public_instruction"
            return result
    result.update(status="supported_role_types", task_role_types=copy.deepcopy(types))
    return result


def _enabled(learning: Mapping[str, Any], benchmark: str) -> bool:
    configured = learning.get(CONFIG_FIELD)
    if configured is None:
        return False
    if configured != HINT_PROTOCOL:
        raise ValueError("unsupported_induction_task_role_hints")
    if benchmark != "alfworld":
        return False
    if learning.get("revision_mode") != "semantic":
        raise ValueError("task_role_hints_require_semantic_induction")
    if learning.get("trajectory_projection") or learning.get("ablation"):
        raise ValueError("task_role_hints_projection_or_ablation_unsupported")
    return True


def augment_induction_payload(
    payload: Mapping[str, Any], learning: Mapping[str, Any], benchmark: str
) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
    if not _enabled(learning, benchmark):
        return payload, None
    if PAYLOAD_FIELD in payload:
        raise ValueError("task_role_hints_payload_already_augmented")
    trajectory = payload.get("completed_trajectory")
    if not isinstance(trajectory, Mapping) or not isinstance(trajectory.get("instruction"), str):
        raise ValueError("task_role_hints_missing_public_instruction")
    instruction = trajectory["instruction"]
    hints = build_task_role_hints(instruction, benchmark=benchmark)
    result = copy.deepcopy(payload)
    result[PAYLOAD_FIELD] = hints
    digest = hashlib.sha256(
        json.dumps(
            hints, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return result, {
        "event": "induction_task_role_hints",
        "protocol": HINT_PROTOCOL,
        "status": hints["status"],
        "task_role_types": copy.deepcopy(hints["task_role_types"]),
        "public_instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "hints_sha256": digest,
        "payload_field": PAYLOAD_FIELD,
        "native_evidence_added": False,
        "generated_proposal_modified": False,
        "semantic_validator_modified_by_hints": False,
    }


def clarify_induction_system(system: str, learning: Mapping[str, Any], benchmark: str) -> str:
    if not _enabled(learning, benchmark):
        return system
    if not isinstance(system, str):
        raise ValueError("task_role_hints_system_must_be_text")
    if _SYSTEM_CLARIFICATION.strip() in system:
        raise ValueError("task_role_hints_system_already_augmented")
    return system + _SYSTEM_CLARIFICATION
