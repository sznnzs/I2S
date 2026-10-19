from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import replace
from itertools import pairwise
from typing import Any, Callable

VERSION = "teacher_observation"
CONFIG_KEY = "teacher_observation_dedup"
RECEIPT_SCHEMA = "teacher-observation-dedup-receipt/1"
VERIFICATION_SCOPE = "source_message_restoration"
MARKER_PREFIX = "[I2S_OBS:"
MARKER = "[I2S_OBS: same exact text as Current observation]"
MIN_OBSERVATION_CHARS = 96
TASK_HEADING = "Task instruction:\n"
CURRENT_HEADING = "Current observation:\n"
HISTORY_HEADING = "Previous interactions:\n"
FINAL_INSTRUCTION = "Return the next action now as the required JSON object."
ACTION_HEADING = re.compile(
    r"(?m)^(?:Admissible actions|Admissible disambiguation choices|"
    r"Available actions and patterns|Available action patterns) \(\d+\):\n"
)
HISTORY_STEP = re.compile(r"(?m)^Step ([1-9]\d*):\nAction: ([^\n]+)\nEnvironment response:\n")
REPAIR_SUFFIX_PREFIXES = (
    "The previous response did not provide an executable action.",
    "The previous action was rejected by the environment.",
    "The previous action did not change the visible state.",
    "The latest environment response indicates that the previous attempt failed.",
    "The small model reported low confidence in its proposed action.",
    "The small model explicitly requested help with this decision.",
    "The small model did not provide a usable confidence assessment.",
    "No reusable procedure has yet been acquired for this task family.",
    "Continue resolving the current task from the latest visible result.",
    "Review the current state and choose the most useful next action.",
    "PUBLIC EXECUTION CHECK:",
)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _messages_hash(messages: Any) -> str | None:
    try:
        text = json.dumps(
            messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return _text_hash(text)
    except (TypeError, ValueError, UnicodeError):
        return None


def _base_receipt(messages: Any, reason: str) -> dict[str, Any]:
    digest = _messages_hash(messages)
    return {
        "schema": RECEIPT_SCHEMA,
        "version": VERSION,
        "applied": False,
        "reason": reason,
        "source_messages_sha256": digest,
        "transformed_messages_sha256": digest,
        "verification_scope": VERIFICATION_SCOPE,
    }


def _pass(messages: Any, reason: str):
    return messages, _base_receipt(messages, reason)


def transform(messages: Any):
    if (
        not isinstance(messages, (list, tuple))
        or len(messages) != 2
        or not all(isinstance(m, dict) for m in messages)
        or [m.get("role") for m in messages] != ["system", "user"]
        or not all(isinstance(m.get("content"), str) for m in messages)
    ):
        return _pass(messages, "unsupported_message_shape")
    if _messages_hash(messages) is None:
        return _pass(messages, "unsupported_message_serialization")
    if any(MARKER_PREFIX in m["content"] for m in messages):
        return _pass(messages, "reserved_marker_already_present")
    user = messages[1]["content"]
    if any(user.count(h) != 1 for h in (TASK_HEADING, CURRENT_HEADING, HISTORY_HEADING)):
        return _pass(messages, "ambiguous_or_missing_prompt_heading")
    if user.count(FINAL_INSTRUCTION) != 1 or not user.endswith(FINAL_INSTRUCTION):
        return _pass(messages, "unknown_final_instruction_boundary")
    action_headings = list(ACTION_HEADING.finditer(user))
    if len(action_headings) != 1:
        return _pass(messages, "ambiguous_or_missing_action_heading")
    task_start = user.index(TASK_HEADING)
    current_heading_start = user.index(CURRENT_HEADING)
    current_start = current_heading_start + len(CURRENT_HEADING)
    action_start = action_headings[0].start()
    history_heading_start = user.index(HISTORY_HEADING)
    history_start = history_heading_start + len(HISTORY_HEADING)
    if not task_start < current_heading_start < action_start:
        return _pass(messages, "unknown_fixed_section_order")
    if user[action_start - 2 : action_start] != "\n\n":
        return _pass(messages, "unknown_current_observation_boundary")
    observation = user[current_start : action_start - 2]
    if len(observation) < MIN_OBSERVATION_CHARS or len(observation) <= len(MARKER):
        return _pass(messages, "observation_too_short")
    if HISTORY_STEP.search(observation):
        return _pass(messages, "history_like_text_inside_current_observation")
    if history_heading_start < task_start:
        layout = "history_before_current"
        history_end = task_start
    elif action_start < history_heading_start:
        layout = "history_after_current"
        history_end = len(user)
    else:
        return _pass(messages, "unknown_history_section_order")
    step_headers = list(HISTORY_STEP.finditer(user, history_start, history_end))
    if not step_headers or step_headers[0].start() != history_start:
        return _pass(messages, "missing_complete_history_step")
    numbers = [int(m.group(1)) for m in step_headers]
    if any(b != a + 1 for a, b in pairwise(numbers)):
        return _pass(messages, "noncontiguous_or_ambiguous_history_steps")
    latest = step_headers[-1]
    duplicate_start = latest.end()
    duplicate_end = duplicate_start + len(observation)
    if user[duplicate_start:duplicate_end] != observation:
        return _pass(messages, "latest_observation_not_exact_current_copy")
    if layout == "history_before_current":
        if user[duplicate_end:task_start] != "\n\n":
            return _pass(messages, "latest_history_observation_has_extra_content")
    elif not any(
        user[duplicate_end:].startswith("\n\n" + prefix) for prefix in REPAIR_SUFFIX_PREFIXES
    ):
        return _pass(messages, "unknown_teacher_instruction_boundary")
    updated_user = user[:duplicate_start] + MARKER + user[duplicate_end:]
    updated = copy.deepcopy(messages)
    updated[1]["content"] = updated_user
    shift = len(MARKER) - len(observation)
    updated_current_start = current_start + (shift if current_start > duplicate_start else 0)
    receipt = {
        **_base_receipt(messages, "exact_latest_observation_reference"),
        "applied": True,
        "message_index": 1,
        "layout": layout,
        "offset_unit": "unicode_codepoints",
        "marker": MARKER,
        "history_last_step": numbers[-1],
        "history_step_count": len(numbers),
        "source_content_sha256": _text_hash(user),
        "transformed_content_sha256": _text_hash(updated_user),
        "transformed_messages_sha256": _messages_hash(updated),
        "observation_sha256": _text_hash(observation),
        "observation_chars": len(observation),
        "source_duplicate_span": [duplicate_start, duplicate_end],
        "transformed_reference_span": [duplicate_start, duplicate_start + len(MARKER)],
        "transformed_current_observation_span": [
            updated_current_start,
            updated_current_start + len(observation),
        ],
        "source_content_utf8_bytes": len(user.encode("utf-8")),
        "transformed_content_utf8_bytes": len(updated_user.encode("utf-8")),
        "removed_utf8_bytes": len(user.encode("utf-8")) - len(updated_user.encode("utf-8")),
        "exact_roundtrip": True,
    }

    if restore(updated, receipt) != messages:
        raise ValueError("teacher_observation_dedup_internal_roundtrip_failure")
    return updated, receipt


def _span(receipt: dict[str, Any], name: str, length: int) -> tuple[int, int]:
    value = receipt.get(name)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(type(x) is int for x in value)
        or not 0 <= value[0] <= value[1] <= length
    ):
        raise ValueError("invalid_receipt_span:" + name)
    return value[0], value[1]


def restore(messages: Any, receipt: dict[str, Any]):
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("version") != VERSION
        or type(receipt.get("applied")) is not bool
    ):
        raise ValueError("unknown_teacher_observation_receipt")
    if _messages_hash(messages) != receipt.get("transformed_messages_sha256"):
        raise ValueError("transformed_messages_hash_mismatch")
    if not receipt.get("applied"):
        if receipt.get("source_messages_sha256") is None or _messages_hash(messages) != receipt.get(
            "source_messages_sha256"
        ):
            raise ValueError("source_messages_hash_mismatch")
        return messages
    if (
        not isinstance(messages, (tuple, list))
        or len(messages) != 2
        or receipt.get("message_index") != 1
        or receipt.get("marker") != MARKER
        or receipt.get("offset_unit") != "unicode_codepoints"
        or not isinstance(messages[1], dict)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise ValueError("invalid_transformed_message_shape")
    user = messages[1]["content"]
    if _text_hash(user) != receipt.get("transformed_content_sha256"):
        raise ValueError("transformed_content_hash_mismatch")
    begin, end = _span(receipt, "transformed_reference_span", len(user))
    current_begin, current_end = _span(receipt, "transformed_current_observation_span", len(user))
    if user[begin:end] != MARKER or user.count(MARKER_PREFIX) != 1:
        raise ValueError("invalid_or_ambiguous_reference_marker")
    if not (end <= current_begin or current_end <= begin):
        raise ValueError("reference_overlaps_current_observation")
    if user[max(0, current_begin - len(CURRENT_HEADING)) : current_begin] != CURRENT_HEADING:
        raise ValueError("current_observation_heading_mismatch")
    observation = user[current_begin:current_end]
    if len(observation) != receipt.get("observation_chars") or _text_hash(
        observation
    ) != receipt.get("observation_sha256"):
        raise ValueError("current_observation_hash_mismatch")
    original_user = user[:begin] + observation + user[end:]
    if receipt.get("source_duplicate_span") != [begin, begin + len(observation)]:
        raise ValueError("source_duplicate_span_mismatch")
    if _text_hash(original_user) != receipt.get("source_content_sha256"):
        raise ValueError("source_content_hash_mismatch")
    if (
        receipt.get("source_content_utf8_bytes") != len(original_user.encode("utf-8"))
        or receipt.get("transformed_content_utf8_bytes") != len(user.encode("utf-8"))
        or receipt.get("removed_utf8_bytes")
        != len(original_user.encode("utf-8")) - len(user.encode("utf-8"))
    ):
        raise ValueError("receipt_byte_accounting_mismatch")
    if (
        receipt.get("exact_roundtrip") is not True
        or receipt.get("verification_scope") != VERIFICATION_SCOPE
    ):
        raise ValueError("receipt_restoration_metadata_mismatch")
    if original_user.count(HISTORY_HEADING) != 1 or original_user.count(TASK_HEADING) != 1:
        raise ValueError("source_history_boundary_mismatch")
    history_start = original_user.index(HISTORY_HEADING) + len(HISTORY_HEADING)
    task_start = original_user.index(TASK_HEADING)
    layout = "history_before_current" if history_start < task_start else "history_after_current"
    history_end = task_start if layout == "history_before_current" else len(original_user)
    headers = list(HISTORY_STEP.finditer(original_user, history_start, history_end))
    if (
        receipt.get("layout") != layout
        or not headers
        or headers[-1].end() != begin
        or receipt.get("history_step_count") != len(headers)
        or receipt.get("history_last_step") != int(headers[-1].group(1))
    ):
        raise ValueError("receipt_latest_history_mismatch")
    original = copy.deepcopy(messages)
    original[1]["content"] = original_user
    if _messages_hash(original) != receipt.get("source_messages_sha256"):
        raise ValueError("source_messages_hash_mismatch")
    return original


class _TeacherPromptBuilder:
    def __init__(
        self, builder: Any, receipt_sink: Callable[[dict[str, Any]], None], context: dict[str, Any]
    ):
        self._builder = builder
        self._receipt_sink = receipt_sink
        self._context = context

    def __getattr__(self, name: str):
        return getattr(self._builder, name)

    def build(self, *args, **kwargs):
        prompt = self._builder.build(*args, **kwargs)
        messages, receipt = transform(prompt.messages)
        expected_step = self._context["call_position"]["step_index"]
        state = args[0] if args else kwargs.get("state")
        if receipt["applied"] and getattr(state, "step_index", None) != expected_step:
            messages, receipt = _pass(prompt.messages, "runtime_state_position_mismatch")
        elif receipt["applied"] and receipt["history_last_step"] != expected_step - 1:
            messages, receipt = _pass(prompt.messages, "history_is_not_previous_runtime_step")
        self._receipt_sink(
            {
                "event": "teacher_observation_deduplication",
                **self._context,
                "receipt": receipt,
                "source_call_id_ownership": False,
            }
        )

        return replace(prompt, messages=messages) if receipt["applied"] else prompt


def teacher_repair_controller(
    controller: Any,
    *,
    receipt_sink: Callable[[dict[str, Any]], None],
    large_model_id: str,
    episode_index: int,
    step_index: int,
    call_slot: int,
):
    if getattr(controller, "large_model_id", None) != large_model_id:
        raise ValueError("teacher_observation_dedup_large_model_mismatch")
    if not callable(receipt_sink):
        raise ValueError("teacher_observation_dedup_receipt_sink_required")
    if (
        any(type(x) is not int or x < 0 for x in (episode_index, step_index, call_slot))
        or step_index < 1
    ):
        raise ValueError("teacher_observation_dedup_invalid_call_position")
    context = {
        "version": VERSION,
        "model_id": large_model_id,
        "purpose": "repair_action",
        "role": "large",
        "call_position": {
            "episode_index": episode_index,
            "step_index": step_index,
            "call_slot": call_slot,
        },
    }
    candidate = copy.copy(controller)
    candidate.prompt_builder = _TeacherPromptBuilder(
        controller.prompt_builder, receipt_sink, context
    )
    return candidate
