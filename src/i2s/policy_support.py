from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .benchmarks.common import normalize_action


def constrain_current_action(request, state, prompt_contract):
    if prompt_contract != "semantic_state" or state.benchmark != "alfworld":
        return request
    actions = list(dict.fromkeys(state.admissible_actions))
    if not actions:
        raise ValueError("semantic_action_guard_requires_current_admissible_actions")
    ref = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skill_id": {"type": "string"},
            "version": {"type": "integer", "minimum": 1},
        },
        "required": ["skill_id", "version"],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "analysis": {"type": "string"},
            "action": {"enum": actions},
            "skill_refs": {"type": "array", "items": ref},
        },
        "required": ["analysis", "action", "skill_refs"],
    }
    return {
        **request,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "current_environment_action", "strict": True, "schema": schema},
        },
    }


def allocate_request_generation(
    generation: Mapping[str, Any], *, estimated_input_tokens: int
) -> dict[str, Any]:
    if generation.get("output_budget_policy") != "fixed_output":
        raise ValueError("unsupported output budget policy")
    max_model_len = int(generation["max_model_len"])
    max_input = int(generation["max_input_tokens"])
    configured_max_new = int(generation["max_new_tokens"])
    estimated_input = int(estimated_input_tokens)
    if estimated_input > max_input:
        raise ValueError(
            f"estimated input tokens {estimated_input} exceed configured input budget {max_input}"
        )
    if estimated_input + configured_max_new > max_model_len:
        raise ValueError("configured output allowance does not fit after the estimated input")
    return {
        **dict(generation),
        "request_max_new_tokens": configured_max_new,
    }


def parse_action_output(
    text: str,
    *,
    benchmark: str,
    parser: str = "json_only",
    max_action_chars: int = 256,
) -> tuple[str | None, str | None, str | None, str | None]:
    if parser in {"strict_action_json", "skill_action_json"}:
        from .json_envelope import normalize_json_envelope

        normalized, envelope = normalize_json_envelope(text)
        base_parser = "action_json" if parser == "strict_action_json" else "json_skill_refs"
        raw, action, mode, error = parse_action_output(
            normalized, benchmark=benchmark, parser=base_parser, max_action_chars=max_action_chars
        )
        return raw, action, (envelope + ":" + mode if envelope and mode else mode), error
    if parser == "structured_recovery":
        try:
            value = json.loads(str(text or ""))
            if not isinstance(value, dict) or set(value) != {"ACTION", "EXPECTED_PROGRESS"}:
                raise ValueError("wrong recovery fields")
            raw = value["ACTION"]
            expected = value["EXPECTED_PROGRESS"]
            if not isinstance(raw, str) or not isinstance(expected, str) or not expected.strip():
                raise ValueError("invalid recovery fields")
            if (
                not raw
                or raw != raw.strip()
                or len(raw) > max_action_chars
                or "\n" in raw
                or "\r" in raw
            ):
                raise ValueError("invalid recovery action")
            return raw, normalize_action(raw, benchmark), "structured_recovery", None
        except (ValueError, TypeError):
            return None, None, "structured_recovery", "structured_recovery_contract_invalid"
    if parser == "json_skill_refs":
        try:
            value = json.loads(str(text or "").strip())
            if isinstance(value, dict):
                value.pop("skill_refs", None)
                text = json.dumps(value)
        except json.JSONDecodeError:
            pass
        parser = "json_only"
    modern_json = parser in {"action_json", "tag_or_json_action"}
    candidates: list[tuple[str, str]] = []
    stripped = str(text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(("json", stripped))
        if modern_json and stripped.endswith("}}"):
            candidates.append(("json_redundant_closing_brace", stripped[:-1]))
    if parser in {"json_or_action_tag", "tag_or_json_action"}:
        for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.I | re.S):
            candidates.append(("fenced_json", match.group(1)))
        match = re.search(r"<action>\s*(.*?)\s*</action>", stripped, re.I | re.S)
        if match:
            candidates.append(("action_tag", json.dumps({"action": match.group(1)})))
    if not candidates:
        return None, None, None, "no_supported_action_object"
    for mode, candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        allowed_keys = (
            ({"action"}, {"analysis", "action"}) if modern_json else ({"analysis", "action"},)
        )
        if not isinstance(value, dict) or set(value) not in allowed_keys:
            continue
        if not isinstance(value.get("action"), str) or (
            "analysis" in value and not isinstance(value.get("analysis"), str)
        ):
            continue
        if modern_json and set(value) == {"action"}:
            mode = mode + "_action_only"
        raw_action = value["action"].strip()
        if not raw_action:
            return raw_action, None, mode, "empty_action"
        if len(raw_action) > max_action_chars:
            return raw_action, None, mode, "action_too_long"
        if "\n" in raw_action or "\r" in raw_action:
            return raw_action, None, mode, "multiline_action"
        return raw_action, normalize_action(raw_action, benchmark), mode, None
    return None, None, None, "invalid_action_json"
