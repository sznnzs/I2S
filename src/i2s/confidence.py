from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfidenceEvidence:
    reported: float | None
    parse_error: str | None
    request_large_model: bool | None = None
    reason: str | None = None


def parse_confidence_assessment(
    text: str,
) -> tuple[float | None, bool | None, str | None, str | None]:
    stripped = str(text or "").strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None, None, None, "invalid_json"
    if not isinstance(value, dict) or set(value) != {
        "request_large_model",
        "confidence",
        "reason",
    }:
        return None, None, None, "invalid_assessment_schema"
    request_large = value["request_large_model"]
    reason = value["reason"]
    if not isinstance(request_large, bool):
        return None, None, None, "request_large_model_not_boolean"
    if not isinstance(reason, str) or not reason.strip():
        return None, None, None, "confidence_reason_not_text"
    if "\n" in reason or "\r" in reason:
        return None, None, None, "confidence_reason_multiline"
    raw = value["confidence"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, None, None, "confidence_not_numeric"
    confidence = float(raw)
    if not math.isfinite(confidence):
        return None, None, None, "confidence_not_finite"
    if not 0.0 <= confidence <= 1.0:
        return None, None, None, "confidence_out_of_range"
    return confidence, request_large, reason.strip(), None


class ConfidenceRouter:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.router_type = str(self.config.get("type", "confidence_threshold"))
        if self.router_type != "confidence_threshold":
            raise ValueError("router.type must be confidence_threshold")
        self.threshold = float(self.config.get("threshold", 0.99))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("router.threshold must be in [0, 1]")
        self.missing_confidence = str(self.config.get("missing_confidence", "repair"))
        if self.missing_confidence not in {"repair", "small"}:
            raise ValueError("router.missing_confidence must be repair or small")
        self.reason_max_chars = int(self.config.get("reason_max_chars", 240))
        if self.reason_max_chars <= 0:
            raise ValueError("router.reason_max_chars must be positive")
        raw_external_context = self.config.get("include_external_context", False)
        if not isinstance(raw_external_context, bool):
            raise ValueError("router.include_external_context must be boolean")
        self.include_external_context = raw_external_context
        raw_episode_history = self.config.get("include_episode_history", False)
        if not isinstance(raw_episode_history, bool):
            raise ValueError("router.include_episode_history must be boolean")
        self.include_episode_history = raw_episode_history

    def evidence(self, text: str) -> ConfidenceEvidence:
        reported, request_large, reason, parse_error = parse_confidence_assessment(text)
        if reason is not None and len(reason) > self.reason_max_chars:
            reported = None
            request_large = None
            reason = None
            parse_error = "confidence_reason_too_long"
        return ConfidenceEvidence(
            reported=reported,
            parse_error=parse_error,
            request_large_model=request_large,
            reason=reason,
        )

    def _threshold(self, override: float | None) -> float:
        threshold = self.threshold if override is None else float(override)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0, 1]")
        return threshold

    def trigger(
        self,
        evidence: ConfidenceEvidence,
        *,
        threshold: float | None = None,
    ) -> str | None:
        effective_threshold = self._threshold(threshold)
        if evidence.parse_error is not None:
            return "confidence_assessment_error" if self.missing_confidence == "repair" else None
        if evidence.request_large_model:
            return "model_requested_large"
        return "low_confidence" if evidence.reported < effective_threshold else None
