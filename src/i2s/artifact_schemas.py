from __future__ import annotations

import re
from typing import Any, Mapping

FROZEN_TASK_SELECTION = "frozen-task-selection"
TASK_SELECTION = "task-selection"
METHOD_EVENT = "method-event"
TASK_RECORD = "task-record"
MODEL_CALL = "model-call"
STEP_RECORD = "step-record"
EPISODE_RECORD = "episode-record"
BENCHMARK_WORKER = "benchmark-worker"


def schema_name(value: Mapping[str, Any]) -> str:
    raw = str(value.get("schema") or value.get("schema_version") or "")
    return re.sub(r"\.v\d+$", "", raw)
