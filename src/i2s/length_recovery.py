from dataclasses import replace

from .schemas import ModelUsage
from .utils import estimate_tokens

PROTOCOL = "length_retry_once"
PURPOSES = {
    "environment_action",
    "repair_action",
    "skill_induction_refinement",
}


def expanded_generation(generation, input_tokens, *, cloud):
    old = int(generation["request_max_new_tokens"])
    target = min(old * 2, 32768 if cloud else 16384)
    max_len = int(generation["max_model_len"])
    limit = min(int(generation["max_input_tokens"]), max_len - target)
    if target <= old or input_tokens > limit:
        return None
    return {
        **generation,
        "max_new_tokens": target,
        "request_max_new_tokens": target,
        "max_input_tokens": limit,
    }


def generate_with_length_recovery(
    model, *, call_id, messages, purpose, generation, seed, enabled, cloud
):
    from .models import ModelInvocationError

    args = dict(call_id=call_id, messages=messages, purpose=purpose, seed=seed)
    first = model.generate(**args, generation=generation)
    if not enabled or purpose not in PURPOSES or first.finish_reason != "length":
        return first

    from .json_envelope import PROTOCOL as JSON_PROTOCOL
    from .json_envelope import load_json_object

    try:
        load_json_object(first.raw_output or "", JSON_PROTOCOL)
        return first
    except (ValueError, TypeError):
        pass
    estimate = estimate_tokens("".join(str(m.get("content", "")) for m in messages))
    required = max(estimate, first.usage.input_tokens or 0)
    expanded = expanded_generation(generation, required, cloud=cloud)
    if expanded is None:
        return first
    error = None
    try:
        second = model.generate(**args, generation=expanded)
    except ModelInvocationError as exc:
        second = exc.call
        error = exc

    def add_known(a, b):
        return a + b if a is not None and b is not None else None

    usage = ModelUsage(
        **{
            k: add_known(getattr(first.usage, k), getattr(second.usage, k))
            for k in ("input_tokens", "output_tokens", "reasoning_tokens")
        }
    )

    combined = replace(
        second,
        generation={**generation, "seed": seed},
        usage=usage,
        latency_seconds=first.latency_seconds + second.latency_seconds,
        retry_count=first.retry_count + second.retry_count + 1,
        generation_attempts=tuple(c.to_dict() for c in (first, second)),
    )
    if error:
        raise ModelInvocationError(str(error), combined) from error
    return combined
