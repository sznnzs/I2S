from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import threading
import time
from dataclasses import replace
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from .call_identity import logical_call_id, logical_call_offset
from .schemas import ModelCall, ModelCallPosition, ModelUsage


def _response_field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _response_usage(response: Any) -> ModelUsage:
    usage = _response_field(response, "usage")
    details = _response_field(usage, "completion_tokens_details")

    def count(value):
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    return ModelUsage(
        input_tokens=count(_response_field(usage, "prompt_tokens")),
        output_tokens=count(_response_field(usage, "completion_tokens")),
        reasoning_tokens=count(_response_field(details, "reasoning_tokens")),
    )


def _response_snapshot(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _response_snapshot(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_response_snapshot(child) for child in value]
    if hasattr(value, "__dict__"):
        return {
            key: _response_snapshot(child)
            for key, child in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _normalize_response_channels(
    content: object,
    reasoning: object,
) -> tuple[str, str | None, str]:
    raw_content = str(content or "")
    raw_reasoning = None if reasoning is None else str(reasoning)
    if raw_content.strip() or not raw_reasoning:
        return raw_content, raw_reasoning, "content"
    end = len(raw_reasoning.rstrip())
    decoder = json.JSONDecoder()
    for start, character in enumerate(raw_reasoning[:end]):
        if character != "{":
            continue
        try:
            value, parsed_end = decoder.raw_decode(raw_reasoning, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        if raw_reasoning[parsed_end:end].strip():
            continue
        return (
            raw_reasoning[start:parsed_end],
            raw_reasoning[:start] or None,
            "reasoning_suffix",
        )
    return raw_content, raw_reasoning, "content"


def _redact_error(value: object) -> str:
    text = str(value)
    text = re.sub(r"(?i)(authorization|api[-_ ]?key)\s*[:=]\s*\S+", r"\1=[redacted]", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
    return text[:2000]


def _base_url(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def resolve_credentials(config: Mapping[str, Any]) -> tuple[str, str]:
    environment = os.environ
    endpoint = str(config.get("base_url") or "").strip()
    if config.get("base_url_env"):
        endpoint = environment.get(str(config["base_url_env"]), "").strip() or endpoint
    if str(config.get("auth", "required")) == "none":
        if not endpoint:
            raise RuntimeError("local no-auth model requires base_url")
        return "local-no-auth", _base_url(endpoint)
    key = environment.get(str(config.get("api_key_env", ""))) if config.get("api_key_env") else None
    if not key:
        raise RuntimeError("missing API credential; set the configured api_key_env variable")
    if not endpoint:
        raise RuntimeError("missing OpenAI-compatible base URL")
    return key, _base_url(endpoint)


def _http_proxy() -> str | None:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip().lower().startswith(("http://", "https://")):
            return value.strip()
    return None


@contextlib.contextmanager
def _request_wall_timeout(seconds: object):
    try:
        limit = float(seconds)
    except (TypeError, ValueError):
        limit = 0.0
    if (
        not math.isfinite(limit)
        or limit <= 0
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _alarm(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"model request exceeded wall timeout ({limit:.3f}s)")

    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


class ModelInvocationError(RuntimeError):
    def __init__(self, message: str, call: ModelCall) -> None:
        super().__init__(message)
        self.call = call


class OpenAICompatibleModel:
    def __init__(self, model_id: str, config: Mapping[str, Any]) -> None:
        self.model_id = model_id
        self.config = dict(config)
        self.model_name = str(config["model"])
        self.role = str(config.get("role", "small"))
        if self.role not in {"small", "large"}:
            raise ValueError(f"model role must be small or large: {model_id}")
        key, endpoint = resolve_credentials(self.config)
        self.base_url = endpoint
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai and httpx are required by the controller environment"
            ) from exc
        proxy_mode = str(
            config.get("proxy_mode", "direct" if config.get("auth") == "none" else "http_env_only")
        )
        proxy = _http_proxy() if proxy_mode == "http_env_only" else None
        timeout_raw = config.get("timeout_seconds", 120.0)
        timeout = None if timeout_raw is None else float(timeout_raw)
        http_client = httpx.Client(
            proxy=proxy,
            timeout=timeout,
            trust_env=False,
        )
        self.client = OpenAI(
            api_key=key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    def generate(
        self,
        *,
        call_id: str,
        messages: tuple[Mapping[str, str], ...],
        purpose: str,
        generation: Mapping[str, Any],
        seed: int,
    ) -> ModelCall:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [dict(item) for item in messages],
            "temperature": float(generation.get("temperature", 0.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "max_tokens": int(
                generation.get("request_max_new_tokens", generation["max_new_tokens"])
            ),
        }
        if bool(self.config.get("send_seed", True)):
            request["seed"] = int(seed)
        extra_body = dict(self.config.get("extra_body") or {})
        if extra_body:
            request["extra_body"] = extra_body
        response_format = generation.get("response_format", self.config.get("response_format"))
        if response_format:
            request["response_format"] = dict(response_format)
        max_retries = int(self.config.get("max_retries", 2))
        backoff = float(self.config.get("retry_backoff_seconds", 1.0))
        started = time.perf_counter()
        retry_count = 0
        physical_attempts: list[dict[str, Any]] = []

        def finish(call: ModelCall) -> ModelCall:
            if not physical_attempts:
                return call
            physical_attempts.append({**call.to_dict(), "attempt_type": "transport"})
            totals = {}
            for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
                values = [attempt["usage"][field] for attempt in physical_attempts]
                totals[field] = sum(values) if all(value is not None for value in values) else None
            return replace(
                call,
                usage=ModelUsage(**totals),
                retry_count=retry_count,
                latency_seconds=time.perf_counter() - started,
                generation_attempts=tuple(physical_attempts),
            )

        while True:
            response = None
            attempt_started = time.perf_counter()
            try:
                if self.config.get("send_request_id_header", False):
                    physical_id = call_id if retry_count == 0 else f"{call_id}:retry{retry_count}"
                    request["extra_headers"] = {"X-Request-Id": physical_id}
                with _request_wall_timeout(self.config.get("request_wall_timeout_seconds")):
                    response = self.client.chat.completions.create(**request)

                if response is None:
                    raise RuntimeError("empty model response")
                choices = getattr(response, "choices", None)
                if not choices:
                    raise RuntimeError("model response contains no choices")
                choice = choices[0]
                content = getattr(choice.message, "content", None)
                if isinstance(content, list):
                    content = "".join(
                        str(
                            getattr(item, "text", "")
                            or (item.get("text", "") if isinstance(item, dict) else "")
                        )
                        for item in content
                    )
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning is None:
                    reasoning = dict(getattr(choice.message, "model_extra", None) or {}).get(
                        "reasoning_content"
                    )
                content, reasoning, raw_output_source = _normalize_response_channels(
                    content, reasoning
                )
                call = ModelCall(
                    call_id=call_id,
                    model_id=self.model_id,
                    model_name=self.model_name,
                    role=self.role,
                    purpose=purpose,
                    messages=messages,
                    generation={**dict(generation), "seed": int(seed)},
                    raw_output=str(content or ""),
                    reasoning_output=(str(reasoning) if reasoning is not None else None),
                    usage=_response_usage(response),
                    latency_seconds=time.perf_counter() - attempt_started,
                    retry_count=0,
                    request_status="response_received",
                    provider_request_id=str(getattr(response, "id", "") or "") or None,
                    finish_reason=str(getattr(choice, "finish_reason", "") or "") or None,
                    response_model=str(getattr(response, "model", "") or "") or None,
                    system_fingerprint=(
                        str(getattr(response, "system_fingerprint", "") or "") or None
                    ),
                    raw_output_source=raw_output_source,
                )
                return finish(call)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = status is None or status in {408, 409, 429} or status >= 500
                error = _redact_error(exc)

                call = ModelCall(
                    call_id=call_id,
                    model_id=self.model_id,
                    model_name=self.model_name,
                    role=self.role,
                    purpose=purpose,
                    messages=messages,
                    generation={**dict(generation), "seed": int(seed)},
                    raw_output=None,
                    reasoning_output=None,
                    usage=_response_usage(response),
                    latency_seconds=time.perf_counter() - attempt_started,
                    retry_count=0,
                    request_status="request_failed",
                    error=f"{type(exc).__name__}: {error}",
                    provider_request_id=str(_response_field(response, "id") or "") or None,
                )
                recorded = {
                    **call.to_dict(),
                    "attempt_type": "transport",
                    "provider_response": _response_snapshot(response),
                }
                if retry_count < max_retries and transient:
                    physical_attempts.append(recorded)
                    delay = backoff * (2**retry_count)
                    retry_count += 1
                    if delay > 0:
                        time.sleep(min(delay, 30.0))
                    continue
                if not physical_attempts:
                    call = replace(call, generation_attempts=(recorded,))
                else:
                    call = finish(call)
                    attempts = [*call.generation_attempts[:-1], recorded]
                    call = replace(call, generation_attempts=tuple(attempts))
                raise ModelInvocationError(call.error or "model invocation failed", call) from exc

    def close(self) -> None:
        self.client.close()


class ModelPool:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        run_id: str,
        generation_seed: int,
        max_steps: int = 1,
        max_calls_per_step: int = 8,
    ) -> None:
        self.config = dict(config)
        self.run_id = run_id
        self.generation_seed = int(generation_seed)
        self.max_steps = int(max_steps)
        self.max_calls_per_step = int(max_calls_per_step)
        if self.max_steps <= 0 or self.max_calls_per_step <= 0:
            raise ValueError("model call position limits must be positive")
        self.models: dict[str, OpenAICompatibleModel] = {}
        self.call_index = 0
        self.lock = threading.Lock()

    def _model(self, model_id: str) -> OpenAICompatibleModel:
        if model_id not in self.models:
            registry = dict(self.config.get("models") or {})
            if model_id not in registry:
                raise KeyError(f"unknown model_id: {model_id}")
            model_config = dict(registry[model_id])
            provider = str(model_config.get("provider", "openai_compatible"))
            if provider != "openai_compatible":
                raise ValueError(f"unsupported model provider: {provider}")
            if model_config.get("scoring_mode") == "vllm_full_label_likelihood":
                from .confidence_likelihood_model import ConfidenceLikelihoodModel

                self.models[model_id] = ConfidenceLikelihoodModel(model_id, model_config)
            else:
                self.models[model_id] = OpenAICompatibleModel(model_id, model_config)
        return self.models[model_id]

    def generate(
        self,
        model_id: str,
        messages: tuple[Mapping[str, str], ...],
        purpose: str,
        generation: Mapping[str, Any],
        *,
        position: ModelCallPosition | None = None,
    ) -> ModelCall:
        if position is None:
            with self.lock:
                self.call_index += 1
                index = self.call_index
            call_id = f"{self.run_id}:c{index:06d}"
            seed = self.generation_seed + index - 1
        else:
            offset = logical_call_offset(
                position,
                max_steps=self.max_steps,
                max_calls_per_step=self.max_calls_per_step,
            )
            call_id = logical_call_id(self.run_id, position)
            seed = self.generation_seed + offset
        try:
            model = self._model(model_id)
        except Exception as exc:
            raw = dict(dict(self.config.get("models") or {}).get(model_id) or {})
            error = f"{type(exc).__name__}: {_redact_error(exc)}"
            call = ModelCall(
                call_id=call_id,
                model_id=model_id,
                model_name=str(raw.get("model") or model_id),
                role=str(raw.get("role") or "unknown"),
                purpose=purpose,
                messages=messages,
                generation={**dict(generation), "seed": seed},
                raw_output=None,
                reasoning_output=None,
                usage=ModelUsage(),
                latency_seconds=0.0,
                retry_count=0,
                request_status="request_failed",
                error=error,
            )
            raise ModelInvocationError(error, call) from exc
        from .length_recovery import generate_with_length_recovery

        raw = dict(dict(self.config.get("models") or {}).get(model_id) or {})
        return generate_with_length_recovery(
            model,
            call_id=call_id,
            messages=messages,
            purpose=purpose,
            generation=generation,
            seed=seed,
            enabled=self.config.get("length_recovery_protocol") == "length_retry_once",
            cloud=bool(raw.get("api_key_env")),
        )

    def generation_for(self, model_id: str, base_generation: Mapping[str, Any]) -> dict[str, Any]:
        registry = dict(self.config.get("models") or {})
        if model_id not in registry:
            raise KeyError(f"unknown model_id: {model_id}")
        result = dict(base_generation)
        result.update(dict(dict(registry[model_id]).get("generation") or {}))
        max_model_len = int(result["max_model_len"])
        max_input = int(result["max_input_tokens"])
        max_new = int(result["max_new_tokens"])
        if max_input + max_new > max_model_len:
            raise ValueError(f"model {model_id} input plus output limits exceed max_model_len")
        if result.get("output_budget_policy") != "fixed_output":
            raise ValueError(f"model {model_id} output_budget_policy must be fixed_output")
        return result

    def public_registry(self, base_generation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for model_id, raw in dict(self.config.get("models") or {}).items():
            config = dict(raw)
            endpoint = str(config.get("base_url") or "")
            resolution_error = None
            if not endpoint:
                try:
                    _credential, endpoint = resolve_credentials(config)
                except Exception as exc:
                    resolution_error = f"{type(exc).__name__}: {_redact_error(exc)}"
            result[model_id] = {
                "model": config.get("model"),
                "role": config.get("role"),
                "reasoning_mode": config.get("reasoning_mode"),
                "provider": config.get("provider", "openai_compatible"),
                "endpoint_host": urlsplit(endpoint).netloc if endpoint else None,
                "credential_source": ("environment" if config.get("api_key_env") else "none"),
                "resolution_error": resolution_error,
                "request_options": {
                    "send_seed": bool(config.get("send_seed", True)),
                    "response_format": dict(config.get("response_format") or {}),
                    "extra_body": dict(config.get("extra_body") or {}),
                    "timeout_seconds": (
                        None
                        if config.get("timeout_seconds", 120.0) is None
                        else float(config.get("timeout_seconds", 120.0))
                    ),
                    "max_retries": int(config.get("max_retries", 2)),
                    "retry_backoff_seconds": float(config.get("retry_backoff_seconds", 1.0)),
                    "proxy_mode": str(
                        config.get(
                            "proxy_mode",
                            "direct" if config.get("auth") == "none" else "http_env_only",
                        )
                    ),
                },
                "effective_generation": (
                    self.generation_for(model_id, base_generation)
                    if base_generation is not None
                    else dict(config.get("generation") or {})
                ),
            }
        return result

    def close(self) -> None:
        for model in self.models.values():
            model.close()
        self.models.clear()
