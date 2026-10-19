from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

BACKENDS = ("local", "deepseek", "gemini")
BENCHMARKS = ("alfworld", "webshop", "scienceworld")
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    if "extends" in value:
        raise ValueError("Use separate run, model, and environment configuration files")
    return value


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand(child) for child in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        if os.environ.get(name):
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"required environment variable is not set: {name}")

    return _VARIABLE.sub(replace, value)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use KEY=VALUE: {override}")
        key, raw = override.split("=", 1)
        parts = key.split(".")
        if not all(parts):
            raise ValueError(f"invalid override key: {key}")
        target = result
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                raise KeyError(f"unknown configuration mapping: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"unknown configuration field: {key}")
        target[parts[-1]] = yaml.safe_load(raw)
    return result


def active_model_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    policy = config["policy"]
    ids = [
        str(policy["model_id"]),
        str(policy["large_model_id"]),
        str(config["router"]["reliability_model_id"]),
    ]
    return tuple(dict.fromkeys(ids))


def validate_config(config: Mapping[str, Any]) -> None:
    for key in (
        "benchmark",
        "model_pool",
        "policy",
        "router",
        "generation",
        "context",
        "execution",
        "seeds",
    ):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"configuration requires a {key} mapping")
    benchmark = config["benchmark"]
    if benchmark.get("name") not in BENCHMARKS:
        raise ValueError(f"benchmark.name must be one of {BENCHMARKS}")
    for key in ("sample_count", "max_steps"):
        value = benchmark.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"benchmark.{key} must be a positive integer")
    if not isinstance(benchmark.get("split"), str) or not benchmark["split"]:
        raise ValueError("benchmark.split must be specified")
    for key in ("sampling", "environment", "generation"):
        value = config["seeds"].get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"seeds.{key} must be a nonnegative integer")
    execution = config["execution"]
    if int(execution.get("episode_workers", 1)) != 1:
        raise ValueError(
            "episodes within a family must remain serial; use execution.family_workers"
        )
    if int(execution.get("family_workers", 1)) < 1:
        raise ValueError("execution.family_workers must be positive")
    if int(execution.get("episode_index_offset", 0)) < 0:
        raise ValueError("execution.episode_index_offset must be nonnegative")
    if int(execution.get("max_calls_per_step", 8)) < 8:
        raise ValueError("execution.max_calls_per_step must reserve all eight policy call slots")
    if config.get("method", "i2s") != "i2s" or config["policy"].get("type") != "i2s":
        raise ValueError("method and policy.type must be i2s")
    if config["context"].get("type") != "skill":
        raise ValueError("context.type must be skill")
    models = config["model_pool"].get("models", {})
    for model_id in active_model_ids(config):
        if not isinstance(models.get(model_id), Mapping):
            raise ValueError(f"model configuration is missing: {model_id}")
        model = models[model_id]
        if model.get("provider", "openai_compatible") != "openai_compatible":
            raise ValueError("only explicit OpenAI-compatible model endpoints are supported")
        if model.get("role") not in {"small", "large"}:
            raise ValueError(f"{model_id}.role must explicitly identify small or large")
        if not model.get("model"):
            raise ValueError(f"{model_id}.model must identify the served model")
        if any(key in model for key in ("api_key", "secret", "token", "credential_module")):
            raise ValueError(
                "use api_key_env for credentials; literal secrets and credential modules are unsupported"
            )
        generation = deep_merge(config["generation"], model.get("generation", {}))
        for key in ("max_input_tokens", "max_new_tokens", "max_model_len"):
            if int(generation.get(key, 0)) < 1:
                raise ValueError(f"{model_id} generation.{key} must be positive")
        if (
            generation["max_input_tokens"] + generation["max_new_tokens"]
            > generation["max_model_len"]
        ):
            raise ValueError(f"{model_id} input and output budgets exceed the context window")


def load_config(
    path: str | Path,
    *,
    benchmark: str | None = None,
    backend: str | None = None,
    model_catalog: str | Path | None = None,
    environment_catalog: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    config = _read(source)
    resolved = isinstance(config.get("model_pool", {}).get("models"), dict)
    if not resolved or model_catalog is not None or backend is not None:
        chosen_backend = str(backend or config.get("backend", "local"))
        if chosen_backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        catalog = _read(Path(model_catalog) if model_catalog else source.parent / "models.yaml")
        try:
            chosen = {
                "qwen3_8b": catalog["actor"],
                "qwen3_8b_router": catalog["scorer"],
                "large": catalog["teachers"][chosen_backend],
            }
        except KeyError as exc:
            raise ValueError(f"model catalog is missing {exc}") from exc
        config.setdefault("model_pool", {})["models"] = copy.deepcopy(chosen)
        config["backend"] = chosen_backend
    if (
        not isinstance(config.get("benchmark"), Mapping)
        or benchmark is not None
        or environment_catalog is not None
    ):
        current_benchmark = config.get("benchmark", "alfworld")
        if isinstance(current_benchmark, Mapping):
            current_benchmark = current_benchmark["name"]
        chosen_benchmark = str(benchmark or current_benchmark)
        catalog = _read(
            Path(environment_catalog)
            if environment_catalog
            else source.parent / "environments.yaml"
        )
        if chosen_benchmark not in catalog:
            raise ValueError(f"environment catalog has no {chosen_benchmark!r} entry")
        environment = copy.deepcopy(catalog[chosen_benchmark])
        for section in ("context", "repair"):
            config[section] = deep_merge(config.get(section, {}), environment.pop(section, {}))
        config["benchmark"] = {**environment, "name": chosen_benchmark}
    if not resolved or benchmark is not None:
        learning = config["policy"].setdefault("learning", {})
        learning.pop("semantic_family_support", None)
        learning.pop("semantic_benchmark_support", None)
        if config["benchmark"]["name"] == "alfworld":
            learning["semantic_family_support"] = "alfworld"
        else:
            learning["semantic_benchmark_support"] = config["benchmark"]["name"]
    config = _expand(apply_overrides(config, overrides))
    config["config_path"] = str(source)
    validate_config(config)
    return config
