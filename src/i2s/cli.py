from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import BACKENDS, BENCHMARKS, load_config, validate_config
from .evaluation import evaluate_run
from .runner import check_runtime, run_experiment, run_families


def _configuration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--models", type=Path, help="model catalog (defaults to sibling models.yaml)"
    )
    parser.add_argument(
        "--environments",
        type=Path,
        help="environment catalog (defaults to sibling environments.yaml)",
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--scenario", help="one native benchmark family/scenario")
    parser.add_argument("--tasks", type=int, help="number of tasks in each selected family")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument(
        "--family-workers", type=int, help="independent family streams to execute concurrently"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an existing field, recorded in config.resolved.yaml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="i2s", description="Intervention-to-Skill benchmark task streams"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run I2S on a configured benchmark")
    _configuration_arguments(run)
    run.add_argument("--run-id", help="unique output directory name")
    run.add_argument(
        "--resume", action="store_true", help="resume a stopped stream from its last committed task"
    )
    run.add_argument(
        "--all-families",
        action="store_true",
        help="run every declared family as an independent stream",
    )
    resolve = commands.add_parser("resolve", help="print the resolved configuration")
    _configuration_arguments(resolve)
    check = commands.add_parser(
        "check", help="validate configuration and required local adapter files"
    )
    _configuration_arguments(check)
    evaluate = commands.add_parser(
        "evaluate", help="rebuild native outcome and output-token metrics from an existing run"
    )
    evaluate.add_argument("run_dir", type=Path)
    return parser


def _load(args: argparse.Namespace) -> dict:
    config = load_config(
        args.config,
        benchmark=args.benchmark,
        backend=args.backend,
        model_catalog=args.models,
        environment_catalog=args.environments,
        overrides=args.overrides,
    )
    if args.scenario is not None:
        config["benchmark"]["scenario"] = args.scenario
    if args.tasks is not None:
        config["benchmark"]["sample_count"] = args.tasks
        if config["benchmark"].get("families"):
            config["benchmark"]["families"] = [
                {
                    **({"scenario": family} if isinstance(family, str) else family),
                    "sample_count": args.tasks,
                }
                for family in config["benchmark"]["families"]
            ]
    for key in ("sampling", "environment", "generation"):
        value = getattr(args, f"{key}_seed")
        if value is not None:
            config["seeds"][key] = value
    if args.family_workers is not None:
        config["execution"]["family_workers"] = args.family_workers
    validate_config(config)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate_run(args.run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["validation"]["valid"] else 1
        config = _load(args)
        if args.command == "resolve":
            print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), end="")
            return 0
        if args.command == "check":
            check_runtime(config)
            print("Configuration and required local adapter files are valid.")
            return 0
        if args.resume and not args.run_id:
            parser.error("--resume requires --run-id")
        if args.all_families and args.scenario:
            parser.error("--scenario and --all-families cannot be combined")
        paths = (
            run_families(config, args.config, args.run_id, resume=args.resume)
            if args.all_families
            else [run_experiment(config, args.config, args.run_id, resume=args.resume)]
        )
        success = True
        for path in paths:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            validation = json.loads((path / "validation.json").read_text(encoding="utf-8"))
            print(f"{manifest['status']}: {path}")
            if manifest.get("stop_reason"):
                print(f"  {manifest['stop_reason']}")
            success = success and manifest["status"] == "completed" and validation["valid"]
        return 0 if success else 1
    except KeyboardInterrupt:
        print(
            "Interrupted; returned model usage and committed method state were preserved.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"i2s: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
