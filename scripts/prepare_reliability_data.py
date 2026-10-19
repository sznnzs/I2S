#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from i2s.training import prepare_data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare confidence calibration data from saved runs."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Completed native training-split run directories",
    )
    parser.add_argument("--output", type=Path, default=Path("data/reliability"))
    parser.add_argument(
        "--exclude-tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Held-out tasks.jsonl inventories that must not overlap any source run",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=10120)
    parser.add_argument(
        "--train-cap", type=int, default=12, help="Maximum training examples per task and label"
    )
    args = parser.parse_args(argv)
    report = prepare_data(
        args.runs,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        train_cap=args.train_cap,
        exclude_tasks=args.exclude_tasks,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
