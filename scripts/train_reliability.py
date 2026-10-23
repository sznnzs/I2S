#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from i2s.training import DEFAULT_MODEL, TrainingConfig, run_evaluation, run_training


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train or evaluate the confidence model.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Public Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--model-id",
        help="Public base-model ID when training from a local model directory",
    )
    parser.add_argument("--train", type=Path, default=Path("data/reliability/train.jsonl"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/reliability/validation.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/reliability_training"))
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Training or evaluation device; honors CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7311)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Threshold for reported metrics; training also reports a validation threshold sweep",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate a confidence checkpoint in --model and write to a new --output directory",
    )
    parser.add_argument(
        "--adapter", type=Path, help="Trained adapter directory, including reliability_config.json"
    )
    args = parser.parse_args(argv)
    common = dict(
        model=args.model,
        model_id=args.model_id,
        validation_path=args.validation,
        output=args.output,
        device=args.device,
        threshold=args.threshold,
    )
    if args.evaluate_only:
        report = run_evaluation(adapter=args.adapter, **common)
    else:
        if args.adapter is not None:
            parser.error(
                "Use --adapter with --evaluate-only; training starts a new calibration adapter"
            )
        config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation=args.gradient_accumulation,
            max_input_tokens=args.max_input_tokens,
            seed=args.seed,
            dtype=args.dtype,
        )
        report = run_training(train_path=args.train, config=config, **common)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
