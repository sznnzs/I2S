from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def command(args: argparse.Namespace) -> list[str]:
    small = args.role == "small"
    model = args.model or os.environ.get(
        "I2S_SMALL_MODEL" if small else "I2S_LARGE_MODEL",
        "Qwen/Qwen3-8B" if small else "Qwen/Qwen3-32B",
    )
    result = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        "Qwen3-8B" if small else "Qwen3-32B",
        "--host",
        args.host,
        "--port",
        str(args.port or (8000 if small else 8001)),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--reasoning-parser",
        "qwen3",
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if small and not args.without_adapter:
        adapter = Path(
            args.adapter or os.environ.get("I2S_ADAPTER", "models/reliability")
        ).expanduser()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter / name).is_file():
                raise FileNotFoundError(
                    f"Calibration adapter file is missing: {adapter / name}. "
                    "Set --adapter or I2S_ADAPTER to the adapter directory."
                )
        settings = json.loads((adapter / "adapter_config.json").read_text())
        if int(settings["r"]) > 8:
            raise ValueError("the default server supports calibration adapter rank at most 8")
        result += [
            "--enable-lora",
            "--max-lora-rank",
            "8",
            "--lora-modules",
            f"i2s-reliability={adapter.resolve()}",
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the calibrated actor or the local teacher.")
    parser.add_argument("role", choices=("small", "large"))
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument(
        "--without-adapter",
        action="store_true",
        help="serve the base actor to collect confidence calibration examples",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.tensor_parallel_size < 1 or args.max_num_seqs < 1:
        parser.error("parallelism and max-num-seqs must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    result = command(args)
    subprocess.run(result, check=True)


if __name__ == "__main__":
    main()
