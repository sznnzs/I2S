from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from i2s.training import DEFAULT_MODEL, check_adapter, fresh_output, public_model_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a complete confidence model checkpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-id")
    parser.add_argument("--adapter", type=Path, default=Path("runs/reliability_training"))
    parser.add_argument("--output", type=Path, default=Path("runs/confidence_model"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    reference = public_model_id(args.model, args.model_id)
    metadata = check_adapter(args.adapter, reference)
    fresh_output(args.output)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map={"": args.device}
    )
    calibrated = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model = calibrated.merge_and_unload(safe_merge=True).eval()
    model.config._name_or_path = reference
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    tokenizer_source = args.adapter if (args.adapter / "tokenizer_config.json").is_file() else args.model
    AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(args.output)
    metadata = {
        **metadata, "checkpoint_format": "full", "merge_dtype": "float32",
        "weight_dtype": "float32",
    }
    (args.output / "reliability_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
