from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import urllib.request
from pathlib import Path

from .io import append_jsonl
from .models import ModelInvocationError, OpenAICompatibleModel
from .confidence_scoring import encode_pair
from .schemas import ModelCall, ModelUsage


def scoring_payload(tokenizer, messages, model, seed, max_input_tokens):
    prompt, labels = encode_pair(tokenizer, list(messages), max_input_tokens=max_input_tokens)
    return {
        "model": model,
        "prompt": [prompt + label for label in labels],
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "prompt_logprobs": 0,
        "echo": False,
    }, {"prompt_length": len(prompt), "labels": labels}


def extract_score(response, info):
    choices = sorted(response["choices"], key=lambda c: c["index"])
    if [c["index"] for c in choices] != [0, 1]:
        raise ValueError("two complete label likelihoods required")
    scores = []
    for choice, label in zip(choices, info["labels"], strict=True):
        entries = choice["prompt_logprobs"]
        score = 0.0
        for i, token in enumerate(label):
            item = entries[info["prompt_length"] + i]
            value = item.get(str(token), item.get(token))
            if value is None:
                raise ValueError("observed label token missing from prompt logprobs")
            score += float(value["logprob"] if isinstance(value, dict) else value)
        if not math.isfinite(score):
            raise ValueError("nonfinite label likelihood")
        scores.append(score)
    lengths = [info["prompt_length"] + len(label) for label in info["labels"]]
    usage = response["usage"]
    if usage["prompt_tokens"] != sum(lengths) or not 0 <= usage["completion_tokens"] <= 2:
        raise ValueError("provider usage does not reconcile with two label prefixes")
    mass = [math.exp(s - max(scores)) for s in scores]
    return mass[1] / sum(mass), scores, lengths


class ConfidenceLikelihoodModel(OpenAICompatibleModel):
    def __init__(self, model_id, config):
        super().__init__(model_id, config)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_path"], local_files_only=bool(config.get("local_files_only", False))
        )
        self.checkpoint = {"checkpoint": config["tokenizer_path"]}
        if config.get("adapter_path"):
            adapter = Path(config["adapter_path"])
            digest = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
            if config.get("adapter_sha256") and digest != config["adapter_sha256"]:
                raise ValueError("Calibration adapter checkpoint hash mismatch")
            self.checkpoint["adapter_sha256"] = digest
            base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
            expected = str(config.get("base_model") or config["tokenizer_path"])
            if base != expected and Path(base).resolve() != Path(expected).resolve():
                raise ValueError("Calibration adapter and tokenizer backbones differ")
        self.audit = (
            Path(config["scoring_audit_path"]) if config.get("scoring_audit_path") else None
        )
        self.audit_lock = threading.Lock()

    def _record(self, value):
        if self.audit is not None:
            with self.audit_lock:
                append_jsonl(self.audit, {**value, **self.checkpoint})

    def generate(self, *, call_id, messages, purpose, generation, seed):
        if purpose != "confidence_assessment":
            raise ValueError("The calibration scorer requires purpose=confidence_assessment")
        started = time.perf_counter()
        raw = None
        fields = dict(
            call_id=call_id,
            model_id=self.model_id,
            model_name=self.model_name,
            role=self.role,
            purpose=purpose,
            messages=messages,
            generation={**dict(generation), "seed": seed},
            retry_count=0,
        )
        try:
            request, info = scoring_payload(
                self.tokenizer, messages, self.model_name, seed, int(generation["max_input_tokens"])
            )
            self._record(
                {"event": "attempt", "call_id": call_id, "request": request, "label_info": info}
            )
            req = urllib.request.Request(
                self.base_url + "/completions",
                data=json.dumps(request).encode(),
                headers={"Content-Type": "application/json", "X-Request-Id": call_id},
            )
            with urllib.request.urlopen(
                req, timeout=self.config.get("timeout_seconds")
            ) as response:
                raw = json.load(response)
            self._record({"event": "response", "call_id": call_id, "response": raw})
            confidence, scores, lengths = extract_score(raw, info)
            fields["generation"].update(
                scoring_transport="vllm_full_label_likelihood",
                scoring_prompt_lengths=lengths,
                scoring_submitted_prompts=2,
                scoring_label_log_likelihoods=scores,
                actual_generated_tail_tokens_charged=raw["usage"]["completion_tokens"],
            )
            return ModelCall(
                **fields,
                raw_output=json.dumps(
                    {
                        "confidence": confidence,
                        "request_large_model": False,
                        "reason": "Calibrated full-label likelihood; apply the configured threshold.",
                    }
                ),
                reasoning_output=None,
                usage=ModelUsage(
                    input_tokens=raw["usage"]["prompt_tokens"],
                    output_tokens=raw["usage"]["completion_tokens"],
                ),
                latency_seconds=time.perf_counter() - started,
                request_status="response_received",
                provider_request_id=raw.get("id"),
                response_model=raw.get("model"),
                finish_reason="label_likelihoods_scored",
                raw_output_source="full_label_likelihood",
            )
        except Exception as exc:
            self._record({"event": "failure", "call_id": call_id, "error_type": type(exc).__name__})
            usage = (raw or {}).get("usage", {})
            call = ModelCall(
                **fields,
                raw_output=None,
                reasoning_output=None,
                usage=ModelUsage(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                ),
                latency_seconds=time.perf_counter() - started,
                request_status="request_failed",
                error=type(exc).__name__,
            )
            raise ModelInvocationError(call.error, call) from exc
