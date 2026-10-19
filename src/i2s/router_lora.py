from __future__ import annotations

import re
from typing import Any

LABELS = ("UNCERTAIN", "CONFIDENT")
SYSTEM = (
    "Assess whether the proposed next environment action is reliable and useful in the "
    "supplied visible state. Judge only this action, not whether the entire task is "
    "solved. Ground exact object identity, destination, possession and ordering in the "
    "task, current observation, admissible actions, and interaction history. "
    "Admissibility alone does not establish useful progress. A repeated information "
    "action whose result is already known may fail to make progress; a valid "
    "acquisition or return to a required source can still be necessary. Use supplied "
    "procedural guidance only when its observable conditions hold. Do not invent hidden"
    " locations or assume a proposed action was executed.\n\nReturn exactly CONFIDENT "
    "when the proposal is a supported useful next action, otherwise UNCERTAIN. These "
    "are the only two reliability labels. Do not explain the answer."
)


def assessment_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    visible = "\n\n".join(str(m["content"]) for m in messages if m["role"] != "system")
    visible = re.sub(r"\n\nReference confidence threshold: [^\n]+", "", visible)
    visible = visible.replace(
        "Decide whether this proposal should be executed by the small model or handed to the large model.",
        "Assess the proposed action using exactly one reliability label.",
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": visible}]


def encode_pair(tokenizer, messages, *, max_input_tokens: int = 8192):
    ids = tokenizer.apply_chat_template(
        assessment_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(ids) > max_input_tokens:
        raise ValueError(
            f"Reliability input {len(ids)} exceeds configured input budget {max_input_tokens}"
        )
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if label_ids != [[12633, 3399, 6836], [38634, 19811]]:
        raise ValueError(f"Unexpected Qwen3 label tokenization: {label_ids}")
    return ids, label_ids


def label_log_likelihoods(model, tokenizer, prompt_ids, label_ids):
    import torch

    max_suffix = max(map(len, label_ids))
    seqs = [
        prompt_ids + label + [tokenizer.pad_token_id] * (max_suffix - len(label))
        for label in label_ids
    ]
    masks = [
        [1] * (len(prompt_ids) + len(label)) + [0] * (max_suffix - len(label))
        for label in label_ids
    ]
    inputs = torch.tensor(seqs, dtype=torch.long, device=model.device)
    attention = torch.tensor(masks, dtype=torch.long, device=model.device)

    output = model(
        input_ids=inputs, attention_mask=attention, use_cache=False, logits_to_keep=max_suffix + 1
    )
    probs = output.logits.float().log_softmax(-1)
    scores = []
    for i, label in enumerate(label_ids):
        target = torch.tensor(label, dtype=torch.long, device=model.device)
        scores.append(probs[i, : len(label)].gather(-1, target[:, None]).sum())
    return torch.stack(scores)
