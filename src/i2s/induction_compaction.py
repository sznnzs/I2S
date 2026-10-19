from __future__ import annotations

import copy
import json
from collections import Counter

from .utils import estimate_tokens


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_payload(payload, raw_reference):
    counts = Counter()

    def key(x):
        if isinstance(x, str) and len(x) >= 48:
            return ("string", x)
        if isinstance(x, (dict, list)) and len(dumps(x)) >= 160:
            return ("json", dumps(x))
        return None

    def count(x):
        k = key(x)
        if k is not None:
            counts[k] += 1
        if isinstance(x, dict):
            for v in x.values():
                count(v)
        elif isinstance(x, list):
            for v in x:
                count(v)

    count(payload)
    pool, identities, dedup = {}, {}, Counter()

    def encode(x):
        k = key(x)
        if k is not None and counts[k] > 1:
            if k in identities:
                dedup[k[0]] += 1
                return {"$ref": identities[k]}
            name = f"f{len(identities) + 1:04d}"
            identities[k] = name
            pool[name] = children(x)
            return {"$ref": name}
        return children(x)

    def children(x):
        if isinstance(x, dict):
            return {k: encode(v) for k, v in x.items()}
        if isinstance(x, list):
            return [encode(v) for v in x]
        return x

    tree = encode(payload)
    compact = {
        "serialization": "lossless-reference-pool-v1",
        "reading_rule": "Replace each object containing only $ref with the named fragments entry recursively. Lists preserve chronological order. References are exact evidence, not summaries.",
        "raw_reference": raw_reference,
        "raw_utf8_bytes": len(dumps(payload).encode()),
        "fragments": pool,
        "payload": tree,
    }
    if expand_payload(compact) != payload:
        raise ValueError("compaction_round_trip_mismatch")
    report = {
        "raw_reference": raw_reference,
        "raw_utf8_bytes": compact["raw_utf8_bytes"],
        "raw_estimated_tokens": estimate_tokens(json.dumps(payload, ensure_ascii=False)),
        "compact_estimated_tokens": estimate_tokens(dumps(compact)),
        "anchors_preserved": len(payload["eligible_anchors"]),
        "eligible_anchor_ids": [a["anchor_id"] for a in payload["eligible_anchors"]],
        "trajectory_events_preserved": len(payload["completed_trajectory"]["steps"]),
        "transitions_preserved": len(payload["transitions"]),
        "deduplicated_fields": dict(dedup),
        "fragment_count": len(pool),
        "round_trip_equal": True,
        "discarded_evidence": 0,
    }
    return compact, report


def expand_payload(compact):
    def expand(x, active=()):
        if isinstance(x, dict) and set(x) == {"$ref"}:
            ref = x["$ref"]
            if ref in active:
                raise ValueError("cyclic_compaction_reference")
            return expand(compact["fragments"][ref], (*active, ref))
        if isinstance(x, dict):
            return {k: expand(v, active) for k, v in x.items()}
        if isinstance(x, list):
            return [expand(v, active) for v in x]
        return copy.deepcopy(x)

    return expand(compact["payload"])


def literal_evidence_index(payload):
    return [
        {
            "anchor_id": a["anchor_id"],
            "correction": a["correction"],
            "support_type": a["support_type"],
            "localized_observations": [
                {"transition_id": t["transition_id"], "observation_after": t["observation_after"]}
                for t in a["localized_evidence"]
            ],
        }
        for a in payload["eligible_anchors"]
    ]
