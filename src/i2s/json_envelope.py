import json
import re

PROTOCOL = "whole_json_fence"


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("ambiguous_duplicate_json_key")
        value[key] = item
    return value


def normalize_json_envelope(text):
    raw = str(text or "").strip()
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```", raw, re.I | re.S)
    if not match:
        return raw, None
    body = match.group("body").strip()
    try:
        value = json.loads(body, object_pairs_hook=_unique_pairs)
    except (ValueError, TypeError):
        return raw, None
    if not isinstance(value, dict):
        return raw, None
    return body, PROTOCOL


def load_json_object(text, protocol=None):
    if protocol not in (None, PROTOCOL):
        raise ValueError("unknown_json_envelope_protocol")
    value = json.loads(normalize_json_envelope(text)[0] if protocol else text)
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value
