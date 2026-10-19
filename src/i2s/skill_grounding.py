from __future__ import annotations

import re

ROLES = {
    "TARGET_OBJECT",
    "OBJECT",
    "DESTINATION",
    "RECEPTACLE",
    "CONTAINER",
    "SOURCE_RECEPTACLE",
    "SOURCE",
    "TOOL",
    "APPLIANCE",
}
PLACEHOLDERS = re.compile(
    r"narrow observable conditions supported here|ordered supported step|grounded fallback if verification fails|exact eligible anchor id|template|placeholder|schema example|insert (?:your|the) (?:text|content)|\b(?:todo|tbd|lorem ipsum)\b",
    re.I,
)
OBSERVABLE = re.compile(
    r"\b(?:instruction|task|requested|visible|observ|current|inventory|carrying|held|closed|open|at|contains|shown|admissible)\w*\b",
    re.I,
)
ACTION = re.compile(
    r"\b(?:go to|open|close|take|move|inventory|look|inspect|check|examine|clean|heat|cool|use|choose|select|return|retry)\b",
    re.I,
)
CONDITION = re.compile(
    r"\b(?:if|when|unless|on rejection|on failure|after rejection|after failure)\b", re.I
)
HIDDEN = re.compile(
    r"\b(?:evaluator|reward|hidden state|ground.truth|will succeed|guaranteed|always located|any task|all tasks)\b",
    re.I,
)
UNSUPPORTED = r"search|put|pickup|pick up|grab|walk|navigate|drop|place|find|locate"


def grounding_issues(content):
    issues = []

    def add(code, field, text, detail):
        row = {"code": code, "field": field, "excerpt": text, "detail": detail}
        if row not in issues:
            issues.append(row)

    fields = {
        k: v
        for k, v in content.items()
        if k in {"applicability", "procedure", "outcome_check", "fallback"}
    }
    for field, value in fields.items():
        parts = value if isinstance(value, list) else [value]
        for text in parts:
            if not isinstance(text, str):
                add("invalid_content_type", field, str(text), "Expected natural-language text.")
                continue
            if PLACEHOLDERS.search(text):
                add(
                    "placeholder_leakage",
                    field,
                    text,
                    "Template or meta-content is not an instruction.",
                )
            if HIDDEN.search(text):
                add(
                    "nonobservable_or_unbounded_claim",
                    field,
                    text,
                    "Scope must use current visible evidence.",
                )
            roles = re.findall(r"<([^<>]+)>", text)
            if any(role not in ROLES for role in roles):
                add(
                    "malformed_entity_role",
                    field,
                    text,
                    "Role names must describe supported object/receptacle/capability roles.",
                )
            if re.search(r"\b(?:go to|open|close) <(?:OBJECT|TARGET_OBJECT)>", text, re.I):
                add(
                    "malformed_entity_role",
                    field,
                    text,
                    "Navigation/open/close targets a receptacle, not the carried target object.",
                )
            if re.search(
                r"\btake <(?:OBJECT|TARGET_OBJECT)> (?:to|on|in)\b|\bmove <(?:OBJECT|TARGET_OBJECT)> (?:from|on|in)\b",
                text,
                re.I,
            ):
                add("impossible_command_template", field, text, "take uses from; move uses to.")

            segments = re.split(r"[.;\n]|\bthen\b|:\s*", text, flags=re.I)
            for segment in segments:
                s = segment.strip(" `\"'-")
                if (
                    re.match(rf"^(?:next,?\s+)?(?:{UNSUPPORTED})\b", s, re.I)
                    or re.search(rf"[`\"](?:{UNSUPPORTED})\s", segment, re.I)
                    or re.search(
                        rf"\b(?:emit|execute|command|action)\s+[`\"]?(?:{UNSUPPORTED})\b",
                        segment,
                        re.I,
                    )
                ):
                    add(
                        "unsupported_action_verb",
                        field,
                        text,
                        "Use admissible ALFWorld primitives; inspection is realized by go/open/observations.",
                    )
            for command in re.findall(r"`([^`]+)`", text):
                if "<" not in command:
                    continue
                c = command.lower()
                if c.startswith("take ") and not re.fullmatch(
                    r"take <(?:target_object|object)> from <(?:source|source_receptacle|receptacle|container)>",
                    c,
                ):
                    add(
                        "impossible_command_template",
                        field,
                        text,
                        "take requires object from source receptacle.",
                    )
                if c.startswith("move ") and not re.fullmatch(
                    r"move <(?:target_object|object)> to <(?:destination|receptacle|container)>", c
                ):
                    add(
                        "impossible_command_template",
                        field,
                        text,
                        "move requires held object to destination receptacle.",
                    )
                if re.match(r"(?:go to|open|close) <(?:object|target_object)>", c):
                    add(
                        "malformed_entity_role",
                        field,
                        text,
                        "Navigation/open/close targets a receptacle, not the carried target object.",
                    )
    applicability = str(content.get("applicability", ""))
    if len(applicability.split()) < 6 or not OBSERVABLE.search(applicability):
        add(
            "nonobservable_applicability",
            "applicability",
            applicability,
            "Require a specific pre/during-execution observable condition.",
        )
    fallback = str(content.get("fallback", ""))
    if not (CONDITION.search(fallback) and ACTION.search(fallback)):
        add(
            "nonactionable_fallback",
            "fallback",
            fallback,
            "Require an observable failure condition and a grounded next operation.",
        )
    return issues
