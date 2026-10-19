from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .utils import json_safe


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            writer.write(text)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_yaml(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        yaml.safe_dump(json_safe(value), allow_unicode=True, sort_keys=False),
    )


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(json_safe(dict(value)), ensure_ascii=False, sort_keys=True))
        writer.write("\n")
        writer.flush()


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    text = "".join(
        json.dumps(json_safe(dict(row)), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return atomic_write_text(path, text)
