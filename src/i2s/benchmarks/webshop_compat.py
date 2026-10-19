from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

UPSTREAM_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
PROFILE_ID = "official_full_catalog"
RETRIEVAL_BACKEND_ID = "pyserini_lucene_bm25_upstream"
REWARD_NLP_BACKEND_ID = "spacy_en_core_web_sm_3.7.1"
SEARCH_PROTOCOL_ID = "native_search_any_page"
UPSTREAM_SOURCE_HASHES = {
    "web_agent_site/engine/engine.py": "9b86ffc82951124f5c4a7eb8103de14100dbee11ba8869951ed6fb8feaeef804",
    "web_agent_site/engine/goal.py": "9703c8583244e2041182eb14856186b998145ab16ce2ba9aef8cb680877c8ed5",
    "web_agent_site/envs/web_agent_text_env.py": "f4efee238c2a69ad76ff5372716a16051c67ea0351c7006491834822ea9afda3",
}


def validate_profile(config: Mapping[str, Any]) -> None:
    expected = {
        "catalog": "full",
        "human_goals": True,
        "environment_profile": PROFILE_ID,
        "retrieval_backend": RETRIEVAL_BACKEND_ID,
        "reward_nlp_backend": REWARD_NLP_BACKEND_ID,
        "search_protocol": SEARCH_PROTOCOL_ID,
    }
    for field, value in expected.items():
        actual = config.get(field, value)
        if actual != value or (field == "human_goals" and actual is not True):
            raise ValueError(f"Official WebShop requires {field}={value!r}")
    if str(config.get("split", "evaluation")) not in {"evaluation", "dev", "train"}:
        raise ValueError("WebShop split must be train, dev, or evaluation")
    if int(config.get("max_actions", 0)) != 0:
        raise ValueError("Official WebShop must expose every public action (max_actions=0)")
    if str(config.get("observation_mode", "text_rich")) not in {"text", "text_rich"}:
        raise ValueError("WebShop observation_mode must be text or text_rich")


def prepare_native_import(source_dir: Path) -> None:
    source_dir = source_dir.expanduser().resolve()
    for relative, expected in UPSTREAM_SOURCE_HASHES.items():
        path = source_dir / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Official WebShop source is missing: {path}. "
                "Run python scripts/setup_environments.py webshop --download."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"WebShop source differs from pinned upstream {UPSTREAM_COMMIT}: {relative}"
            )
    loaded = sys.modules.get("web_agent_site")
    if loaded is not None:
        package_file = getattr(loaded, "__file__", None)
        if (
            package_file is None
            or Path(package_file).resolve().parent != source_dir / "web_agent_site"
        ):
            raise RuntimeError(
                "A different WebShop checkout is already imported; use a fresh benchmark worker"
            )
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def bind_native_data_paths(data_dir: Path) -> Any:
    data_dir = data_dir.expanduser().resolve()
    paths = {
        "DEFAULT_FILE_PATH": data_dir / "items_shuffle.json",
        "DEFAULT_ATTR_PATH": data_dir / "items_ins_v2.json",
        "HUMAN_ATTR_PATH": data_dir / "items_human_ins.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Full WebShop data file is missing: {path}")
    from web_agent_site import utils
    from web_agent_site.engine import engine

    for name, path in paths.items():
        setattr(utils, name, str(path))
        setattr(engine, name, str(path))
    return engine


def native_search_document(product: Mapping[str, Any]) -> dict[str, Any]:
    option_text = ", and ".join(
        f"{name}: {', '.join(values)}" for name, values in product.get("options", {}).items()
    )
    contents = " ".join(
        [product["Title"], product["Description"], product["BulletPoints"][0], option_text]
    ).lower()
    return {"id": product["asin"], "contents": contents, "product": product}


def validate_full_index(searcher: Any, *, product_count: int, empty_document_count: int) -> None:
    if product_count < 1_000_000:
        raise ValueError(
            "WebShop requires the complete product catalog (at least one million products)"
        )
    expected = product_count - empty_document_count
    if empty_document_count < 0 or expected <= 0 or searcher.num_docs != expected:
        raise ValueError(
            f"WebShop index does not cover the full catalog: {searcher.num_docs} documents; expected {expected}"
        )
