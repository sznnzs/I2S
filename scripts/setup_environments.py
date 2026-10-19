#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from i2s.benchmarks.webshop_compat import (
    UPSTREAM_COMMIT,
    bind_native_data_paths,
    native_search_document,
    prepare_native_import,
    validate_full_index,
)

ALFWORLD_CONFIG_URL = (
    "https://raw.githubusercontent.com/alfworld/alfworld/"
    "1558ba46d078279ecb4c5d33a6cdffc96714a2d2/configs/base_config.yaml"
)
WEBSHOP_REPOSITORY = "https://github.com/princeton-nlp/WebShop.git"
WEBSHOP_DATA = {
    "items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
    "items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}


def _download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=60) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    temporary.replace(destination)


def prepare_alfworld(data_dir: Path, config_path: Path) -> None:
    downloader = Path(sys.executable).with_name("alfworld-download")
    if not downloader.is_file():
        raise RuntimeError("Install alfworld==0.4.2 in the active Python environment first")
    environment = dict(os.environ, ALFWORLD_DATA=str(data_dir.resolve()))
    subprocess.run(
        [sys.executable, str(downloader), "--data-dir", str(data_dir.resolve())],
        env=environment,
        check=True,
    )
    _download(ALFWORLD_CONFIG_URL, config_path)


def prepare_webshop_source(source_dir: Path) -> None:
    if source_dir.exists():
        prepare_native_import(source_dir)
        return
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="webshop-source-", dir=source_dir.parent) as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "remote", "add", "origin", WEBSHOP_REPOSITORY],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            check=True,
        )
        checkout.rename(source_dir)
    prepare_native_import(source_dir)


def prepare_webshop_data(data_dir: Path) -> None:
    import gdown

    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, file_id in WEBSHOP_DATA.items():
        destination = data_dir / filename
        if destination.is_file():
            continue
        temporary = destination.with_suffix(destination.suffix + ".partial")
        result = gdown.download(id=file_id, output=str(temporary), quiet=False, resume=True)
        if result is None or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"WebShop download did not complete: {filename}")
        temporary.replace(destination)


def build_webshop_index(source_dir: Path, data_dir: Path, index_dir: Path, threads: int) -> None:
    if threads <= 0:
        raise ValueError("--threads must be positive")
    if index_dir.exists():
        raise FileExistsError(
            f"Index destination already exists: {index_dir}. Use another --webshop-index path."
        )
    prepare_native_import(source_dir)
    engine = bind_native_data_paths(data_dir)
    products, product_map, product_prices, attributes = engine.load_products(
        str(data_dir / "items_shuffle.json"), num_products=None, human_goals=True
    )
    if len(products) < 1_000_000:
        raise ValueError("WebShop requires the complete catalog; the loaded catalog is too small")
    count = len(products)
    empty_documents = 0
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="webshop-index-", dir=index_dir.parent) as temporary:
        staging = Path(temporary)
        resources = staging / "documents"
        resources.mkdir()
        shards = [
            (resources / f"part-{i:02d}.jsonl").open("w", encoding="utf-8") for i in range(threads)
        ]
        try:
            for index, product in enumerate(products):
                document = native_search_document(product)
                if not document["contents"].strip():
                    empty_documents += 1
                shards[index % len(shards)].write(json.dumps(document, ensure_ascii=False) + "\n")
        finally:
            for shard in shards:
                shard.close()
        del products, product_map, product_prices, attributes
        gc.collect()
        staged_index = staging / "index"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                str(resources),
                "--index",
                str(staged_index),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(threads),
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ],
            check=True,
        )
        from pyserini.search.lucene import LuceneSearcher

        searcher = LuceneSearcher(str(staged_index))
        validate_full_index(searcher, product_count=count, empty_document_count=empty_documents)
        close = getattr(searcher, "close", None)
        if callable(close):
            close()
        staged_index.rename(index_dir)
    print(f"WebShop index prepared: {count - empty_documents} searchable products")


def check_installation(benchmark: str, args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    modules = {
        "alfworld": ("alfworld", "textworld", "yaml"),
        "scienceworld": ("scienceworld",),
        "webshop": (
            "gym",
            "numpy",
            "torch",
            "spacy",
            "en_core_web_sm",
            "pyserini",
            "jnius",
            "flask",
            "bs4",
            "selenium",
            "cleantext",
            "thefuzz",
            "rank_bm25",
        ),
    }
    for name in modules[benchmark]:
        if importlib.util.find_spec(name) is None:
            errors.append(f"Missing Python dependency: {name}")
    if benchmark in {"scienceworld", "webshop"} and shutil.which("java") is None:
        errors.append(
            "Java is not on PATH; activate the Conda environment or configure JAVA_HOME and PATH"
        )
    if benchmark == "alfworld":
        required = [args.alfworld_config]
        required.extend(
            args.data_dir / "alfworld" / name
            for name in (
                "json_2.1.1/train",
                "json_2.1.1/valid_seen",
                "json_2.1.1/valid_unseen",
                "logic/alfred.pddl",
                "logic/alfred.twl2",
            )
        )
        errors.extend(
            f"Missing ALFWorld dependency: {path}" for path in required if not path.exists()
        )
    elif benchmark == "webshop":
        required = [args.data_dir / "webshop" / name for name in WEBSHOP_DATA]
        errors.extend(f"Missing WebShop data: {path}" for path in required if not path.is_file())
        try:
            prepare_native_import(args.webshop_source)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
        if not args.webshop_index.is_dir() or not any(args.webshop_index.glob("segments_*")):
            errors.append(f"Missing Lucene index: {args.webshop_index}; run with --build-index")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install benchmark assets and build search indexes."
    )
    parser.add_argument(
        "benchmark",
        choices=("all", "alfworld", "webshop", "scienceworld"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--alfworld-config", type=Path, default=Path("configs/environments/alfworld.yaml")
    )
    parser.add_argument("--webshop-source", type=Path, default=Path("third_party/WebShop"))
    parser.add_argument("--webshop-index", type=Path, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download official source and full benchmark datasets",
    )
    parser.add_argument(
        "--build-index", action="store_true", help="Build a new full-catalog WebShop Lucene index"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="CPU threads and document shards for indexing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.alfworld_config = args.alfworld_config.expanduser().resolve()
    args.webshop_source = args.webshop_source.expanduser().resolve()
    args.webshop_index = (
        args.webshop_index.expanduser().resolve()
        if args.webshop_index is not None
        else args.data_dir / "webshop/search_index"
    )
    selected = (
        ("alfworld", "scienceworld", "webshop") if args.benchmark == "all" else (args.benchmark,)
    )
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.build_index and "webshop" not in selected:
        raise ValueError("--build-index requires benchmark=webshop or all")
    for benchmark in selected:
        if args.download and benchmark == "alfworld":
            prepare_alfworld(args.data_dir / "alfworld", args.alfworld_config)
        elif args.download and benchmark == "webshop":
            prepare_webshop_source(args.webshop_source)
            prepare_webshop_data(args.data_dir / "webshop")
        if args.build_index and benchmark == "webshop":
            build_webshop_index(
                args.webshop_source, args.data_dir / "webshop", args.webshop_index, args.threads
            )
        errors = check_installation(benchmark, args)
        if errors:
            for error in errors:
                print(f"{benchmark}: {error}", file=sys.stderr)
            return 1
        print(f"{benchmark}: dependencies and paths present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
