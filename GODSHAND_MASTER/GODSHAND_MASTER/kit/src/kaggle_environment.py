#!/usr/bin/env python3
"""Validate the immutable Kaggle environment lock without installing anything.

The notebook must install requirements-kaggle-cu121.txt before it clones ComfyUI.
This CPU-testable validator makes both the lock and node-schema contract explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mock_logging import log

REQUIRED_NODES = {"UnetLoaderGGUF", "WanImageToVideo", "SaveImage"}
SHA_LEN = 40


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    torch = lock.get("torch", {})
    repos = lock.get("repositories", {})
    if torch.get("index_url") != "https://download.pytorch.org/whl/cu121":
        raise ValueError("ENV_LOCK_TORCH_INDEX")
    if (torch.get("torch"), torch.get("torchvision"), torch.get("torchaudio")) != ("2.4.1", "0.19.1", "2.4.1"):
        raise ValueError("ENV_LOCK_TORCH_VERSIONS")
    for name in ("ComfyUI", "ComfyUI-GGUF", "ComfyUI-VideoHelperSuite"):
        revision = repos.get(name, {}).get("revision", "")
        if len(revision) != SHA_LEN or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"ENV_LOCK_REVISION:{name}")
    if set(lock.get("required_node_schemas", ())) != REQUIRED_NODES:
        raise ValueError("ENV_LOCK_NODE_SCHEMAS")
    log("INFO", "validated", stage="environment_lock", gate="A1/A3/A4",
        lock=str(path), repositories=sorted(repos))
    return lock


def validate_object_info(object_info: dict) -> None:
    missing = sorted(REQUIRED_NODES.difference(object_info))
    if missing:
        raise RuntimeError("GRAPH_SCHEMA_MISSING:" + ",".join(missing))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(load_lock(args.lock), sort_keys=True))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
