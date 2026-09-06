#!/usr/bin/env python3
"""Immutable model-asset lock, download, and verification helpers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

from mock_logging import log

REQUIRED_FIELDS = {"repo_id", "revision", "filename", "bytes", "sha256", "destination", "required"}
BLOCK = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or not isinstance(lock.get("assets"), list) or not lock["assets"]:
        raise ValueError("ASSET_LOCK_SCHEMA")
    for asset in lock["assets"]:
        if set(asset) != REQUIRED_FIELDS or len(asset["revision"]) != 40 or len(asset["sha256"]) != 64:
            raise ValueError("ASSET_LOCK_SCHEMA")
        if not isinstance(asset["bytes"], int) or asset["bytes"] <= 0 or Path(asset["destination"]).is_absolute():
            raise ValueError("ASSET_LOCK_SCHEMA")
    return lock


def verify_asset(asset: dict, root: Path) -> Path:
    path = root / asset["destination"]
    if not path.is_file():
        raise RuntimeError(f"ASSET_MISSING:{asset['destination']}")
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    log("INFO" if actual_size == asset["bytes"] and actual_hash == asset["sha256"] else "ERROR",
        "asset_verify", stage="assets", gate="C1/C2/C3", asset=asset["destination"],
        expected_bytes=asset["bytes"], actual_bytes=actual_size,
        expected_sha256=asset["sha256"], actual_sha256=actual_hash)
    if actual_size != asset["bytes"]:
        raise RuntimeError(f"ASSET_SIZE_MISMATCH:{asset['destination']}")
    if actual_hash != asset["sha256"]:
        raise RuntimeError(f"ASSET_SHA256_MISMATCH:{asset['destination']}")
    return path


def verify_lock(lock_path: Path, root: Path) -> list[Path]:
    lock = load_lock(lock_path)
    verified = []
    for asset in lock["assets"]:
        try:
            verified.append(verify_asset(asset, root))
        except RuntimeError:
            if asset["required"]:
                raise
    return verified


def download_asset(asset: dict, root: Path, token: str | None = None) -> Path:
    """Download to .partial, verify both invariants, then atomically rename."""
    dest = root / asset["destination"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    url = "https://huggingface.co/{}/resolve/{}/{}".format(asset["repo_id"], asset["revision"], asset["filename"])
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=BLOCK)
        verify_asset({**asset, "destination": str(partial.relative_to(root))}, root)
        os.replace(partial, dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return dest
