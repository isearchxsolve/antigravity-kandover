#!/usr/bin/env python3
"""WAN engine identity is a delivery contract, never a silent fallback."""
from __future__ import annotations
import json
import os
from pathlib import Path
from mock_logging import log


def wan_setup_failed(detail: str) -> str:
    if os.environ.get("ALLOW_ENGINE_FALLBACK") == "1":
        os.environ["KIT_ENGINE"] = "ltxv"
        log("ERROR", "engine_fallback_opt_in", stage="engine", gate="H1", detail=detail)
        return "ltxv"
    log("ERROR", "engine_setup_failed", stage="engine", gate="H1", detail=detail)
    raise RuntimeError("WAN_ASSET_SETUP_FAILED")


def write_shot_manifest(output_dir: Path, *, shot_id: str, engine: str = "wan", **details) -> Path:
    if engine != "wan":
        raise RuntimeError("ENGINE_NOT_WAN")
    path = output_dir / "shot_manifest.json"
    payload = {"shot_id": shot_id, "engine": engine, **details}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log("INFO", "shot_manifest", stage="engine", gate="H1", shot_id=shot_id,
        engine=engine, manifest=str(path), **details)
    return path


def assert_wan_manifest(path: Path) -> None:
    try:
        engine = json.loads(path.read_text(encoding="utf-8")).get("engine")
    except Exception as exc:
        raise RuntimeError(f"SHOT_MANIFEST_INVALID:{path}") from exc
    if engine != "wan":
        raise RuntimeError(f"ENGINE_NOT_WAN:{path}")
