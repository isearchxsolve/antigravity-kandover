#!/usr/bin/env python3
"""Real, one-shot WAN smoke gate.

This program deliberately refuses mock mode.  It is safe to invoke on CPU only
to verify that refusal; the remaining checks require a running real ComfyUI
server and are therefore intentionally not exercised by the CPU suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from mock_logging import log

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUIRED_NODES = {"UnetLoaderGGUF", "WanImageToVideo", "SaveImage"}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def refuse_mock() -> None:
    if os.environ.get("COMFY_MOCK") == "1":
        log("ERROR", "refused", stage="gpu_smoke", gate="E2",
            reason="COMFY_MOCK=1")
        raise RuntimeError("MOCK_FORBIDDEN: gpu smoke requires COMFY_MOCK=0")


def validate_png(path: Path) -> None:
    if not path.is_file() or path.read_bytes()[:8] != PNG_SIGNATURE:
        raise RuntimeError(f"GPU_SMOKE_PNG_SIGNATURE_INVALID:{path}")


def run_smoke(base_url: str, graph: Path, output_png: Path) -> dict:
    """Validate real server/model prerequisites and a real PNG output.

    Graph execution is queued by the production launcher after its graph-specific
    fixture substitutions.  This gate verifies the server schema, CUDA telemetry
    and output signature without treating any mock artifact as proof.
    """
    refuse_mock()
    if not graph.is_file():
        raise RuntimeError(f"GPU_SMOKE_GRAPH_MISSING:{graph}")
    system = _get_json(base_url.rstrip("/") + "/system_stats")
    devices = ((system.get("devices") or system.get("system") or {}).get("devices")
               if isinstance(system, dict) else None) or []
    if not devices:
        raise RuntimeError("GPU_SMOKE_CUDA_ALLOCATION_UNVERIFIED:no device telemetry")
    object_info = _get_json(base_url.rstrip("/") + "/object_info")
    missing = sorted(REQUIRED_NODES.difference(object_info))
    if missing:
        raise RuntimeError("GPU_SMOKE_NODE_SCHEMA_MISSING:" + ",".join(missing))
    # The real runner must supply a PNG emitted by SaveImage for the locked,
    # 64-aligned tiny fixture.  Signature verification prevents a text/mock file.
    validate_png(output_png)
    result = {"status": "PASS", "device_count": len(devices),
              "graph": str(graph), "output_png": str(output_png),
              "nodes": sorted(REQUIRED_NODES)}
    log("INFO", "passed", stage="gpu_smoke", gate="E2", **result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GODSHAND real WAN smoke gate")
    ap.add_argument("--base-url", default=os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188"))
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--output-png", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run_smoke(args.base_url, args.graph, args.output_png), sort_keys=True))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
