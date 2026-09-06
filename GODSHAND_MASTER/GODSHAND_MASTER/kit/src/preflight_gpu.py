#!/usr/bin/env python3
"""Asset preflight used by Gate 3; does not launch inference."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from assets import load_lock, verify_lock, download_asset
from mock_logging import log

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("."))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    try:
        lock = load_lock(args.assets)
        if args.download:
            for asset in lock["assets"]:
                try: download_asset(asset, args.work, os.environ.get("HF_TOKEN"))
                except Exception:
                    if asset["required"]: raise
        if args.verify:
            verified = verify_lock(args.assets, args.work)
        else:
            verified = []
        result = {"status":"PASS", "assets": [str(p) for p in verified]}
        log("INFO", "preflight_pass", stage="assets", gate="C1/C2/C3", **result)
        print(json.dumps(result, sort_keys=True)); return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr); return 2
if __name__ == "__main__": raise SystemExit(main())
