#!/usr/bin/env python3
"""Small structured evidence logger for CPU/mock safety tests.

Logging is intentionally opt-in for production compatibility, but mock mode and
tests enable it by default.  Records never include environment values, tokens,
or request headers.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def enabled() -> bool:
    return (os.environ.get("KIT_VERBOSE") == "1" or
            os.environ.get("COMFY_MOCK") == "1" or
            "PYTEST_CURRENT_TEST" in os.environ or
            "UNITTEST_CURRENT_TEST" in os.environ)


def _log_path() -> Path:
    root = Path(os.environ.get("KIT_LOG_DIR", "logs"))
    root.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("KIT_MOCK_RUN_ID")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        os.environ["KIT_MOCK_RUN_ID"] = run_id
    return root / f"mock_run_{run_id}.jsonl"


def log(level: str, event: str, *, stage: str, gate: str | None = None,
        shot_id: str | None = None, **fields) -> dict:
    """Emit one JSONL evidence record to stderr and the per-run log file."""
    if not enabled():
        return {}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(), "event": event, "stage": stage,
        "gate": gate, "shot_id": shot_id,
        "mode": "mock" if os.environ.get("COMFY_MOCK") == "1" else "real",
        **fields,
    }
    line = json.dumps(record, sort_keys=True, default=str)
    print(line, file=sys.stderr, flush=True)
    with _log_path().open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return record
