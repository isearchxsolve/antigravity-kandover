"""CPU-only contracts for Wave 2 notebook command injection.

These tests deliberately exercise only fail-closed/mock-safe paths.  They do
not start ComfyUI, allocate CUDA, download model weights, or render media.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


KIT = Path(__file__).resolve().parents[1]
SRC = KIT / "src"
ROOT = KIT.parents[2]


class TestNotebookStructure(unittest.TestCase):
    def test_production_notebooks_are_json_and_changed_cells_parse(self):
        targets = {
            ROOT / "rainman-wan-MERGED-TUNED-FIXED.ipynb": (1,),
            ROOT / "GODSHAND_COMPLETE_GATED_PIPELINE.ipynb": (3, 4, 6, 7, 9, 10),
        }
        for path, cells in targets.items():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for number in cells:
                source = "".join(notebook["cells"][number]["source"])
                ast.parse(source)
                self.assertIn("# wave2:", source, f"{path.name} cell {number}")


class TestInjectedCliContracts(unittest.TestCase):
    def test_environment_lock_command_from_notebook(self):
        result = subprocess.run(
            [sys.executable, "src/kaggle_environment.py", "--lock", "kaggle-environment.lock.json"],
            cwd=KIT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"torch": {', result.stdout)

    def test_gpu_smoke_command_fails_closed_in_mock_mode(self):
        temp = Path(tempfile.mkdtemp())
        graph = temp / "flf2v.json"; graph.write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "src/gpu_smoke.py", "--base-url", "http://127.0.0.1:8188",
             "--graph", str(graph), "--output-png", str(temp / "gpu_smoke.png")],
            cwd=KIT, env={**os.environ, "COMFY_MOCK": "1"}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("MOCK_FORBIDDEN", result.stderr)

    def test_preflight_command_fails_closed_without_locked_assets(self):
        temp = Path(tempfile.mkdtemp())
        result = subprocess.run(
            [sys.executable, "src/preflight_gpu.py", "--assets", "assets.lock.json",
             "--work", str(temp), "--verify"],
            cwd=KIT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("ASSET_", result.stderr)

    def test_qc_commands_from_notebook_have_expected_exit_contracts(self):
        shots = Path(tempfile.mkdtemp())
        report = shots / "qc_report.json"
        result = subprocess.run(
            [sys.executable, "src/qc.py", "--shots-dir", str(shots), "--report", str(report)],
            cwd=KIT, env={**os.environ, "COMFY_MOCK": "1", "KIT_CONFIG": "config_confrontation.yaml"},
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report.is_file())
        missing = shots / "missing-master.mp4"
        result = subprocess.run(
            [sys.executable, "src/qc.py", "--master", str(missing)], cwd=KIT,
            env={**os.environ, "KIT_CONFIG": "config_confrontation.yaml"}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("MASTER_MISSING", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
