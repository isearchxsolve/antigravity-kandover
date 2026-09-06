"""CPU-only regression contracts for GODSHAND ranked gap fixes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
SRC = KIT / "src"


class TestE2GpuSmoke(unittest.TestCase):
    def test_gpu_smoke_refuses_mock_mode(self):
        graph = Path(tempfile.mkdtemp()) / "tiny.json"
        graph.write_text("{}", encoding="utf-8")
        env = {**os.environ, "COMFY_MOCK": "1", "KIT_VERBOSE": "1",
               "KIT_LOG_DIR": str(graph.parent / "logs")}
        result = subprocess.run(
            [sys.executable, str(SRC / "gpu_smoke.py"), "--graph", str(graph),
             "--output-png", str(graph.parent / "unused.png")],
            env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("MOCK_FORBIDDEN", result.stderr)
        records = list((graph.parent / "logs").glob("mock_run_*.jsonl"))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_text(encoding="utf-8").splitlines()[0])["stage"], "gpu_smoke")


class TestA1A3A4EnvironmentLock(unittest.TestCase):
    def test_kaggle_environment_lock_is_immutable_and_schema_complete(self):
        sys.path.insert(0, str(SRC))
        import kaggle_environment
        lock = kaggle_environment.load_lock(KIT / "kaggle-environment.lock.json")
        self.assertEqual(lock["torch"]["torch"], "2.4.1")
        with self.assertRaisesRegex(RuntimeError, "GRAPH_SCHEMA_MISSING:SaveImage"):
            kaggle_environment.validate_object_info({"UnetLoaderGGUF": {}, "WanImageToVideo": {}})


class TestB1B2VramReserve(unittest.TestCase):
    def test_mem_get_info_uses_free_and_reserve_blocks_enqueue(self):
        from unittest.mock import patch
        sys.path.insert(0, str(SRC))
        import render_motion
        class Cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def mem_get_info(_device): return (512 * 1024 * 1024, 16384 * 1024 * 1024)
            @staticmethod
            def empty_cache(): pass
        class Torch: cuda = Cuda()
        with patch.object(render_motion, "torch", Torch()), patch.dict(os.environ, {"COMFY_MOCK": "0"}, clear=False):
            self.assertEqual(render_motion.vram_free_mb(), 512)
            self.assertEqual(render_motion.vram_total_mb(), 16384)
            self.assertFalse(render_motion.ensure_vram(500))


class TestC1C2C3Assets(unittest.TestCase):
    def test_lock_schema_and_hash_mismatch_fail(self):
        sys.path.insert(0, str(SRC))
        import assets
        lock = assets.load_lock(KIT / "assets.lock.json")
        self.assertGreaterEqual(len(lock["assets"]), 7)
        root = Path(tempfile.mkdtemp())
        asset = {"repo_id": "unit/repo", "revision": "a" * 40, "filename": "x.bin",
                 "bytes": 3, "sha256": "0" * 64, "destination": "models/x.bin", "required": True}
        target = root / asset["destination"]; target.parent.mkdir(parents=True); target.write_bytes(b"bad")
        with self.assertRaisesRegex(RuntimeError, "ASSET_SHA256_MISMATCH:models/x.bin"):
            assets.verify_asset(asset, root)


class TestE1F1QcCli(unittest.TestCase):
    def test_notebook_shots_dir_command_writes_report(self):
        root = Path(tempfile.mkdtemp())
        env = {**os.environ, "COMFY_MOCK": "1", "KIT_VERBOSE": "1", "KIT_CONFIG": "config_confrontation.yaml",
               "KIT_LOG_DIR": str(root / "logs")}
        command = [sys.executable, "src/qc.py", "--shots-dir", str(root)]
        result = subprocess.run(command, cwd=KIT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = root / "qc_report.json"
        self.assertTrue(report.is_file())
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["records"], [])

    def test_master_cli_missing_file_is_probe_only_and_fatal(self):
        missing = Path(tempfile.mkdtemp()) / "missing.mp4"
        result = subprocess.run([sys.executable, "src/qc.py", "--master", str(missing)], cwd=KIT,
                                text=True, capture_output=True, env={**os.environ, "KIT_CONFIG": "config_confrontation.yaml"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("MASTER_MISSING", result.stderr)


class TestE4MasterDispatch(unittest.TestCase):
    def test_orchestrator_uses_probe_only_qc_master_command(self):
        source = (KIT / "run.py").read_text(encoding="utf-8")
        self.assertIn('"qc.py"), "--master", master_path', source)
        self.assertNotIn("run.py _qc_master", source)


class TestH1EnginePolicy(unittest.TestCase):
    def test_wan_failure_defaults_to_fatal_and_manifest_requires_wan(self):
        sys.path.insert(0, str(SRC))
        import engine_policy
        with self.assertRaisesRegex(RuntimeError, "WAN_ASSET_SETUP_FAILED"):
            engine_policy.wan_setup_failed("missing locked asset")
        root = Path(tempfile.mkdtemp())
        manifest = engine_policy.write_shot_manifest(root, shot_id="s01", engine="wan")
        engine_policy.assert_wan_manifest(manifest)


class TestB3B4B5WorkerContract(unittest.TestCase):
    def test_two_t4_profile_and_per_worker_local_binding(self):
        sys.path.insert(0, str(SRC))
        import worker_contract
        self.assertEqual(worker_contract.validate_kaggle_t4s([{"name":"Tesla T4"}, {"name":"Tesla T4"}], "kaggle_2xt4_zero_oom"), 2)
        env = worker_contract.worker_env(1, {"fp8": True, "model_offload": True, "sequential_offload": False})
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["KIT_PHYSICAL_GPU_ID"], env["KIT_LOCAL_CUDA_DEVICE"]), ("1", "1", "0"))
        self.assertIn("--fp8_e4m3fn-unet", env["COMFY_ARGS"])


class TestH2F3FatalMaster(unittest.TestCase):
    def test_missing_master_is_fatal_and_stage_status_is_success_only(self):
        source = (KIT / "run.py").read_text(encoding="utf-8")
        self.assertIn('raise SystemExit("FINAL_MASTER_MISSING")', source)
        self.assertIn('"last_successful_stage"', source)
        self.assertIn('subprocess.run(list(a), check=True)', (SRC / "stitch.py").read_text(encoding="utf-8"))


class TestG1G3G5G6DeliverySafety(unittest.TestCase):
    def test_disk_math_zip_policy_and_audio_args(self):
        sys.path.insert(0, str(SRC))
        import delivery_safety as ds
        self.assertEqual(ds.estimate_working_bytes(10, 100, 100, 2), 750000)
        with self.assertRaisesRegex(RuntimeError, "DISK_SHORTFALL:extract"):
            ds.require_disk("extract", 100, 20, free_bytes=119)
        self.assertEqual(ds.zip_policy(Path("master.mov")), __import__("zipfile").ZIP_STORED)
        self.assertIn("1:a?", ds.audio_encode_args(True))
        self.assertNotIn("1:a?", ds.audio_encode_args(False))
        self.assertIn("-shortest", ds.audio_encode_args(True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
