#!/usr/bin/env python3
"""Unit tests for VRAM Governor v2 - Zero OOM on Kaggle 2xT4."""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add src to path (works from the repo root or the kit layout)
_here = os.path.dirname(__file__)
for _candidate in (
    os.path.join(_here, '..', 'src'),
    os.path.join(_here, 'GODSHAND_MASTER', 'GODSHAND_MASTER', 'kit', 'src'),
):
    sys.path.insert(0, os.path.abspath(_candidate))
import vram_governor as vg


class TestVRAMGovernor(unittest.TestCase):
    """Unit tests for VRAM Governor core functions."""

    @patch('vram_governor.subprocess.run')
    def test_query_smi_parses_nvidia_smi_output(self, mock_run):
        """Test _query_smi parses nvidia-smi CSV output correctly."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, Tesla T4, 16384, 2048, 14336, 45, 10\n1, Tesla T4, 16384, 8192, 8192, 55, 50"
        )
        
        gpus = vg._query_smi()
        
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["index"], 0)
        self.assertEqual(gpus[0]["name"], "Tesla T4")
        self.assertEqual(gpus[0]["total_mb"], 16384)
        self.assertEqual(gpus[0]["used_mb"], 2048)
        self.assertEqual(gpus[0]["free_mb"], 14336)
        self.assertEqual(gpus[0]["temp_c"], 45)
        self.assertEqual(gpus[0]["util_pct"], 10)
        
        self.assertEqual(gpus[1]["index"], 1)
        self.assertEqual(gpus[1]["free_mb"], 8192)

    @patch('vram_governor.subprocess.run')
    def test_query_smi_handles_errors(self, mock_run):
        """Test _query_smi handles nvidia-smi errors gracefully."""
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        
        gpus = vg._query_smi()
        self.assertEqual(gpus, [])

    @patch('vram_governor.subprocess.run')
    def test_query_smi_handles_malformed_output(self, mock_run):
        """Test _query_smi handles malformed nvidia-smi output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="malformed,output\n0, Tesla T4, 16384, 2048, 14336, 45, 10"
        )
        
        gpus = vg._query_smi()
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["index"], 0)

    @patch('vram_governor._query_smi')
    def test_query_smi_robust_retries(self, mock_query):
        """Test _query_smi_robust retries on failure."""
        mock_query.side_effect = [[], [], [{"index": 0, "free_mb": 10000}]]
        
        gpus = vg._query_smi_robust()
        
        self.assertEqual(len(gpus), 1)
        self.assertEqual(mock_query.call_count, 3)

    @patch('vram_governor._query_smi_robust')
    def test_free_mb_returns_free_vram(self, mock_robust):
        """Test free_mb returns free VRAM for specific GPU."""
        mock_robust.return_value = [
            {"index": 0, "free_mb": 14336},
            {"index": 1, "free_mb": 8192},
        ]
        
        self.assertEqual(vg.free_mb(0), 14336)
        self.assertEqual(vg.free_mb(1), 8192)
        self.assertEqual(vg.free_mb(2), 0)  # Non-existent GPU

    @patch('vram_governor._query_smi_robust')
    def test_probe_all_returns_all_gpus(self, mock_robust):
        """Test probe_all returns all GPU info."""
        mock_robust.return_value = [
            {"index": 0, "name": "Tesla T4", "free_mb": 14336, "temp_c": 45, "util_pct": 10},
            {"index": 1, "name": "Tesla T4", "free_mb": 8192, "temp_c": 55, "util_pct": 50},
        ]
        
        gpus = vg.probe_all()
        self.assertEqual(len(gpus), 2)

    def test_pick_regime_t4_plenty(self):
        """Test regime selection for T4_PLENTY (>=12GB free)."""
        self.assertEqual(vg.pick_regime(14336), vg.REGIME_T4_PLENTY)
        self.assertEqual(vg.pick_regime(12288), vg.REGIME_T4_PLENTY)
        self.assertEqual(vg.pick_regime(16384), vg.REGIME_T4_PLENTY)

    def test_pick_regime_t4_tight(self):
        """Test regime selection for T4_TIGHT (8-12GB free)."""
        self.assertEqual(vg.pick_regime(10240), vg.REGIME_T4_TIGHT)
        self.assertEqual(vg.pick_regime(9216), vg.REGIME_T4_TIGHT)
        self.assertEqual(vg.pick_regime(8192), vg.REGIME_T4_TIGHT)

    def test_pick_regime_t4_danger(self):
        """Test regime selection for T4_DANGER (4-8GB free)."""
        self.assertEqual(vg.pick_regime(6144), vg.REGIME_T4_DANGER)
        self.assertEqual(vg.pick_regime(5120), vg.REGIME_T4_DANGER)
        self.assertEqual(vg.pick_regime(4096), vg.REGIME_T4_DANGER)

    def test_pick_regime_t4_critical(self):
        """Test regime selection for T4_CRITICAL (<4GB free)."""
        self.assertEqual(vg.pick_regime(3072), vg.REGIME_T4_CRITICAL)
        self.assertEqual(vg.pick_regime(2048), vg.REGIME_T4_CRITICAL)
        self.assertEqual(vg.pick_regime(0), vg.REGIME_T4_CRITICAL)

    @patch('vram_governor.free_mb')
    def test_resolve_params_t4_plenty(self, mock_free_mb):
        """Test parameter resolution for T4_PLENTY regime."""
        mock_free_mb.return_value = 14336
        
        cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
        params = vg.resolve_params(cfg, gpu_index=0)
        
        self.assertEqual(params["regime"], vg.REGIME_T4_PLENTY)
        self.assertEqual(params["width"], 1280)
        self.assertEqual(params["height"], 704)
        self.assertEqual(params["steps"], 30)
        self.assertEqual(params["seconds"], 6.0)
        self.assertTrue(params["attention_slicing"])
        self.assertTrue(params["vae_tiling"])
        self.assertEqual(params["vae_tile_size"], 512)
        self.assertFalse(params["fp8"])
        self.assertFalse(params["model_offload"])
        self.assertEqual(params["offload_level"], "none")

    @patch('vram_governor.free_mb')
    def test_resolve_params_t4_tight(self, mock_free_mb):
        """Test parameter resolution for T4_TIGHT regime."""
        mock_free_mb.return_value = 10240
        
        cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
        params = vg.resolve_params(cfg, gpu_index=0)
        
        self.assertEqual(params["regime"], vg.REGIME_T4_TIGHT)
        self.assertTrue(params["model_offload"])
        self.assertTrue(params["sequential_offload"])
        self.assertTrue(params["cpu_offload_clip"])
        self.assertEqual(params["vae_tile_size"], 256)

    @patch('vram_governor.free_mb')
    def test_resolve_params_t4_danger(self, mock_free_mb):
        """Test parameter resolution for T4_DANGER regime."""
        mock_free_mb.return_value = 6144
        
        cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
        params = vg.resolve_params(cfg, gpu_index=0)
        
        self.assertEqual(params["regime"], vg.REGIME_T4_DANGER)
        self.assertTrue(params["fp8"])
        self.assertTrue(params["model_offload"])
        self.assertTrue(params["sequential_offload"])
        self.assertTrue(params["cpu_offload_clip"])
        self.assertTrue(params["cpu_offload_vae"])
        self.assertEqual(params["offload_level"], "model")
        self.assertEqual(params["steps"], 18)  # max(8, 30-12)
        self.assertEqual(params["seconds"], 4.0)  # min(6, 4)
        self.assertEqual(params["vae_tile_size"], 128)

    @patch('vram_governor.free_mb')
    def test_resolve_params_t4_critical(self, mock_free_mb):
        """Test parameter resolution for T4_CRITICAL regime."""
        mock_free_mb.return_value = 2048
        
        cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
        params = vg.resolve_params(cfg, gpu_index=0)
        
        self.assertEqual(params["regime"], vg.REGIME_T4_CRITICAL)
        self.assertTrue(params["fp8"])
        self.assertTrue(params["model_offload"])
        self.assertTrue(params["sequential_offload"])
        self.assertTrue(params["cpu_offload_clip"])
        self.assertTrue(params["cpu_offload_vae"])
        self.assertEqual(params["offload_level"], "full")
        self.assertEqual(params["steps"], 10)  # max(4, 30-20)
        self.assertEqual(params["seconds"], 3.0)  # min(6, 3)
        self.assertEqual(params["width"], 768)  # min(1280, 768)
        self.assertEqual(params["height"], 512)  # min(704, 512)
        self.assertEqual(params["vae_tile_size"], 64)

    @patch('vram_governor.free_mb')
    def test_resolve_params_uses_shot_overrides(self, mock_free_mb):
        """Test that shot-specific overrides are respected."""
        mock_free_mb.return_value = 14336
        
        cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
        shot = {"width": 1920, "height": 1080, "steps": 50, "dur": 10}
        
        params = vg.resolve_params(cfg, gpu_index=0, shot=shot)
        
        self.assertEqual(params["width"], 1920)
        self.assertEqual(params["height"], 1080)
        self.assertEqual(params["steps"], 50)
        self.assertEqual(params["seconds"], 6.0)  # Capped by max_clip_seconds

    @patch('vram_governor.free_mb')
    def test_resolve_params_always_enables_attention_slicing(self, mock_free_mb):
        """Test that attention slicing is ALWAYS enabled for T4."""
        for free_mb in [14336, 10240, 6144, 2048]:
            mock_free_mb.return_value = free_mb
            cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
            params = vg.resolve_params(cfg, gpu_index=0)
            self.assertTrue(params["attention_slicing"], f"Failed for free_mb={free_mb}")

    @patch('vram_governor.free_mb')
    def test_resolve_params_always_enables_vae_tiling(self, mock_free_mb):
        """Test that VAE tiling is ALWAYS enabled for T4."""
        for free_mb in [14336, 10240, 6144, 2048]:
            mock_free_mb.return_value = free_mb
            cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
            params = vg.resolve_params(cfg, gpu_index=0)
            self.assertTrue(params["vae_tiling"], f"Failed for free_mb={free_mb}")

    @patch('vram_governor.free_mb')
    @patch('vram_governor.time.sleep')
    @patch('vram_governor.time.time')
    def test_wait_for_vram_returns_when_available(self, mock_time, mock_sleep, mock_free_mb):
        """Test wait_for_vram returns when enough VRAM is available."""
        mock_time.side_effect = [0, 0, 0, 0, 0, 100]
        mock_free_mb.side_effect = [2048, 2048, 2048, 2048, 8192]
        
        result = vg.wait_for_vram(8192, gpu_index=0, timeout_sec=300, poll_sec=5)
        
        self.assertEqual(result, 8192)
        self.assertEqual(mock_free_mb.call_count, 5)

    @patch('vram_governor.free_mb')
    @patch('vram_governor.time.sleep')
    @patch('vram_governor.time.time')
    def test_wait_for_vram_times_out(self, mock_time, mock_sleep, mock_free_mb):
        """Test wait_for_vram returns current free VRAM on timeout."""
        mock_time.side_effect = [0, 5, 10, 15, 20, 301]
        mock_free_mb.return_value = 2048
        
        result = vg.wait_for_vram(8192, gpu_index=0, timeout_sec=300, poll_sec=5)
        
        self.assertEqual(result, 2048)

    def test_safety_margin_mb_default(self):
        """Test safety_margin_mb returns default value."""
        with patch.dict(os.environ, {}, clear=True):
            margin = vg.safety_margin_mb()
            self.assertEqual(margin, 2048)

    def test_safety_margin_mb_from_env(self):
        """Test safety_margin_mb reads from environment."""
        with patch.dict(os.environ, {"KIT_VRAM_SAFETY_MB": "4096"}):
            margin = vg.safety_margin_mb()
            self.assertEqual(margin, 4096)

    def test_regime_constants_defined(self):
        """Test that all regime constants are defined."""
        self.assertEqual(vg.REGIME_T4_PLENTY, "t4_plenty")
        self.assertEqual(vg.REGIME_T4_TIGHT, "t4_tight")
        self.assertEqual(vg.REGIME_T4_DANGER, "t4_danger")
        self.assertEqual(vg.REGIME_T4_CRITICAL, "t4_critical")

    def test_safe_reserve_mb_constant(self):
        """Test SAFE_RESERVE_MB constant."""
        self.assertEqual(vg.SAFE_RESERVE_MB, 2048)


class TestVRAMGovernorIntegration(unittest.TestCase):
    """Integration tests for VRAM Governor with mocked nvidia-smi."""

    @patch('vram_governor.subprocess.run')
    def test_full_probe_and_resolve_flow(self, mock_run):
        """Test full flow from probe to parameter resolution."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, Tesla T4, 16384, 2048, 14336, 45, 10\n1, Tesla T4, 16384, 8192, 8192, 55, 50"
        )
        
        # Probe all GPUs
        gpus = vg.probe_all()
        self.assertEqual(len(gpus), 2)
        
        # Resolve params for GPU 0 (plenty of VRAM)
        cfg = {"width": 1280, "height": 704, "steps": 30, "max_clip_seconds": 6}
        params0 = vg.resolve_params(cfg, gpu_index=0)
        self.assertEqual(params0["regime"], vg.REGIME_T4_PLENTY)
        
        # Resolve params for GPU 1 (tight VRAM)
        params1 = vg.resolve_params(cfg, gpu_index=1)
        self.assertEqual(params1["regime"], vg.REGIME_T4_TIGHT)


if __name__ == "__main__":
    unittest.main()