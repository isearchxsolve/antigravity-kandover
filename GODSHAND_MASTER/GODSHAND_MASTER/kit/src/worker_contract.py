"""GPU worker binding and explicit Comfy control contract."""
from __future__ import annotations
import os
from mock_logging import log

def validate_kaggle_t4s(gpus, profile: str) -> int:
    expected = 2 if profile.startswith("kaggle_2xt4") else 1
    if len(gpus) != expected or (expected == 2 and any("T4" not in g.get("name", "") for g in gpus)):
        raise RuntimeError("GPU_T4_COUNT")
    return expected

def worker_env(physical_gpu_id: int, controls: dict) -> dict:
    env = {"CUDA_VISIBLE_DEVICES": str(physical_gpu_id), "KIT_PHYSICAL_GPU_ID": str(physical_gpu_id),
           "KIT_LOCAL_CUDA_DEVICE": "0", "KIT_NUM_GPUS": "1",
           "COMFY_ARGS": " ".join(flag for enabled, flag in ((controls.get("fp8"), "--fp8_e4m3fn-unet"),
                         (controls.get("model_offload"), "--lowvram"),
                         (controls.get("sequential_offload"), "--cpu")) if enabled)}
    log("INFO", "worker_binding", stage="worker", gate="B3/B4/B5", physical_gpu_id=physical_gpu_id,
        local_cuda_device=0, controls=controls, comfy_args=env["COMFY_ARGS"])
    return env
