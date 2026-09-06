"""CPU-testable capacity, archive, audio-map, and delivery-manifest guards."""
from __future__ import annotations
import os, shutil, zipfile
from pathlib import Path
from mock_logging import log

def estimate_working_bytes(frames, width, height, working_copies=2, output_bytes=0):
    return int(frames * width * height * 3 * working_copies * 1.25 + output_bytes)

def require_disk(stage, estimated_bytes, reserve_bytes, path=".", free_bytes=None):
    free = shutil.disk_usage(path).free if free_bytes is None else free_bytes
    required = int(estimated_bytes) + int(reserve_bytes)
    log("INFO" if free >= required else "ERROR", "disk_budget", stage=stage, gate="G1/G3", free_bytes=free, required_bytes=required)
    if free < required: raise RuntimeError(f"DISK_SHORTFALL:{stage}")

def zip_policy(path: Path): return zipfile.ZIP_STORED if path.suffix.lower() in {".mp4", ".mov"} else zipfile.ZIP_DEFLATED

def audio_encode_args(has_audio: bool):
    return ["-map", "0:v:0"] + (["-map", "1:a?", "-c:a", "aac"] if has_audio else []) + ["-shortest"]

def atomic_zip(destination: Path, inputs: list[Path], reserve_bytes=0, free_bytes=None):
    total = sum(p.stat().st_size for p in inputs)
    require_disk("zip", int(total * 1.1), reserve_bytes, destination.parent, free_bytes)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with zipfile.ZipFile(partial, "w") as archive:
        for item in inputs: archive.write(item, item.name, compress_type=zip_policy(item))
        if archive.testzip(): raise RuntimeError("ZIP_CORRUPT")
    os.replace(partial, destination)
    return destination
