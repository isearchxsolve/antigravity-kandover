# GODSHAND Deep-Dive Gap Analysis — Fresh Kaggle 4K Run

## 1. Executive verdict

**Verdict: NOT ready for a fresh Kaggle 4K shot.** This merged working tree is ready only for a carefully instrumented GPU test after the critical fixes below; it is not ready to be called render-ready.

The single most likely first *real* failure is an incompatible live ComfyUI/PyTorch/CUDA environment assembled at `rainman-wan-MERGED-TUNED-FIXED.ipynb`, cell 1, source lines **4965-4981**: it performs `git pull`, installs an unpinned upstream `requirements.txt`, and installs unpinned support packages without selecting a tested torch CUDA wheel set. The current “GPU smoke” will not expose that problem because it resets `COMFY_MOCK=1` in `kit/src/selftest.py:219-230`.

There are deterministic code defects after that point: the VRAM check reads total VRAM as free (`kit/src/render_motion.py:58-78`), the production QC command passes an unsupported argument (`rainman…ipynb`, cell 1, source line **5900**), and the final-master QC command accidentally runs the whole orchestrator (`…:5916` → `kit/run.py:55-174`). Historical reports describe fixes, but their integration does not match the current interfaces.

No CUDA, diffusion, Kaggle kernel, or video render was attempted in this audit.

### Interpretation of “Verified”

`N` means the current tree does not implement the proposed remediation, or the remediation cannot be proven by CPU-only inspection. It does **not** mean a GPU failure was simulated. A line number in an `.ipynb` is reported as `cell N, source L` (the executable source-line coordinate).

## 2. Gap table

| ID | Severity | File / symbol / line | Gap | Concrete suggested fix | Verified Y/N |
|---|---|---|---|---|---|
| A1 | CRITICAL | `rainman-wan-MERGED-TUNED-FIXED.ipynb`, cell 1:L4965-4981, `sh()` install block | A live `git pull` and `pip install -r requirements.txt` make ComfyUI and its dependency graph change on every fresh run. | Replace `git pull` with `git clone` at a recorded ComfyUI commit SHA; write `requirements-kaggle-cu121.txt` with hashes; fail unless `pip freeze` matches its lock manifest. | N |
| A2 | MAJOR | `kit/requirements.txt:L3-35` | Every kit dependency is a lower bound (`>=`), so a fresh resolver can upgrade NumPy, OpenCV, Pydantic, or FFmpeg bindings past tested API/ABI combinations. | Replace every `>=` with exact pins generated from the known-good Kaggle install; install with `pip install --require-hashes -r requirements-kaggle-cu121.txt`. | N |
| A3 | CRITICAL | `rainman…ipynb`, cell 1:L4965-4981; `kit/requirements.txt` (no torch family entries) | There is no authoritative `torch`/`torchvision`/`torchaudio`/CUDA index selection. An upstream ComfyUI install may retain Kaggle’s torch or replace it with a mismatched wheel; this is the principal fresh-kernel killer. | Install first and assert before any Comfy install: `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1`; record `torch.__version__`, `torch.version.cuda`, and CUDA availability. Do not let later `pip` resolve torch transitively (`--no-deps` where appropriate). | N |
| A4 | CRITICAL | `rainman…ipynb`, cell 1:L4973-4981 | `ComfyUI-GGUF` and VideoHelperSuite are cloned at their moving default branches; the runtime graph relies on their node schemas. | Clone each custom-node repository at a reviewed commit SHA, install its locked requirements, and validate required node schemas (`UnetLoaderGGUF`, `WanImageToVideo`, `SaveImage`) before model loading. | N |
| A5 | MAJOR | `take_0_postprocess_4k.ipynb`, cell 1:L69-75 and L188-194; `upscale_hd_kaggle_4k.ipynb`, cell 1:L52-63/L111-121 | GFPGAN, BasicSR, Real-ESRGAN, and RIFE are installed unpinned at runtime. The notebook patches a private third-party source file instead of selecting compatible releases. | Add a separately pinned 4K requirements lock: `gfpgan==1.3.8`, `basicsr==1.4.2`, `facexlib==0.3.0`, `realesrgan==0.3.0`, the torch bundle above, and a pinned RIFE binary/package checksum. Validate imports before extracting frames. | N |
| B1 | CRITICAL | `kit/src/render_motion.py`, `vram_mb()`/`vram_free_mb()`:L58-78 | `torch.cuda.mem_get_info()` returns `(free,total)`, but `vram_free_mb()` returns index 1. The 500 MB guard therefore sees 16 GB total even when the T4 has no free memory. | Implement `free, total = torch.cuda.mem_get_info(device); return free // MiB, total // MiB`; add a unit test with mocked `(512*MiB, 16384*MiB)` proving `ensure_vram(500)` fails below reserve. | N |
| B2 | CRITICAL | `kit/src/vram_governor.py:L18,L196-197`; `kit/src/render_motion.py:L247,L590-596` | `SAFE_RESERVE_MB=2048` is never applied to a scheduling decision, and 500 MB is both too small and currently measured incorrectly. | Require `free_mb >= estimated_peak_mb + safety_margin_mb()` before enqueue; make the profile provide a per-model peak estimate. Treat unknown telemetry as fatal in real mode, not as the critical fallback regime. | N |
| B3 | CRITICAL | `kit/src/vram_governor.py:L113-182`; `kit/src/render_motion.py:L250-259` | Governor outputs `fp8`, model/sequential/CPU offload, attention slicing, and VAE tiling, but the renderer forwards only resolution, steps, and seconds. The advertised controls do nothing to ComfyUI. | Convert every resolved control into explicit ComfyUI launch flags or graph/node inputs; after `/system_stats`, write the applied regime into `shot_manifest.json`. Fail if a requested node/input is unavailable. | N |
| B4 | MAJOR | `rainman…ipynb`, cell 1:L5519-5528; `kit/config_confrontation.yaml:L104,L128-143` | The selected profile says two GPUs, but `KIT_NUM_GPUS` defaults to `1`, silently discarding T4 #1. | Set `KIT_NUM_GPUS` from the active config profile and assert exactly two visible Tesla T4s for the Kaggle profile; otherwise select a named single-T4 profile and record it. | N |
| B5 | MAJOR | `rainman…ipynb`, cell 1:L5531-5541; `kit/src/render_motion.py:L231,L581-598` | Each worker is restricted with `CUDA_VISIBLE_DEVICES=<physical id>`, but its scheduler reads global `nvidia-smi` indexes and independently picks an index. Telemetry/regime selection can describe a different physical GPU from the process’s CUDA device. | Give each worker immutable `KIT_PHYSICAL_GPU_ID` and `KIT_LOCAL_CUDA_DEVICE=0`; bypass the in-worker multi-GPU scheduler. Query only the assigned physical GPU via NVML/`nvidia-smi -i`. | N |
| B6 | MAJOR | `kit/src/render_motion.py:L372-415` | OOM fallback reduces duration/resolution while retaining the same shot completion semantics, and has a `max(288, …)` height floor that is not 64-aligned. A degraded shot can become the permanent resume artifact. | Round both dimensions with a helper `align64(max(minimum, value))`; write fallback tier/actual dimensions to a manifest; mark degraded outputs `needs_review` and reject them unless an explicit `ALLOW_DEGRADED_SHOTS=1` is set. | N |
| C1 | CRITICAL | `rainman…ipynb`, cell 1:L5001-5006 and L5056-5068 | Model selection is “first currently available candidate” from mutable HF repo listings. A renamed/revoked file or reordered listing changes the model under the same runbook. | Commit `assets.lock.json` containing repo ID, immutable revision SHA, exact filename, expected byte count, SHA-256, destination, and required/optional status. Download using `revision=<sha>`. | N |
| C2 | CRITICAL | `rainman…ipynb`, cell 1:L5069-5109 | HF downloads and LoRA files have no SHA-256 or size validation. Presence of a partial/corrupt/checkpoint-wrong file is accepted. | After every download, require `path.stat().st_size == expected_size` and stream SHA-256 match; atomically rename `.partial` only after both checks. | N |
| C3 | CRITICAL | `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`, cell 4:L26-50 | “Model Downloads” only instantiates `ComfyClient`; `ComfyClient.__init__` does not download anything (`kit/src/comfy_client.py:L18-36`). Gate 3 can pass with no models. | Replace the cell with `python src/preflight_gpu.py --assets assets.lock.json --download --verify`, then require its signed JSON result to list every required asset. | N |
| C4 | MAJOR | `rainman…ipynb`, cell 1:L4989-4998 and L5044-5046 | Models are placed under `/kaggle/temp` and cache/split directories are deleted. Fresh Kaggle sessions do not preserve that disk, while the “resume” scheme restores only `final.mp4`, not an offline model bundle. | Publish a Kaggle dataset containing the verified model tree (or a legally distributable manifest/cache), mount it read-only, and copy/link it without cleanup. Make Internet-off mode an explicit tested profile. | N |
| C5 | MAJOR | `rainman…ipynb`, cell 1:L92-95 and L5089-5105 | `KIT_WAN_STEPS` is set to `40` before LoRA discovery. Later `setdefault()` does not change it to 4 even if 4-step LoRAs are found; a LoRA failure also silently changes the intended model regime. | Compute `steps/cfg` once after asset verification with ordinary assignment; require both LoRAs for the `wan_4step` profile, otherwise fail with `ASSET_LORA_PAIR_MISSING` or select an explicit `wan_no_lora_20step` profile. | N |
| D1 | MAJOR | `kit/src/render_motion.py:L304-306` | Every shot uses the same `42+n` take seeds. There is no base seed in config, shot hash, seed ledger, or link to the exact generated output, so a prior take cannot be reproduced or audited. | Add `base_seed` to config; derive `seed = stable_u64(base_seed + '\0' + shot_id + '\0' + take_index) % 2**32`; persist config hash, graph hash, model-lock hash, seed, and output SHA-256 per take. | N |
| D2 | MAJOR | `kit/src/render_motion.py:L80-90`; `rainman…ipynb`, cell 1:L104-105 | Python/NumPy/Torch RNGs are not all seeded, `torch.backends.cudnn.benchmark=True` enables nondeterministic selection, and no deterministic-algorithm policy is recorded. | In a dedicated reproducibility setup before model creation: seed `random`, NumPy, CPU/CUDA torch; set `cudnn.benchmark=False`, `cudnn.deterministic=True`, and capture whether `torch.use_deterministic_algorithms(True, warn_only=True)` is supported by the workflow. | N |
| D3 | MINOR | `rainman…ipynb`, cell 1:L4965-4981/L5001-5024 | Dynamic git/HF dependencies mean the same seed does not imply the same generation environment. | Include the environment lock, Comfy/custom-node commit SHAs, asset hashes, graph JSON SHA-256, and image/container fingerprint in the reproducibility manifest. | N |
| E1 | CRITICAL | `rainman…ipynb`, cell 1:L5900; `kit/src/qc.py:L337-345` | The production notebook calls `qc.py --shots-dir`, but current QC accepts only `--clips`. Completion deterministically hits argparse failure rather than QC. | Either change the notebook to `qc.py` with `KIT_CONFIG`/`paths.work` set, or add `--shots-dir` to `qc.py` and validate it is a directory. Add a CLI integration test for the exact command. | N |
| E2 | CRITICAL | `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`, cell 3:L26-34; `kit/src/selftest.py:L219-230` | The GPU-smoke gate claims real inference but selftest unconditionally overwrites `COMFY_MOCK=1`. It proves only ffmpeg mock output and can give green before ComfyUI/models exist. | Add a dedicated `src/gpu_smoke.py` mode that refuses `COMFY_MOCK=1`, probes `/system_stats`, loads one locked WAN graph with a 64-aligned tiny fixture, verifies CUDA allocation/model node output, and checks an output PNG signature. | N |
| E3 | MAJOR | `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`, cell 10:L30-33; `kit/src/vram_governor.py:L20-24` | Gate 9 references `vg.VRAM_REGIMES`, which does not exist in the merged governor. Theatrical mode crashes after other work. | Export an immutable `VRAM_REGIMES` mapping from the governor, or display the existing `REGIME_T4_*` constants. Add `python -c "import vram_governor; ..."` to quick checks. | N |
| E4 | CRITICAL | `rainman…ipynb`, cell 1:L5911-5919; `kit/run.py:L55-174` | The notebook launches `python run.py _qc_master <master>`. `run.py` has no CLI dispatch, so it enters `main()` and may restart preflight/forge/cost-meter/full rendering recursively instead of inspecting the master. | Move `_qc_master` into `src/qc.py` as `qc.py --master PATH` and invoke it; alternatively add an `argparse` subcommand to `run.py` that exits after the probe. Test the exact subprocess call. | N |
| E5 | MAJOR | `add_qc_gate.py:L5,L20`; `add_validator_to_notebook.py:L5,L15-19`; production notebook cell 1:L5847-5848 | Patch helpers mutate a specific notebook and report text success, but do not assert that replacement occurred; integration validator exists in the production notebook, while QC integration has drifted. | Convert patch scripts to idempotent build steps that require `replace_count == 1`, then parse the output notebook and execute a CLI contract test against every injected command. | N |
| E6 | MAJOR | `test_vram_governor.py:L21-298`; `test_gpu_throttle_oom.py:L62-265` | The new tests mock SMI and search source text; they do not assert GPU-index binding, `mem_get_info` semantics, Comfy graph control application, or real smoke preconditions. | Keep CPU tests, but add hardware-contract tests which mock both CUDA and SMI tuple semantics and a Kaggle-only smoke result schema. Require that schema before full render. | N |
| F1 | CRITICAL | `rainman…ipynb`, cell 1:L5900-5902; `kit/src/qc.py:L239-376` | The complete-shot QC gate is presently unusable because of the unsupported CLI (E1); therefore its purported hard gate is absent. | Implement the E1 CLI fix and make the notebook raise with the QC report path and shot IDs on nonzero exit. | N |
| F2 | MAJOR | `kit/src/qc.py:L253-290` | Low resolution, duration/fps mismatch, freeze, blur, flicker, loudness, silence, and low color variance are only WARN. A visibly broken shot may still pass the pipeline. | Add a profile-specific severity policy: make resolution/fps/duration and sustained freeze hard FAIL; make sharpness/flicker/loudness thresholds configurable per shot and require an explicit waiver manifest for WARN-only delivery. | N |
| F3 | CRITICAL | `kit/run.py:L166-174` | Missing final master is printed as “skipping final gate,” followed by `DONE`; no nonzero exit prevents shipment. | Replace lines 167-174 with `if not exists: raise SystemExit("FINAL_MASTER_MISSING")`; then invoke `qc.py --master` and require video, audio policy, duration, expected dimensions/fps/pix_fmt, decode, and checksum. | N |
| F4 | MAJOR | `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`, cell 7:L23-48 | Deliverables check merely prints missing clip files and suppresses ffprobe exceptions; it always prints Gate 6 passed. | Accumulate missing/unprobeable clips and raise `RuntimeError` with names; require expected shot count, nonzero sizes, ffprobe exit 0, duration tolerance, and manifest hash match. | N |
| F5 | MAJOR | `kit/src/qc.py:L214-335`; `ONLINE_KAGGLE_OFFLINE_4K_RUNBOOK.md:L355-358` | QC has no reference-frame/identity continuity, prompt/render provenance, scene semantic, watermark/text, audio intelligibility, or model-output signature test. Existing CV metrics cannot prove artistic or semantic correctness. | Add optional per-shot reference keyframes/identity embeddings and CLIP/prompt scores with calibrated thresholds; add OCR/watermark checks; retain mandatory human review as a formal signed gate. | N |
| G1 | MAJOR | `rainman…ipynb`, cell 1:L7-16 and L5745-5749 | Disk checks only print warnings and continue below 3–5 GB. They do not estimate the massive PNG/ProRes/zip peak. | Implement `require_disk(stage, estimated_bytes, reserve_bytes)` that raises before extraction/ESRGAN/ProRes/zip; estimate `frames × width × height × 3 × working_copies × 1.25` plus output sizes. | N |
| G2 | MAJOR | `rainman…ipynb`, cell 1:L18-67 and L5940-5948 | Cleanup deletes HF/Comfy caches and raw outputs while swallowing exceptions. It can invalidate a resume/download state and hides permission or disk errors. | Use per-run directories plus a manifest of reclaimable files; delete only files whose SHA-256 was copied into durable output; fail/report on unexpected cleanup errors. Never delete model assets selected by `assets.lock.json`. | N |
| G3 | MAJOR | `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`, cell 9:L86-97; cell 7:L50-63 | Zip packaging starts with no free-space check and uses deflate on already compressed H.264/ProRes. A large 4K master can fill working disk after successful rendering. | Calculate total input bytes and require `free >= inputs + 10% + reserve`; use `ZIP_STORED` for `.mp4/.mov`, write to `.partial`, verify `ZipFile.testzip()`, then atomic rename. | N |
| G4 | MAJOR | `take_0_postprocess_4k.ipynb`, cell 1:L22-29; complete notebook cell 8:L29-34 | Automatic input selection takes the first recursive MP4/newest zip, not an explicitly declared package with manifest. It can upscale an unrelated artifact. | Require `INPUT_MANIFEST`/explicit path; reject more than one candidate; validate the selected clip IDs and checksums against a chunk manifest before processing. | N |
| G5 | MAJOR | `take_0_postprocess_4k.ipynb`, cell 1:L292-312 | Both final encodes unconditionally map `1:a`; an input without audio fails. They also omit `-shortest`, so generated video/audio duration can diverge. | Detect audio with ffprobe; conditionally add `-map 1:a?` and audio codec; use `-shortest` after validated retiming; run a final ffprobe audio/video duration tolerance check. | N |
| G6 | MAJOR | `take_0_postprocess_4k.ipynb`, cell 1:L301-312; complete notebook cell 8:L201-212 | The files are described as “12-bit ProRes 4444 XQ” but encoded as `yuv444p10le` (10-bit). Review H.264 is 8-bit `yuv420p`. No encoder/pix_fmt capability preflight exists. | Correct all labels to 10-bit 4:4:4, or select an FFmpeg-supported 12-bit input/pixel format only after `ffmpeg -h encoder=prores_ks` and ffprobe verification. Declare review/master codec, pix_fmt, color tags, and audio policy in a delivery manifest. | N |
| H1 | CRITICAL | `rainman…ipynb`, cell 1:L5109-5112 | Any Wan asset failure silently switches to LTXV and continues. A successful output can be from the wrong film engine. | Make engine fallback opt-in (`ALLOW_ENGINE_FALLBACK=1`); default to `raise RuntimeError("WAN_ASSET_SETUP_FAILED")`; write `engine=wan` into each manifest and assert it before export. | N |
| H2 | CRITICAL | `rainman…ipynb`, cell 1:L5904-5920 | `stitch.py` return code is checked only to decide whether to copy files; a stitch failure does not raise, and master QC failure is logged as WARN. The script continues to status/zip handling. | Invoke with `check=True` or test `returncode` and raise; make master QC fatal; write a machine-readable `stage_status.json` after every successful stage only. | N |
| H3 | MAJOR | `kit/src/vram_governor.py:L27-67`; `kit/src/thermal_monitor.py:L34-39`; `kit/src/gpu_scheduler.py:L86-89` | SMI/telemetry exceptions are swallowed and cooldown timeouts proceed. In real GPU mode, missing/invalid telemetry converts into permissive behavior rather than a precise abort. | Add `strict=True` for GPU mode: failure to sample expected GPU raises `GPU_TELEMETRY_UNAVAILABLE`; cooldown timeout raises `GPU_THERMAL_TIMEOUT`; preserve stderr/command result in diagnostic JSON. | N |
| H4 | MAJOR | `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`, cell 6:L53-62 | The log scan passes when no expected log files exist. It does not prove the actual server/shard logs were collected. | Require log paths registered when each Comfy/render process starts; failure to find every required log is `LOG_EVIDENCE_MISSING`, not pass. Scan incrementally after each shot, not only after a 2–3 hour run. | N |
| H5 | MAJOR | `rainman…ipynb`, cell 1:L5769-5804 | `_verify_or_die()` validates a few graph strings and paths but not version lock, asset hash/size, CUDA build, runtime node schema, output type, seed manifest, disk budget, or actual GPU availability. | Replace it with the staged `preflight_gpu.py` harness specified below; write `preflight_result.json` and require `status == "PASS"` before starting a Comfy server or render worker. | N |

## 3. Ranked attack order — fix before the next Kaggle run

The ranking uses likelihood of blocking/misleading a fresh GPU run multiplied by cost-to-fix. “Cost” here means engineering effort, not Kaggle time.

1. **E2 — make GPU smoke genuinely real.** Stop selftest from overriding `COMFY_MOCK`; require a server, CUDA, a locked WAN model, and a real output signature. Without this, every later gate can be falsely green.
2. **A1/A3/A4 — create one pinned Kaggle CUDA environment.** Commit the Comfy/custom-node SHAs and a tested `cu121` torch bundle before any model or node install.
3. **B1/B2 — correct free-VRAM accounting and enforce the 2 GB reserve.** The current check cannot prevent an OOM.
4. **C1/C2/C3 — lock, download, and verify model assets.** Replace “client initialized” and dynamic candidate discovery with immutable asset records and hashes.
5. **E1/F1 — repair the QC CLI contract.** Update `qc.py` or notebook invocation, then test the exact subprocess command.
6. **E4 — replace the recursive final-master command.** Implement `qc.py --master`; never invoke `run.py _qc_master …`.
7. **H1 — prohibit silent WAN→LTXV fallback.** A wrong engine must stop the run with an explicit asset/model error.
8. **B3/B4/B5 — bind T4 behavior to actual worker processes.** Apply the governor output to Comfy, select two GPUs from config, and eliminate physical/local GPU-index ambiguity.
9. **H2/F3/F4 — make output/stitch/master evidence fatal.** Missing master, failed stitch, missing clips, and absent logs must raise rather than print a success banner.
10. **G1/G3/G5/G6 — make the 4K path capacity- and codec-safe.** Budget disk before PNG/ProRes/zip work; harden optional audio; correct 10-bit/12-bit claims and verify actual ffprobe fields.

## 4. Minimal fail-fast harness specification

Add a new, standalone `kit/src/preflight_gpu.py`. It must run **after** the locked environment and assets are present but **before** launching ComfyUI or dispatching a shot. It is a preflight/instrumentation program, not a renderer.

Its interface should be:

```text
python src/preflight_gpu.py \
  --profile kaggle_2xt4_zero_oom \
  --assets assets.lock.json \
  --config config_confrontation.yaml \
  --work /kaggle/working \
  --result /kaggle/working/preflight_result.json
```

Required behavior:

1. Exit nonzero at the first failed stage, with a stable error code such as `ENV_TORCH_CUDA_MISMATCH`, `GPU_T4_COUNT`, `VRAM_HEADROOM`, `ASSET_SHA256`, `GRAPH_SCHEMA`, `DISK_BUDGET`, or `SEED_POLICY`.
2. Emit a small JSON record containing only diagnostic metadata — never HF tokens or secret values.
3. Refuse `COMFY_MOCK=1`, an unknown engine, missing logs/manifests, or an unpinned dependency fingerprint when running under the Kaggle profile.
4. Start the real smoke only after this harness reports PASS. The smoke must be one 64-aligned WAN prompt, one locked asset set, one assigned GPU, and an output PNG signature/ffprobe check; it must not use the mock client.

Illustrative implementation skeleton:

```python
#!/usr/bin/env python3
"""Fail-fast Kaggle GPU preflight. No diffusion call occurs here."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

MiB = 1024 * 1024
T4_TOTAL_MIN_MB = 15_000
T4_RESERVE_MB = 2_048

def die(code, detail, result):
    result.update(status="FAIL", code=code, detail=detail)
    print(f"[FAIL:{code}] {detail}", file=sys.stderr)
    raise SystemExit(2)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * MiB), b""):
            h.update(block)
    return h.hexdigest()

def run_text(cmd, code, result):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        die(code, p.stderr[-1200:] or p.stdout[-1200:], result)
    return p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    a = ap.parse_args()
    result = {"status": "STARTED", "profile": a.profile, "stages": []}

    if os.environ.get("COMFY_MOCK") == "1":
        die("MOCK_FORBIDDEN", "Kaggle GPU preflight refuses COMFY_MOCK=1", result)
    if os.environ.get("KIT_ENGINE") not in (None, "wan"):
        die("ENGINE_NOT_WAN", f"KIT_ENGINE={os.environ['KIT_ENGINE']!r}", result)

    import torch
    expected = {"torch": "2.4.1", "cuda": "12.1"}
    if not torch.cuda.is_available():
        die("CUDA_UNAVAILABLE", f"torch={torch.__version__}, cuda={torch.version.cuda}", result)
    if not torch.__version__.startswith(expected["torch"]) or torch.version.cuda != expected["cuda"]:
        die("ENV_TORCH_CUDA_MISMATCH", f"got torch={torch.__version__} cuda={torch.version.cuda}", result)
    result["stages"].append("torch_cuda")

    smi_csv = run_text([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits"
    ], "GPU_QUERY", result)
    gpus = [line.split(",") for line in smi_csv.splitlines() if line.strip()]
    if len(gpus) != 2 or any("T4" not in row[1] or int(row[2]) < T4_TOTAL_MIN_MB
                             or int(row[3]) < T4_RESERVE_MB for row in gpus):
        die("GPU_T4_COUNT_OR_HEADROOM", smi_csv, result)
    # Require exactly two Tesla T4 devices and each has >= 2GB reserve.
    # Also check CUDA_VISIBLE_DEVICES maps exactly one requested worker physical GPU.

    lock = json.loads(a.assets.read_text(encoding="utf-8"))
    for asset in lock["assets"]:
        p = Path(asset["path"])
        if not p.is_file():
            die("ASSET_MISSING", asset["path"], result)
        if p.stat().st_size != asset["bytes"]:
            die("ASSET_SIZE", f"{p}: {p.stat().st_size} != {asset['bytes']}", result)
        if sha256(p) != asset["sha256"]:
            die("ASSET_SHA256", str(p), result)
    result["stages"].append("assets")

    # Read YAML config; require WAN graph, 64-aligned requested W/H, model peak
    # budget + reserve <= free VRAM of the assigned T4, and a positive disk budget.
    # Verify every required Comfy node through /object_info only AFTER server launch.
    # Create a deterministic manifest plan; no image/video generation in this program.
    base_seed = 20260817
    first_shot = "smoke_001"
    seed = int.from_bytes(hashlib.sha256(f"{base_seed}:{first_shot}:0".encode()).digest()[:4], "big")
    result["seed_plan"] = {"base_seed": base_seed, "smoke_001_take_0": seed}

    if shutil.disk_usage(a.work).free < lock["disk_budget_bytes"]:
        die("DISK_BUDGET", "insufficient working disk for worst-case PNG/ProRes/zip peak", result)
    result.update(status="PASS", code="OK")
    a.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

The production notebook should then use stages with an explicit success artifact:

```text
environment lock install
  -> preflight_gpu.py (writes preflight_result.json)
  -> launch pinned ComfyUI per assigned physical T4
  -> graph schema/version check
  -> real WAN smoke (writes gpu_smoke_result.json; COMFY_MOCK must be false)
  -> full render, per-shot manifest/checkpoint after every accepted shot
  -> fatal per-shot QC
  -> fatal stitch + qc.py --master
  -> 4K disk preflight + fatal ffprobe delivery verification + archive verification
```

The harness intentionally does not claim to predict a model’s exact peak VRAM. The locked profile must supply an empirically measured minimum free-VRAM threshold for its precise model, graph, resolution, steps, and batch size; the first real smoke validates that threshold before the 435-shot run.

## 5. Honest constraints

This was a CPU-only static analysis. I did **not** run CUDA, `nvidia-smi`, ComfyUI, Wan, any Hugging Face download, a Kaggle kernel, GPU smoke, diffusion forward pass, or 4K postprocess. I did not touch any `.mp4`, rebuild an `.ipynb`, remove evidence, or read/print a secret.

Consequently, this report cannot verify actual driver compatibility, torch/xformers wheel availability on the next Kaggle image, model access permissions, HF bandwidth, node runtime schemas, real VRAM peak/fragmentation, temperature behavior, diffusion output correctness, temporal quality, identity preservation, codec availability, 4K disk peak, or artistic taste. Those are exactly the jobs of the pinned preflight, a genuine one-shot GPU smoke, staged manifests, automated QC, and human review.

The old CPU suites demonstrate selected source and mock/ffmpeg paths only. They are valuable regression tests, but they do not prove that this unified merged tree has rendered on a real Kaggle T4.
