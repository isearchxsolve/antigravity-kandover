# GODSHAND Wave 2 Integration Report

Date: 2026-08-18  
Scope: CPU-only integration and verification. No CUDA, diffusion, production render, or repository media artifact was invoked or changed.

## Outcome

**CPU integration is complete and green.** `render_motion()` now passes its real module/end-to-end mock acceptance test, and both production notebooks are wired to the Wave-1 fail-closed contracts. The remaining evidence that cannot be generated on this host is explicitly limited to a real Kaggle two-T4 / ComfyUI / WAN execution.

## Phase A — `render_motion()` restoration

`kit/src/render_motion.py` was audited in full. Its repaired contiguous `render_motion()` body contains the render-take loop, low-denoise graph wiring, identity-aware passing-take selection, `_do_postprocess(...)` call (including the NEED_INTERP guard), final output write, manifest write, and thermal snapshots. `_run_with_oom_fallback()` is a module-level peer and retains 64-aligned resolution tiers.

Evidence: `python edge_case_integration_test.py` passed **36/36**. In A7, with `COMFY_MOCK=1` and `POST_MODE=cpu`, the test emitted the postprocess interpolation-skip diagnostic and created `work/e01/final.mp4` in its temporary fixture. No diffusion or GPU inference occurred.

## Notebook integration matrix

| Gap ID | Notebook / cell | Integrated change | CPU verification |
|---|---|---|---|
| A1/A3/A4 | `rainman-wan-MERGED-TUNED-FIXED.ipynb`, main cell | CUDA 12.1 torch bundle is installed/asserted first; ComfyUI and custom nodes clone at lockfile SHAs; lock/schema validator runs; no live pulls. | Main cell AST parses; environment-lock CLI contract passes. |
| C1/C2/C3 | Rainman main cell | Replaced candidate discovery with `assets.lock.json` preflight/download/verify and persists `preflight_result.json`. | Asset preflight fails closed on absent locked assets; Wave-1 hash test passes. |
| H1 | Rainman main cell | WAN is set as the engine and setup errors use `WAN_ASSET_SETUP_FAILED` by default; only explicit `ALLOW_ENGINE_FALLBACK=1` can select the legacy branch. | Engine-policy test and notebook AST pass. |
| B4/B5 | Rainman main cell | Requires the `kaggle_2xt4` two-Tesla-T4 topology; every server/render worker gets `KIT_PHYSICAL_GPU_ID`, `KIT_LOCAL_CUDA_DEVICE=0`, and the Wave-1 control contract. | Worker contract test passes. |
| G1 | Rainman main cell | `require_disk(...)` guards locked asset download, rendering, ESRGAN, and zip creation. | Delivery safety test passes; AST marker present. |
| H5 | Rainman main cell | Verify gate requires staged `preflight_result.json` with `status == "PASS"`. | Notebook AST and contract test pass. |
| E1 | Rainman main cell | Per-shot QC invokes `qc.py --shots-dir ... --report ...`; failures include report path and failed shot IDs. | Exact mock-safe QC command writes report and exits 0 on empty fixture. |
| E4/H2 | Rainman main cell | Stitch uses `check=True`; missing master is fatal; master QC uses `qc.py --master`; stage status is written only after master QC succeeds. | Exact missing-master command exits 2 with `MASTER_MISSING`; no `_qc_master` recursion remains. |
| E2 | `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`, Gate 2/cell 3 | Replaced `selftest --motion` with real-only `gpu_smoke.py` and `COMFY_MOCK=0`. | Exact command with safe mock fixture exits 2 with `MOCK_FORBIDDEN`. |
| C3 | Complete pipeline Gate 3/cell 4 | Replaced the no-op client initialization with locked asset preflight; every required lock entry must be in the PASS result. | Exact safe absent-asset preflight exits 2 with `ASSET_*`. |
| H4 | Complete pipeline Gate 5/cell 6 | Registered log paths are mandatory; missing files raise `LOG_EVIDENCE_MISSING`. | Cell AST parses; fail-closed condition is present. |
| F4 | Complete pipeline Gate 6/cell 7 | Accumulates missing, empty, unprobeable, duration-drift, and shot-count failures before raising. | Cell AST parses; explicit ffprobe/duration/expected-shot contract is present. |
| G3 | Complete pipeline Gates 6 and 8/cells 7 and 9 | Checks free space, uses `ZIP_STORED` for video, writes `.partial`, runs `testzip()`, then atomically renames. | Delivery safety policy test passes; both cells AST parse. |
| E3 | Complete pipeline Gate 9/cell 10 | Iterates `REGIME_T4_PLENTY`, `REGIME_T4_TIGHT`, `REGIME_T4_DANGER`, and `REGIME_T4_CRITICAL`—the governor’s exported constants. | Cell AST parses; stale `vg.VRAM_REGIMES` reference removed. |

Every changed Python cell is JSON-valid and `ast.parse` clean. The added `kit/tests/test_notebook_contracts.py` passed all **5** CPU tests, including the injected-command contracts above.

## Full CPU suite evidence

| Command | Result |
|---|---:|
| `python GODSHAND_MASTER/GODSHAND_MASTER/kit/tests/test_gap_fixes.py -v` | 11/11 passed |
| `python -m unittest ...kit.tests.test_notebook_contracts -v` | 5/5 passed |
| `python test_vram_governor.py` | 24/24 passed |
| `python test_gpu_throttle_oom.py` | 32/32 passed |
| `python comprehensive_smoke_test.py` | 10/10 sections passed |
| `python edge_case_integration_test.py` | 36/36 passed |

Known counted checks total **108** (11 + 5 + 24 + 32 + 36), plus all **10** comprehensive smoke sections. The comprehensive test initially timed out because its intentionally frozen FFmpeg fixture used an unbounded `loop` filter. `comprehensive_smoke_test.py` now supplies `-t <duration>` to bound that test-only fixture; the rerun passed in approximately 140 seconds.

## Residual GPU-unverifiable evidence

- Real Kaggle kernel availability with exactly two Tesla T4 GPUs.
- Locked CUDA 12.1 wheel resolution and the exact locked ComfyUI/custom-node revisions on Kaggle.
- Real ComfyUI `/system_stats` CUDA telemetry, required WAN node schemas, and real SaveImage PNG output for `gpu_smoke.py`.
- WAN weight download authorization/hash verification against Hugging Face, actual WAN graph execution, VRAM/thermal controls, and full-film quality.
- Real stitch/master outputs, media QC of production clips, and disk capacity on the Kaggle volume.

## Kaggle-readiness verdict

**Ready for a fail-closed Kaggle validation run; not yet real-GPU certified.** The notebooks will now stop rather than silently switch engines, skip missing assets/logs/deliverables, accept an unverified master, or publish an unvalidated archive. The first Kaggle execution must supply the residual real-hardware evidence listed above before claiming production rendering success.
