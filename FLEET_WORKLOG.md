# GODSHAND Gap-Fix Worklog

### [step 1] Read the source-of-truth gap analysis
- Read `GODSHAND_DEEP_DIVE_GAP_ANALYSIS.md` in full before making any code changes; its ranked attack order is the implementation sequence.
- Evidence: report identifies E2, A1/A3/A4, B1/B2, C1/C2/C3, E1/F1, E4, H1, B3/B4/B5, H2/F3/F4, and G1/G3/G5/G6 as the required top-ten work items.
- Next: inventory the existing kit source, notebooks, and CPU test surface without running CUDA, diffusion, or rendering.

### [step 2] E2 real GPU-smoke gate and mock refusal
- Added `kit/src/gpu_smoke.py` with a hard `COMFY_MOCK=1` refusal, real-only `/system_stats` and `/object_info` probes, required WAN node-schema validation, and PNG signature validation; retained `selftest.py` as explicitly mock-only.
- Evidence: `TestE2GpuSmoke.test_gpu_smoke_refuses_mock_mode` passed (`python -m unittest ...test_gap_fixes.py -v`): subprocess exited 2 with `MOCK_FORBIDDEN` and produced parseable JSONL evidence.
- GPU-unverifiable residue: the real server, CUDA allocation, locked WAN graph execution, and SaveImage output can only be exercised on the intended Kaggle T4 environment.
- Next: pin the Kaggle CUDA/Comfy/custom-node environment (A1/A3/A4).

### [step 3] A1/A3/A4 immutable Kaggle environment manifest
- Added `kit/requirements-kaggle-cu121.txt`, `kit/kaggle-environment.lock.json`, and `src/kaggle_environment.py`; the lock records exact CUDA 12.1 torch-family pins, the current recorded immutable ComfyUI/custom-node SHAs, date/rationale, and the three required node schemas.
- Evidence: `TestA1A3A4EnvironmentLock.test_kaggle_environment_lock_is_immutable_and_schema_complete` passed; it validates exact pins/SHA format and proves a missing node schema raises `GRAPH_SCHEMA_MISSING`.
- Remaining: the source production notebook still requires a serialized cell replacement to call the new validator/install sequence; real wheel availability and node runtime schemas remain Kaggle-only verification.
- Next: correct VRAM tuple semantics and apply the 2GB reserve.

### [step 4] B1/B2 free VRAM and mandatory reserve
- Corrected `render_motion.vram_free_mb()` to read `(free,total)` index 0, added `vram_total_mb()`, and made `ensure_vram()` require `estimated_peak_mb + SAFE_RESERVE_MB`; `vram_governor.require_headroom()` now fails unknown telemetry in real mode and records the decision.
- Evidence: `TestB1B2VramReserve.test_mem_get_info_uses_free_and_reserve_blocks_enqueue` passed with mocked `(512 MiB, 16384 MiB)`: read free=512, total=16384, and rejected a 500MB estimate because 2,548MB is required.
- Next: add immutable asset lock/download/verify support and CPU hash-mismatch coverage.

### [step 5] C1/C2/C3 asset lock, verifier, and Gate-3 preflight helper
- Added `kit/assets.lock.json` (WAN high/low experts, UMT5, VAE, paired LoRAs, and RealESRGAN) with immutable revisions, exact bytes/SHA-256/destinations; added `src/assets.py` atomic `.partial` downloader and streaming verifier plus `src/preflight_gpu.py --assets --download --verify`.
- Evidence: `TestC1C2C3Assets.test_lock_schema_and_hash_mismatch_fail` passed; malformed content produced `ASSET_SHA256_MISMATCH:models/x.bin` and structured expected-vs-actual evidence.
- Next: repair the QC subprocess contract. Initial attempt to create a synthetic video fixture failed because `ffmpeg` is absent on this Windows CPU host: `FileNotFoundError: [WinError 2] The system cannot find the file specified`; the revised integration test uses the exact CLI against an empty mock shots directory and is re-running now.

### [step 6] E1/F1 `qc.py --shots-dir` contract
- Added `--shots-dir`, `--report`, and `--verbose` to `src/qc.py`; it validates the directory, writes a JSON QC report, emits structured per-check evidence, and reports the report path plus failed shot IDs on exit. Added focused `--master` dispatch needed by E4 without invoking `run.py`.
- Evidence: `TestE1F1QcCli.test_notebook_shots_dir_command_writes_report` and `test_master_cli_missing_file_is_probe_only_and_fatal` passed using the notebook's exact `python src/qc.py --shots-dir ...` shape and a master probe-only failure.
- Host limitation: ffmpeg is not installed, so media-decode branches are covered through existing mock/unit-level checks only; no render attempt was made.
- Next: wire the final-master contract into the orchestrator and prohibit fallback.

### [step 7] E4 master-QC dispatch and H1 WAN engine policy
- Changed `kit/run.py` to invoke `qc.py --master` instead of re-entering the orchestrator; added `src/engine_policy.py` to make WAN setup failure fatal by default and require `engine=wan` in `shot_manifest.json`.
- Initial regression run caught a real implementation bug: `TypeError: mock_logging.log() got multiple values for keyword argument 'shot_id'` in `engine_policy.write_shot_manifest`; corrected the duplicate context field before re-running.
- Next: re-run the H1/E4 regression test, then bind the governor controls to workers.

### [step 8] B3/B4/B5 worker contract, H2/F3 fail-fast, and G safety helpers
- Added `src/worker_contract.py` for Tesla-T4 topology and immutable physical-to-local GPU binding, recorded resolved controls in the shot manifest, and made `run.py` write `stage_status.json` only after a successful stage and raise `FINAL_MASTER_MISSING`.
- Added `src/delivery_safety.py` with capacity math, reserve guard, MP4/MOV stored-zip policy, atomic archive construction, and conditional-audio/`-shortest` encoding arguments; all are covered by the focused CPU suite.
- Evidence: focused suite passed 11/11 (`python -m unittest ...test_gap_fixes.py -v`); legacy `test_gpu_throttle_oom.py` initially failed due `GPU_TELEMETRY_UNAVAILABLE` on its CPU mock, then passed 32/32 after strict telemetry was scoped to explicit real GPU mode (`COMFY_MOCK=0`/`KIT_REQUIRE_GPU=1`).
- Next: run the available aggregate CPU suite and document blockers honestly.

### [step 9] Aggregate CPU-suite result and stop condition
- `run_all_suites.py --skip gpu_smoke` with `GODSHAND_KIT_ROOT` set to this workspace produced: 2 pass (`cinema4k`, `selftest_quick`), 3 fail (`edge_case`, `comprehensive`, initially `gpu_throttle_oom`), 1 explicit GPU skip. Re-running `test_gpu_throttle_oom.py` after the telemetry fix passed 32/32.
- Actual remaining failure output: `FileNotFoundError: [WinError 2] The system cannot find the file specified` for `ffmpeg` in both `edge_case_integration_test.py` and `comprehensive_smoke_test.py`; no rendering/diffusion/GPU test was attempted.
- Stop condition: this host cannot complete the requested green full media suite, and the minified production notebooks still require their requested cell-level replacements. `GODSHAND_GAP_FIX_REPORT.md` records the implemented work and remaining unclosed requirements without claiming completion.

### [step 10] Delivered honest implementation report
- Wrote `GODSHAND_GAP_FIX_REPORT.md` with per-gap files/tests/residue, structured logging coverage, test inventory/counts, readiness verdict, and explicit CPU-unverifiable/incomplete work.
- Evidence: `python -m py_compile` passed for every added/modified safety module; the focused gap suite remains green at 11/11 and legacy throttle/VRAM suite at 32/32.
- Handoff blocker remains unchanged: no `ffmpeg` executable for the existing media suites, and production notebook cells have not been integrated with the new helpers.

### [step 11] Wave 2 integration started — Phase A source audit
- Read `kit/src/render_motion.py` in full before editing. The previously orphaned render-take loop, low-denoise graph wiring, identity-aware take selection, `_do_postprocess(...)` call, final output path, thermal snapshot, and manifest write are presently contiguous inside `render_motion()` (lines 209–387); `_run_with_oom_fallback()` is a module-level peer (lines 390–442) and keeps 64-aligned fallback dimensions.
- Next: execute the CPU/mock edge-case integration suite to establish the A7 baseline with the newly available ffmpeg tools; no GPU or diffusion invocation is authorized.

### [step 12] Phase A accepted — `render_motion()` runtime restoration verified
- No additional source edit was necessary: the current `render_motion()` already has the restored contiguous body and module-level OOM wrapper found in step 11.
- Evidence: `GODSHAND_KIT_ROOT=...\\kit python edge_case_integration_test.py` exited 0: **36 passed, 0 failed**. A1–A6 structural checks passed and A7 ran with `COMFY_MOCK=1`, `POST_MODE=cpu`, emitted the postprocess NEED_INTERP-skip diagnostic, and created the temporary fixture `work/e01/final.mp4`.
- Next: atomically replace and validate every listed fail-closed block in `rainman-wan-MERGED-TUNED-FIXED.ipynb` before opening the second production notebook.

### [step 13] B1 complete — `rainman-wan-MERGED-TUNED-FIXED.ipynb` atomically rewired
- Replaced the live-pull/unpinned setup with the CUDA 12.1 lock-first install, locked repository checkout/revision assertions, and node-lock validator (`wave2:A1/A3/A4`). Replaced mutable WAN/LTX asset candidate discovery with the locked atomic asset preflight and `preflight_result.json`; default WAN setup failure now raises via `wan_setup_failed` (`wave2:C1/C2/C3`, `wave2:H1`).
- Added the two-Tesla-T4 worker topology assertion and immutable physical-to-local (`0`) bindings, fail-closed disk guards, staged preflight verification, QC JSON-report diagnostics, probe-only master QC, fatal stitch/missing-master handling, and success-only `stage_status.json` (`wave2:B4/B5`, `G1`, `H5`, `E1`, `E4`, `H2`). Corrected two pre-existing malformed `SaveImage` dictionary lines so the whole cell is parseable.
- Evidence: notebook JSON parses and `ast.parse` passes for its only code cell; no live `git pull`, `list_repo_files`, or `run.py _qc_master` remains. The temporary formatting attempt initially failed with `IOException` because it read and wrote the same notebook path; it was retried through a validated temporary file before patching.
- Next: atomically complete B2 and execute the injected-command CPU contracts.

### [step 14] B2 complete — complete gated notebook rewired and CPU contract coverage added
- Rewired Gate 2 to `gpu_smoke.py` real-only mode (`wave2:E2`), Gate 3 to locked asset preflight requiring every required asset (`wave2:C3`), Gate 5 to fail closed for missing registered logs (`wave2:H4`), Gate 6 to aggregate malformed/missing/unprobeable expected clips and atomically package (`wave2:F4/G3`), Gate 8 to stored-video `.partial`/`testzip()`/atomic archive delivery (`wave2:G3`), and Gate 9 to iterate the governor's actual exported regime constants (`wave2:E3`).
- Added `kit/tests/test_notebook_contracts.py`. Evidence: `python -m unittest GODSHAND_MASTER.GODSHAND_MASTER.kit.tests.test_notebook_contracts -v` exited 0 (**5 tests**): both notebooks JSON-valid and all changed Python cells AST-parse; environment lock, real-only GPU-smoke mock refusal, asset-preflight failure, shots QC report, and fatal master-QC CLI contracts passed.
- Next: run all requested CPU suites, then write the Wave 2 report with exact counts and remaining real-GPU-only verification.

### [step 15] Phase C complete — full CPU suite green and Wave 2 report written
- Focused suites: `test_gap_fixes.py` **11/11**, `test_notebook_contracts.py` **5/5**, `test_vram_governor.py` **24/24**, and `test_gpu_throttle_oom.py` **32/32** all exited 0. `edge_case_integration_test.py` was rerun after all notebook work and exited 0 with **36 passed, 0 failed**, including A7 producing its temporary `work/e01/final.mp4` fixture under COMFY_MOCK.
- `comprehensive_smoke_test.py` initially timed out at 120 seconds and again remained silent after 120 seconds; inspection found its `gen_freeze_video()` test fixture used `loop=loop=1` without an output duration. Added the intended `-t str(duration)` fixture bound. Evidence: rerun completed in ~140 seconds and printed `ALL COMPREHENSIVE SMOKE TESTS PASSED` (10 sections). This is a CPU-only FFmpeg fixture correction, not a production render.
- Wrote `GODSHAND_WAVE2_REPORT.md` with the per-gap integration table, CLI-contract results, A7 evidence, full-suite counts, residual real-GPU-only evidence, and the fail-closed Kaggle-readiness verdict.
- Next: hand off for the required real Kaggle 2xT4 validation; no CUDA/diffusion/render was attempted in this integration run.
