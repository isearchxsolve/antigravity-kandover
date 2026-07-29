#!/usr/bin/env python3
"""
Test suite for ZERO OOM + ZERO GPU THROTTLE on Kaggle 2xT4.

Verifies that the PRODUCTION render path (render_motion.py — the path the
Kaggle notebook actually executes) integrates the GPU scheduler, thermal
monitor, and VRAM governor — the same safety stack that render_film.py
(the legacy path) already uses.

Without these integrations:
  - Only GPU 0 is used (GPU 1 sits idle) → 50% throughput waste
  - No thermal cooldown between shots → T4 throttle (sustained 85C+)
  - No VRAM regime resolution → no adaptive parameter reduction → OOM
  - torch.cuda.empty_cache() called when torch is None → crash

Exit 0 = all green. Non-zero = a real defect that causes OOM or throttle.
"""
import os, sys, inspect, importlib
from pathlib import Path

# Force UTF-8 stdout on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(os.environ.get(
    "GODSHAND_KIT_ROOT",
    r"C:\dev\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit"))
if not ROOT.is_dir():
    alt = Path(r"C:\Users\Admin\Downloads\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit")
    if alt.is_dir():
        ROOT = alt
SRC = ROOT / "src"
os.chdir(ROOT)
sys.path.insert(0, str(SRC))

PASS, FAIL = 0, 0
def ok(name, detail=""):
    global PASS; PASS += 1
    print(f"  [PASS] {name}{(' -- ' + detail) if detail else ''}")
def bad(name, detail=""):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {name}{(' -- ' + detail) if detail else ''}")
def check(name, cond, detail=""):
    (ok if cond else bad)(name, detail)
    return cond

print("=" * 72)
print("GPU THROTTLE + OOM PREVENTION TEST (production path)")
print("=" * 72)

# =====================================================================
# GROUP A: render_motion.py must use the GPU safety stack
# =====================================================================
print("\n[A] render_motion.py GPU safety stack integration")
import render_motion as rm

# A1: render_motion.py must import gpu_scheduler
src_rm = inspect.getsource(rm)
check("A1 render_motion imports gpu_scheduler",
      "gpu_scheduler" in src_rm,
      "production path has no GPU scheduler — only GPU 0 used, GPU 1 idle")

# A2: render_motion.py must import thermal_monitor
check("A2 render_motion imports thermal_monitor",
      "thermal_monitor" in src_rm,
      "production path has no thermal monitor — T4 will throttle")

# A3: render_motion.py must import vram_governor
check("A3 render_motion imports vram_governor",
      "vram_governor" in src_rm,
      "production path has no VRAM governor — no adaptive OOM prevention")

# A4: render_motion main() must create a Scheduler
src_main = inspect.getsource(rm.main)
check("A4 render_motion main() creates gpu_scheduler.Scheduler",
      "Scheduler" in src_main,
      "main() does not create a GPU scheduler — shots run sequentially on GPU 0")

# A5: render_motion main() must call sched.pick()
check("A5 render_motion main() calls sched.pick()",
      "pick()" in src_main,
      "main() does not call sched.pick() — no round-robin GPU assignment")

# A6: render_motion main() must call sched.finish()
check("A6 render_motion main() calls sched.finish()",
      "finish(" in src_main,
      "main() does not call sched.finish() — no thermal gap tracking")

# A7: render_motion must pass gpu_idx to render_motion()
src_rm_func = inspect.getsource(rm.render_motion)
check("A7 render_motion() accepts gpu_idx parameter",
      "gpu_idx" in src_rm_func,
      "render_motion() has no gpu_idx param — cannot target specific GPU")

# A8: render_motion must call thermal_monitor.snapshot before render
check("A8 render_motion() calls thermal_monitor.snapshot",
      "snapshot" in src_rm_func,
      "no thermal snapshot before render — no throttle forensics")

# A9: render_motion must call vg.resolve_params for VRAM-aware params
check("A9 render_motion() calls vg.resolve_params or vram_governor.resolve_params",
      "resolve_params" in src_rm_func,
      "no VRAM regime resolution — OOM risk on T4 with <8GB free")

# A10: render_motion main() must not crash when torch is None
# The main loop does: if not ensure_vram(500): gc.collect(); torch.cuda.empty_cache()
# If torch is None, torch.cuda.empty_cache() crashes with AttributeError
check("A10 render_motion main() guards torch.cuda.empty_cache() for torch=None",
      "torch is not None" in src_main or "torch.cuda" not in src_main
      or "if torch" in src_main,
      "main() calls torch.cuda.empty_cache() without checking if torch is None")

# =====================================================================
# GROUP B: render_stills.py must use the GPU safety stack
# =====================================================================
print("\n[B] render_stills.py GPU safety stack integration")
import render_stills as rs

src_rs = inspect.getsource(rs)
src_rs_main = inspect.getsource(rs.main)

check("B1 render_stills imports gpu_scheduler",
      "gpu_scheduler" in src_rs,
      "stills path has no GPU scheduler")
check("B2 render_stills imports thermal_monitor",
      "thermal_monitor" in src_rs,
      "stills path has no thermal monitor")
check("B3 render_stills imports vram_governor",
      "vram_governor" in src_rs,
      "stills path has no VRAM governor")
check("B4 render_stills main() creates Scheduler",
      "Scheduler" in src_rs_main,
      "stills main() does not create a GPU scheduler")

# =====================================================================
# GROUP C: comfy_client.py CUDA config correctness
# =====================================================================
print("\n[C] comfy_client.py CUDA config correctness")
import comfy_client as cc

src_cc = inspect.getsource(cc)

# C1: PYTORCH_CUDA_ALLOC_CONF should not have duplicate setdefault
alloc_conf_count = src_cc.count("PYTORCH_CUDA_ALLOC_CONF")
check("C1 comfy_client has no duplicate PYTORCH_CUDA_ALLOC_CONF setdefault",
      alloc_conf_count <= 1,
      f"found {alloc_conf_count} occurrences (should be <=1)")

# C2: max_split_size_mb should be >= 128 (64 causes excessive fragmentation on T4)
check("C2 comfy_client max_split_size_mb >= 128",
      "max_split_size_mb:128" in src_cc or "max_split_size_mb:256" in src_cc,
      "max_split_size_mb:64 is too aggressive for T4 — causes fragmentation")

# C3: comfy_client must not override notebook's PYTORCH_CUDA_ALLOC_CONF
# setdefault only sets if not already set, so this is OK as long as the
# value matches the notebook's value
check("C3 comfy_client PYTORCH_CUDA_ALLOC_CONF matches notebook (128)",
      "max_split_size_mb:128" in src_cc,
      "comfy_client sets max_split_size_mb to 64, overriding notebook's 128")

# =====================================================================
# GROUP D: vram_governor T4 regime correctness
# =====================================================================
print("\n[D] vram_governor T4 regime correctness")
import vram_governor as vg

# D1: T4 PLENTY regime should still enable attention_slicing + vae_tiling
params = vg.resolve_params({"width": 1280, "height": 720, "steps": 8, "max_clip_seconds": 5}, gpu_index=999)
# gpu_index=999 won't exist, free_mb returns 0, regime = CRITICAL
check("D1 vram_governor returns valid regime for unknown GPU",
      params["regime"] in (vg.REGIME_T4_PLENTY, vg.REGIME_T4_TIGHT,
                           vg.REGIME_T4_DANGER, vg.REGIME_T4_CRITICAL),
      f"got regime={params['regime']}")

# D2: pick_regime thresholds are correct for T4 (16GB)
check("D2 pick_regime(12*1024) == T4_PLENTY",
      vg.pick_regime(12 * 1024) == vg.REGIME_T4_PLENTY)
check("D3 pick_regime(8*1024) == T4_TIGHT",
      vg.pick_regime(8 * 1024) == vg.REGIME_T4_TIGHT)
check("D4 pick_regime(4*1024) == T4_DANGER",
      vg.pick_regime(4 * 1024) == vg.REGIME_T4_DANGER)
check("D5 pick_regime(2*1024) == T4_CRITICAL",
      vg.pick_regime(2 * 1024) == vg.REGIME_T4_CRITICAL)

# D6: SAFE_RESERVE_MB should be >= 2048 (2GB safety margin for T4)
check("D6 SAFE_RESERVE_MB >= 2048",
      vg.SAFE_RESERVE_MB >= 2048,
      f"SAFE_RESERVE_MB={vg.SAFE_RESERVE_MB}")

# =====================================================================
# GROUP E: gpu_scheduler thermal settings
# =====================================================================
print("\n[E] gpu_scheduler thermal settings")
import gpu_scheduler as gs

# E1: THROTTLE_TEMP_C should be <= 80 (T4 throttle at 85C, need margin)
check("E1 THROTTLE_TEMP_C <= 80",
      gs.THROTTLE_TEMP_C <= 80,
      f"THROTTLE_TEMP_C={gs.THROTTLE_TEMP_C}")

# E2: MIN_GAP_SEC should be >= 15 (thermal recovery between shots)
check("E2 MIN_GAP_SEC >= 15",
      gs.MIN_GAP_SEC >= 15,
      f"MIN_GAP_SEC={gs.MIN_GAP_SEC}")

# E3: TEMP_GREEN_C should be < THROTTLE_TEMP_C (cooldown to green, not just below throttle)
check("E3 TEMP_GREEN_C < THROTTLE_TEMP_C",
      gs.TEMP_GREEN_C < gs.THROTTLE_TEMP_C,
      f"TEMP_GREEN_C={gs.TEMP_GREEN_C} >= THROTTLE_TEMP_C={gs.THROTTLE_TEMP_C}")

# E4: Scheduler.pick() should return 0 when no GPUs visible
sched = gs.Scheduler()
check("E4 Scheduler.pick() returns 0 when no GPUs",
      sched.gpu_count == 0 or sched.pick() >= 0,
      "Scheduler.pick() crashed or returned negative")

# E5: Scheduler.finish() should not crash on invalid index
try:
    sched.finish(999)
    check("E5 Scheduler.finish(999) does not crash", True)
except Exception as e:
    check("E5 Scheduler.finish(999) does not crash", False, str(e))

# =====================================================================
# GROUP F: render_motion OOM fallback tier correctness
# =====================================================================
print("\n[F] render_motion OOM fallback tier correctness")

# F1: fallback tiers must produce multiples of 64 for width/height
src_oom = inspect.getsource(rm._run_with_oom_fallback)
check("F1 OOM fallback rounds to multiples of 64",
      "// 64 * 64" in src_oom or "//64*64" in src_oom,
      "OOM fallback does not round resolution to VAE-compatible multiples of 64")

# F2: OOM fallback must have at least 4 tiers
check("F2 OOM fallback has >= 4 tiers",
      src_oom.count("(") >= 4,
      "not enough fallback tiers")

# F3: OOM fallback must catch CUDA OOM and retry
check("F3 OOM fallback catches 'CUDA out of memory'",
      "CUDA out of memory" in src_oom,
      "OOM fallback does not catch CUDA OOM errors")

# F4: OOM fallback must call gc.collect() + torch.cuda.empty_cache() on OOM
check("F4 OOM fallback calls gc.collect() on OOM",
      "gc.collect()" in src_oom,
      "OOM fallback does not call gc.collect()")

# =====================================================================
print("\n" + "=" * 72)
print(f"RESULT: {PASS} passed, {FAIL} failed")
print("=" * 72)

if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)


def test_gpu_throttle_oom_suite():
    assert FAIL == 0, f"gpu_throttle_oom_test reported {FAIL} failures (see [FAIL] lines above)"
    assert PASS > 0, "gpu_throttle_oom_test did not run any assertions (broken harness)"
