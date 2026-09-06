# GODSHAND Kit - Thermal Management Architecture for Kaggle 2xT4 Zero Throttle

## Problem Statement
Two T4 GPUs (16GB VRAM each) sitting side-by-side in a Kaggle notebook chassis share thermal coupling. Running both at 100% simultaneously causes thermal throttling (83°C+), reducing performance and risking OOM from extended render times.

## Design Goals
- **ZERO GPU THROTTLE**: Never exceed 78°C on either GPU
- **ZERO OOM**: Aggressive VRAM management with T4-specific regimes
- **ZERO COMPROMISE**: Hollywood quality maintained through intelligent scheduling
- **ZERO DEFECTS**: Predictive thermal management, mandatory cooldowns

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GODSHAND THERMAL ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │    GPU 0 (T4)   │    │    GPU 1 (T4)   │  ← Thermal Coupling │
│  │  16GB VRAM      │    │  16GB VRAM      │                    │
│  │  Temp: 70°C     │    │  Temp: 72°C     │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────┐                   │
│  │         THERMAL MONITOR DAEMON          │                   │
│  │  • 5s polling interval                  │                   │
│  │  • Per-GPU thermal profile logging      │                   │
│  │  • Predictive temperature modeling      │                   │
│  │  • Throttle detection (util 100% + temp)│                   │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │         GPU SCHEDULER (Staggered)       │                   │
│  │  • Round-robin with thermal scoring     │                   │
│  │  • Mandatory 20s gap between shots      │                   │
│  │  • Cooldown to GREEN zone (≤70°C)       │                   │
│  │  • Predictive cooldown (YELLOW ≥74°C)   │                   │
│  │  • Consecutive shot penalty             │                   │
│  └─────────────────┬───────────────────────┘                   │
│                    │                                            │
│           ┌────────┴────────┐                                   │
│           ▼                 ▼                                   │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │  VRAM GOVERNOR  │  │  RENDER PIPELINE│                       │
│  │  • T4 Regimes   │  │  • Shot dispatch│                       │
│  │  • Auto-downgrade│  │  • Rejection    │                       │
│  │  • FP8/Offload  │  │    sampling     │                       │
│  └─────────────────┘  └─────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Thermal Thresholds (AGGRESSIVE for ZERO THROTTLE)

| Zone | Temp Range | Action |
|------|------------|--------|
| **GREEN** | ≤ 70°C | Safe to dispatch next shot |
| **YELLOW** | 71-74°C | Predictive cooldown - wait for GREEN |
| **ORANGE** | 75-78°C | Mandatory cooldown - block dispatch |
| **RED/THROTTLE** | ≥ 79°C | Emergency cooldown - 300s timeout |

## GPU Scheduler Algorithm

```
pick_gpu():
    for each GPU:
        score = temp * 10 + consecutive_shots * 50 + max(0, gap_needed) * 2
    pick GPU with LOWEST score
    cooldown(selected_gpu)  # Block until GREEN + gap satisfied
    return selected_gpu

cooldown(gpu_idx):
    while True:
        gap_ok = (now - last_finish[gpu_idx]) >= MIN_GAP_SEC (20s)
        temp_ok = temp[gpu_idx] <= TEMP_GREEN_C (70°C)
        predictive_ok = temp[gpu_idx] < TEMP_YELLOW_C (74°C) or not PREDICTIVE_COOLDOWN
        
        if gap_ok and temp_ok and predictive_ok:
            break
        if timeout > 300s:
            warn and proceed
        sleep(2s)
```

## VRAM Governor - T4 Specific Regimes

| Regime | Free VRAM | Config |
|--------|-----------|--------|
| **T4_PLENTY** | ≥ 12GB | fp16, attn_slice, VAE tile 512 |
| **T4_TIGHT** | 8-12GB | fp16, model_offload, sequential_offload, cpu_offload_clip, VAE tile 256 |
| **T4_DANGER** | 4-8GB | fp8, full offload, steps-12, 4s max, VAE tile 128 |
| **T4_CRITICAL** | < 4GB | fp8, full offload, steps-20, 3s max, 768x512, VAE tile 64 |

## Thermal Monitor Daemon

- **Polling**: 5 second intervals
- **Logging**: JSONL to `/tmp/thermal_profile.jsonl`
- **Per-shot snapshots**: before/after each shot
- **Throttle detection**: sustained 100% util + rising temp
- **Emergency brake**: blocks dispatch if any GPU ≥ 79°C

## Integration Points

1. **render_film.py** → `scheduler.pick()` → `scheduler.finish()`
2. **render_shot()** → `vram_governor.resolve_params()` → `thermal_monitor.snapshot()`
3. **Daemon** → continuous logging → post-run forensics

## Environment Variables (Tunable)

```bash
KIT_GPU_GAP_SEC=20              # Min seconds between shots on same GPU
KIT_GPU_THROTTLE_TEMP_C=78      # Hard throttle ceiling
KIT_GPU_TEMP_GREEN_C=70         # Safe dispatch temperature
KIT_GPU_TEMP_YELLOW_C=74        # Predictive cooldown trigger
KIT_GPU_COOLDOWN_POLL_SEC=2     # Cooldown poll interval
KIT_GPU_COOLDOWN_TIMEOUT_SEC=300 # Max cooldown wait
KIT_PREDICTIVE_COOLDOWN=1       # Enable predictive cooldown
KIT_VRAM_SAFETY_MB=2048         # VRAM safety margin
KIT_SHOT_RETRIES=3              # Retry attempts per take
KIT_THERMAL_LOG=/tmp/thermal_profile.jsonl
```

## Post-Run Forensics

Thermal profile JSONL enables:
- Throttle event detection
- Per-shot thermal delta analysis
- GPU utilization efficiency
- Cooldown effectiveness metrics
- Correlation with render quality metrics