# GODSHAND Kit - VRAM Governor Architecture for Kaggle 2xT4 Zero OOM

## Problem Statement
Kaggle provides 2x T4 GPUs with 16GB VRAM each. Wan 2.2 / Phantom / FLF2V models with high-res video generation (1280x704, 30 steps, 5s clips) can easily exceed 16GB VRAM, causing OOM crashes. Zero OOM requires aggressive, predictive VRAM management.

## Design Goals
- **ZERO OOM**: Never crash from VRAM exhaustion
- **ZERO COMPROMISE**: Maintain Hollywood quality where possible
- **AUTO-DOWNGRADE**: Graceful degradation when VRAM is tight
- **T4-SPECIFIC**: Optimized for T4 16GB architecture

## VRAM Governor Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VRAM GOVERNOR                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PROBE LAYER (nvidia-smi)                   │   │
│  │  • Robust query with 5 retries, 1s delay                │   │
│  │  • Returns: index, name, total, used, free, temp, util  │   │
│  │  • Pure stdlib - no torch/nvidia-ml-py dependency       │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              REGIME RESOLVER                            │   │
│  │  • Maps free VRAM → T4-specific regime                  │   │
│  │  • 4 regimes: PLENTY, TIGHT, DANGER, CRITICAL           │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PARAMETER RESOLVER                         │   │
│  │  • Input: base config + shot params + GPU free VRAM     │   │
│  │  • Output: mutated params fitting VRAM regime           │   │
│  │  • Keys: width, height, steps, seconds, offload flags   │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│           ┌────────────────┼────────────────┐                   │
│           ▼                ▼                ▼                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  COMFYUI     │  │  MODEL       │  │  VAE/CLIP    │            │
│  │  WORKFLOW    │  │  LOADING     │  │  OFFLOAD     │            │
│  │  OVERRIDES   │  │  STRATEGY    │  │  STRATEGY    │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## T4-Specific VRAM Regimes

### Regime Thresholds (16GB T4)

```
┌─────────────────┬──────────────┬────────────────────────────────────┐
│ Regime          │ Free VRAM    │ Trigger Condition                  │
├─────────────────┼──────────────┼────────────────────────────────────┤
│ T4_PLENTY       │ ≥ 12 GB      │ free_mb >= 12288                   │
│ T4_TIGHT        │ 8-12 GB      │ free_mb >= 8192                    │
│ T4_DANGER       │ 4-8 GB       │ free_mb >= 4096                    │
│ T4_CRITICAL     │ < 4 GB       │ free_mb < 4096                     │
└─────────────────┴──────────────┴────────────────────────────────────┘
```

### Regime Configurations

#### T4_PLENTY (≥ 12GB free) - "Full Quality"
```python
{
    "precision": "fp16",
    "attention_slicing": True,
    "vae_tiling": True,
    "vae_tile_size": 512,
    "model_offload": False,
    "sequential_offload": False,
    "cpu_offload_clip": False,
    "cpu_offload_vae": False,
    "steps": base_steps,           # No reduction
    "seconds": base_seconds,       # No reduction
    "width": base_width,           # No reduction
    "height": base_height,         # No reduction
    "offload_level": "none",
    "fp8": False
}
```

#### T4_TIGHT (8-12GB free) - "Model Offload"
```python
{
    "precision": "fp16",
    "attention_slicing": True,
    "vae_tiling": True,
    "vae_tile_size": 256,
    "model_offload": True,         # Enable model CPU offload
    "sequential_offload": True,    # Sequential layer offload
    "cpu_offload_clip": True,      # Offload CLIP to CPU
    "cpu_offload_vae": False,
    "steps": base_steps,
    "seconds": base_seconds,
    "width": base_width,
    "height": base_height,
    "offload_level": "model",
    "fp8": False
}
```

#### T4_DANGER (4-8GB free) - "FP8 + Aggressive Offload"
```python
{
    "precision": "fp8",            # Switch to FP8 quantization
    "attention_slicing": True,
    "vae_tiling": True,
    "vae_tile_size": 128,          # Smaller VAE tiles
    "model_offload": True,
    "sequential_offload": True,
    "cpu_offload_clip": True,
    "cpu_offload_vae": True,       # Offload VAE to CPU
    "steps": max(8, base_steps - 12),  # Reduce steps
    "seconds": min(base_seconds, 4.0), # Cap duration
    "width": base_width,
    "height": base_height,
    "offload_level": "model",
    "fp8": True
}
```

#### T4_CRITICAL (< 4GB free) - "Maximum Offload + Resolution Drop"
```python
{
    "precision": "fp8",
    "attention_slicing": True,
    "vae_tiling": True,
    "vae_tile_size": 64,           # Minimal VAE tiles
    "model_offload": True,
    "sequential_offload": True,
    "cpu_offload_clip": True,
    "cpu_offload_vae": True,
    "steps": max(4, base_steps - 20),  # Minimal steps
    "seconds": min(base_seconds, 3.0), # 3s max clips
    "width": min(base_width, 768),     # Resolution drop
    "height": min(base_height, 512),
    "offload_level": "full",
    "fp8": True
}
```

## Parameter Resolution Flow

```
resolve_params(cfg, gpu_index, shot):
    1. free = free_mb(gpu_index)           # Probe current free VRAM
    2. regime = pick_regime(free)          # Map to T4 regime
    3. base_params = extract from cfg/shot # width, height, steps, seconds
    4. params = apply_regime(base_params, regime)
    5. params["regime"] = regime
    6. params["free_mb"] = free
    7. params["gpu_index"] = gpu_index
    8. return params
```

## ComfyUI Workflow Integration

The resolved parameters map to ComfyUI workflow overrides:

| Parameter | ComfyUI Node | Notes |
|-----------|--------------|-------|
| `width` / `height` | EmptyLatentVideo / WanVideoSampler | Video dimensions |
| `steps` | WanVideoSampler | Denoising steps |
| `seconds` | WanVideoSampler | Clip duration |
| `attention_slicing` | WanVideoModelLoader | `attention_slicing` arg |
| `vae_tiling` / `vae_tile_size` | WanVideoVAE / VAELoader | `tile_size` arg |
| `fp8` | WanVideoModelLoader | `fp8` arg |
| `model_offload` | WanVideoModelLoader | `offload_model` arg |
| `sequential_offload` | WanVideoModelLoader | `sequential_offload` arg |
| `cpu_offload_clip` | CLIPLoader | `device="cpu"` |
| `cpu_offload_vae` | VAELoader | `device="cpu"` |

## Safety Margin

```python
SAFE_RESERVE_MB = 2048  # 2GB reserved for:
                        # - PyTorch allocator fragmentation
                        # - CUDA context overhead
                        # - Temporary buffers during generation
                        # - ComfyUI workspace tensors
```

Effective usable VRAM = `free_mb - SAFE_RESERVE_MB`

## Wait-for-VRAM Mechanism

```python
wait_for_vram(min_free_mb, gpu_index, timeout_sec=300, poll_sec=5.0):
    """Block until sufficient VRAM is available."""
    while timeout not exceeded:
        free = free_mb(gpu_index)
        if free >= min_free_mb:
            return free
        sleep(poll_sec)
    return free_mb(gpu_index)  # Return whatever's available
```

Used before dispatching heavy shots to ensure VRAM headroom.

## Environment Variables

```bash
KIT_VRAM_SAFETY_MB=2048    # Safety reserve (MB)
KIT_LOW_VRAM=1             # Force low VRAM mode (set by governor)
```

## Integration with GPU Scheduler

```
Scheduler.pick() 
    → selects GPU with most free VRAM (via scoring)
    → calls VRAM Governor.resolve_params() for that GPU
    → passes params to render_shot()
    → render_shot() applies params to ComfyUI workflow
    → after shot: VRAM freed, next pick() sees updated free VRAM
```

## Post-Run Telemetry

Each shot logs:
```json
{
  "shot_id": "s001",
  "gpu_index": 0,
  "regime": "t4_tight",
  "free_mb_before": 9216,
  "free_mb_after": 14336,
  "params_applied": {...}
}
```

Enables analysis of:
- Regime distribution across shots
- VRAM recovery between shots
- Effectiveness of offload strategies
- Correlation with render quality

## Edge Cases Handled

1. **nvidia-smi failure**: Returns 0 free VRAM → triggers CRITICAL regime (safe fallback)
2. **VRAM fragmentation**: Safety margin absorbs allocator fragmentation
3. **Concurrent processes**: Probe runs immediately before each shot
4. **Model caching**: ComfyUI keeps models in VRAM; governor accounts via `free_mb`
5. **Multi-GPU**: Each GPU probed independently; regime per-GPU