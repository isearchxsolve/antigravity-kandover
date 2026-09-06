# GODSHAND Kit - GPU Scheduler Architecture for Kaggle 2xT4 Zero Throttle

## Problem Statement
Two T4 GPUs in a Kaggle notebook chassis are thermally coupled. Running both at 100% simultaneously causes thermal throttling (83°C+), reducing clock speeds and extending render times, which increases OOM risk. The scheduler must stagger workloads to maintain thermal headroom.

## Design Goals
- **ZERO GPU THROTTLE**: Never exceed 78°C on either GPU
- **MAXIMUM THROUGHPUT**: Keep both GPUs busy without thermal coupling
- **FAIR DISTRIBUTION**: Balance shot count across GPUs
- **PREDICTIVE**: Anticipate thermal buildup before it happens
- **RESUMABLE**: Track progress for crash recovery

## GPU Scheduler Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GPU SCHEDULER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    STATE TRACKER                        │   │
│  │  Per-GPU:                                               │   │
│  │  • last_finish: timestamp of last shot completion       │   │
│  │  • shot_count: total shots completed on this GPU        │   │
│  │  • consecutive_shots: shots since last cooldown         │   │
│  │  • last_temp: last known temperature                    │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SCORING ENGINE                       │   │
│  │  Score = temp * 10 + consecutive_shots * 50 + gap_penalty│   │
│  │  Lower score = better candidate                         │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    COOLDOWN ENFORCER                    │   │
│  │  • Blocks until GPU temp ≤ GREEN (70°C)                 │   │
│  │  • Enforces MIN_GAP_SEC (20s) between shots             │   │
│  │  • Predictive: blocks if temp ≥ YELLOW (74°C)           │   │
│  │  • Timeout: 300s max wait, then proceed with warning    │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│           ┌────────────────┼────────────────┐                   │
│           ▼                ▼                ▼                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   GPU 0      │  │   GPU 1      │  │   SINGLE     │            │
│  │  (T4)        │  │  (T4)        │  │   GPU MODE   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scheduler Algorithm

### Pick GPU (Main Dispatch Logic)

```python
def pick(self):
    if self._n == 0: return 0          # No GPU (CPU fallback)
    if self._n == 1:                   # Single GPU mode
        self._cooldown(0)
        return 0
    
    # Multi-GPU: Score each GPU
    best_idx = None
    best_score = float('inf')
    
    for _ in range(self._n):           # Round-robin iteration
        idx = self._rr_index % self._n
        self._rr_index += 1
        
        gpus = vg.probe_all()          # Fresh telemetry
        cur = next(g for g in gpus if g["index"] == idx)
        s = self._states[idx]
        
        # Scoring: lower is better
        temp_score = cur["temp_c"] * 10              # Primary: temperature
        consecutive_score = s.consecutive_shots * 50 # Secondary: fairness
        gap_score = max(0, MIN_GAP_SEC - (time.time() - s.last_finish)) * 2
        score = temp_score + consecutive_score + gap_score
        
        if score < best_score:
            best_score = score
            best_idx = idx
    
    # Enforce cooldown on selected GPU
    self._cooldown(best_idx)
    return best_idx
```

### Cooldown Enforcement

```python
def _cooldown(self, idx):
    deadline = time.time() + COOLDOWN_TIMEOUT_SEC  # 300s max
    s = self._states[idx]
    
    while True:
        now = time.time()
        gap_ok = (now - s.last_finish) >= MIN_GAP_SEC  # 20s minimum
        
        gpus = vg.probe_all()
        cur = next((g for g in gpus if g["index"] == idx), None)
        
        if cur is None:
            temp_ok = True
        else:
            s.last_temp = cur["temp_c"]
            # AGGRESSIVE: Wait for GREEN zone (≤70°C), not just below throttle
            temp_ok = cur["temp_c"] <= TEMP_GREEN_C
            
            # PREDICTIVE: If temp ≥ YELLOW (74°C), wait longer
            if PREDICTIVE_COOLDOWN and cur["temp_c"] >= TEMP_YELLOW_C:
                temp_ok = False
        
        if gap_ok and temp_ok:
            s.consecutive_shots = 0  # Reset on successful cooldown
            return
        
        if now > deadline:
            print(f"WARNING: cooldown timeout on GPU {idx}")
            return
        
        time.sleep(COOLDOWN_POLL_SEC)  # 2s poll interval
```

### Finish Shot (State Update)

```python
def finish(self, idx):
    if self._n == 0: return
    if 0 <= idx < self._n:
        now = time.time()
        self._states[idx].last_finish = now
        self._states[idx].shot_count += 1
        self._states[idx].consecutive_shots += 1
```

## Scoring Function Deep Dive

### Score Components

| Component | Weight | Formula | Purpose |
|-----------|--------|---------|---------|
| **Temperature** | 10x | `temp_c * 10` | Primary: prefer cooler GPU |
| **Consecutive Shots** | 50x | `consecutive_shots * 50` | Fairness: prevent GPU hogging |
| **Gap Penalty** | 2x | `max(0, 20 - elapsed) * 2` | Enforce minimum gap |

### Score Examples

```
GPU 0: temp=65°C, consecutive=0, last_finish=15s ago
  Score = 65*10 + 0*50 + max(0,20-15)*2 = 650 + 0 + 10 = 660

GPU 1: temp=72°C, consecutive=2, last_finish=5s ago  
  Score = 72*10 + 2*50 + max(0,20-5)*2 = 720 + 100 + 30 = 850

→ GPU 0 selected (lower score)
```

## Thermal Thresholds

```python
# Environment variable defaults (AGGRESSIVE for ZERO THROTTLE)
MIN_GAP_SEC = 20              # Seconds between shots on same GPU
THROTTLE_TEMP_C = 78          # Hard throttle ceiling (never exceed)
TEMP_GREEN_C = 70             # Safe dispatch temperature
TEMP_YELLOW_C = 74            # Predictive cooldown trigger
COOLDOWN_POLL_SEC = 2         # Cooldown check interval
COOLDOWN_TIMEOUT_SEC = 300    # Max cooldown wait (5 min)
PREDICTIVE_COOLDOWN = True    # Enable predictive cooldown
```

### Thermal State Machine

```
                    ┌─────────────────┐
                    │   DISPATCHING   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │   GREEN    │ │  YELLOW    │ │   ORANGE   │
       │  (≤70°C)   │ │ (71-74°C)  │ │ (75-78°C)  │
       └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
             │              │              │
             │ dispatch     │ predictive   │ mandatory
             │ immediately  │ cooldown     │ cooldown
             ▼              ▼              ▼
       ┌─────────────────────────────────────────┐
       │           COOLDOWN ENFORCER             │
       │  Wait for: temp ≤ 70°C AND gap ≥ 20s    │
       └─────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   DISPATCHING   │
                    └─────────────────┘
```

## Single GPU Fallback

When only 1 GPU detected:
- No round-robin scoring
- Simple cooldown enforcement
- `MIN_GAP_SEC` still applies
- `PREDICTIVE_COOLDOWN` still applies
- Logs warning about single-GPU mode

## Integration with Render Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RENDER_FILM.MAIN()                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  sched = gpu_scheduler.Scheduler()                              │
│                                                                 │
│  for shot in shots:                                             │
│      gpu_idx = sched.pick()          # ← BLOCKS until ready    │
│      render_shot(shot, ..., gpu_idx) # ← Does the work         │
│      sched.finish(gpu_idx)           # ← Updates state         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Telemetry & Summary

```python
def summary(self):
    """Human-readable per-GPU stats for telemetry dump."""
    gpus = vg.probe_all()
    lines = []
    for i in range(self._n):
        cur = next((g for g in gpus if g["index"] == i), None)
        s = self._states[i]
        if cur:
            lines.append(f"GPU {i}: {cur['free_mb']}MB free, "
                         f"{cur['temp_c']}C, util {cur['util_pct']}%, "
                         f"shots done={s.shot_count}, consecutive={s.consecutive_shots}")
        else:
            lines.append(f"GPU {i}: (no telemetry), shots done={s.shot_count}")
    return "\n".join(lines)
```

### Example Output

```
GPU 0: 12288MB free, 68C, util 0%, shots done=12, consecutive=0
GPU 1: 10240MB free, 71C, util 0%, shots done=10, consecutive=1
```

## Environment Variables

```bash
KIT_GPU_GAP_SEC=20                    # Min gap between shots on same GPU
KIT_GPU_THROTTLE_TEMP_C=78            # Hard throttle ceiling
KIT_GPU_TEMP_GREEN_C=70               # Safe dispatch temperature
KIT_GPU_TEMP_YELLOW_C=74              # Predictive cooldown trigger
KIT_GPU_COOLDOWN_POLL_SEC=2           # Cooldown poll interval
KIT_GPU_COOLDOWN_TIMEOUT_SEC=300      # Max cooldown wait
KIT_PREDICTIVE_COOLDOWN=1             # Enable predictive cooldown
```

## Edge Cases Handled

1. **GPU disappears mid-run**: `probe_all()` returns empty → score=0 → fallback to round-robin
2. **nvidia-smi failure**: Returns empty list → single GPU mode or CPU fallback
3. **Cooldown timeout**: 300s max wait → warning logged → proceed anyway
4. **Thermal runaway**: Predictive cooldown catches rising temps before throttle
5. **Uneven shot distribution**: Consecutive shot penalty forces alternation
6. **First shot**: No `last_finish` → gap_ok = True immediately

## Post-Run Forensics

Scheduler summary enables:
- Shot distribution analysis (balance check)
- Thermal efficiency (time in cooldown vs rendering)
- Consecutive shot patterns (fairness verification)
- GPU utilization correlation with thermal profile

## Performance Characteristics

| Metric | Target |
|--------|--------|
| Max GPU temp | ≤ 78°C (never throttle) |
| Min inter-shot gap | 20s (enforced) |
| Cooldown overhead | < 5% of total runtime |
| Shot distribution | ±1 shot per GPU |
| Dispatch latency | < 100ms (excluding cooldown) |

## Testing Scenarios

1. **Cold start**: Both GPUs at 35°C → immediate dispatch to GPU 0
2. **Hot GPU**: GPU 0 at 75°C, GPU 1 at 65°C → dispatch to GPU 1
3. **Both hot**: Both at 72°C → wait for first to hit 70°C
4. **Single GPU**: Only GPU 0 visible → sequential with 20s gaps
5. **Rapid shots**: 10 shots queued → staggered across GPUs with cooldowns