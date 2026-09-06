#!/bin/bash
# entrypoint.sh - GODSHAND Kit Entrypoint for Kaggle 2xT4
# ZERO GPU THROTTLE | ZERO OOM | HOLLYWOOD QUALITY

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================
log_info "=========================================="
log_info "GODSHAND KIT - KAGGLE 2xT4 ZERO THROTTLE"
log_info "=========================================="
log_info "Environment validation..."

# Check NVIDIA GPUs
log_info "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
log_info "Detected ${GPU_COUNT} GPU(s)"

if [ "$GPU_COUNT" -lt 1 ]; then
    log_error "No NVIDIA GPUs detected! Exiting."
    exit 1
fi

if [ "$GPU_COUNT" -eq 1 ]; then
    log_warn "Only 1 GPU detected. Running in single-GPU mode (no thermal staggering)."
    export KIT_GPU_GAP_SEC=0
    export KIT_PREDICTIVE_COOLDOWN=0
fi

# Check VRAM per GPU
log_info "Checking VRAM per GPU..."
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | while IFS=',' read -r idx total free; do
    total_gb=$((total / 1024))
    free_gb=$((free / 1024))
    log_info "  GPU ${idx}: ${free_gb}GB free / ${total_gb}GB total"
    if [ "$free_gb" -lt 8 ]; then
        log_warn "  GPU ${idx}: LOW VRAM (${free_gb}GB free) - enabling aggressive offload"
        export KIT_LOW_VRAM=1
    fi
done

# Check system memory
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
log_info "System RAM: ${TOTAL_MEM}GB"
if [ "$TOTAL_MEM" -lt 16 ]; then
    log_warn "Low system RAM (${TOTAL_MEM}GB). Enabling aggressive CPU offload."
    export KIT_LOW_VRAM=1
fi

# Check disk space
DISK_FREE=$(df -BG /workspace | awk 'NR==2 {print $4}' | sed 's/G//')
log_info "Workspace free space: ${DISK_FREE}GB"
if [ "$DISK_FREE" -lt 20 ]; then
    log_warn "Low disk space (${DISK_FREE}GB). Cleanup recommended."
fi

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
log_info "=========================================="
log_info "GODSHAND KIT CONFIGURATION"
log_info "=========================================="
log_info "GPU Gap: ${KIT_GPU_GAP_SEC}s"
log_info "Throttle Temp: ${KIT_GPU_THROTTLE_TEMP_C}°C"
log_info "Green Temp: ${KIT_GPU_TEMP_GREEN_C}°C"
log_info "Yellow Temp: ${KIT_GPU_TEMP_YELLOW_C}°C"
log_info "Cooldown Poll: ${KIT_GPU_COOLDOWN_POLL_SEC}s"
log_info "Cooldown Timeout: ${KIT_GPU_COOLDOWN_TIMEOUT_SEC}s"
log_info "Predictive Cooldown: ${KIT_PREDICTIVE_COOLDOWN}"
log_info "VRAM Safety Margin: ${KIT_VRAM_SAFETY_MB}MB"
log_info "Low VRAM Mode: ${KIT_LOW_VRAM}"
log_info "Shot Retries: ${KIT_SHOT_RETRIES}"
log_info "Thermal Log: ${KIT_THERMAL_LOG}"
log_info "PyTorch CUDA Alloc: ${PYTORCH_CUDA_ALLOC_CONF}"
log_info "=========================================="

# =============================================================================
# PYTORCH CUDA MEMORY CONFIGURATION
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_BENCHMARK=1
export TORCH_CUDNN_DETERMINISTIC=0

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
log_info "Setting up directories..."
mkdir -p /workspace/ComfyUI/models
mkdir -p /workspace/ComfyUI/custom_nodes
mkdir -p /workspace/hf_cache
mkdir -p /workspace/work
mkdir -p /tmp/thermal_logs
mkdir -p /opt/kit/work

# Symlink ComfyUI models if needed
if [ ! -L /opt/ComfyUI/models ] && [ -d /workspace/ComfyUI/models ]; then
    ln -sf /workspace/ComfyUI/models /opt/ComfyUI/models
fi

# =============================================================================
# COMFYUI SETUP
# =============================================================================
log_info "Setting up ComfyUI..."

# Install ComfyUI if not present
if [ ! -d "/opt/ComfyUI" ]; then
    log_info "Installing ComfyUI..."
    cd /opt
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd /opt/ComfyUI
    pip install -r requirements.txt
fi

# Install custom nodes for Wan 2.2 / Phantom / FLF2V
log_info "Installing ComfyUI custom nodes..."
cd /opt/ComfyUI/custom_nodes

# ComfyUI-Manager (for easy node management)
if [ ! -d "ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ComfyUI-Manager && pip install -r requirements.txt && cd ..
fi

# WanVideo wrapper
if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
    cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt && cd ..
fi

# VideoHelperSuite
if [ ! -d "VideoHelperSuite" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git VideoHelperSuite
    cd VideoHelperSuite && pip install -r requirements.txt && cd ..
fi

# Advanced ControlNet
if [ ! -d "ComfyUI-Advanced-ControlNet" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git ComfyUI-Advanced-ControlNet
    cd ComfyUI-Advanced-ControlNet && pip install -r requirements.txt && cd ..
fi

# Face restore (CodeFormer)
if [ ! -d "ComfyUI-CodeFormer" ]; then
    git clone https://github.com/sczhou/CodeFormer.git ComfyUI-CodeFormer
    cd ComfyUI-CodeFormer && pip install -r requirements.txt && cd ..
fi

# Upscalers
if [ ! -d "ComfyUI-UltimateSDUpscale" ]; then
    git clone https://github.com/ssitu/ComfyUI-UltimateSDUpscale.git ComfyUI-UltimateSDUpscale
    cd ComfyUI-UltimateSDUpscale && pip install -r requirements.txt && cd ..
fi

# =============================================================================
# MODEL DOWNLOADS (if not cached)
# =============================================================================
log_info "Checking model cache..."

# Wan 2.2 models
WAN_MODEL_DIR="/workspace/ComfyUI/models/diffusion_models"
mkdir -p "$WAN_MODEL_DIR"

# Check for Wan 2.2 S2V model
if [ ! -f "$WAN_MODEL_DIR/wan2.2_s2v_fp16.safetensors" ] && [ ! -f "$WAN_MODEL_DIR/wan2.2_s2v_fp8.safetensors" ]; then
    log_warn "Wan 2.2 S2V model not found. Download required."
    log_info "Expected at: $WAN_MODEL_DIR/wan2.2_s2v_fp16.safetensors"
fi

# Check for Phantom model
if [ ! -f "$WAN_MODEL_DIR/phantom_fp16.safetensors" ] && [ ! -f "$WAN_MODEL_DIR/phantom_fp8.safetensors" ]; then
    log_warn "Phantom model not found. Download required."
fi

# Check for FLF2V model
if [ ! -f "$WAN_MODEL_DIR/flf2v_fp16.safetensors" ] && [ ! -f "$WAN_MODEL_DIR/flf2v_fp8.safetensors" ]; then
    log_warn "FLF2V model not found. Download required."
fi

# VAE
VAE_DIR="/workspace/ComfyUI/models/vae"
mkdir -p "$VAE_DIR"
if [ ! -f "$VAE_DIR/wan_vae_fp16.safetensors" ]; then
    log_warn "Wan VAE not found at $VAE_DIR/wan_vae_fp16.safetensors"
fi

# Text encoders
CLIP_DIR="/workspace/ComfyUI/models/text_encoders"
mkdir -p "$CLIP_DIR"

# Upscalers
UPSCALE_DIR="/workspace/ComfyUI/models/upscale_models"
mkdir -p "$UPSCALE_DIR"

# =============================================================================
# THERMAL MONITOR DAEMON
# =============================================================================
log_info "Starting thermal monitor daemon..."
cd /opt/kit
python3 -c "
import thermal_monitor
import time
import threading

def monitor_loop():
    while True:
        for i in range(2):
            thermal_monitor.snapshot(i, 'daemon')
        time.sleep(5)

t = threading.Thread(target=monitor_loop, daemon=True)
t.start()
print('[thermal_monitor] Daemon started')
" &
THERMAL_PID=$!
log_info "Thermal monitor daemon started (PID: $THERMAL_PID)"

# =============================================================================
# COMFYUI SERVER
# =============================================================================
log_info "Starting ComfyUI server..."
cd /opt/ComfyUI

# Start ComfyUI with optimized settings for T4
python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --cuda-malloc \
    --disable-auto-launch \
    --max-split-size-mb 128 \
    &

COMFY_PID=$!
log_info "ComfyUI server started (PID: $COMFY_PID)"

# =============================================================================
# WAIT FOR COMFYUI READY
# =============================================================================
log_info "Waiting for ComfyUI to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
        log_success "ComfyUI is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        log_error "ComfyUI failed to start within 120 seconds"
        kill $COMFY_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# =============================================================================
# GPU SCHEDULER PRE-FLIGHT CHECK
# =============================================================================
log_info "Running GPU scheduler pre-flight check..."
cd /opt/kit
python3 -c "
import gpu_scheduler
s = gpu_scheduler.Scheduler()
print(f'[gpu_scheduler] GPUs: {s.gpu_count}')
print(s.summary())
"

# =============================================================================
# VRAM GOVERNOR PRE-FLIGHT CHECK
# =============================================================================
log_info "Running VRAM governor pre-flight check..."
cd /opt/kit
python3 -c "
import vram_governor as vg
gpus = vg.probe_all()
for g in gpus:
    regime = vg.pick_regime(g['free_mb'])
    print(f'[vram_governor] GPU {g[\"index\"]}: {g[\"free_mb\"]}MB free, {g[\"temp_c\"]}C, {g[\"util_pct\"]}% util -> {regime}')
"

# =============================================================================
# THERMAL MONITOR PRE-FLIGHT CHECK
# =============================================================================
log_info "Running thermal monitor pre-flight check..."
cd /opt/kit
python3 -c "
import thermal_monitor
for i in range(2):
    row = thermal_monitor.snapshot(i, 'preflight')
    if row:
        print(f'[thermal_monitor] GPU {i}: {row[\"temp_c\"]}C, {row[\"util_pct\"]}% util, {row[\"free_mb\"]}MB free')
"

# =============================================================================
# RUN GODSHAND RENDER PIPELINE
# =============================================================================
log_info "=========================================="
log_info "STARTING GODSHAND RENDER PIPELINE"
log_info "=========================================="
log_info "Config: config_confrontation.yaml"
log_info "Shots: shots_confrontation.json"
log_info "Work dir: /opt/kit/work"
log_info "=========================================="

cd /opt/kit

# Run the render pipeline
python3 -m src.render_film \
    --config config_confrontation.yaml \
    --shots shots_confrontation.json \
    --work-dir /opt/kit/work \
    2>&1 | tee /opt/kit/work/render.log

RENDER_EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# POST-RENDER TELEMETRY
# =============================================================================
log_info "=========================================="
log_info "POST-RENDER TELEMETRY"
log_info "=========================================="

# GPU Scheduler summary
python3 -c "
import gpu_scheduler
s = gpu_scheduler.Scheduler()
print(s.summary())
"

# Thermal profile dump
log_info "Thermal profile:"
if [ -f /tmp/thermal_profile.jsonl ]; then
    tail -20 /tmp/thermal_profile.jsonl
fi

# VRAM final state
python3 -c "
import vram_governor as vg
gpus = vg.probe_all()
for g in gpus:
    print(f'[vram_governor] GPU {g[\"index\"]}: {g[\"free_mb\"]}MB free, {g[\"temp_c\"]}C, {g[\"util_pct\"]}% util')
"

# =============================================================================
# CLEANUP
# =============================================================================
log_info "Shutting down..."
kill $COMFY_PID 2>/dev/null || true
kill $THERMAL_PID 2>/dev/null || true
wait $COMFY_PID 2>/dev/null || true
wait $THERMAL_PID 2>/dev/null || true

# Final GPU state
log_info "Final GPU state:"
nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits

if [ $RENDER_EXIT_CODE -eq 0 ]; then
    log_success "=========================================="
    log_success "GODSHAND RENDER PIPELINE COMPLETED SUCCESSFULLY"
    log_success "ZERO GPU THROTTLE | ZERO OOM | HOLLYWOOD QUALITY"
    log_success "=========================================="
    exit 0
else
    log_error "=========================================="
    log_error "GODSHAND RENDER PIPELINE FAILED (exit code: $RENDER_EXIT_CODE)"
    log_error "=========================================="
    exit $RENDER_EXIT_CODE
fi