# Dockerfile for Kaggle 2xT4 Hollywood-Quality Zero-Compromise Zero-Defect Zero-GPU-Throttle Deployment
# Base: NVIDIA CUDA 12.4 + Python 3.10 (slim for smaller image)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# AGGRESSIVE THERMAL / VRAM ENVIRONMENT VARIABLES (ZERO GPU THROTTLE ON 2xT4)
# =============================================================================
# Aggressive thermal settings for ZERO GPU THROTTLE on 2xT4
ENV KIT_GPU_GAP_SEC=20
ENV KIT_GPU_THROTTLE_TEMP_C=78
ENV KIT_GPU_TEMP_GREEN_C=70
ENV KIT_GPU_TEMP_YELLOW_C=74
ENV KIT_GPU_COOLDOWN_POLL_SEC=2
ENV KIT_GPU_COOLDOWN_TIMEOUT_SEC=300
ENV KIT_PREDICTIVE_COOLDOWN=1

# Aggressive VRAM safety margin for ZERO OOM on 2xT4 (16GB each)
ENV KIT_VRAM_SAFETY_MB=2048
ENV KIT_VRAM_SAFETY_MB=2048

# PyTorch / CUDA optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_BENCHMARK=1
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV TORCH_ALLOW_TF32_CUDNN_OVERRIDE=1

# ComfyUI / PyTorch memory optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_BENCHMARK=1

# Disable telemetry / telemetry noise
ENV COMFYUI_TELEMETRY=0
ENV DO_NOT_TRACK=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1

# Python optimization
ENV PYTHONOPTIMIZE=1
ENV PYTHONHASHSEED=0

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# PYTORCH WITH CUDA 12.4 (T4 optimized)
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# =============================================================================
# COMFYUI + DEPENDENCIES (T4-OPTIMIZED)
# =============================================================================
RUN pip install --no-cache-dir \
    comfyui==0.2.7 \
    xformers==0.0.28 \
    --index-url https://download.pytorch.org/whl/cu124

# ComfyUI dependencies
RUN pip install --no-cache-dir \
    einops==0.8.0 \
    kornia==0.7.2 \
    safetensors==0.4.5 \
    accelerate==0.33.0 \
    transformers==4.44.0 \
    tokenizers==0.19.1 \
    huggingface-hub==0.24.0 \
    safetensors==0.4.5 \
    Pillow==10.4.0 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    tqdm==4.66.5 \
    pyyaml==6.0.1 \
    requests==2.32.3 \
    tqdm==4.66.5 \
    psutil==5.9.8 \
    psutil==5.9.8 \
    pyyaml==6.0.1 \
    requests==2.32.3 \
    tqdm==4.66.5 \
    numpy==1.26.4

# ComfyUI custom nodes dependencies
RUN pip install --no-cache-dir \
    comfyui-impact-pack==2024.10.1 \
    comfyui-impact-subpack==2024.10.1 \
    comfyui-impact-pack==2024.10.1 \
    comfyui-impact-subpack==2024.10.1 \
    comfyui-impact-pack==2024.10.1 \
    comfyui-impact-subpack==2024.10.1

# Additional ComfyUI nodes for Wan 2.2 / Phantom / FLF2V
RUN pip install --no-cache-dir \
    comfyui-wan-video==2024.12.1 \
    comfyui-phantom==2024.12.1 \
    comfyui-flf2v==2024.12.1 \
    comfyui-video-helpers==2024.12.1 \
    comfyui-video-helpers-suite==2024.12.1

# Video processing
RUN pip install --no-cache-dir \
    imageio==2.35.1 \
    imageio-ffmpeg==0.5.1 \
    decord==0.6.0 \
    av==11.0.0

# Audio processing
RUN pip install --no-cache-dir \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    pydub==0.25.1

# Image processing
RUN pip install --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    pillow==10.4.0 \
    scikit-image==0.24.0

# Utilities
RUN pip install --no-cache-dir \
    pyyaml==6.0.1 \
    pydantic==2.8.2 \
    pydantic-settings==2.4.0 \
    python-dotenv==1.0.1 \
    tqdm==4.66.5 \
    psutil==5.9.8 \
    requests==2.32.3 \
    aiohttp==3.9.5 \
    aiofiles==23.2.1

# =============================================================================
# COMFYUI INSTALLATION
# =============================================================================
WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
WORKDIR /opt/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# CUSTOM NODES FOR WAN 2.2 / PHANTOM / FLF2V
# =============================================================================
WORKDIR /opt/ComfyUI/custom_nodes

# ComfyUI-Impact-Pack (essential for Wan/Phantom)
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-Impact-Pack
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI-VideoHelperSuite
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI-WanVideoWrapper (Wan 2.2)
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI-Phantom
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-Phantom.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-Phantom
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI-FLF2V
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-FLF2V.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-FLF2V
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI-Advanced-ControlNet
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-Advanced-ControlNet
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI-Face-Enhance (for face restore)
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_FaceEnhance.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI_FaceEnhance
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI-Upscale (for upscaling)
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/chflame163/ComfyUI-Upscale.git
WORKDIR /opt/ComfyUI/custom_nodes/ComfyUI-Upscale
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# MODEL DIRECTORIES
# =============================================================================
WORKDIR /opt/ComfyUI
RUN mkdir -p models/checkpoints models/vae models/loras models/embeddings \
    models/controlnet models/upscale_models models/clip models/unet \
    models/clip_vision models/gligen models/photomaker models/ipadapter \
    models/wan models/phantom models/flf2v models/face_restore

# =============================================================================
# COPY KIT SOURCE CODE
# =============================================================================
WORKDIR /opt/kit
COPY kit/src/ ./src/
COPY kit/config_confrontation.yaml ./config_confrontation.yaml
COPY kit/shots_confrontation.json ./shots_confrontation.json
COPY kit/requirements.txt ./requirements.txt 2>/dev/null || true

# Install kit-specific requirements
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# =============================================================================
# ENTRYPOINT & THERMAL MONITORING
# =============================================================================
WORKDIR /opt/kit

# Create entrypoint script
RUN cat > /opt/kit/entrypoint.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "GODSHAND KAGGLE 2xT4 DEPLOYMENT"
echo "HOLLYWOOD QUALITY | ZERO COMPROMISE | ZERO DEFECTS | ZERO GPU THROTTLE"
echo "=========================================="
echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv
echo ""
echo "Environment:"
echo "  KIT_GPU_GAP_SEC=${KIT_GPU_GAP_SEC}"
echo "  KIT_GPU_THROTTLE_TEMP_C=${KIT_GPU_THROTTLE_TEMP_C}"
echo "  KIT_GPU_TEMP_GREEN_C=${KIT_GPU_TEMP_GREEN_C}"
echo "  KIT_GPU_TEMP_YELLOW_C=${KIT_GPU_TEMP_YELLOW_C}"
echo "  KIT_VRAM_SAFETY_MB=${KIT_VRAM_SAFETY_MB}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo ""

# Start thermal monitor in background
python3 -m src.thermal_monitor --daemon &
THERMAL_PID=$!
echo "[entrypoint] Thermal monitor started (PID: $THERMAL_PID)"

# Start ComfyUI server in background
cd /opt/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 --enable-cors-header &
COMFY_PID=$!
echo "[entrypoint] ComfyUI server started (PID: $COMFY_PID)"

# Wait for ComfyUI to be ready
echo "[entrypoint] Waiting for ComfyUI to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "[entrypoint] ComfyUI is ready!"
        break
    fi
    sleep 2
done

# Run the kit orchestrator
cd /opt/kit
echo "[entrypoint] Starting GODSHAND render pipeline..."
python3 -m src.render_film --config config_confrontation.yaml --shots shots_confrontation.json

# Cleanup
echo "[entrypoint] Render complete. Shutting down..."
kill $COMFY_PID 2>/dev/null || true
kill $THERMAL_PID 2>/dev/null || true
wait $COMFY_PID 2>/dev/null || true
wait $THERMAL_PID 2>/dev/null || true

echo "[entrypoint] Done."
EOF

RUN chmod +x /opt/kit/entrypoint.sh

# =============================================================================
# HEALTH CHECK
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8188/system_stats > /dev/null 2>&1 || exit 1

# =============================================================================
# ENTRYPOINT
# =============================================================================
ENTRYPOINT ["/opt/kit/entrypoint.sh"]