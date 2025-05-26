# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    NEMO_MODEL_PATH=/app/models/stt_hi_conformer_ctc_medium.nemo \
    KENLM_MODEL_PATH=/app/models/stt_hi_conformer_ctc_medium_kenlm.bin \
    MAX_FILE_SIZE=52428800 \
    MAX_DURATION=60 \
    MAX_WORKERS=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libsndfile1 \
    libsox-dev \
    sox \
    ffmpeg \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install KenLM Python bindings
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

# Upgrade pip and wheel
RUN pip install --upgrade pip setuptools wheel Cython

# Install core dependencies
RUN pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    typing-extensions==4.8.0 \
    python-dotenv==1.0.0 \
    numba==0.58.1 \
    pyctcdecode==0.5.0 \
    torchaudio

# Install NeMo toolkit and ASR/NLP dependencies
RUN pip install "nemo_toolkit[all]==1.22.0" \
 && pip install \
    pytorch-lightning==2.0.9 \
    torchmetrics==1.2.0 \
    transformers==4.35.2 \
    datasets==2.14.6 \
    tokenizers==0.15.0 \
    sentencepiece==0.1.99 \
    sacrebleu==2.3.1 \
    editdistance==0.6.2 \
    jiwer==3.0.3 \
    inflect==7.0.0 \
    youtokentome==1.0.6 \
    matplotlib==3.7.3 \
    tensorboard==2.15.1 \
    wandb==0.16.0

# Create directories
RUN mkdir -p /app/models /app/temp

# Copy application code
COPY fastapi_hindi_asr.py .
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "fastapi_hindi_asr:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
