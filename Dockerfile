# ==============================================================================
# Anot Medical ASR — Production Service Container (CPU Base)
# ==============================================================================

FROM python:3.12-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies: FFmpeg, libsndfile, git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for Docker layer caching
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchaudio==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers==4.49.0 \
    fastapi==0.142.2 \
    uvicorn==0.54.0 \
    python-multipart==0.0.32 \
    pydantic==2.13.5 \
    soundfile==0.13.1 \
    numpy>=1.26.0 \
    faster-whisper==1.2.1

# Copy application source code
COPY app/ ./app/
COPY audio/ ./audio/
COPY decoding/ ./decoding/
COPY language_model/ ./language_model/
COPY evaluation/ ./evaluation/

# Create temporary upload directory
RUN mkdir -p temp/asr_uploads && chmod 777 temp/asr_uploads

# Service runs on port 8000
EXPOSE 8000

# Healthcheck targeting the safe /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Production entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
