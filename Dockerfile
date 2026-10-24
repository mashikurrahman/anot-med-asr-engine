# syntax=docker/dockerfile:1
# ==============================================================================
# Anot Medical ASR — Production image (Whisper engine, Railway/Docker)
# ==============================================================================
# Serves the FastAPI transcription service with the Whisper large-v3 family via
# faster-whisper (CTranslate2, CPU). The ASR model is BAKED INTO the image at
# build time so there is no multi-GB download on first request (fast cold start).
#
# Defaults are tuned for a Railway Hobby service (<= 8 GB RAM):
#   - engine = whisper, model = distil-large-v3 (~2 GB RAM, ~2x faster than
#     large-v3, ~1% accuracy cost). Switch to large-v3 by overriding
#     ANOT_WHISPER_MODEL and rebuilding (it fits in 8 GB).
# ==============================================================================
FROM python:3.12-slim

# ffmpeg is REQUIRED (AudioService converts m4a/mp3/etc. -> 16 kHz wav).
# libsndfile1 for soundfile; libgomp1 for CTranslate2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----
# faster-whisper (CTranslate2) is the production ASR runtime. torch/transformers
# are needed only because the import chain references the MedASR path; the MedASR
# weights are never downloaded when ANOT_ASR_ENGINE=whisper.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    transformers==4.49.0 \
    fastapi==0.142.2 \
    "uvicorn[standard]==0.34.0" \
    python-multipart==0.0.32 \
    pydantic==2.13.5 \
    soundfile==0.13.1 \
    numpy>=1.26.0

# ---- Pre-bake the ASR model into the image (no runtime download) ----
ARG WHISPER_MODEL=distil-large-v3
ENV ANOT_WHISPER_MODEL=${WHISPER_MODEL}
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

# ---- Application source (engine only; research/data trees are gitignored) ----
COPY app/ ./app/
COPY audio/ ./audio/
COPY decoding/ ./decoding/
COPY language_model/ ./language_model/
COPY evaluation/evaluate_transcript.py ./evaluation/evaluate_transcript.py
COPY evaluation/medical_entities.json ./evaluation/medical_entities.json
COPY accuracy/ ./accuracy/

# ---- Production runtime configuration ----
ENV ANOT_ASR_ENGINE=whisper \
    ANOT_ASR_ENV=production \
    ANOT_ASR_DEBUG=false \
    ANOT_ASR_HOST=0.0.0.0 \
    ANOT_WHISPER_COMPUTE_TYPE=int8 \
    ANOT_WHISPER_BEAM_SIZE=1 \
    ANOT_WHISPER_CPU_THREADS=0 \
    ANOT_AUDIO_ENHANCE=true \
    ANOT_ASR_MAX_CONCURRENT_JOBS=1 \
    PYTHONUNBUFFERED=1
# Speed: beam_size=1 is ~1.6x faster than 5 on CPU (minor accuracy cost).
# Set ANOT_WHISPER_BEAM_SIZE=5 in Railway for maximum accuracy.
# NOTE: set ANOT_ASR_API_KEY and ANOT_CORS_ORIGINS in the Railway dashboard.

EXPOSE 8000
# Railway injects $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
