#!/usr/bin/env python3
"""
Anot Medical ASR — Modal serverless GPU deployment
==================================================
Runs the SAME FastAPI engine on a serverless GPU (Whisper large-v3, float16).
~10-30x real-time: a 3-minute clip transcribes in seconds, at full large-v3
accuracy. Scales to zero when idle, so you pay only while transcribing.

Deploy:
    pip install modal
    modal setup
    modal secret create anot-asr ANOT_ASR_API_KEY=<your-long-secret>
    modal deploy modal_app.py

Then use the printed https URL in the console (tools/webui) or your app, with
the X-API-Key header set to the secret above.

Notes:
- GPU: T4 (16 GB) is plenty for large-v3 and the cheapest option here.
- faster-whisper uses CTranslate2 (CUDA + cuDNN 9); the base image provides them.
- torch/transformers are installed only to satisfy the engine's import chain
  (the MedASR path); Whisper itself does not use them.
"""

import modal

app = modal.App("anot-asr-gpu")

image = (
    # CUDA 12 + cuDNN 9 runtime (required by CTranslate2 / faster-whisper on GPU).
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "faster-whisper==1.2.1",
        "fastapi==0.115.6",
        "uvicorn[standard]==0.34.0",
        "python-multipart==0.0.20",
        "pydantic==2.10.4",
        "soundfile==0.13.1",
        "numpy>=1.26.0",
        "transformers==4.49.0",
    )
    # torch (CPU wheel) only to satisfy imports; Whisper runs on GPU via CTranslate2.
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
    # Pre-bake the model into the image (no runtime download / fast cold start).
    .run_commands(
        "python -c \"from faster_whisper import WhisperModel; "
        "WhisperModel('large-v3', device='cpu', compute_type='int8')\""
    )
    # GPU runtime configuration.
    .env({
        "ANOT_ASR_ENGINE": "whisper",
        "ANOT_ASR_ENV": "production",
        "ANOT_ASR_DEBUG": "false",
        "ANOT_WHISPER_MODEL": "large-v3",
        "ANOT_WHISPER_DEVICE": "cuda",
        "ANOT_WHISPER_COMPUTE_TYPE": "float16",
        "ANOT_WHISPER_BEAM_SIZE": "5",   # GPU is fast; use max accuracy
        "ANOT_AUDIO_ENHANCE": "true",
        "PYTHONUNBUFFERED": "1",
    })
    # Copy the engine source (research/data trees are excluded).
    .add_local_dir("app", remote_path="/root/app", ignore=["**/__pycache__"])
    .add_local_dir("audio", remote_path="/root/audio", ignore=["**/__pycache__"])
    .add_local_dir("decoding", remote_path="/root/decoding", ignore=["**/__pycache__"])
    # Exclude models/ — it holds the PHI-derived n-gram LM. The engine falls back
    # to the phrase-library hotwords (no clinic-mined terms), which is fine.
    .add_local_dir("language_model", remote_path="/root/language_model",
                   ignore=["models", "models/**", "**/__pycache__"])
    .add_local_dir("accuracy", remote_path="/root/accuracy", ignore=["**/__pycache__"])
    .add_local_file("evaluation/evaluate_transcript.py", "/root/evaluation/evaluate_transcript.py")
    .add_local_file("evaluation/medical_entities.json", "/root/evaluation/medical_entities.json")
)


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=300,          # keep warm 5 min after last request, then scale to 0
    timeout=60 * 30,              # allow long recordings
    secrets=[modal.Secret.from_name("anot-asr")],  # provides ANOT_ASR_API_KEY (+ optional CORS)
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from app.main import app as fastapi_engine  # lifespan loads Whisper on GPU
    return fastapi_engine
