#!/usr/bin/env python3
"""
Anot Medical ASR — Production Service Configuration
===================================================
Stage 11: Centralized, environment-driven configuration for the Anot ASR API.
"""

import os
import pathlib
from typing import Optional, Set
from pydantic import BaseModel, Field

# Base directories
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _load_env_file():
    """Load key-value pairs from .env if present into os.environ."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_file()


class Settings(BaseModel):
    """Production service settings with safe environment-variable overrides."""

    # Service metadata
    SERVICE_NAME: str = "anot-asr"
    VERSION: str = "1.1.0-console"
    ENVIRONMENT: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_ENV", "development"))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("ANOT_ASR_DEBUG", "false").lower() == "true")

    # Network & host (private / local only by default)
    HOST: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_HOST", "127.0.0.1"))
    PORT: int = Field(default_factory=lambda: int(os.getenv("ANOT_ASR_PORT", "8000")))
    API_PREFIX: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_API_PREFIX", "/api/v1"))

    # Hardware & Model
    DEVICE: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_DEVICE", "cpu").lower())
    MODEL_ID: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_MODEL_ID", "google/medasr"))

    # ASR engine selection.
    #   "whisper" (default, recommended): Whisper large-v3 via faster-whisper.
    #       Best accuracy on conversational patient-visit audio (validated ~97-98%
    #       clinical accuracy). Local, batch, no per-minute cost.
    #   "medasr": Google MedASR (CTC). Dictation-only; retained for pure doctor
    #       dictation where it is fast and competitive.
    # Repo default is "medasr" so the test suite runs offline without the large
    # Whisper weights; PRODUCTION should set ANOT_ASR_ENGINE=whisper (see
    # .env.example) for best accuracy on patient-visit audio.
    ASR_ENGINE: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_ENGINE", "medasr").lower())
    WHISPER_MODEL_SIZE: str = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_MODEL", "large-v3"))
    WHISPER_DEVICE: str = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_DEVICE", "cpu").lower())
    WHISPER_COMPUTE_TYPE: str = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_COMPUTE_TYPE", "int8"))
    # Beam size: 5 = most accurate, 1 = ~1.6x faster on CPU with minor accuracy cost.
    WHISPER_BEAM_SIZE: int = Field(default_factory=lambda: int(os.getenv("ANOT_WHISPER_BEAM_SIZE", "5")))
    # CPU threads for CTranslate2 (0 = auto-detect all cores). Set to the vCPU
    # count on constrained hosts to use all available cores.
    WHISPER_CPU_THREADS: int = Field(default_factory=lambda: int(os.getenv("ANOT_WHISPER_CPU_THREADS", "0")))
    # Language: "en" (default), a specific code like "bn" (Bengali), or "auto"
    # to let Whisper detect per recording. Can also be set per request.
    WHISPER_LANGUAGE: str = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_LANGUAGE", "en"))
    # Explicit clinical context prompt. Empty (default) => the multilingual phrase
    # library supplies a language-appropriate prompt automatically.
    WHISPER_INITIAL_PROMPT: str = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_INITIAL_PROMPT", ""))
    # Comma-separated domain terms to bias spelling (drugs, clinic terms). Mined
    # from local notes or set manually; empty string disables.
    # "true" = send no initial prompt / hotwords. The fine-tuned model was trained and
    # validated without them (prompting did not help it), so deploy it with this on.
    WHISPER_PLAIN: bool = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_PLAIN", "false").lower() == "true")
    # Fix high-confidence drug-name mishearings (albedo -> albuterol) after ASR.
    # Remove "uh"/"um" fillers from the final transcript (raw_transcript keeps them).
    STRIP_FILLERS: bool = Field(default_factory=lambda: os.getenv("ANOT_STRIP_FILLERS", "false").lower() == "true")
    # Turn spoken "period"/"comma"/"enter" dictation commands into punctuation / new lines.
    DICTATION_COMMANDS: bool = Field(default_factory=lambda: os.getenv("ANOT_DICTATION_COMMANDS", "false").lower() == "true")
    TERM_CORRECTION: bool = Field(default_factory=lambda: os.getenv("ANOT_TERM_CORRECTION", "false").lower() == "true")
    # Real per-word timestamps/probabilities (used by /v1/listen). Costs a little extra decode time.
    WHISPER_WORD_TIMESTAMPS: bool = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_WORD_TIMESTAMPS", "false").lower() == "true")
    # Replace non-English stretches (English-only engine) with a "[Non-English speech, m:ss-m:ss]"
    # marker instead of letting the decoder invent English for them. English-language runs only.
    NON_ENGLISH_MARK: bool = Field(default_factory=lambda: os.getenv("ANOT_NON_ENGLISH_MARK", "false").lower() == "true")
    # What to do with a stretch that is not English: "mark" leaves a placeholder, "translate" translates it into English (labelled).
    NON_ENGLISH_MODE: str = Field(default_factory=lambda: os.getenv("ANOT_NON_ENGLISH_MODE", "mark").lower())
    # Second pass over loud stretches the first pass left untranscribed (quiet / distant / child speech). Off by default.
    RECOVER_GAPS: bool = Field(default_factory=lambda: os.getenv("ANOT_RECOVER_GAPS", "false").lower() == "true")
    RECOVER_REL_LEVEL: float = Field(default_factory=lambda: float(os.getenv("ANOT_RECOVER_REL_LEVEL", "0.2")))
    LANG_WINDOW_SEC: float = Field(default_factory=lambda: float(os.getenv("ANOT_LANG_WINDOW_SEC", "15")))
    # Windows shorter than this are not language-checked (too little speech to judge); they inherit the neighbour.
    LANG_MIN_SEC: float = Field(default_factory=lambda: float(os.getenv("ANOT_LANG_MIN_SEC", "8")))
    LANG_EN_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("ANOT_LANG_EN_THRESHOLD", "0.4")))
    # SOAP note generation (Anthropic API). Disabled until ANTHROPIC_API_KEY is set. Transcripts are PHI:
    # use an Anthropic account covered by a BAA (and zero data retention if your policy requires it).
    ANTHROPIC_API_KEY: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    SOAP_MODEL: str = Field(default_factory=lambda: os.getenv("ANOT_SOAP_MODEL", "claude-haiku-5-5"))
    SOAP_MAX_TOKENS: int = Field(default_factory=lambda: int(os.getenv("ANOT_SOAP_MAX_TOKENS", "16000")))
    # Reasoning depth for the Claude 5.x models (ignored for older ones, which do not accept it).
    SOAP_EFFORT: str = Field(default_factory=lambda: os.getenv("ANOT_SOAP_EFFORT", "medium"))
    SOAP_TIMEOUT_SECONDS: float = Field(default_factory=lambda: float(os.getenv("ANOT_SOAP_TIMEOUT", "120")))
    # Batched GPU decoding (about 2x faster, same accuracy on the benchmark). Leave off on CPU hosts.
    WHISPER_BATCHED: bool = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_BATCHED", "false").lower() == "true")
    WHISPER_BATCH_SIZE: int = Field(default_factory=lambda: int(os.getenv("ANOT_WHISPER_BATCH_SIZE", "16")))
    # Speaker labels ("Speaker 1", "Speaker 2") in visit transcripts, from pyannote diarization. The model is
    # gated on Hugging Face: accept its terms once and provide a read token at image build time.
    DIARIZE: bool = Field(default_factory=lambda: os.getenv("ANOT_DIARIZE", "false").lower() == "true")
    DIARIZE_MODEL: str = Field(default_factory=lambda: os.getenv("ANOT_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"))
    DIARIZE_MAX_SPEAKERS: int = Field(default_factory=lambda: int(os.getenv("ANOT_DIARIZE_MAX_SPEAKERS", "0")))
    HF_TOKEN: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    WHISPER_HOTWORDS: str = Field(default_factory=lambda: os.getenv("ANOT_WHISPER_HOTWORDS", ""))
    # Anti-hallucination / silence-rejection thresholds (faster-whisper defaults
    # tightened slightly for clinical audio).
    WHISPER_NO_SPEECH_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("ANOT_WHISPER_NO_SPEECH_THRESHOLD", "0.6")))
    WHISPER_LOGPROB_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("ANOT_WHISPER_LOGPROB_THRESHOLD", "-1.0")))
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("ANOT_WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")))
    # Per-segment average log-prob below this is flagged low-confidence in output.
    WHISPER_LOW_CONFIDENCE_LOGPROB: float = Field(default_factory=lambda: float(os.getenv("ANOT_WHISPER_LOW_CONFIDENCE_LOGPROB", "-0.8")))

    # ---- Audio front-end enhancement (label-free accuracy lever) ----
    AUDIO_ENHANCE: bool = Field(default_factory=lambda: os.getenv("ANOT_AUDIO_ENHANCE", "true").lower() == "true")
    AUDIO_HIGHPASS: bool = Field(default_factory=lambda: os.getenv("ANOT_AUDIO_HIGHPASS", "true").lower() == "true")
    AUDIO_HIGHPASS_HZ: float = Field(default_factory=lambda: float(os.getenv("ANOT_AUDIO_HIGHPASS_HZ", "70.0")))
    AUDIO_LOUDNESS_NORM: bool = Field(default_factory=lambda: os.getenv("ANOT_AUDIO_LOUDNESS_NORM", "true").lower() == "true")
    AUDIO_TARGET_DBFS: float = Field(default_factory=lambda: float(os.getenv("ANOT_AUDIO_TARGET_DBFS", "-20.0")))
    AUDIO_DENOISE: bool = Field(default_factory=lambda: os.getenv("ANOT_AUDIO_DENOISE", "false").lower() == "true")
    # Windowed AGC for recordings whose level drifts (distant/moving mic). Off
    # until validated on the human-verified benchmark.
    AUDIO_GAIN_RIDE: bool = Field(default_factory=lambda: os.getenv("ANOT_AUDIO_GAIN_RIDE", "false").lower() == "true")

    # Default inference options
    MODE_DEFAULT: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_MODE_DEFAULT", "dictation"))
    DECODER_DEFAULT: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_DECODER_DEFAULT", "greedy"))
    BEAM_WIDTH_DEFAULT: int = Field(default_factory=lambda: int(os.getenv("ANOT_ASR_BEAM_WIDTH", "4")))
    LM_ENABLED_DEFAULT: bool = Field(default_factory=lambda: os.getenv("ANOT_ASR_LM_ENABLED", "false").lower() == "true")
    LM_WEIGHT_DEFAULT: float = Field(default_factory=lambda: float(os.getenv("ANOT_ASR_LM_WEIGHT", "0.05")))
    MEDICAL_WEIGHT_DEFAULT: float = Field(default_factory=lambda: float(os.getenv("ANOT_ASR_MEDICAL_WEIGHT", "0.05")))
    NORMALIZE_DEFAULT: bool = Field(default_factory=lambda: os.getenv("ANOT_ASR_NORMALIZE", "true").lower() == "true")

    # Resource Protection Limits
    MAX_UPLOAD_MB: int = Field(default_factory=lambda: int(os.getenv("ANOT_ASR_MAX_UPLOAD_MB", "500")))
    MAX_AUDIO_DURATION_SECONDS: float = Field(default_factory=lambda: float(os.getenv("ANOT_ASR_MAX_DURATION_SECONDS", "14400.0")))  # 4 hours
    MAX_CONCURRENT_JOBS: int = Field(default_factory=lambda: int(os.getenv("ANOT_ASR_MAX_CONCURRENT_JOBS", "2")))

    @property
    def MAX_DURATION_SECONDS(self) -> float:
        return self.MAX_AUDIO_DURATION_SECONDS

    @property
    def BEAM_WIDTH(self) -> int:
        return self.BEAM_WIDTH_DEFAULT

    @property
    def LM_ENABLED(self) -> bool:
        return self.LM_ENABLED_DEFAULT

    # Temporary Audio Storage
    TEMP_DIR: pathlib.Path = Field(default_factory=lambda: pathlib.Path(os.getenv("ANOT_ASR_TEMP_DIR", str(PROJECT_ROOT / "temp" / "asr_uploads"))))

    # Supported audio extensions
    SUPPORTED_EXTENSIONS: Set[str] = {".wav", ".m4a", ".mp3", ".flac", ".mp4", ".aac", ".ogg", ".webm", ".opus"}

    # Security & Auth boundary. Set ANOT_ASR_API_KEY to require an X-API-Key
    # header on transcription endpoints (strongly recommended in any deployment).
    # Extra keys for outside clients: "name:key[:daily_limit], name2:key2:50". Each can be revoked and limited on its own.
    CLIENT_KEYS: str = Field(default_factory=lambda: os.getenv("ANOT_ASR_CLIENT_KEYS", ""))
    API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("ANOT_ASR_API_KEY", None))

    # Allowed CORS origins (comma-separated), e.g. "https://app.example.com".
    CORS_ORIGINS: str = Field(default_factory=lambda: os.getenv("ANOT_CORS_ORIGINS", ""))

    @property
    def cors_origins_list(self) -> list:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


# Singleton settings instance
settings = Settings()
