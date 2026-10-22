# Deploying the Anot ASR Engine (Railway, test/beta)

A working, secured **English** transcription service you can test with real audio.
(Bangla and formal accuracy certification are out of scope for this test deploy.)

## What this deploys
- FastAPI service, **Whisper engine** (faster-whisper, CPU).
- Model **baked into the image** (no multi-GB runtime download).
- Default model **`distil-large-v3`** (~2 GB RAM, ~2× faster) — fits Railway
  **Hobby (≤ 8 GB RAM)** comfortably. Switch to `large-v3` by setting the build
  arg / `ANOT_WHISPER_MODEL=large-v3` and redeploying (also fits 8 GB).

## Steps

1. **Create a Railway project** from the private repo
   `mashikurrahman/anot-med-asr-engine` → "Deploy from GitHub repo". Railway
   auto-detects `Dockerfile` / `railway.toml`.

2. **Set environment variables** (Railway → Variables):
   - `ANOT_ASR_API_KEY` = a long random secret (required to lock the API)
   - `ANOT_CORS_ORIGINS` = `https://<your-web-app-domain>`
   - (optional) `ANOT_WHISPER_MODEL` = `large-v3` for max accuracy
   - (optional) `ANOT_AUDIO_DENOISE` = `false` (keep off; it hurt accuracy in tests)
   `ANOT_ASR_ENGINE=whisper` and host/port are already set in the image.

3. **Deploy.** First build is slow (bakes the model). Healthcheck is `/ready`;
   it turns green once the model is loaded in memory.

4. **Smoke-test** (replace URL + key):
   ```bash
   curl -sS https://<your-service>.up.railway.app/health
   curl -sS -X POST https://<your-service>.up.railway.app/api/v1/transcribe \
     -H "X-API-Key: <your-key>" \
     -F "file=@visit.m4a" -F "engine=whisper" -F "mode=conversation"
   ```
   Response includes `transcript`, per-segment `confidence`, and timings.

5. **Wire into your web app**: swap the Deepgram call for
   `POST /api/v1/transcribe` (send the `X-API-Key` header), take `transcript`,
   and pass it to Claude Haiku for the SOAP note exactly as today.

## Expectations / limits (Hobby)
- CPU transcription ≈ **1.3× real-time** (a 30-min visit ≈ ~20–40 min). Batch,
  not live. `ANOT_ASR_MAX_CONCURRENT_JOBS=1` by default — raise only if RAM allows.
- The service loads the model once and holds ~2–4 GB RAM continuously; on Hobby's
  usage billing, stop/sleep the service when not testing to control cost.
- It is still PHI: the API key + HTTPS are the minimum. Full HIPAA/compliance
  (audit logs, BAAs, retention) is separate, pre-production work.

## Troubleshooting
- **Uploads fail / "ffmpeg not found"**: the image installs ffmpeg; if you build
  a custom image, keep it.
- **OOM / killed on boot**: you're on `large-v3` with too little RAM — set
  `ANOT_WHISPER_MODEL=distil-large-v3` and redeploy.
- **401 Unauthorized**: send the `X-API-Key` header matching `ANOT_ASR_API_KEY`.
