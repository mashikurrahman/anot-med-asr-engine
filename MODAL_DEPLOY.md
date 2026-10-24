# Deploy the engine on a serverless GPU (Modal)

Runs Whisper **large-v3 on GPU** — ~10–30× real-time (a 3-min clip in seconds),
full accuracy, and **scales to zero** so you pay only while transcribing.

## One-time setup
1. Create a free account at **modal.com** (comes with monthly credits).
2. Install + authenticate the CLI (run from this repo folder):
   ```bash
   pip install modal
   modal setup
   ```
3. Create your API-key secret (choose a long random value — this is what the
   `X-API-Key` header must match):
   ```bash
   modal secret create anot-asr ANOT_ASR_API_KEY=your-long-random-secret
   ```
   *(Optional: add CORS for a browser UI by appending `ANOT_CORS_ORIGINS=https://your-app`.)*

## Deploy
```bash
modal deploy modal_app.py
```
Modal prints a public URL like `https://<you>--anot-asr-gpu-fastapi-app.modal.run`.

## Use it
Exactly like the Railway service — same API:
```bash
curl -sS https://<your-modal-url>/health
curl -sS -X POST https://<your-modal-url>/api/v1/transcribe \
  -H "X-API-Key: your-long-random-secret" \
  -F "file=@visit.m4a" -F "engine=whisper" -F "mode=conversation"
```
Or paste the Modal URL into the web console (tools/webui → "Advanced: service URL").

## What to expect
- **Speed:** a 3-min clip transcribes in ~5–15 s (vs ~3–6 min on Railway CPU).
- **Accuracy:** full `large-v3`, beam 5 (GPU makes beam 5 essentially free).
- **Cost:** pay per GPU-second only while transcribing; idle = ~$0 (scales to 0).
  A T4 runs large-v3 fast and is the cheapest option here.
- **Cold start:** the first request after idle spins up a container (~10–30 s);
  warm requests are instant. The model is baked into the image, so no download.

## Tuning (in modal_app.py)
- `gpu="T4"` → `"L4"` or `"A10G"` for more speed on long files.
- `scaledown_window=300` → how long to stay warm after the last request.
- `ANOT_WHISPER_BEAM_SIZE` → already 5 (max accuracy; cheap on GPU).

## Troubleshooting
- **cuDNN/CUDA errors on first deploy:** CTranslate2 4.8 needs CUDA 12 + cuDNN 9,
  which the base image provides. If a future faster-whisper changes this, bump the
  `nvidia/cuda:...-cudnn9-runtime` tag to match.
- **401:** the `X-API-Key` header must equal the `ANOT_ASR_API_KEY` secret value.
