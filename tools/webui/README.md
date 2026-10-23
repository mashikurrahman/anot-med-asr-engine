# Anot ASR Console (test UI)

A single-page web UI to test the deployed ASR engine: drag in a recording, get a
transcript with per-segment confidence.

## Run it (Windows)
Double-click **`serve.bat`** — it serves the page on `http://localhost:3000/` and
opens your browser.

Or manually:
```powershell
cd tools/webui
python -m http.server 3000
```
then open http://localhost:3000/

## Use it
1. **Service URL** is pre-filled with your Railway domain (editable).
2. Paste your **API key** (`ANOT_ASR_API_KEY`) — saved in this browser only.
3. The status dot turns green when the service is up and the model is loaded.
4. Drag a recording (`.m4a`, `.wav`, …), pick **mode** (conversation/dictation)
   and **engine** (whisper), click **Transcribe**.
5. See the transcript, timing, and a per-segment view (low-confidence segments
   are highlighted).

## Why serve on port 3000?
The API restricts browser requests (CORS) to `http://localhost:3000` by default.
Serving the page there lets the browser call the API. To use a different origin,
set `ANOT_CORS_ORIGINS` in Railway to that origin.

## Notes
- A ~1-minute clip takes ~1–1.5 min on CPU (batch, not live).
- Audio goes directly from your browser to your own service — nothing else sees it.
- This is a test tool, not the production frontend.
