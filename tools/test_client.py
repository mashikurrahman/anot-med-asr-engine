#!/usr/bin/env python3
"""
Anot ASR — deployed-engine test client
======================================
Point-and-shoot tester for the deployed Railway service. Sends one file or a
whole folder of recordings to /api/v1/transcribe and prints the transcript,
per-segment confidence, and timing. Also checks /health and /ready first.

Usage:
    python tools/test_client.py --url https://<service>.up.railway.app \
        --key <ANOT_ASR_API_KEY> --path "C:/path/to/audio_or_folder" \
        [--mode conversation] [--engine whisper] [--out results.jsonl]

Nothing is stored server-side beyond the request; treat any real audio as PHI.
"""

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".mp4", ".aac", ".ogg"}


def _get(url: str, key: str | None):
    req = urllib.request.Request(url, method="GET")
    if key:
        req.add_header("X-API-Key", key)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post_file(base: str, key: str | None, path: pathlib.Path, mode: str, engine: str):
    """Multipart upload using only the stdlib."""
    boundary = "----anot" + str(int(time.time() * 1000))
    fields = {"mode": mode, "engine": engine}
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n".encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(base.rstrip("/") + "/api/v1/transcribe", data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if key:
        req.add_header("X-API-Key", key)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read().decode("utf-8"))
    data["_client_seconds"] = round(time.perf_counter() - t0, 1)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL of the deployed service")
    ap.add_argument("--key", default=None, help="ANOT_ASR_API_KEY (if auth enabled)")
    ap.add_argument("--path", required=True, help="Audio file or folder")
    ap.add_argument("--mode", default="conversation", choices=["conversation", "dictation"])
    ap.add_argument("--engine", default="whisper", choices=["whisper", "medasr"])
    ap.add_argument("--out", default=None, help="Optional JSONL output path")
    args = ap.parse_args()

    # 1. Health / readiness
    try:
        hs, hb = _get(args.url.rstrip("/") + "/health", args.key)
        rs, rb = _get(args.url.rstrip("/") + "/ready", args.key)
        print(f"/health -> {hs} {hb}")
        print(f"/ready  -> {rs} {rb}")
    except urllib.error.HTTPError as e:
        print(f"Readiness check failed: {e.code} {e.read().decode('utf-8', 'ignore')[:200]}")
        if e.code == 503:
            print("Service is still loading the model — wait and retry.")
        sys.exit(1)
    except Exception as e:
        print(f"Cannot reach service: {e}")
        sys.exit(1)

    # 2. Collect files
    p = pathlib.Path(args.path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        print("No audio files found.")
        sys.exit(1)
    print(f"\nTranscribing {len(files)} file(s) via engine={args.engine}, mode={args.mode}...\n")

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None
    for f in files:
        try:
            res = _post_file(args.url, args.key, f, args.mode, args.engine)
        except urllib.error.HTTPError as e:
            print(f"[{f.name}] ERROR {e.code}: {e.read().decode('utf-8','ignore')[:200]}\n")
            continue
        conf = res.get("confidence", {})
        perf = res.get("performance", {})
        print(f"=== {f.name} ===")
        print(f"  lang={res.get('language')} dur={res.get('duration_seconds')}s "
              f"rtf={perf.get('rtf')} client={res.get('_client_seconds')}s "
              f"low_conf={conf.get('low_confidence_segments')}/{conf.get('segments_total')}")
        print(f"  {res.get('transcript','')[:600]}\n")
        if out_fh:
            out_fh.write(json.dumps({"file": f.name, **res}, ensure_ascii=False) + "\n")
    if out_fh:
        out_fh.close()
        print(f"Saved transcripts -> {args.out}")


if __name__ == "__main__":
    main()
