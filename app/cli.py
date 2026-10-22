#!/usr/bin/env python3
"""
Anot Medical ASR — Production Command Line Interface
====================================================
Stage 11: Local CLI invoking the same production service and pipelines as the API.
"""

import argparse
import asyncio
import json
import pathlib
import sys

from app.core.config import settings
from app.services.asr_service import ASRService


def cli_transcribe(args_list=None) -> int:
    """Programmatic entrypoint for transcribe command."""
    parser = argparse.ArgumentParser(description="Transcribe clinical audio")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--mode", "-m", type=str, default="dictation", choices=["dictation", "conversation"])
    parser.add_argument("--decoder", "-d", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam-width", "-b", type=int, default=4)
    parser.add_argument("--lm", action="store_true")
    parser.add_argument("--lm-weight", type=float, default=0.05)
    parser.add_argument("--medical-weight", type=float, default=0.05)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", "-o", type=str, default=None, help="Save result to JSON file")

    args = parser.parse_args(args_list)
    audio_path = pathlib.Path(args.audio_path).resolve()
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    asr = ASRService.get_instance()
    asr.initialize()

    result = asyncio.run(
        asr.transcribe_audio_file(
            audio_path=audio_path,
            mode=args.mode,
            decoder=args.decoder,
            beam_width=args.beam_width,
            language_model=args.lm,
            lm_weight=args.lm_weight,
            medical_weight=args.medical_weight,
            normalize=args.normalize
        )
    )

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    elif not args.out:
        print("\n" + "=" * 80)
        print(f"  ANOT MEDICAL ASR TRANSCRIPT [{result['mode'].upper()}]")
        print("=" * 80)
        print(f"Request ID: {result['request_id']}")
        print(f"Duration:   {result['duration_seconds']}s")
        print(f"Decoder:    {result['decoder']['type']}")
        print("-" * 80)
        print(result["transcript"])
        print("=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Anot Medical ASR — Production Service CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Transcribe command
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe a clinical audio file")
    trans_parser.add_argument("audio_path", type=str, help="Path to audio file (.wav, .m4a, .mp3, etc.)")
    trans_parser.add_argument("--mode", "-m", type=str, default="dictation", choices=["dictation", "conversation"], help="Workflow mode")
    trans_parser.add_argument("--decoder", "-d", type=str, default="greedy", choices=["greedy", "beam"], help="CTC decoder")
    trans_parser.add_argument("--beam-width", "-b", type=int, default=4, help="Beam width if beam decoder selected")
    trans_parser.add_argument("--lm", action="store_true", help="Enable Medical Language Model rescoring")
    trans_parser.add_argument("--lm-weight", type=float, default=0.05, help="LM prior weight")
    trans_parser.add_argument("--medical-weight", type=float, default=0.05, help="Medical lexicon boost weight")
    trans_parser.add_argument("--no-normalize", action="store_true", help="Disable clinical normalization")
    trans_parser.add_argument("--json", action="store_true", help="Output full structured JSON response")

    # Health command
    subparsers.add_parser("health", help="Check service and model status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    asr = ASRService.get_instance()

    if args.command == "health":
        print(f"Service: {settings.SERVICE_NAME} v{settings.VERSION}")
        print(f"Device:  {settings.DEVICE.upper()}")
        print(f"Model:   {settings.MODEL_ID}")
        print(f"Ready:   {asr.model.is_loaded()}")
        return

    if args.command == "transcribe":
        audio_path = pathlib.Path(args.audio_path).resolve()
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)

        # Initialize model once
        asr.initialize()

        # Run transcription asynchronously
        result = asyncio.run(
            asr.transcribe_audio_file(
                audio_path=audio_path,
                mode=args.mode,
                decoder=args.decoder,
                beam_width=args.beam_width,
                language_model=args.lm,
                lm_weight=args.lm_weight,
                medical_weight=args.medical_weight,
                normalize=not args.no_normalize
            )
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 80)
            print(f"  ANOT MEDICAL ASR TRANSCRIPT [{result['mode'].upper()}]")
            print("=" * 80)
            print(f"Request ID: {result['request_id']}")
            print(f"Duration:   {result['duration_seconds']}s")
            print(f"Decoder:    {result['decoder']['type']}")
            print("-" * 80)
            print(result["transcript"])
            print("=" * 80)


if __name__ == "__main__":
    main()
