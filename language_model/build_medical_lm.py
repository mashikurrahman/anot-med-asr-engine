#!/usr/bin/env python3
"""
Anot Medical ASR — Build the in-domain medical rescoring LM
===========================================================
Stage 20 CLI: discover local in-domain text -> de-identify -> train an
interpolated Witten-Bell n-gram LM -> write ARPA to language_model/models/.

The output ARPA is PHI-derived and stays gitignored. Run locally:

    python language_model/build_medical_lm.py --order 3 --min-count 2
"""

import argparse
import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from language_model.corpus_builder import build_corpus, discover_sources, mine_hotwords
from language_model.medical_lexicon import MedicalLexicon
from language_model.ngram_trainer import NGramTrainer

DEFAULT_OUT = _PROJECT_ROOT / "language_model" / "models" / "medical.arpa"
DEFAULT_HOTWORDS = _PROJECT_ROOT / "language_model" / "models" / "medical_hotwords.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Anot in-domain medical rescoring LM.")
    ap.add_argument("--order", type=int, default=3, help="n-gram order (default 3)")
    ap.add_argument("--min-count", type=int, default=2, help="min count for orders>=2 (default 2)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="output ARPA path")
    args = ap.parse_args()

    sources = discover_sources(_PROJECT_ROOT)
    print(f"Discovered {len(sources)} in-domain text source(s).")
    corpus = build_corpus(sources)
    total_tokens = sum(len(s) for s in corpus)
    print(f"Corpus: {len(corpus)} sentences, {total_tokens} tokens (de-identified).")

    if not corpus:
        print("No corpus text found. Ensure clinical notes / references are present locally.")
        sys.exit(1)

    trainer = NGramTrainer(order=args.order, min_count=args.min_count).fit(corpus)
    print("Model summary:", trainer.summary())

    arpa = trainer.to_arpa()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(arpa, encoding="utf-8")
    print(f"[ARPA written] {out_path}  ({len(arpa)} bytes)")

    # Mine clinic-specific medical hotwords (lexicon-validated; PHI-safe).
    lexicon = MedicalLexicon(entities_json_path=str(
        _PROJECT_ROOT / "evaluation" / "medical_entities.json"))
    hotwords = mine_hotwords(corpus, medical_lexicon=lexicon, top_n=80)
    hw_path = pathlib.Path(DEFAULT_HOTWORDS)
    hw_path.write_text(" ".join(hotwords), encoding="utf-8")
    print(f"[Hotwords written] {hw_path}  ({len(hotwords)} terms)")
    print(f"  top terms: {', '.join(hotwords[:15])}")
    print("NOTE: these artifacts are PHI-derived and gitignored. Keep them local.")


if __name__ == "__main__":
    main()
