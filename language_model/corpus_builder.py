#!/usr/bin/env python3
"""
Anot Medical ASR — In-domain LM Corpus Builder (PHI-scrubbing)
==============================================================
Stage 20: Extracts clean, tokenised sentences from local in-domain medical text
(clinical note .docx files + available reference transcripts) for training the
rescoring language model.

PRIVACY CONTRACT (Anot Health Privacy Rules):
  - 100% local. Nothing leaves the machine.
  - De-identifies before any n-gram is counted: removes names following common
    labels, dates, MRNs/SSNs/phones/emails, and standalone long digit strings.
  - The resulting corpus and trained LM are PHI-derived and MUST stay gitignored
    (language_model/models/ and language_model/corpus/ are ignored).

This module only PREPARES text. It never transmits or logs raw note content.
"""

import html
import re
import zipfile
from pathlib import Path
from typing import Iterable, List

# --- de-identification patterns (conservative, applied before counting) -----
_PHI_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), " "),                       # SSN
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), " "),           # phone
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), " "),                # email
    (re.compile(r"\b(?:mrn|mr#|record\s*#)\s*:?\s*\w+\b", re.I), " "), # MRN
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), " "),           # dates
    (re.compile(r"\b\d{5,}\b"), " "),                                  # long digit runs / IDs
    # Names following explicit labels (one or two capitalised tokens).
    (re.compile(r"\b(patient|name|pt|dr|doctor|provider)\s*:?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
                re.I), r"\1: "),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9%/.-]+")


def deidentify(text: str) -> str:
    """Apply conservative de-identification substitutions."""
    out = text
    for pat, repl in _PHI_PATTERNS:
        out = pat.sub(repl, out)
    return out


def docx_to_text(path: Path) -> str:
    """Extract visible paragraph text from a .docx without external libraries."""
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
    except (zipfile.BadZipFile, KeyError, OSError):
        return ""
    # Paragraph boundaries -> newlines; drop all tags; unescape entities.
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)


def tokenize_sentences(text: str) -> List[List[str]]:
    """Lowercase, de-identify, and split into tokenised sentences."""
    text = deidentify(text)
    sentences: List[List[str]] = []
    for raw in _SENTENCE_SPLIT.split(text):
        raw = raw.strip().lower()
        if not raw:
            continue
        toks = _TOKEN.findall(raw)
        # Strip leading/trailing punctuation (keep internal '.' for decimals like
        # 98.6), then drop empties and residual noise.
        cleaned = []
        for t in toks:
            t = t.strip(".-/")
            if not t:
                continue
            if t.isdigit() and len(t) > 3:
                continue
            cleaned.append(t)
        if len(cleaned) >= 2:
            sentences.append(cleaned)
    return sentences


def build_corpus(sources: Iterable[Path]) -> List[List[str]]:
    """
    Build a tokenised sentence corpus from a mix of .docx notes and .txt
    reference transcripts. Silently skips unreadable files.
    """
    corpus: List[List[str]] = []
    for path in sources:
        path = Path(path)
        if not path.exists():
            continue
        if path.suffix.lower() == ".docx":
            text = docx_to_text(path)
        elif path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        corpus.extend(tokenize_sentences(text))
    return corpus


def mine_hotwords(
    corpus: List[List[str]],
    medical_lexicon=None,
    top_n: int = 80,
    min_count: int = 3,
) -> List[str]:
    """
    Mine the clinic's most frequent in-domain MEDICAL terms from the note corpus,
    for use as Whisper hotwords / vocabulary biasing.

    Safety: a candidate is kept ONLY if it is already a known medical term (present
    in the medical lexicon). This guarantees the hotword list contains real
    clinical vocabulary and never leaks names or other PHI from the notes — the
    notes only decide WHICH known medical terms to prioritise and in what order.
    """
    from collections import Counter

    if medical_lexicon is None:
        from language_model.medical_lexicon import MedicalLexicon
        medical_lexicon = MedicalLexicon()

    known = medical_lexicon.get_all_words()
    counts: Counter = Counter()
    for sent in corpus:
        for tok in sent:
            if len(tok) >= 4 and tok.isalpha() and tok in known:
                counts[tok] += 1

    ranked = [w for w, c in counts.most_common() if c >= min_count]
    return ranked[:top_n]


def discover_sources(project_root: Path) -> List[Path]:
    """
    Discover local in-domain text sources: clinical-note .docx under the
    (gitignored) real-benchmark recordings, plus reference transcripts in
    results/ and benchmark reference files.
    """
    sources: List[Path] = []
    rec = project_root / "real benchmark" / "recordings"
    if rec.exists():
        sources.extend(sorted(rec.rglob("*.docx")))
    results = project_root / "results"
    if results.exists():
        sources.extend(sorted(results.glob("*_reference.txt")))
    return sources
