# Anot Medical ASR — Medical Language Model Subsystem

**Stage**: 9B — Medical LM Rescoring Experiment  
**Status**: Engineering Validation (NOT production-ready)

## Architecture

```
CTC Acoustic Logits (MedASR LasrForCTC, 105M params)
        ↓
Native CTC Prefix Beam Search (decoding/beam_decoder.py)
        ↓
Subword-to-Word Bridge (SentencePiece ▁ boundary detection)
        ↓
ARPA n-gram Language Model Scoring (pure Python, no kenlm)
        ↓
Medical Lexicon Vocabulary Boost
        ↓
Combined Rescoring:
    combined = acoustic + lm_weight × lm_score + medical_weight × lexicon_score
        ↓
Final Transcript
```

## Modules

| File | Purpose |
|------|---------|
| `arpa_lm.py` | Pure Python ARPA n-gram reader with backoff weights and incremental scoring |
| `medical_lexicon.py` | Configurable medical vocabulary prior with category-weighted scoring |
| `lm_rescorer.py` | Subword-to-word bridge + combined LM/lexicon beam scorer + CLI |

## Subword Compatibility

MedASR uses a **SentencePiece Unigram tokenizer** (`LasrTokenizer`) with 613 tokens (512 base subwords + 100 extra + specials). Medical words are decomposed into character and subword pieces:

| Word | Token IDs | Pieces |
|------|-----------|--------|
| patient | [208, 8, 23, 75] | pa, t, i, ent |
| hypertension | [4, 70, 20, 202, 252, 5, 162] | ▁, h, y, per, ten, s, ion |
| lisinopril | [241, 5, 36, 13, 411, 42] | li, s, in, o, pri, l |
| blood pressure | [51, 128, 13, 15, 59, 340, 61, 10] | b, lo, o, d, p, ress, ur, e |

**Problem**: The ARPA LM operates at the **word level**, while the beam decoder operates at the **subword level**.

**Solution**: The `SubwordWordBridge` class accumulates subword tokens and detects word boundaries via the `▁` (U+2581) SentencePiece prefix. When a new `▁`-prefixed token arrives, the previous word is considered complete and scored by the ARPA LM.

## ARPA Language Model

### Built-in Small Medical ARPA

A hand-crafted trigram model for engineering validation:
- **Order**: 3 (trigram)
- **1-grams**: ~120 entries
- **2-grams**: ~85 entries
- **3-grams**: ~45 entries
- **Vocabulary**: Common clinical dictation patterns

This is NOT trained on patient data. It uses generic medical language patterns only.

### Integration with Google's KenLM Asset

The Google MedASR repository includes:
- `lm_6.kenlm` (704 MB, 6-gram)
- `lm_6.arpa.xz` (240 MB compressed ARPA)

**Current Status**: The local cache has an incomplete download (469 MB / 704 MB). The `kenlm` Python package requires C++ compilation on Windows. The pure Python `arpa_lm.py` reader can parse uncompressed ARPA format files when the asset becomes available.

## Medical Lexicon

12 categories covering ~300+ unique medical words:
- Medications (56 drugs)
- Dosage units
- Frequencies  
- Routes of administration
- Diagnoses
- Symptoms
- Anatomy
- Lab tests
- Vitals
- Procedures
- Common medical abbreviations

Automatically merges with `evaluation/medical_entities.json` for project-specific vocabulary.

## Scoring Formula

```
combined_score = acoustic_score + lm_weight × lm_score + medical_weight × lexicon_score
```

Where:
- `acoustic_score`: Raw CTC log probability from MedASR frame logits
- `lm_score`: ARPA n-gram log probability of the completed word in context
- `lexicon_score`: Medical vocabulary category-weighted boost (0 for non-medical words)

### Recommended Starting Weights

| Parameter | Default | Range |
|-----------|---------|-------|
| `lm_weight` | 0.1 | 0.05 – 0.50 |
| `medical_weight` | 0.05 | 0.01 – 0.20 |

**CRITICAL**: Keep weights low. Acoustic evidence must remain dominant. The LM is a rescorer, NOT a replacement for acoustic decoding.

## Usage

### Programmatic API

```python
from language_model.lm_rescorer import create_default_rescorer

rescorer = create_default_rescorer(
    tokenizer_or_processor=processor,
    lm_weight=0.1,
    medical_weight=0.05,
    beam_width=8
)

result = rescorer.decode(logits, processor, clean_special_tokens=True)
print(result["text"])
print(result["lm_config"])
```

### CLI

```bash
python -m language_model.lm_rescorer --info
python -m language_model.lm_rescorer --lm-weight 0.1 --medical-weight 0.05 --beam-width 8
```

## Safety

- The LM is a **rescorer**. It influences candidate ranking but does NOT invent clinical facts.
- Acoustic evidence remains dominant when weights are properly configured.
- The system does NOT perform clinical reasoning, diagnosis, or treatment recommendation.
- No patient data was used to construct the built-in ARPA model.
