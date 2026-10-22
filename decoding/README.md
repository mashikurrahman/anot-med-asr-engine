# Anot Medical ASR — CTC Decoding Subsystem

This package provides deterministic CTC decoding engines for Google's MedASR (`LasrForCTC`) acoustic model.

## 1. Modules

- `greedy_decoder.py`: Pure frame-level argmax CTC decoding with duplicate collapse and blank removal. Mirrors the authoritative production baseline.
- `beam_decoder.py`: Pure Python/NumPy CTC Prefix Beam Search (Graves et al., Hannun et al.) with configurable beam widths, probability pruning, temperature scaling, and language model rescoring interface.

## 2. Mathematical Formulation

### Greedy CTC Decoding
At each time frame $t \in [1, T]$:
$$\pi_t = \arg\max_{k \in \mathcal{V}} P(k | x_t)$$
The output transcript $\mathbf{y} = \mathcal{B}(\boldsymbol{\pi})$ collapses consecutive identical tokens and drops the blank token $\epsilon$ (index 0).

### CTC Prefix Beam Search
Prefix probability tracking decomposes prefix $\mathbf{p}$ into paths ending in blank $P_b(\mathbf{p}, t)$ and paths ending in non-blank $P_{nb}(\mathbf{p}, t)$:
- Total probability: $P(\mathbf{p}, t) = P_b(\mathbf{p}, t) + P_{nb}(\mathbf{p}, t)$
- Blank extension: $P_b(\mathbf{p}, t) = P(\mathbf{p}, t-1) \cdot P(\epsilon | x_t)$
- Non-blank extension by label $c$:
  - If $c = \mathbf{p}_{|\mathbf{p}|}$:
    - $P_{nb}(\mathbf{p}, t) = P_{nb}(\mathbf{p}, t-1) \cdot P(c | x_t)$ (collapse)
    - $P_{nb}(\mathbf{p} \cdot c, t) = P_b(\mathbf{p}, t-1) \cdot P(c | x_t)$ (distinct repeated symbol)
  - If $c \neq \mathbf{p}_{|\mathbf{p}|}$:
    - $P_{nb}(\mathbf{p} \cdot c, t) = P(\mathbf{p}, t-1) \cdot P(c | x_t)$

Pruning keeps the top $B$ highest total probability prefixes at each time step.

## 3. Environment & Compatibility Audit (Stage 9 Investigation)

1. **KenLM / PyCTCDecode Evaluation**:
   - Google's published demonstration notebook references `kenlm==0.3.0` and a fork of `pyctcdecode`.
   - On Windows 11 with Python 3.12, `pyctcdecode` on PyPI enforces `numpy<2.0.0, >=1.15.0`.
   - The active project environment relies on `numpy 2.5.3` and `torch 2.14.1+xpu` (Intel SYCL/oneMKL). Forcing a downgrade to NumPy 1.x creates severe binary incompatibility hazards with Intel XPU acceleration.
   - The remote Hugging Face repository provides `lm_6.kenlm` (704 MB) and `lm_6.arpa.xz` (240 MB). The local cache contains an interrupted download (`.incomplete` 469 MB), requiring offline evaluation.

2. **Native Implementation Strategy**:
   - `beam_decoder.py` implements CTC prefix beam search natively in pure Python/NumPy, guaranteeing 100% platform portability across Windows/Linux/macOS without external C++ compilation or package downgrade hazards.
