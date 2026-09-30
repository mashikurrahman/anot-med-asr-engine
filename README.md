# Anot Medical ASR

A local, privacy-first medical speech-to-text system and clinical evaluation framework developed for **Anot Health**, powered by Google's **MedASR** (`google/medasr`).

---

## Overview

Anot Medical ASR enables on-device medical dictation and transcription without sending sensitive patient audio to external cloud providers. The project includes:
- **Local MedASR Inference**: Conformer-based medical ASR (105M parameters) running locally on Intel CPU (and validated on Intel Arc GPU).
- **CPU vs. XPU Benchmarks**: Performance profiling demonstrating real-time factor (RTF) of ~0.061 on Intel Core i7-13700H (16x faster than real-time).
- **Anot Medical ASR Evaluation Engine**: A multi-dimensional clinical transcription evaluation framework that analyzes normalized Word Error Rate (WER), clinical entity accuracy, dosage equivalence, critical numeric entities, and acoustic disfluencies.

---

## Hardware & Environment Specifications

- **OS**: Windows 11
- **Python**: 3.12.10
- **PyTorch**: 2.14.1+xpu
- **Transformers**: 5.18.0
- **Model**: `google/medasr` (`LasrForCTC` / `LasrProcessor`, 16 kHz mono)
- **CPU**: Intel Core i7-13700H
- **GPU**: Intel Arc A370M (4 GB VRAM)

---

## Project Structure

```text
Anot med asr/
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
├── evaluation/                    # Clinical Evaluation Engine
│   ├── README.md                  # Comprehensive evaluation documentation
│   ├── evaluate_transcript.py     # Evaluation CLI and core analysis engine
│   ├── medical_entities.json      # Configurable medical entity vocabulary & mappings
│   └── reports/                   # Evaluation reports & JSON metrics
│       ├── anot_test_01_evaluation_report.txt
│       └── anot_test_01_metrics.json
├── results/                       # Test results and reference transcripts
│   ├── anot_test_01_raw.txt       # Raw MedASR output for Test 01
│   ├── anot_test_01_report.txt    # Inference performance report
│   ├── anot_test_01_evaluation.txt# Initial manual/rule-based evaluation
│   ├── anot_test_01_reference.txt # Ground truth transcript for Test 01
│   └── anot_test_02_reference.txt # Reference transcript for Test 02
├── test audio/                    # Sample clinical audio recordings
│   └── Recording (2).m4a          # 62.36s clinical dictation
└── tests/                         # Test suites and benchmarking scripts
    ├── benchmark_cpu_vs_xpu.py    # Hardware benchmark (CPU vs. Intel Arc A370M)
    ├── benchmark_results.json     # Benchmark metrics
    ├── run_stage6_evaluation.py   # Stage 6 execution pipeline
    ├── test_evaluation.py         # Unit and integration tests (10/10 passing)
    └── test_medasr_xpu.py         # XPU device validation script
```

---

## Quick Start

### 1. Run Clinical Evaluation
Compare an ASR hypothesis against a reference transcript:
```bash
python evaluation/evaluate_transcript.py \
  --reference results/anot_test_01_reference.txt \
  --hypothesis results/anot_test_01_raw.txt \
  --output evaluation/reports/anot_test_01_evaluation_report.txt \
  --json-output evaluation/reports/anot_test_01_metrics.json
```

### 2. Run Test Suite
```bash
python -m unittest tests/test_evaluation.py -v
```

### 3. Run CPU vs. GPU Benchmark
```bash
python tests/benchmark_cpu_vs_xpu.py
```

---

## Benchmark Highlights

| Metric | CPU (Intel Core i7-13700H) | GPU (Intel Arc A370M XPU) |
|---|---|---|
| **Audio Duration** | 43.80 s | 43.80 s |
| **Inference Time** | **2.81 s** | 8.74 s |
| **Real-Time Factor (RTF)** | **0.0641** | 0.1995 |
| **Throughput** | **15.6x faster than real-time** | 5.0x faster than real-time |
| **Output Parity** | 100% Identical | 100% Identical |

*Note: For single-stream CTC inference, CPU outperforms mobile Arc GPU due to lower kernel launch and memory transfer overhead.*

---

## Evaluation Engine Features

- **No Single Aggregate Score**: Separates Word Error Rate, clinical entity accuracy, and numeric fidelity to avoid misleading safety claims.
- **Dosage Unit Equivalence**: Recognizes `500 milligrams` and `500 mg` as identical, preventing false dosage penalties.
- **Critical Numeric Mismatches**: Flags vital sign and patient age errors separately from lexical substitutions.
- **False-Start / Stutter Handling**: Preserves and flags acoustic disfluencies (e.g., `isupro- lisinopril`) without penalizing valid medications.

---

## Disclaimer

This software is for preliminary development, benchmarking, and research purposes only. It is **not** clinically certified, approved by regulatory authorities, or intended for unmonitored diagnostic or therapeutic decision-making.
