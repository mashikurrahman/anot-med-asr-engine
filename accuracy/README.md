# Anot Medical ASR — Accuracy Comparison Framework

This directory contains the clinical accuracy benchmarking and differential comparison subsystem.

## 1. Modules

- `metrics.py`: Computes Word Error Rate (WER), character-level alignment, and stratifies clinical errors across:
  - Patient demographics (age, gender)
  - Medication regimens (omissions, additions, dosage number discrepancies, frequency differences)
  - Vital signs (blood pressure, pulse, SpO2, temperature)
  - Medical terminology (diagnoses, symptoms, procedures, anatomy, labs)
  - Clinical duration and temporal references
- `compare_runs.py`: Produces side-by-side comparison tables between decoding strategies, model versions, or external ASR baselines without computing misleading single-score aggregations.

## 2. Reusability

Designed for comparing:
1. Current MedASR Greedy decoding vs Beam Search
2. Beam Search without LM vs Beam Search with Medical LM
3. Baseline MedASR vs Future Fine-Tuned MedASR checkpoints
4. MedASR vs Alternative Local Speech Engines
