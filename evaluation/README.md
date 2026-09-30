# Anot Medical ASR Evaluation Engine

A modular evaluation framework designed for **Anot Health** to benchmark local medical speech-to-text transcriptions (such as Google's MedASR) against ground-truth clinical reference transcripts.

---

## 1. Overview & Objectives

In clinical documentation, transcription errors are not equal. While standard speech recognition systems rely exclusively on **Word Error Rate (WER)**, evaluating medical speech requires granular analysis of clinical entities, critical numbers, medication dosages, and acoustic disfluencies.

The **Anot Medical ASR Evaluation Engine** provides:
1. **Standard General ASR Metrics** (WER, Substitutions, Deletions, Insertions).
2. **Medical Terminology Extraction & Comparison** (Diagnoses, Symptoms, Procedures, Anatomy, Lab Tests, Vitals).
3. **Critical Numeric Entity Comparison** (Patient age, blood pressure, pulse, O2 saturation, temperature, weight, durations).
4. **Structured Medication Evaluation** (Drug name, dosage amount, unit equivalence, frequency, route).
5. **Acoustic False-Start & Stutter Analysis** (Recognizing speaker disfluencies like `isupro- lisinopril` or `ator- atorvastatin`).
6. **Potentially Clinically Significant Mismatches** (Demographic, dosage, or omission errors).
7. **Strict Separation of Metrics** (Deliberately avoids single composite "accuracy scores" in favor of transparent, measurable dimensions).

---

## 2. Directory Structure

```text
evaluation/
├── README.md                  # Comprehensive engine documentation
├── evaluate_transcript.py     # Main CLI evaluation engine
├── medical_entities.json      # Configurable medical vocabulary & normalization rules
└── reports/                   # Destination directory for evaluation reports
    ├── anot_test_01_evaluation_report.txt
    └── anot_test_01_metrics.json

tests/
└── test_evaluation.py         # Unit and integration test suite
```

---

## 3. How to Run

### Command-Line Interface (CLI)

Run the evaluation engine against a reference transcript and an ASR hypothesis transcript:

```bash
python evaluation/evaluate_transcript.py \
  --reference results/anot_test_01_reference.txt \
  --hypothesis results/anot_test_01_raw.txt \
  --output evaluation/reports/anot_test_01_evaluation_report.txt
```

### Command Options

- `--reference`, `-r` *(Required)*: Path to the plain-text reference transcript.
- `--hypothesis`, `-y` *(Required)*: Path to the raw ASR output transcript.
- `--output`, `-o` *(Optional)*: File path to save the formatted text report. If omitted, the report is printed to standard output.
- `--entities`, `-e` *(Optional)*: Path to a custom `medical_entities.json` vocabulary file (defaults to `evaluation/medical_entities.json`).
- `--json-output`, `-j` *(Optional)*: Path to export structured metrics as machine-readable JSON.

### Running Automated Tests

Run the test suite via Python's `unittest`:

```bash
python -m unittest tests/test_evaluation.py -v
```

---

## 4. Metric Definitions

### 1. General ASR Metrics (Word Error Rate & Levenshtein Alignment)
- **Reference Word Count ($N_{ref}$)**: Total words in normalized reference.
- **Hypothesis Word Count ($N_{hyp}$)**: Total words in normalized hypothesis.
- **Substitutions ($S$)**: Count of words in reference replaced by different words in hypothesis.
- **Deletions ($D$)**: Count of words in reference omitted in hypothesis.
- **Insertions ($I$)**: Count of words in hypothesis not present in reference.
- **Word Error Rate (WER)**: Standard formula:
  $$\text{WER} = \frac{S + D + I}{N_{ref}}$$
- **Preprocessing / Normalization**: Lowercases text, strips CTC tokens (e.g. `</s>`, `{comma}`), removes punctuation, normalizes whitespace, and normalizes common spoken number words (e.g. `two` $\rightarrow$ `2`).

### 2. Medical Terminology Comparison
Cross-references recognized words against configured categories in `evaluation/medical_entities.json`:
- **Diagnoses** (e.g. `hypertension`, `type 2 diabetes mellitus`)
- **Symptoms** (e.g. `persistent cough`, `shortness of breath`, `wheezing`, `crackles`)
- **Procedures** (e.g. `chest x-ray`)
- **Anatomy** (e.g. `lungs`, `chest`, `bilaterally`)
- **Laboratory Tests** (e.g. `complete blood count`)
- **Vital Signs** (e.g. `blood pressure`, `pulse`, `oxygen saturation`)

*Entities are parsed using longest-match first to avoid redundant nested sub-matches (e.g., `persistent cough` supersedes standalone `cough`).*

### 3. Medication Evaluation & Unit Equivalence
For each detected medication, the engine extracts:
- **Drug Name**: Validated against pharmaceutical list.
- **Dosage Amount**: Numeric value (e.g., `500`, `10`, `20`).
- **Dosage Unit**: Normalized into canonical forms using `dosage_units` mapping:
  - `milligrams` / `milligram` / `mg` $\rightarrow$ canonical `mg`
  - `micrograms` / `microgram` / `mcg` $\rightarrow$ canonical `mcg`
  - `milliliters` / `milliliter` / `ml` $\rightarrow$ canonical `ml`
- **Dosage Comparison**:
  - **Equivalent Dosage Notation**: When the numerical value and canonical unit match, but lexical representation differs (e.g., `500 milligrams` vs `500 mg`), it is classified as an **Equivalent Dosage Notation**, NOT a dosage error.
  - **Critical Dosage Mismatch**: When the numeric dosage differs (e.g., `500 mg` vs `250 mg`), it is flagged as a high-risk error.
- **Frequency**: Normalized using clinical mappings (e.g., `twice daily` vs `bid`, `at bedtime` vs `qhs`).

### 4. Critical Numeric Entity Comparison
Extracts and isolates numerical values across distinct clinical categories:
- **Patient Age**: e.g., `58-year-old` vs `50-year-old`. Discrepancies are flagged as **`CRITICAL NUMERIC MISMATCH (patient_age)`**.
- **Blood Pressure**: Normalizes `138 over 84` and `138/84` into systolic `138` and diastolic `84`. Classified as **Equivalent Numeric Notation** when values match.
- **Pulse**: Evaluates numeric beats per minute (`78 bpm`).
- **Oxygen Saturation**: Normalizes `97 percent` and `97%` into value `97`. Classified as **Equivalent Numeric Notation** when values match.
- **Clinical Durations**: Distinguishes clinical timelines (e.g. `three-week history`, `follow up in two weeks`) from dates or patient ages.

### 5. False-Start / Stutter Acoustic Disfluency Analysis
Identifies speaker disfluencies where words were interrupted, stuttered, or abandoned:
- **Abandoned False Starts**: e.g., `isupro- lisinopril` (speaker initiated a wrong syllable before self-correcting to Lisinopril).
- **Stutter Repetitions**: e.g., `ator- atorvastatin` (speaker stuttered initial syllable before completing Atorvastatin).
- **Clinical Impact**: Identifying these artifacts prevents the evaluation engine from penalizing the subsequent medication or falsely classifying the fragment as an unrecognized or hallucinated drug.

### 6. Potentially Clinically Significant Mismatches
Filters errors by clinical severity:
- **CRITICAL**: Medication omission, dosage number alteration, wrong drug name.
- **HIGH**: Patient demographic mismatch (age), vital sign alteration, omitted diagnosis.
- **MEDIUM**: Dosing frequency discrepancy, non-critical clinical duration discrepancy.

---

## 5. Why WER Alone Is Insufficient for Medical ASR

Word Error Rate (WER) is the de-facto metric in general speech recognition, but relying on WER alone in clinical environments is fundamentally flawed:

1. **Equal Weighting of Unequal Words**:
   - In standard WER, transcribing *"the patient"* as *"a patient"* is penalized as **1 substitution** (1 error).
   - Transcribing *"58-year-old"* as *"50-year-old"* is also penalized as **1 substitution** (1 error).
   - In clinical reality, confusing patient age can invalidate patient identification, insurance records, or age-dependent dosing protocols, while swapping "the" and "a" is clinically harmless.
2. **Penalizing Standard Clinical Shorthand**:
   - An ASR model transcribing *"500 milligrams"* as *"500 mg"* receives a WER penalty of **1 substitution** (or deletion/insertion depending on tokenization), despite *"500 mg"* being the preferred clinical standard in electronic health records (EHRs).
   - Similarly, transcribing *"138 over 84"* as *"138/84"* incurs a WER penalty even though *"138/84"* is the exact notation desired in clinical charts.
3. **Obscuring Catastrophic Dosage Errors**:
   - Transcribing *"metformin 500 mg"* as *"metformin 50 mg"* yields a tiny ~1% change in document WER, but represents a 10-fold underdose that could lead to severe hyperglycemia.
4. **False-Start Misinterpretation**:
   - Natural spoken clinician dictations contain false starts and self-corrections (e.g. *"isupro- lisinopril"*). Traditional scoring treats *"isupro-"* as a hallucinated drug or an uncorrected error; clinical evaluation must contextualize it as a captured disfluency.

---

## 6. Design Philosophy: No Single "Accuracy Score"

The engine intentionally does **NOT** produce a single aggregate "overall accuracy score" (e.g. "94% accurate"). In clinical medicine, aggregate percentages create a dangerous illusion of safety:
- A transcript with "99% accuracy" that gets a single medication dosage wrong (e.g. 10x overdose) is clinically catastrophic.
- Separate, measurable, categorized dimensions ensure clinicians and ML engineers can independently inspect terminology recall, numeric fidelity, and formatting equivalents.

---

## 7. Limitations & Clinical Disclaimer

- **Development Benchmark Only**: This framework is intended for engineering evaluation and development benchmarking.
- **Vocabulary Scope**: The vocabulary in `medical_entities.json` contains representative common medications, conditions, and procedures. It is **not** clinically exhaustive, certified, or intended to substitute for formal clinical pharmacopeia (e.g., RxNorm, SNOMED CT, ICD-10).
- **Supervision**: Never use this framework or MedASR output for unmonitored patient care. All AI-generated clinical documentation must undergo licensed clinician review.
