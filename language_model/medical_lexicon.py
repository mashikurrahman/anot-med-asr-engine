#!/usr/bin/env python3
"""
Anot Medical ASR — Medical Lexicon Vocabulary Prior
====================================================
Stage 9B: Configurable medical vocabulary scoring for CTC beam rescoring.

Provides a domain-specific vocabulary boost for medical terms appearing in
beam search candidates. The lexicon acts as a soft prior, NOT a hard replacement.

CRITICAL DESIGN PRINCIPLE:
    The lexicon MUST NOT force medical words into transcripts.
    It provides a configurable additive score that is ALWAYS dominated
    by acoustic evidence when the weights are properly configured.

Categories:
    MEDICATIONS, DOSAGES, UNITS, FREQUENCIES, ROUTES, DIAGNOSES,
    SYMPTOMS, ANATOMY, LAB_TESTS, VITALS, PROCEDURES,
    COMMON_MEDICAL_ABBREVIATIONS, NUMERIC_PATTERNS
"""

import json
import re
import pathlib
from typing import Dict, List, Set, Optional, Any, Tuple


class MedicalLexicon:
    """
    Medical vocabulary prior for CTC beam search rescoring.
    
    Each word in the lexicon receives a configurable boost score when
    it appears as a beam search candidate. The boost is additive and
    controllable via a global weight parameter.
    """

    # Default categories with representative terms
    DEFAULT_LEXICON = {
        "medications": [
            "metformin", "lisinopril", "atorvastatin", "amlodipine",
            "levothyroxine", "metoprolol", "omeprazole", "losartan",
            "albuterol", "gabapentin", "hydrochlorothiazide", "sertraline",
            "simvastatin", "montelukast", "amoxicillin", "pantoprazole",
            "furosemide", "prednisone", "escitalopram", "aspirin",
            "ibuprofen", "acetaminophen", "clopidogrel", "warfarin",
            "apixaban", "insulin", "glipizide", "rosuvastatin",
            "carvedilol", "diltiazem", "verapamil", "spironolactone",
            "propranolol", "atenolol", "ramipril", "enalapril",
            "candesartan", "valsartan", "irbesartan", "telmisartan",
            "empagliflozin", "dapagliflozin", "sitagliptin", "liraglutide",
            "semaglutide", "dulaglutide", "pioglitazone", "glyburide",
            "cephalexin", "azithromycin", "ciprofloxacin", "doxycycline",
            "fluconazole", "acyclovir", "oseltamivir", "hydroxychloroquine",
            # Allergy / respiratory / ENT (clinic-relevant) and common outpatient
            # medications — generic names and frequently-spoken brand names.
            "fluticasone", "mometasone", "budesonide", "beclomethasone",
            "triamcinolone", "azelastine", "olopatadine", "ciclesonide",
            "cetirizine", "levocetirizine", "loratadine", "desloratadine",
            "fexofenadine", "diphenhydramine", "chlorpheniramine", "hydroxyzine",
            "pseudoephedrine", "phenylephrine", "guaifenesin", "dextromethorphan",
            "benzonatate", "ipratropium", "tiotropium", "umeclidinium",
            "salmeterol", "formoterol", "vilanterol", "theophylline", "cromolyn",
            "epinephrine", "methylprednisolone", "dexamethasone", "mometasone",
            "clavulanate", "clindamycin", "levofloxacin", "moxifloxacin",
            "cefdinir", "cefuroxime", "cefpodoxime", "sulfamethoxazole",
            "trimethoprim", "nitrofurantoin", "metronidazole", "valacyclovir",
            "nystatin", "ketoconazole", "mupirocin", "hydrocortisone",
            "clobetasol", "tacrolimus", "pimecrolimus", "mometasone",
            "tamsulosin", "finasteride", "allopurinol", "tramadol",
            "cyclobenzaprine", "meloxicam", "naproxen", "diclofenac",
            "duloxetine", "venlafaxine", "bupropion", "fluoxetine", "paroxetine",
            "citalopram", "trazodone", "mirtazapine", "buspirone", "alprazolam",
            "lorazepam", "clonazepam", "zolpidem", "pregabalin", "topiramate",
            "lamotrigine", "levetiracetam", "pravastatin", "lovastatin",
            "ezetimibe", "fenofibrate", "gemfibrozil", "bisoprolol", "nebivolol",
            "hydralazine", "clonidine", "doxazosin", "prazosin", "isosorbide",
            "nitroglycerin", "digoxin", "amiodarone", "rivaroxaban", "dabigatran",
            "edoxaban", "enoxaparin", "heparin", "glimepiride", "canagliflozin",
            "exenatide", "tirzepatide", "montelukast", "roflumilast", "mepolizumab",
            "benralizumab", "dupilumab", "omalizumab", "prednisolone", "fluticasone",
            # Frequently-spoken brand names
            "flonase", "nasonex", "nasacort", "astelin", "patanase", "zyrtec",
            "claritin", "allegra", "benadryl", "xyzal", "clarinex", "singulair",
            "ventolin", "proair", "proventil", "symbicort", "advair", "breo",
            "trelegy", "spiriva", "dupixent", "xolair", "nucala", "fasenra",
            "augmentin", "zithromax", "amoxil", "bactrim", "keflex", "medrol",
            "flovent", "pulmicort", "qvar", "dulera", "sudafed", "mucinex",
            "ozempic", "mounjaro", "jardiance", "farxiga", "trulicity",
            "synthroid", "lantus", "humalog", "novolog",
        ],
        "dosages": [
            "milligrams", "milligram", "mg",
            "micrograms", "microgram", "mcg",
            "grams", "gram",
            "milliliters", "milliliter", "ml",
            "units", "unit",
            "milliequivalents", "milliequivalent", "meq",
            "percent", "percentage",
            "tablets", "tablet", "capsules", "capsule",
        ],
        "units": [
            "mg", "mcg", "ml", "cc", "iu",
            "mmhg", "bpm", "kg", "lbs", "cm",
            "mmol", "mg/dl", "meq/l",
        ],
        "frequencies": [
            "daily", "twice", "three", "four",
            "bedtime", "needed", "hourly", "weekly", "monthly",
            "morning", "evening", "night",
            "bid", "tid", "qid", "prn",
            "once", "every",
        ],
        "routes": [
            "oral", "sublingual", "intravenous", "intramuscular",
            "subcutaneous", "inhalation", "inhaled", "topical",
            "transdermal", "rectal", "ophthalmic", "nasal",
        ],
        "diagnoses": [
            "hypertension", "diabetes", "mellitus", "asthma",
            "pneumonia", "bronchitis", "copd", "emphysema",
            "hyperlipidemia", "hypercholesterolemia",
            "coronary", "artery", "disease",
            "atrial", "fibrillation", "flutter",
            "congestive", "failure",
            "gastroesophageal", "reflux",
            "chronic", "kidney", "renal",
            "hypothyroidism", "hyperthyroidism",
            "osteoporosis", "osteoarthritis", "arthritis",
            "anemia", "thrombocytopenia",
            "depression", "anxiety", "insomnia",
            "neuropathy", "polyneuropathy",
            "dementia", "alzheimer",
            "cirrhosis", "hepatitis",
            "pancreatitis", "diverticulitis",
            "sepsis", "cellulitis",
        ],
        "symptoms": [
            "cough", "dyspnea", "wheezing", "crackles",
            "chest", "pain", "fever", "chills",
            "fatigue", "headache", "dizziness", "vertigo",
            "nausea", "vomiting", "diarrhea", "constipation",
            "edema", "swelling", "palpitations",
            "hemoptysis", "orthopnea", "syncope",
            "paresthesia", "numbness", "tingling",
            "pruritus", "rash", "erythema",
            "shortness", "breath",
            "abdominal", "tenderness",
            "weakness", "malaise", "lethargy",
        ],
        "anatomy": [
            "lungs", "lung", "heart", "chest", "abdomen",
            "throat", "extremities", "bilateral", "bilaterally",
            "airway", "trachea", "bronchi", "bronchial",
            "liver", "kidney", "spleen", "pancreas",
            "cervical", "thoracic", "lumbar", "sacral",
            "anterior", "posterior", "lateral", "medial",
            "proximal", "distal", "superior", "inferior",
            "hepatic", "renal", "pulmonary", "cardiac",
            "cranial", "femoral", "tibial", "radial",
        ],
        "lab_tests": [
            "hemoglobin", "hematocrit", "platelet", "platelets",
            "leukocyte", "leukocytes", "neutrophils", "lymphocytes",
            "creatinine", "bun", "electrolytes", "potassium", "sodium",
            "glucose", "cholesterol", "triglycerides",
            "troponin", "bnp", "procalcitonin",
            "bilirubin", "albumin", "transaminase",
            "urinalysis", "culture", "sensitivity",
            "hemoglobin", "a1c", "hba1c",
        ],
        "vitals": [
            "blood", "pressure", "systolic", "diastolic",
            "pulse", "heart", "rate", "oxygen", "saturation",
            "temperature", "respiratory", "weight", "height", "bmi",
        ],
        "procedures": [
            "x-ray", "xray", "radiograph", "radiography",
            "ct", "scan", "computed", "tomography",
            "mri", "magnetic", "resonance", "imaging",
            "ecg", "ekg", "electrocardiogram", "electrocardiography",
            "echocardiogram", "echocardiography",
            "ultrasound", "ultrasonography",
            "bronchoscopy", "spirometry", "colonoscopy",
            "endoscopy", "catheterization", "angiography",
            "biopsy", "aspiration", "intubation",
        ],
        "common_medical_abbreviations": [
            "bp", "hr", "rr", "spo2", "o2",
            "bid", "tid", "qid", "prn", "qhs",
            "po", "iv", "im", "sq", "sl",
            "cbc", "bmp", "cmp", "lfts",
            "wbc", "rbc", "hgb", "hct", "plt",
            "esr", "crp",
            "ct", "mri", "ekg", "ecg",
            "copd", "chf", "cad", "dm",
            "htn", "afib", "dvt", "pe",
            "gerd", "ckd", "esrd",
            "uti", "uri",
        ],
    }

    # Category-specific boost magnitudes (relative weights)
    DEFAULT_CATEGORY_WEIGHTS = {
        "medications": 1.5,       # Strong boost for medication names
        "dosages": 1.0,
        "units": 1.0,
        "frequencies": 0.8,
        "routes": 0.8,
        "diagnoses": 1.2,
        "symptoms": 0.8,
        "anatomy": 0.6,
        "lab_tests": 1.0,
        "vitals": 0.8,
        "procedures": 1.0,
        "common_medical_abbreviations": 0.5,
    }

    def __init__(
        self,
        lexicon: Optional[Dict[str, List[str]]] = None,
        category_weights: Optional[Dict[str, float]] = None,
        entities_json_path: Optional[str] = None
    ):
        """
        Initialize the medical lexicon.
        
        Args:
            lexicon: Custom lexicon dict {category: [words]}. Uses DEFAULT_LEXICON if None.
            category_weights: Per-category score multipliers. Uses defaults if None.
            entities_json_path: Path to medical_entities.json to merge with lexicon.
        """
        self.lexicon = lexicon if lexicon is not None else dict(self.DEFAULT_LEXICON)
        self.category_weights = category_weights if category_weights is not None else dict(self.DEFAULT_CATEGORY_WEIGHTS)

        # Merge from evaluation/medical_entities.json if available
        if entities_json_path:
            self._merge_entities_json(entities_json_path)

        # Build flat lookup: word -> (category, weight)
        self._word_lookup: Dict[str, List[Tuple[str, float]]] = {}
        self._build_lookup()

    def _merge_entities_json(self, path: str) -> None:
        """Merge terms from the project's medical_entities.json."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                entities = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        category_map = {
            "medications": "medications",
            "diagnoses": "diagnoses",
            "symptoms": "symptoms",
            "procedures": "procedures",
            "anatomy": "anatomy",
            "laboratory_tests": "lab_tests",
            "vital_signs": "vitals",
        }

        for src_key, dst_key in category_map.items():
            items = entities.get(src_key, [])
            if isinstance(items, list):
                if dst_key not in self.lexicon:
                    self.lexicon[dst_key] = []
                for item in items:
                    if isinstance(item, str):
                        # Split multi-word terms into individual words for lookup
                        words = item.lower().split()
                        for w in words:
                            if w not in self.lexicon[dst_key]:
                                self.lexicon[dst_key].append(w)

        # Also merge dosage units, frequencies, routes from their dict structures
        for key in ["dosage_units", "frequency_expressions", "routes"]:
            items = entities.get(key, {})
            if isinstance(items, dict):
                target = {"dosage_units": "dosages", "frequency_expressions": "frequencies", "routes": "routes"}.get(key, key)
                if target not in self.lexicon:
                    self.lexicon[target] = []
                for canonical, aliases in items.items():
                    for alias in ([canonical] + (aliases if isinstance(aliases, list) else [])):
                        if isinstance(alias, str):
                            for w in alias.lower().split():
                                if w not in self.lexicon[target]:
                                    self.lexicon[target].append(w)

    def _build_lookup(self) -> None:
        """Build the flat word -> [(category, weight)] lookup table."""
        self._word_lookup.clear()
        for category, words in self.lexicon.items():
            weight = self.category_weights.get(category, 1.0)
            for word in words:
                w_lower = word.lower().strip()
                if not w_lower:
                    continue
                if w_lower not in self._word_lookup:
                    self._word_lookup[w_lower] = []
                self._word_lookup[w_lower].append((category, weight))

    def score_word(self, word: str) -> float:
        """
        Return the medical lexicon boost score for a word.
        
        Returns the maximum category weight if the word appears in the lexicon.
        Returns 0.0 if the word is not in the lexicon.
        
        The returned score is meant to be multiplied by a global medical_weight
        parameter in the rescorer.
        """
        w_lower = word.lower().strip()
        entries = self._word_lookup.get(w_lower, [])
        if not entries:
            return 0.0
        return max(weight for _, weight in entries)

    def score_word_detailed(self, word: str) -> Dict[str, Any]:
        """Return detailed scoring info for a word."""
        w_lower = word.lower().strip()
        entries = self._word_lookup.get(w_lower, [])
        if not entries:
            return {"word": w_lower, "in_lexicon": False, "score": 0.0, "categories": []}
        return {
            "word": w_lower,
            "in_lexicon": True,
            "score": max(weight for _, weight in entries),
            "categories": [(cat, wt) for cat, wt in entries]
        }

    def contains(self, word: str) -> bool:
        """Check if a word is in the medical lexicon."""
        return word.lower().strip() in self._word_lookup

    def get_all_words(self) -> Set[str]:
        """Return all words in the lexicon."""
        return set(self._word_lookup.keys())

    def summary(self) -> Dict[str, Any]:
        """Return lexicon statistics."""
        return {
            "total_unique_words": len(self._word_lookup),
            "categories": {cat: len(words) for cat, words in self.lexicon.items()},
            "category_weights": dict(self.category_weights),
        }
