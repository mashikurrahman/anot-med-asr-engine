#!/usr/bin/env python3
"""
Tests for the multilingual medical phrase library and language-aware builders
(language_model/phrase_library.py).
"""

import unittest

from language_model.phrase_library import (
    build_hotwords,
    build_initial_prompt,
    supported_languages,
    _normalize_lang,
)


class TestPhraseLibrary(unittest.TestCase):
    def test_supported_languages(self):
        langs = supported_languages()
        self.assertIn("en", langs)
        self.assertIn("bn", langs)

    def test_language_normalization(self):
        self.assertEqual(_normalize_lang(None), "en")
        self.assertEqual(_normalize_lang("auto"), "en")
        self.assertEqual(_normalize_lang("EN"), "en")
        self.assertEqual(_normalize_lang("bn"), "bn")
        self.assertEqual(_normalize_lang("ben"), "bn")
        self.assertEqual(_normalize_lang("en-US"), "en")

    def test_english_hotwords_include_lay_and_clinical(self):
        hw = build_hotwords("en").lower()
        self.assertIn("shortness of breath", hw)   # clinical
        self.assertIn("stuffy nose", hw)           # lay expression
        self.assertIn("prednisone", hw)            # medication

    def test_hotwords_capped(self):
        hw = build_hotwords("en", max_terms=20)
        self.assertLessEqual(len(hw.split()), 40)  # phrases contain spaces; term count <= 20

    def test_bengali_hotwords_present(self):
        hw = build_hotwords("bn")
        self.assertTrue(hw)
        self.assertIn("blood pressure", hw.lower())  # English code-switch terms
        self.assertIn("জ্বর", hw)                    # Bangla term (fever)

    def test_prompts_per_language_nonempty(self):
        self.assertGreater(len(build_initial_prompt("en")), 10)
        self.assertGreater(len(build_initial_prompt("bn")), 10)
        self.assertNotEqual(build_initial_prompt("en"), build_initial_prompt("bn"))


if __name__ == "__main__":
    unittest.main()
