#!/usr/bin/env python3
"""Tests for app/services/soap_service.py (no network: the Anthropic client is faked)."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx

from app.services.soap_service import (NOT_DOCUMENTED, build_output_config, SoapGenerationError, SoapService, build_user_message,
                                       parse_soap)

GOOD = {"subjective": "- Foot pain x15 days", "objective": "- Swelling, inner heel", "assessment": "- Likely bone spur",
        "plan": "- X-ray\n- Prednisone taper\n- Kenalog injection", "flags": ["Dose of prednisone unclear"]}


def _response(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=50))


class FakeClient:
    def __init__(self, result=None, error=None):
        self.calls, self._result, self._error = [], result, error
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._result


def _run(coro):
    return asyncio.run(coro)


class TestParseSoap(unittest.TestCase):
    def test_valid(self):
        note = parse_soap(json.dumps(GOOD))
        self.assertEqual(note["plan"], GOOD["plan"])
        self.assertEqual(note["flags"], ["Dose of prednisone unclear"])

    def test_blank_section_becomes_not_documented(self):
        note = parse_soap(json.dumps({**GOOD, "objective": "  "}))
        self.assertEqual(note["objective"], NOT_DOCUMENTED)

    def test_missing_section_or_bad_json_rejected(self):
        for raw in ("not json", "[]", json.dumps({k: v for k, v in GOOD.items() if k != "plan"})):
            with self.assertRaises(SoapGenerationError):
                parse_soap(raw)

    def test_flags_default_to_empty_list(self):
        note = parse_soap(json.dumps({**GOOD, "flags": "oops"}))
        self.assertEqual(note["flags"], [])


class TestGenerate(unittest.TestCase):
    def test_success_wraps_transcript_and_uses_schema(self):
        client = FakeClient(_response(json.dumps(GOOD)))
        note = _run(SoapService(client).generate("Doctor: take prednisone.", "REQ-1"))
        call = client.calls[0]
        self.assertIn("<transcript>", call["messages"][0]["content"])
        self.assertEqual(call["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(note["subjective"], GOOD["subjective"])
        self.assertIn("model", note)

    def test_empty_transcript_rejected_before_calling_model(self):
        client = FakeClient(_response("{}"))
        with self.assertRaises(SoapGenerationError) as cm:
            _run(SoapService(client).generate("   "))
        self.assertEqual(cm.exception.status_code, 422)
        self.assertEqual(client.calls, [])

    def test_truncated_and_refused_responses(self):
        for stop in ("max_tokens", "refusal"):
            with self.assertRaises(SoapGenerationError):
                _run(SoapService(FakeClient(_response(json.dumps(GOOD), stop))).generate("text"))

    def test_api_errors_become_safe_messages_without_transcript_text(self):
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        for exc, code in ((anthropic.RateLimitError("secret-transcript", response=httpx.Response(429, request=req),
                                                    body=None), 429),
                          (anthropic.AuthenticationError("secret-transcript", response=httpx.Response(401, request=req),
                                                         body=None), 503),
                          (anthropic.APIConnectionError(request=req), 502)):
            with self.assertRaises(SoapGenerationError) as cm:
                _run(SoapService(FakeClient(error=exc)).generate("secret-transcript"))
            self.assertEqual(cm.exception.status_code, code)
            self.assertNotIn("secret-transcript", str(cm.exception))

    def test_not_configured_without_key(self):
        with patch("app.services.soap_service.settings") as s:
            s.ANTHROPIC_API_KEY = ""
            with self.assertRaises(SoapGenerationError) as cm:
                _run(SoapService().generate("hello"))
            self.assertEqual(cm.exception.status_code, 503)

    def test_switched_off_even_with_a_key(self):
        with patch("app.services.soap_service.settings") as s:
            s.ANTHROPIC_API_KEY = "sk-ant-test"
            s.SOAP_ENABLED = False
            self.assertFalse(SoapService.is_configured())
            with self.assertRaises(SoapGenerationError) as cm:
                _run(SoapService().generate("hello"))
            self.assertEqual(cm.exception.status_code, 503)

    def test_on_when_key_and_switch_are_set(self):
        with patch("app.services.soap_service.settings") as s:
            s.ANTHROPIC_API_KEY = "sk-ant-test"
            s.SOAP_ENABLED = True
            self.assertTrue(SoapService.is_configured())

    def test_effort_only_for_5x_models(self):
        self.assertEqual(build_output_config("claude-haiku-5-5", "medium")["effort"], "medium")
        self.assertEqual(build_output_config("claude-sonnet-5-5", "low")["effort"], "low")
        self.assertNotIn("effort", build_output_config("claude-haiku-4-5", "medium"))
        self.assertNotIn("effort", build_output_config("claude-haiku-5-5", ""))
        self.assertEqual(build_output_config("claude-haiku-4-5", "medium")["format"]["type"], "json_schema")

    def test_user_message_wraps_transcript(self):
        self.assertEqual(build_user_message("  hi  "), "<transcript>\nhi\n</transcript>")


if __name__ == "__main__":
    unittest.main()
