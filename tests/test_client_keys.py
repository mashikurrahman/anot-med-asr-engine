#!/usr/bin/env python3
"""Tests for per-client API keys and daily usage limits (app/security/client_keys.py + verify_auth_boundary)."""

import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.security.client_keys import DailyUsage, parse_client_keys
from app.security.privacy import enforce_daily_limit, verify_auth_boundary


class _Req:
    def __init__(self):
        self.state = type("S", (), {})()


class TestParse(unittest.TestCase):
    def test_names_keys_and_limits(self):
        keys = parse_client_keys("acme:abc123:50, beta:def456")
        self.assertEqual(keys["abc123"], ("acme", 50))
        self.assertEqual(keys["def456"], ("beta", None))

    def test_blank_and_malformed_entries_are_ignored(self):
        self.assertEqual(parse_client_keys(""), {})
        self.assertEqual(parse_client_keys("nokey, :x, a:"), {})
        self.assertEqual(parse_client_keys("a:k:notanumber"), {"k": ("a", None)})

    def test_zero_limit_means_unlimited(self):
        self.assertEqual(parse_client_keys("a:k:0")["k"], ("a", None))


class TestDailyUsage(unittest.TestCase):
    def test_counts_per_client_per_day_and_stops_at_limit(self):
        day = ["2026-11-13"]
        u = DailyUsage(today=lambda: day[0])
        self.assertTrue(u.allow("acme", 2))
        self.assertTrue(u.allow("acme", 2))
        self.assertFalse(u.allow("acme", 2))
        self.assertTrue(u.allow("beta", 2))  # another client is unaffected
        day[0] = "2026-11-14"
        self.assertTrue(u.allow("acme", 2))  # new day, fresh count

    def test_no_limit_always_allows(self):
        u = DailyUsage(today=lambda: "d")
        self.assertTrue(all(u.allow("a", None) for _ in range(500)))


def _auth(x_api_key=None, authorization=None, master="master-key", clients="acme:client-key:1"):
    req = _Req()
    with patch("app.security.privacy.settings") as s:
        s.API_KEY = master
        s.CLIENT_KEYS = clients
        asyncio.run(verify_auth_boundary(request=req, api_key=x_api_key, authorization=authorization))
    return req


class TestAuthWithClientKeys(unittest.TestCase):
    def test_client_key_is_accepted_and_named(self):
        self.assertEqual(_auth(x_api_key="client-key").state.client_name, "acme")

    def test_master_key_still_works_and_is_unlimited(self):
        req = _auth(x_api_key="master-key")
        self.assertEqual(req.state.client_name, "master")
        self.assertIsNone(req.state.client_limit)

    def test_deepgram_style_header_works_for_client_keys(self):
        self.assertEqual(_auth(authorization="Token client-key").state.client_name, "acme")

    def test_unknown_key_rejected(self):
        with self.assertRaises(HTTPException) as cm:
            _auth(x_api_key="nope")
        self.assertEqual(cm.exception.status_code, 401)

    def test_client_keys_alone_are_enough_to_require_auth(self):
        with self.assertRaises(HTTPException):
            _auth(master=None, x_api_key="nope")
        self.assertEqual(_auth(master=None, x_api_key="client-key").state.client_name, "acme")


class TestEnforceLimit(unittest.TestCase):
    def test_over_limit_is_429(self):
        req = _Req()
        req.state.client_name, req.state.client_limit = "acme", 1
        with patch("app.security.privacy._usage", DailyUsage(today=lambda: "d")):
            asyncio.run(enforce_daily_limit(req))
            with self.assertRaises(HTTPException) as cm:
                asyncio.run(enforce_daily_limit(req))
        self.assertEqual(cm.exception.status_code, 429)

    def test_unlimited_client_is_never_blocked(self):
        req = _Req()
        req.state.client_name, req.state.client_limit = "master", None
        with patch("app.security.privacy._usage", DailyUsage(today=lambda: "d")):
            for _ in range(5):
                asyncio.run(enforce_daily_limit(req))


if __name__ == "__main__":
    unittest.main()
