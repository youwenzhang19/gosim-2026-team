"""LLM on/off and ask_json fallback (no network)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent as agent_mod
from llm_client import LLMClient, api_key


class LlmToggleTests(unittest.TestCase):
    def tearDown(self):
        for name in (
            "OBSERVER_MODEL_DISABLED", "OPENAI_API_KEY", "KIMI_API_KEY",
            "OPENAI_BASE_URL", "OPENAI_MODEL",
        ):
            os.environ.pop(name, None)

    def test_model_disabled_flag(self):
        os.environ["OBSERVER_MODEL_DISABLED"] = "1"
        self.assertTrue(agent_mod.model_disabled())
        os.environ.pop("OBSERVER_MODEL_DISABLED")
        self.assertFalse(agent_mod.model_disabled())

    def test_api_key_prefers_openai_then_kimi(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("KIMI_API_KEY", None)
        self.assertEqual(api_key(), "")
        os.environ["KIMI_API_KEY"] = "kimi-test"
        self.assertEqual(api_key(), "kimi-test")
        os.environ["OPENAI_API_KEY"] = "openai-test"
        self.assertEqual(api_key(), "openai-test")

    def test_ask_json_no_key_returns_none(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("KIMI_API_KEY", None)
        client = LLMClient(log=lambda *_: None)
        self.assertFalse(client.enabled)
        self.assertIsNone(client.ask_json("sys", {"a": 1}, 600.0))

    def test_ask_json_low_wall_returns_none(self):
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        client = LLMClient(log=lambda *_: None)
        # Below ASK_JSON_WALL_RESERVE_SECONDS → no attempt, rules path.
        self.assertIsNone(client.ask_json("sys", {"a": 1}, 10.0))

    def test_ask_json_http_failure_falls_back(self):
        os.environ["OPENAI_API_KEY"] = "test-key-not-used"
        client = LLMClient(log=lambda *_: None, max_retries=1)
        with mock.patch.object(client, "_request", side_effect=TimeoutError("boom")):
            self.assertIsNone(client.ask_json("sys", {"a": 1}, 600.0))

    def test_rules_only_agent_without_key(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("KIMI_API_KEY", None)
        os.environ.pop("OBSERVER_MODEL_DISABLED", None)
        # Minimal public init stub — Planner needs a real survey payload; skip
        # full ObserverAgent if fixture unavailable and only check RulesOnly.
        adv = agent_mod.RulesOnly()
        self.assertIsNone(adv.confirm_report({}, 100.0, 1.0))
        self.assertEqual(adv.poll(), (None, None))


if __name__ == "__main__":
    unittest.main()
