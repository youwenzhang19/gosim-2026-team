#!/usr/bin/env python3
"""Deterministic / no-real-key wrapper around the official python example agent.

Sets a placeholder OPENAI_API_KEY so agent.py's startup check passes, then stubs
LLMClient.ask_json to always return None. Planning then stays on the rule-based
path (anchor search / fiber fill / exposure) with no network calls.

Usage (from kit root):
  python3 runner/run_local.py --card local-cards/L1 \\
    --agent "python3 smoke_deterministic_agent.py" --agent-cwd python \\
    --out /tmp/gosim-l1-smoke-out
"""
from __future__ import annotations

import os
import sys
from typing import Optional

# Satisfy require_api_key() without a real credential.
os.environ.setdefault("OPENAI_API_KEY", "smoke-no-real-key")
# Point away from any production endpoint in case a stub is missed.
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9")

from agent_core import llm_client  # noqa: E402


def _ask_json_noop(self, system_prompt: str, user_payload: dict, wallclock_remaining_seconds: float) -> Optional[dict]:
    self.log("llm: stubbed (deterministic smoke); rule-based path")
    return None


llm_client.LLMClient.ask_json = _ask_json_noop  # type: ignore[method-assign]

import agent  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(agent.main())
