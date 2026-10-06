"""A tiny OpenAI-compatible chat client, configured through environment variables.
Defaults to the Kimi Coding Plan endpoint, since that is what this example ships with;
any other OpenAI-compatible `/chat/completions` endpoint works too by overriding the
base URL and model:

    OPENAI_BASE_URL   default https://api.kimi.com/coding/v1 (see README for the
                      api.kimi.ai alternative for accounts outside mainland China)
    OPENAI_MODEL      default "k3"
    OPENAI_API_KEY    bearer token (KIMI_API_KEY is also accepted)

See https://www.kimi.com/code/docs/en/ for the Kimi Coding Plan API. This client never
sets or overrides the HTTP User-Agent header -- whatever Python's standard library
sends by default is left alone.

Waiting for the model is not charged to the CPU budget (see clock.py), but it does use
real time, and each card has a 30-minute real-time cap. So model use is bounded by real
time: every attempt has a timeout (default 8 s), one question gives up after 18 s in
total, no call starts in the last 5 minutes before the cap, and a run makes at most 64
requests. HTTP 429 (rate limit) and 5xx answers, timeouts and network errors are retried
with exponential backoff plus random jitter, honouring `Retry-After`. In the hidden final
a card's 3 repeats run at the same time on the same key, so 429s are to be expected; the
jitter keeps the repeats from retrying in lockstep. If a question still fails, that one
planning step falls back to its rule-based answer -- the next night's calls run normally.

Standard library only (urllib) so the example has zero third-party dependencies.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "k3"
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)

WALL_RESERVE_SECONDS = 300.0   # no new model call this close to the real-time cap
QUESTION_DEADLINE_SECONDS = 18.0  # one question, all retries and backoff included
MAX_BACKOFF_SECONDS = 20.0


class RetryableError(Exception):
    """429, 5xx, timeout or network trouble: worth another try after a pause."""

    def __init__(self, reason: str, retry_after: Optional[float] = None):
        super().__init__(reason)
        self.retry_after = retry_after


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    try:
        return max(0.0, float(value)) if value else None
    except ValueError:
        return None  # an HTTP date is allowed too; we simply use our own backoff then


class MissingAPIKeyError(RuntimeError):
    pass


def require_api_key() -> None:
    """Raise MissingAPIKeyError if neither OPENAI_API_KEY nor KIMI_API_KEY is set.
    Called once at process startup, before reading anything from stdin."""
    if not (os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()):
        raise MissingAPIKeyError("missing API key: set OPENAI_API_KEY")


class LLMClient:
    def __init__(self, log=lambda text: None, call_timeout_seconds: float = 8.0,
                 max_calls: int = 64, max_attempts: int = 2):
        self.log = log
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/") or DEFAULT_BASE_URL
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()
        self.model = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
        self.call_timeout_seconds = call_timeout_seconds
        self.max_calls = max_calls
        self.max_attempts = max_attempts
        self.calls_made = 0

    def _attempt(self, system_prompt: str, user_payload: dict, timeout: float) -> dict:
        """One HTTP attempt. Raises RetryableError for problems worth retrying, any other
        exception for problems that will not go away (bad key, bad request)."""
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            # No "temperature": Kimi Coding Plan models (k3 / kimi-for-coding) reject any
            # value but 1 with HTTP 400, so leave it to the provider's default. Reasoning
            # models spend tokens thinking before the JSON answer, hence the roomy cap.
            "max_tokens": 1024,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise RetryableError(f"HTTP {exc.code}", _retry_after_seconds(exc.headers.get("Retry-After")))
            raise  # 400/401/403/404: retrying will not help
        except (urllib.error.URLError, OSError) as exc:  # timeout, connection reset, DNS ...
            raise RetryableError(type(exc).__name__)
        text = data["choices"][0]["message"]["content"] or ""
        match = _JSON_OBJECT.search(text)
        if not match:
            raise RetryableError("no JSON object in model reply")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise RetryableError("model reply was not a JSON object")
        return parsed

    def ask_json(self, system_prompt: str, user_payload: dict, wall_left_seconds: float) -> Optional[dict]:
        """One planning question, answered as exactly one JSON object, or None so the
        caller's rule-based answer takes over for this step. `wall_left_seconds` is the
        real time left before the card's cap (`wallclock.wall_remaining_seconds`)."""
        if not self.api_key:
            return None
        deadline = time.monotonic() + min(QUESTION_DEADLINE_SECONDS, wall_left_seconds - WALL_RESERVE_SECONDS)
        for attempt in range(1, self.max_attempts + 1):
            time_left = deadline - time.monotonic()
            if time_left < 2.0:
                self.log("llm: no time left for this question; using the rule-based path")
                return None
            if self.calls_made >= self.max_calls:
                self.log("llm: call cap reached for this run; using the rule-based path")
                return None
            self.calls_made += 1
            try:
                return self._attempt(system_prompt, user_payload, min(self.call_timeout_seconds, time_left))
            except RetryableError as exc:
                # Exponential backoff with full jitter (1, 2, 4 ... s, randomised), or the
                # server's Retry-After when it sends one.
                pause = exc.retry_after if exc.retry_after is not None else random.uniform(0, 2.0 ** (attempt - 1))
                pause = min(pause, MAX_BACKOFF_SECONDS)
                self.log(f"llm: attempt {attempt}/{self.max_attempts} failed ({exc}); retry in {pause:.1f}s")
                if attempt < self.max_attempts and time.monotonic() + pause < deadline - 2.0:
                    time.sleep(pause)  # sleeping is waiting: no CPU budget is charged
                else:
                    break
            except Exception as exc:  # noqa: BLE001 - never let the model break a decision
                self.log(f"llm: call failed ({type(exc).__name__}); not retrying")
                break
        self.log("llm: no answer for this question; this step falls back to its rule-based answer")
        return None
