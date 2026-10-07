"""Bounded OpenAI-compatible chat client for the pro agent (standard library only)."""
from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "k3"
STAGES = ("night_plan", "fault_review", "operations", "confirm_report",
          "adaptive_forecast", "strategy_proposal")
MAX_RECENT_CALLS = 64
QUESTION_DEADLINE_SECONDS = 90.0
# Sync ask_json (v4-compatible one-shot): keep each question short so a hang
# never burns the card's wall clock; background submit() still uses the longer
# QUESTION_DEADLINE_SECONDS / operations budgets above.
ASK_JSON_DEADLINE_SECONDS = 18.0
ASK_JSON_HTTP_TIMEOUT_SECONDS = 8.0
ASK_JSON_WALL_RESERVE_SECONDS = 300.0
OPERATIONS_TOTAL_TIMEOUT_SECONDS = 360.0
OPERATIONS_HTTP_TIMEOUT_SECONDS = 240.0
OPERATIONS_MAX_TOKENS = 12000
DEFAULT_MAX_TOKENS = 2000
MAX_OUTPUT_TOKENS = OPERATIONS_MAX_TOKENS
MAX_WORKERS = 4
OPERATIONS_WORKERS = 2
MAX_ATTEMPTS_PER_REQUEST = 3
MAX_BACKOFF_SECONDS = 1.0
WALLCLOCK_RESERVE_SECONDS = 30.0
_STAGE_PRIORITY = {
    "night_plan": 0,
    "fault_review": 0,
    "confirm_report": 1,
    "operations": 0,
    "adaptive_forecast": 10,
    "strategy_proposal": 20,
}
_FINISH_REASONS = ("stop", "length", "content_filter", "tool_calls", "function_call", "missing", "other")
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_SAFE_TAGS = set(STAGES) | {"model_call"}


def api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()


def load_dotenv(path: str) -> None:
    """Fill missing environment variables from a local .env (for local runs only)."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip().strip('"').strip("'")
                if name and value and not os.environ.get(name):
                    os.environ[name] = value
    except OSError:
        pass


def _bounded_float(value, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return min(high, max(low, parsed))


def _bounded_int(value, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return min(high, max(low, parsed))


def _safe_tag(tag: str) -> str:
    value = str(tag)
    return value if value in _SAFE_TAGS else "model_call"


def _stage_for_tag(tag: str) -> str:
    value = str(tag)
    if value in STAGES:
        return value
    return "operations"


def _finish_reason(value) -> str:
    if value is None:
        return "missing"
    value = str(value).strip().lower()
    return value if value in _FINISH_REASONS else "other"


class _InvalidResponse(ValueError):
    def __init__(self, message: str, finish_reason: str):
        super().__init__(message)
        self.finish_reason = finish_reason


class ModelReply:
    """Small Future-like result that also preserves the pro agent's Call interface."""

    def __init__(self, tag: str):
        self.tag = _safe_tag(tag)
        self.stage = _stage_for_tag(tag)
        self.answer = None
        self.error = None
        self.seconds = 0.0
        self.attempts_made = 0
        self.finish_reason = None
        self._deadline = None
        self._logged = False
        self._done = threading.Event()

    def _finish(self, answer=None, error=None, seconds: float = 0.0, finish_reason=None) -> None:
        if self._done.is_set():
            return
        self.answer = answer
        self.error = error
        self.seconds = max(0.0, float(seconds))
        self.finish_reason = finish_reason
        self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, seconds: float) -> bool:
        return self._done.wait(max(0.0, float(seconds)))

    def result(self, timeout=None):
        if not self._done.wait(None if timeout is None else max(0.0, float(timeout))):
            raise TimeoutError("model reply is not ready")
        if self.error == "timeout":
            raise TimeoutError("model request timed out")
        if self.error is not None:
            raise RuntimeError(f"model request failed ({self.error})")
        return self.answer

    def exception(self, timeout=None):
        if not self._done.wait(None if timeout is None else max(0.0, float(timeout))):
            raise TimeoutError("model reply is not ready")
        if self.error == "timeout":
            return TimeoutError("model request timed out")
        if self.error is not None:
            return RuntimeError(f"model request failed ({self.error})")
        return None


# Kept for code that imports the pro client's original handle name.
Call = ModelReply


class LLMClient:
    def __init__(self, log=lambda text: None, call_timeout: float = 90.0, max_calls: int | None = None,
                 max_retries: int = 3, max_in_flight: int = 4):
        self.log = log
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/") or DEFAULT_BASE_URL
        self.key = api_key()
        self.enabled = bool(self.key)
        self.disabled = not self.enabled
        self.model = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
        self.call_timeout = _bounded_float(call_timeout, QUESTION_DEADLINE_SECONDS, 0.1,
                                           QUESTION_DEADLINE_SECONDS)
        try:
            self.max_calls = None if max_calls is None else max(0, int(max_calls))
        except (TypeError, ValueError, OverflowError):
            self.max_calls = None
        # max_retries historically acts as a total-attempt count in this client.
        self.max_retries = _bounded_int(max_retries, MAX_ATTEMPTS_PER_REQUEST, 1,
                                        MAX_ATTEMPTS_PER_REQUEST)
        self.max_in_flight = _bounded_int(max_in_flight, MAX_WORKERS, 1, MAX_WORKERS)
        self._operations_worker_count = min(OPERATIONS_WORKERS, self.max_in_flight)
        self._optional_worker_count = max(1, self.max_in_flight - self._operations_worker_count)
        self._operations_reserve = min(OPERATIONS_WORKERS, max(1, self.max_in_flight // 2))
        self._optional_in_flight_limit = max(1, self.max_in_flight - self._operations_reserve)
        self.calls: list[ModelReply] = []
        self.ok = 0
        self.failed = 0
        self.rejected = 0
        self.timeouts = 0
        self.deadline_before_attempt = 0
        self.retries = 0
        self.no_key = 0
        self.queue_full = 0
        self._attempts = 0
        self._finish_reasons = {reason: 0 for reason in _FINISH_REASONS}
        self._by_stage = {
            stage: {"attempts": 0, "success": 0, "failure": 0, "rejected": 0,
                    "timeout": 0, "deadline_before_attempt": 0, "retries": 0,
                    "finish_reason": {reason: 0 for reason in _FINISH_REASONS}}
            for stage in STAGES
        }
        self._lock = threading.RLock()
        self._jobs: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.max_in_flight)
        self._operations_jobs: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.max_in_flight)
        self._workers = {"optional": [], "operations": []}
        self._sequence = 0
        self._closed = False

    @property
    def calls_made(self) -> int:
        """Number of outbound HTTP attempts, for Operations metrics."""
        with self._lock:
            return self._attempts

    def _request(self, system: str, user, timeout: float, max_tokens: int = DEFAULT_MAX_TOKENS,
                 include_finish_reason: bool = False):
        user_text = user if isinstance(user, str) else json.dumps(
            user, ensure_ascii=False, separators=(",", ":")
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        choice = data["choices"][0]
        finish_reason = _finish_reason(choice.get("finish_reason"))
        text = choice["message"]["content"] or ""
        match = _JSON_OBJECT.search(text)
        if not match:
            raise _InvalidResponse("no JSON object in the reply", finish_reason)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise _InvalidResponse("invalid JSON object in the reply", finish_reason) from exc
        if not isinstance(parsed, dict):
            raise _InvalidResponse("reply is not a JSON object", finish_reason)
        if include_finish_reason:
            return parsed, finish_reason
        return parsed

    def in_flight(self) -> int:
        with self._lock:
            return sum(1 for call in self.calls if not call.done())

    def _prune_calls_locked(self) -> None:
        completed = [call for call in self.calls if call.done()]
        keep_completed = {id(call) for call in completed[-MAX_RECENT_CALLS:]}
        self.calls[:] = [call for call in self.calls if not call.done() or id(call) in keep_completed]

    def _reject(self, error: str, stage: str) -> None:
        with self._lock:
            if error == "no_key":
                self.no_key += 1
            else:
                self.rejected += 1
                self._by_stage[stage]["rejected"] += 1
                if error == "deadline_before_attempt":
                    self.deadline_before_attempt += 1
                    self._by_stage[stage]["deadline_before_attempt"] += 1
                if error == "queue_full":
                    self.queue_full += 1

    def _start_worker_locked(self) -> None:
        for lane, target_count in (("operations", self._operations_worker_count),
                                   ("optional", self._optional_worker_count)):
            self._workers[lane] = [worker for worker in self._workers[lane] if worker.is_alive()]
            while len(self._workers[lane]) < target_count:
                index = len(self._workers[lane]) + 1
                worker = threading.Thread(target=self._run, args=(lane,),
                                          name=f"llm-{lane}-{index}", daemon=True)
                self._workers[lane].append(worker)
                worker.start()

    def _submit(self, tag: str, system: str, user, timeout: float, max_tokens: int):
        stage = _stage_for_tag(tag)
        if not self.enabled:
            self._reject("no_key", stage)
            return None, "no_key"
        timeout_cap = (OPERATIONS_TOTAL_TIMEOUT_SECONDS if stage == "operations"
                       else self.call_timeout)
        timeout = _bounded_float(timeout, timeout_cap, 0.0, timeout_cap)
        if timeout <= 0:
            self._reject("deadline_before_attempt", stage)
            return None, "timeout"
        max_tokens = _bounded_int(max_tokens, DEFAULT_MAX_TOKENS, 1, MAX_OUTPUT_TOKENS)
        with self._lock:
            if self._closed:
                self._reject("closed", stage)
                return None, "closed"
            if self.max_calls is not None and self._attempts >= self.max_calls:
                self._reject("attempt_limit", stage)
                return None, "attempt_limit"
            in_flight = sum(1 for call in self.calls if not call.done())
            if in_flight >= self.max_in_flight:
                self._reject("queue_full", stage)
                return None, "queue_full"
            if stage != "operations":
                optional_in_flight = sum(1 for call in self.calls
                                         if not call.done() and call.stage != "operations")
                if optional_in_flight >= self._optional_in_flight_limit:
                    self._reject("queue_full", stage)
                    return None, "queue_full"
            reply = ModelReply(tag)
            reply._deadline = time.monotonic() + timeout
            jobs = self._operations_jobs if stage == "operations" else self._jobs
            self._sequence += 1
            priority = _STAGE_PRIORITY.get(stage, 30)
            try:
                jobs.put_nowait((priority, self._sequence, (reply, system, user, max_tokens)))
            except queue.Full:
                self._reject("queue_full", stage)
                return None, "queue_full"
            self.calls.append(reply)
            self._prune_calls_locked()
            self._start_worker_locked()
            return reply, None

    def submit(self, tag: str, system: str, user: dict, wallclock_left: float):
        """Queue one background call; None when no key, time, or queue budget is available."""
        stage = _stage_for_tag(tag)
        if not self.enabled:
            self._reject("no_key", stage)
            return None
        wall_left = _bounded_float(wallclock_left, 0.0, 0.0, 86400.0)
        stage_timeout = (OPERATIONS_TOTAL_TIMEOUT_SECONDS if stage == "operations"
                         else self.call_timeout)
        timeout = min(stage_timeout, wall_left - WALLCLOCK_RESERVE_SECONDS)
        if timeout < 5.0:
            self._reject("deadline_before_attempt", stage)
            return None
        max_tokens = OPERATIONS_MAX_TOKENS if stage == "operations" else DEFAULT_MAX_TOKENS
        reply, _error = self._submit(tag, system, user, timeout, max_tokens)
        return reply

    def ask_json(self, system_prompt: str, user_payload: dict, wall_left_seconds: float):
        """v4-compatible synchronous one-shot JSON question.

        Returns a dict on success, or None so the caller keeps its rule-based
        answer for this step only. Never raises into the planner. Uses the same
        OPENAI_*/KIMI_* env wiring as background submit(); keys are never
        hardcoded. Bounded by ASK_JSON_DEADLINE_SECONDS and a wall reserve so a
        dead endpoint cannot idle out a whole season.
        """
        stage = "confirm_report"
        if not self.enabled:
            self._reject("no_key", stage)
            try:
                self.log("llm: ask_json skipped (no API key); rules decide")
            except Exception:
                pass
            return None
        wall_left = _bounded_float(wall_left_seconds, 0.0, 0.0, 86400.0)
        budget = wall_left - ASK_JSON_WALL_RESERVE_SECONDS
        if budget < 2.0:
            self._reject("deadline_before_attempt", stage)
            try:
                self.log("llm: ask_json no wall budget; rules decide")
            except Exception:
                pass
            return None
        deadline = time.monotonic() + min(ASK_JSON_DEADLINE_SECONDS, budget)
        last_error = "no_answer"
        for attempt in range(self.max_retries):
            remaining = deadline - time.monotonic()
            if remaining < 2.0:
                last_error = "timeout"
                break
            claim_error = self._claim_attempt(stage)
            if claim_error:
                last_error = claim_error
                break
            try:
                answer = self._request(
                    system_prompt, user_payload,
                    min(ASK_JSON_HTTP_TIMEOUT_SECONDS, remaining),
                    DEFAULT_MAX_TOKENS, include_finish_reason=False,
                )
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                try:
                    exc.close()
                except Exception:
                    pass
                retryable = status == 429 or status >= 500 or status == 408
                last_error = f"http_{status}"
                if not retryable or attempt + 1 >= self.max_retries:
                    break
            except (TimeoutError, urllib.error.URLError, OSError, _InvalidResponse,
                    ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = type(exc).__name__
                if attempt + 1 >= self.max_retries:
                    break
            else:
                if isinstance(answer, dict):
                    with self._lock:
                        self.ok += 1
                        self._by_stage[stage]["success"] += 1
                    return answer
                last_error = "invalid_response"
                break
            delay = min(MAX_BACKOFF_SECONDS, 0.25 * (2 ** attempt))
            if time.monotonic() + delay >= deadline - 1.0:
                last_error = "timeout"
                break
            with self._lock:
                self.retries += 1
                self._by_stage[stage]["retries"] += 1
            time.sleep(delay)
        with self._lock:
            self.failed += 1
            self._by_stage[stage]["failure"] += 1
            if last_error == "timeout":
                self.timeouts += 1
                self._by_stage[stage]["timeout"] += 1
        try:
            self.log(f"llm: ask_json failed ({last_error}); rules decide")
        except Exception:
            pass
        return None

    def _claim_attempt(self, stage: str):
        with self._lock:
            if self.max_calls is not None and self._attempts >= self.max_calls:
                return "attempt_limit"
            self._attempts += 1
            self._by_stage[stage]["attempts"] += 1
            return None

    def _run(self, lane: str) -> None:
        jobs = self._operations_jobs if lane == "operations" else self._jobs
        while True:
            try:
                _priority, _sequence, job = jobs.get(timeout=0.1)
            except queue.Empty:
                if self._closed:
                    return
                continue
            reply, system, user, max_tokens = job
            try:
                if time.monotonic() >= reply._deadline:
                    self._reject("deadline_before_attempt", reply.stage)
                    reply._finish(error="timeout")
                    with self._lock:
                        self._prune_calls_locked()
                    continue
                self._perform(reply, system, user, max_tokens)
            finally:
                jobs.task_done()

    def _perform(self, reply: ModelReply, system: str, user, max_tokens: int) -> None:
        started = time.monotonic()
        error = None
        answer = None
        finish_reason = None
        for attempt in range(self.max_retries):
            remaining = reply._deadline - time.monotonic()
            if remaining <= 0:
                error = "timeout"
                break
            claim_error = self._claim_attempt(reply.stage)
            if claim_error:
                error = claim_error
                break
            reply.attempts_made += 1
            try:
                http_timeout = min(
                    OPERATIONS_HTTP_TIMEOUT_SECONDS if reply.stage == "operations" else self.call_timeout,
                    remaining,
                )
                answer, finish_reason = self._request(system, user, http_timeout, max_tokens,
                                                       include_finish_reason=True)
                reply.finish_reason = finish_reason
                self._record_finish_reason(reply.stage, finish_reason)
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                try:
                    exc.close()
                except Exception:
                    pass
                if status == 408:
                    error, retryable = "timeout", True
                elif status == 429:
                    error, retryable = "http_429", True
                elif 500 <= status <= 599:
                    error, retryable = "http_5xx", True
                else:
                    error, retryable = "http_error", False
            except TimeoutError:
                error, retryable = "timeout", True
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", None)
                error = "timeout" if isinstance(reason, TimeoutError) or time.monotonic() >= reply._deadline \
                    else "network_error"
                retryable = True
            except OSError:
                error = "timeout" if time.monotonic() >= reply._deadline else "network_error"
                retryable = True
            except _InvalidResponse as exc:
                error, retryable = "invalid_response", True
                finish_reason = exc.finish_reason
                reply.finish_reason = finish_reason
                self._record_finish_reason(reply.stage, finish_reason)
            except (ValueError, KeyError, IndexError, TypeError):
                error, retryable = "invalid_response", True
            else:
                if time.monotonic() > reply._deadline:
                    error = "timeout"
                    answer = None
                else:
                    error = None
                break

            if not retryable or attempt + 1 >= self.max_retries:
                break
            delay = min(MAX_BACKOFF_SECONDS, 0.25 * (2 ** attempt))
            remaining = reply._deadline - time.monotonic()
            if remaining <= delay:
                error = "timeout"
                break
            with self._lock:
                self.retries += 1
                self._by_stage[reply.stage]["retries"] += 1
            time.sleep(delay)

        elapsed = time.monotonic() - started
        if answer is not None and error is None:
            with self._lock:
                self.ok += 1
                self._by_stage[reply.stage]["success"] += 1
                reply._finish(answer=answer, seconds=elapsed, finish_reason=finish_reason)
                self._prune_calls_locked()
            return
        if error is None:
            error = "invalid_response"
        if reply.attempts_made == 0:
            rejected_as = "deadline_before_attempt" if error == "timeout" else error
            self._reject(rejected_as, reply.stage)
        else:
            with self._lock:
                self.failed += 1
                self._by_stage[reply.stage]["failure"] += 1
                if error == "timeout":
                    self.timeouts += 1
                    self._by_stage[reply.stage]["timeout"] += 1
        reply._finish(error=error, seconds=elapsed, finish_reason=finish_reason)
        with self._lock:
            self._prune_calls_locked()

    def _record_finish_reason(self, stage: str, reason: str) -> None:
        reason = _finish_reason(reason)
        with self._lock:
            self._finish_reasons[reason] += 1
            self._by_stage[stage]["finish_reason"][reason] += 1

    def collect(self, call):
        """Return a finished reply, logging only a static stage and safe error category."""
        if call is None or not call.done():
            return None
        if not call._logged:
            call._logged = True
            if call.answer is None:
                try:
                    self.log(f"llm: {call.tag} failed ({call.error}); rules decide")
                except Exception:
                    pass
        return call.answer

    def metrics_summary(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "success": self.ok,
                "failure": self.failed,
                "rejected": self.rejected,
                "timeout": self.timeouts,
                "deadline_before_attempt": self.deadline_before_attempt,
                "retries": self.retries,
                "no_key": self.no_key,
                "attempts": self._attempts,
                "attempt_limit": self.max_calls,
                "queue_full": self.queue_full,
                "finish_reason": dict(self._finish_reasons),
                "by_stage": {
                    stage: {name: dict(value) if isinstance(value, dict) else value
                            for name, value in values.items()}
                    for stage, values in self._by_stage.items()
                },
            }

    def close(self, wait_seconds: float = 0.0) -> None:
        """Stop the daemon worker after queued work, optionally waiting a bounded time."""
        with self._lock:
            self._closed = True
            workers = [worker for lane in self._workers.values() for worker in lane]
        deadline = time.monotonic() + min(QUESTION_DEADLINE_SECONDS, max(0.0, float(wait_seconds)))
        for worker in workers:
            if worker is threading.current_thread():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(remaining)
