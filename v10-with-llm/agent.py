#!/usr/bin/env python3
"""GOSIM v10 with LLM (must-observe + required sprint). Standard library only.

Protocol: participant-agent-protocol-v4. Derived from GOSIM 2026 Python Pro
(ab27e5ef…, CC BY-NC 4.0) plus v9/v10 planner work (required sprint, dynamic
anchors, determinism, must-observe). LLM wiring matches v4's official
OpenAI-compatible env injection (OPENAI_BASE_URL / OPENAI_MODEL /
OPENAI_API_KEY or KIMI_API_KEY) — never hardcode keys.

One JSON object per line on stdin, one per line on stdout, logs on stderr.

Planner (planner.py) stays deterministic and LLM-free: required sprint, dynamic
anchors, must-observe short probes, sorted tie-breaks. Model stages live in
advisor.py / operations.py / llm_client.py and only overlay advice when a key
is present and OBSERVER_MODEL_DISABLED is not set. Any model miss or timeout
leaves the rule default for that step; the season never waits empty on LLM.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import timedelta

from advisor import Advisor
from adaptive_forecast import AdaptiveForecasts
from calibration import OnlineOutcomeCalibrator, ScienceGainCalibrator
from llm_client import LLMClient, api_key, load_dotenv
from operations import OperationsAdvisor
from planner import Planner
from policy_adapter import PolicyAdapter
from request_bundle import select_bundle
from short_horizon import rerank
from skymath import format_utc, parse_utc

WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}
# Whole-sky overnight pessimism: rain/storm only. overcast/haze are too often false positives.
BAD_KINDS = {"rain", "storm"}
PROTOCOL = "participant-agent-protocol-v4"


def _env(name: str, default: float) -> float:
    return type(default)(os.environ.get(f"PRO_{name}", default))


# --- fault reporting (see _fault_verdict) ---
E_LOW_FREE = _env("E_LOW_FREE", 0.9)      # free probe: E below this in 3 of the last 4 hours
E_LOW_FREE2 = _env("E_LOW_FREE2", 0.9)    # ... threshold while both free probes are left
E_FREE_HOURS = _env("E_FREE_HOURS", 4)
E_LOW = _env("E_LOW", 0.85)               # paid probe: low hours must span two nights ...
E_PAID_LOW = _env("E_PAID_LOW", 0.75)     # ... the median of the last 12 hourly E below this ...
E_PAID_HOURS = _env("E_PAID_HOURS", 12)
E_RECOVER = _env("E_RECOVER", 0.95)       # after a false probe, wait until E is back above this
MAX_PAID_FALSE = _env("MAX_PAID", 6)
PERSIST_NIGHTS = _env("PERSIST_NIGHTS", 3)
E_PAID_STEP = _env("E_PAID_STEP", 0.05)   # ... minus this per paid false probe so far
MAX_FALSE_REPORTS = 8
# The participant guide: an earthquake (announced in the bulletin) lowers instrument efficiency, the loss fades
# night by night, and a report does not repair it. So E drops right after an earthquake are not reportable, and
# while its effect may last only a new step down in E (a fresh drop from the preceding hours) is fault evidence.
QUAKE_HOLD_HOURS = _env("QUAKE_HOLD_HOURS", 12.0)   # no probes this long after an earthquake notice appears
QUAKE_STEP = _env("QUAKE_STEP", 0.8)                # step: median E of the last 3 rows < this x the 9 rows before
QUAKE_TAIL_HOURS = _env("QUAKE_TAIL_HOURS", 24.0)   # the earthquake period lasts this long after its last notice
PAID_SPACING_HOURS = 20.0
MIN_REPORT_SPACING_HOURS = 2.0
# --- pace ---
PACE_SAFETY = _env("PACE_SAFETY", 0.75)
# --- model ---
MODEL_WAIT_MAX = _env("MODEL_WAIT_MAX", 20.0)
MODEL_FAULT_HIGH = _env("MODEL_FAULT_HIGH", 0.6)  # fault review at or above this: report more readily tonight
MODEL_FAULT_LOW = _env("MODEL_FAULT_LOW", 0.15)   # ... at or below this: paid reports need the strongest evidence
SCALE_STEP = _env("SCALE_STEP", 0.7)
LOG_WAIT_MAX = _env("LOG_WAIT_MAX", 240.0)
LOG_READER = _env("LOG_READER", 1)             # 0: do not read staff notes
MODEL_FREE_PROBE = _env("MODEL_FREE_PROBE", 0)   # 1: a high fault review may also spend a free probe on a low scale            # with a likely fault: report when scale stays below this x ref
FIXED_LEVEL = _env("FIXED_LEVEL", -1)     # development only: pin the search level (deterministic runs)   # spend at most this share of the remaining wall clock


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


class RulesOnly:
    """The advisor's interface without a model (OBSERVER_MODEL_DISABLED=1): every rule default stands."""
    night_date = None

    def start_night(self, night_date, *_args):
        self.night_date = night_date
        return None, None

    def poll(self, wallclock_left=None):
        return None, None

    def confirm_report(self, *_args):
        return None


def model_disabled() -> bool:
    """The platform sets OBSERVER_MODEL_DISABLED=1 for an evaluation started with 「本次不提供模型」 / --no-model."""
    return os.environ.get("OBSERVER_MODEL_DISABLED") == "1"


class ObserverAgent:
    def __init__(self, init: dict, rules_only: bool = False):
        started = time.monotonic()
        self.planner = Planner(init, log=log)
        self.rules_only = rules_only or not api_key()
        self.client = None if self.rules_only else LLMClient(log=log)
        self.advisor = RulesOnly() if self.rules_only else Advisor(self.client, log=log)
        self.operations = OperationsAdvisor(self.client, log=log,
            utc_offset_hours=float(init.get("site", {}).get("utc_offset_hours", 0.0)),
            survey_start_utc=init["survey"]["start_utc"],
            survey_end_utc=init["survey"]["end_utc"]) if LOG_READER else None
        self.calibrator = OnlineOutcomeCalibrator()
        self.gain_calibrator = ScienceGainCalibrator()
        self.planner.gain_calibrator = self.gain_calibrator if os.environ.get("V7_GAIN_ENABLED", "0") == "1" else None
        self.forecasts = AdaptiveForecasts(self.client if os.environ.get("V8_FORECAST_ENABLED", "1") == "1" else None, log=log)
        self.policy_adapter = PolicyAdapter(self.client if os.environ.get("V8_POLICY_ENABLED", "1") == "1" else None, log=log)
        self.calibration_pending = None
        self.state_revision = 0
        self.operation_report_evidence = []
        self.consumed_report_evidence = set()
        self.audit = {"night_plan_accepted": 0, "fault_review_accepted": 0,
                      "operation_waits": 0, "operation_report_uses": 0,
                      "rerank_changes": 0, "rerank_fallbacks": 0, "feedback_unknown": 0,
                      "v8_proposal_changes": 0, "v8_proposal_fallbacks": 0,
                      "v8_forecast_submissions": 0, "v8_policy_submissions": 0,
                      "v8_bundle_selections": 0, "v8_bundle_changes": 0,
                      "v8_bundle_fallbacks": 0, "v8_last_bundle": None,
                      "v8_last_selection": None}
        self.correct_report_at = None            # time of the last correct report
        self.model_wait = 0.0                    # wall seconds spent waiting for the model (not planning cost)
        self.fault_likely = None                 # tonight's model estimate that an instrument fault is active
        self.scale_hours: dict = {}              # hour -> [planner.scale samples] (for the model's fault table)
        self.start = parse_utc(init["survey"]["start_utc"])
        self.forecast_notices: list = []
        self.night_seen = None
        self.observes = 0
        # fault reporting state
        reporting = init["scoring"].get("reporting", {})
        self.public_reporting = dict(reporting)
        self.public_required = dict(init["scoring"].get("required", {}))
        self.free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.reports = 0
        self.correct_reports = 0
        self.false_reports = 0
        self.false_since_correct = 0
        self.paid_false = 0
        self.last_report_hours = -1e9
        self.ref_from_hours = -1e9
        self.episode_blocked = False
        self.blocked_at_hour = -1
        self.quake_on = False
        self.quake_onset_hours = -1e9
        self.quake_last_hours = -1e9
        # pace state
        self.cost_ema = [0.0, 0.0, 0.0, 0.0]     # CPU seconds per observe decision at each search level
        self.wall_ema = [0.0, 0.0, 0.0, 0.0]     # real seconds per observe decision (own turn, model waits excluded)
        self.turn_end = None
        self.decisions = 0
        self.engine_ema = None
        self.sim_step_ema = None
        self.last_now = None
        log(f"pro: {len(self.planner.ids)} targets, {sum(self.planner.required)} required, "
            f"{len(self.planner.nights)} nights; init {time.monotonic() - started:.2f}s; model {self.client.model if self.client else 'none (rules only)'}")

    # --- decision loop ------------------------------------------------------------------------------

    def respond(self, payload: dict) -> dict:
        started = time.monotonic()
        cpu_started = time.process_time()
        if self.turn_end is not None:   # engine time between our turns (charged only by the old real-time clock)
            gap = started - self.turn_end
            if 0.0 <= gap < 5.0:
                self.engine_ema = gap if self.engine_ema is None else 0.95 * self.engine_ema + 0.05 * gap
        model_before = self.model_wait
        level = self.planner.fast_level
        action = self._respond(payload)
        # the platform charges CPU time inside our turns; waiting for the model is free of CPU, so keep it
        # out of the real-time estimate as well
        cpu = time.process_time() - cpu_started
        wall = time.monotonic() - started - (self.model_wait - model_before)
        if action.get("action") == "observe" and level < 4:
            for ema, cost in ((self.cost_ema, cpu), (self.wall_ema, wall)):
                c = ema[level]
                ema[level] = cost if c == 0.0 else 0.9 * c + 0.1 * cost
                for k in range(level + 1, 4):   # cheaper levels not measured yet: a third of the level above
                    if ema[k] == 0.0 or ema[k] > ema[k - 1]:
                        ema[k] = ema[k - 1] / 3.0
        self.turn_end = time.monotonic()
        action.setdefault("decision_source", "v8-rules" if self.rules_only else "v8-model-assisted")
        return action

    def _respond(self, payload: dict) -> dict:
        now = parse_utc(payload["now_utc"])
        if self.last_now is not None and self.planner.current_night(now) is not None:
            step = (now - self.last_now).total_seconds()
            if 0 < step <= 3600:
                self.sim_step_ema = step if self.sim_step_ema is None else 0.95 * self.sim_step_ema + 0.05 * step
        self.last_now = now
        hours = (now - self.start).total_seconds() / 3600.0
        planner = self.planner
        resynced = any(m.get("record_type") == "state_resync" for m in payload.get("new_messages", []))
        if resynced:
            self.state_revision += 1
            self.calibrator.reset()
            self.gain_calibrator.reset()
            self.calibration_pending = None
            self.forecasts.reset()
            self.policy_adapter.reset()
            self.scale_hours.clear()
            self.fault_likely = None
            self.night_seen = None
            self.advisor = RulesOnly() if self.rules_only else Advisor(self.client, log=log)
        else:
            self.forecasts.observe_result(payload.get("last_result"), now)
            self._calibration_feedback(payload.get("last_result"), now)
        for message in payload.get("new_messages", []):
            if message.get("record_type") == "forecast":
                self.forecast_notices = message.get("notices", [])
        last = payload.get("last_result") or {}
        if last.get("action") == "report":
            self._on_report_result(last, hours)
        planner.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        quake = any(kind == "earthquake" for kind, _ in planner.notices)
        if quake and not self.quake_on:
            self.quake_onset_hours = hours
            log(f"pro: earthquake notice at {payload['now_utc']}")
        self.quake_on = quake
        if quake:
            self.quake_last_hours = hours
        planner.on_requests(payload.get("active_requests", []))
        planner.on_result(payload.get("last_result"), now, hours)
        self._pace(payload, now)
        reader_action = self._read_notes(payload, now, hours)
        report = self._report_from_note(hours, payload)
        if report is not None:
            return report
        night = planner.current_night(now)
        if night is not None:
            night_index, night_start, night_end = night
            if self.night_seen != night_index:
                self.night_seen = night_index
                self._night_advice(night_start, payload, hours)
            else:
                self._apply_advice(*self.advisor.poll(self._clock(payload)[1]))
        self._adaptive_advice(payload, now)
        if reader_action is not None:
            return reader_action

        if night is None:
            nxt = planner.next_night_start(now)
            wait_until = self._clip_wait_to_report(now, nxt)
            if wait_until is not None:
                reason = "staff note: instrument report due" if wait_until != nxt else \
                    "daytime: sleep until the next night"
                return {"action": "wait", "until_utc": format_utc(wait_until), "reason": reason}
            if nxt is None:
                return {"action": "finish", "reason": "no observing night left"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "daytime: sleep until the next night"}
        self.scale_hours.setdefault(int(hours), []).append(self.planner.scale)
        if (night_end - now).total_seconds() < planner.min_exposure:
            nxt = planner.next_night_start(now)
            wait_until = self._clip_wait_to_report(now, nxt)
            if wait_until is None:
                return {"action": "finish", "reason": "survey over"}
            reason = "staff note: instrument report due" if wait_until != nxt else "night ending"
            return {"action": "wait", "until_utc": format_utc(wait_until), "reason": reason}
        if planner.site_closed():
            seconds, report_due = self._clip_slot_wait_to_report(
                now, self._to_next_slot(now, night_start))
            reason = "staff note: instrument report due" if report_due else \
                "bulletin: rain/storm over the whole sky"
            return {"action": "wait", "duration_seconds": seconds, "reason": reason}
        report = self._maybe_report(hours, payload)
        if report is not None:
            return report
        boundary = self.operations.next_boundary(now) if self.operations is not None and \
            os.environ.get("V7_OPERATIONS_APPLY", "1") == "1" and \
            hasattr(self.operations, "next_boundary") else None
        if boundary is not None and boundary < night_end:
            if (boundary - now).total_seconds() < planner.min_exposure:
                return {"action": "wait", "until_utc": format_utc(boundary),
                        "reason": "staff note: approaching an operations transition"}
            night_end = boundary
        action = planner.plan(now, night_end, night_index, hours)
        if action is None:
            seconds, report_due = self._clip_slot_wait_to_report(
                now, self._to_next_slot(now, night_start))
            reason = "staff note: instrument report due" if report_due else "nothing useful is up"
            return {"action": "wait", "duration_seconds": seconds, "reason": reason}
        bundle_selected = False
        if planner.requests and os.environ.get("V8_BUNDLE_ENABLED", "1") == "1":
            cpu_left, wall_left, _ = self._clock(payload)
            allowance = min(0.06, max(0.0, cpu_left / self._decisions_left(now) * 0.18))
            if cpu_left > 30.0 and wall_left > 60.0 and allowance > 0:
                selected, summary = select_bundle(
                    planner, action, now, night_end, night_index, hours,
                    time.process_time() + allowance)
                if summary.get("used"):
                    bundle_selected = selected == action or planner.record_action(
                        selected, now, night_end, night_index, hours)
                    if bundle_selected:
                        self.audit["v8_bundle_selections"] += 1
                        self.audit["v8_bundle_changes"] += int(selected != action)
                        action = selected
                    else:
                        summary = {**summary, "used": False, "fallback": True,
                                   "reason": "commit_rejected"}
                if summary.get("fallback"):
                    self.audit["v8_bundle_fallbacks"] += 1
                self.audit["v8_last_bundle"] = summary
        if not bundle_selected and os.environ.get("V7_HORIZON_ENABLED", "0") == "1":
            cpu_left, wall_left, _ = self._clock(payload)
            allowance = min(0.04, max(0.0, cpu_left / self._decisions_left(now) * 0.12))
            if cpu_left > 30.0 and wall_left > 60.0 and allowance > 0:
                selected, summary = rerank(planner, action, now, night_end, night_index, hours,
                                           time.process_time() + allowance)
                if selected != action:
                    if planner.record_action(selected, now, night_end, night_index, hours):
                        action = selected
                        self.audit["rerank_changes"] += 1
                    else:
                        selected = action
                if summary.get("fallback"):
                    self.audit["rerank_fallbacks"] += 1
        if not bundle_selected and os.environ.get("V8_POLICY_APPLY", "1") == "1":
            cpu_left, wall_left, _ = self._clock(payload)
            allowance = min(0.04, max(0.0, cpu_left / self._decisions_left(now) * 0.12))
            if cpu_left > 30.0 and wall_left > 60.0 and allowance > 0:
                selected, summary = self.policy_adapter.select(
                    planner, action, now, night_end, night_index, hours,
                    time.process_time() + allowance, self.forecasts)
                if selected != action:
                    if planner.record_action(selected, now, night_end, night_index, hours):
                        action = selected
                        self.audit["v8_proposal_changes"] += 1
                    else:
                        summary = {**summary, "changed": False, "fallback": True,
                                   "reason": "commit_rejected"}
                if summary.get("fallback"):
                    self.audit["v8_proposal_fallbacks"] += 1
                self.audit["v8_last_selection"] = summary
        self._freeze_prediction(payload, action, now, night_end, night_index, hours)
        self.observes += 1
        action["reason"] = f"{len(action['assignments'])} fibres, program {action['program']}"
        return action

    # --- public staff facts --------------------------------------------------------------------------

    def _adaptive_advice(self, payload, now):
        """Grade old forecasts before submitting future predictions or proposals."""
        public_context = {**payload, "forecast_notices": self.forecast_notices}
        wall_left = self._clock(payload)[1]
        started = time.monotonic()
        submitted = self.forecasts.update(public_context, self.planner, wall_left)
        if submitted:
            self.audit["v8_forecast_submissions"] += 1
            self.forecasts.wait(min(2.0, self._model_wait_budget(payload)))
            self.forecasts.update(public_context, self.planner, self._clock(payload)[1])
        policy_submitted = self.policy_adapter.update(
            public_context, self.planner, self._clock(payload)[1], self.forecasts.context())
        if policy_submitted:
            self.audit["v8_policy_submissions"] += 1
            self.policy_adapter.wait(min(2.0, self._model_wait_budget(payload)))
            self.policy_adapter.update(public_context, self.planner, self._clock(payload)[1],
                                       self.forecasts.context())
        self.model_wait += time.monotonic() - started

    def _clip_wait_to_report(self, now, until):
        operations = self.operations
        if (operations is None or os.environ.get("V7_OPERATIONS_APPLY", "1") != "1"
                or not hasattr(operations, "next_report_at")):
            return until
        report_at = operations.next_report_at(now)
        if report_at is not None and report_at > now and (until is None or report_at < until):
            return report_at
        return until

    def _clip_slot_wait_to_report(self, now, seconds: int) -> tuple[int, bool]:
        until = self._clip_wait_to_report(now, now + timedelta(seconds=seconds))
        if until is not None and until < now + timedelta(seconds=seconds):
            return max(1, int((until - now).total_seconds())), True
        return seconds, False

    def _read_notes(self, payload: dict, now, hours: float):
        """Read critical public facts before scheduling optional model work."""
        operations = self.operations
        if operations is None:
            return None
        night = self.planner.current_night(now)
        submitted = operations.update(payload, now, night[2] if night else now, self._clock(payload)[1])
        if submitted:
            started = time.monotonic()
            operations.wait(self._note_wait_budget(payload, now))
            self.model_wait += time.monotonic() - started
        operations.collect()
        applied = os.environ.get("V7_OPERATIONS_APPLY", "1") == "1"
        self.planner.log_avoid = operations.avoid_now(now) if applied else set()
        self.operation_report_evidence = operations.report_evidence(now) if applied else []
        end = operations.closed_until(now) if applied else None
        if applied and night is not None and end is None and len(self.planner.log_avoid) == 8:
            end = now + timedelta(seconds=self._to_next_slot(now, night[1]))
        if night is not None and end is not None:
            boundary = operations.next_boundary(now) if hasattr(operations, "next_boundary") else None
            if boundary is not None:
                end = min(end, boundary)
            seconds = int(min((end - now).total_seconds(), (night[2] - now).total_seconds(), 3600))
            if seconds <= 0:
                return None
            seconds = max(1, seconds)
            self.audit["operation_waits"] += 1
            return {"action": "wait", "duration_seconds": seconds, "reason": "staff note: site closed"}
        return None

    def _note_wait_budget(self, payload: dict, now) -> float:
        """How long a new note may hold the run. Reading models can take minutes, and a note often announces
        something a few nights ahead, so waiting for the answer pays when real time is to spare: half of the
        real time the planner will not need, spread over the notes expected until the end (at the rate seen so far)."""
        _, wall_left, _ = self._clock(payload)
        per_decision = max(self.wall_ema[self.planner.fast_level], 0.05) + (self.engine_ema or 0.02)
        spare = wall_left - 1.5 * self._decisions_left(now) * per_decision - 60.0
        end = self.planner.nights[-1][1] if self.planner.nights else now
        done = max(3 * 86400.0, (now - self.start).total_seconds())   # rate estimate needs a few days
        notes_left = 1.0 + (self.operations.summary().get("notes_seen", 0) if self.operations else 0) * max(0.0, (end - now).total_seconds()) / done
        return max(0.0, min(LOG_WAIT_MAX, 0.5 * spare / notes_left))

    def _to_next_slot(self, now, night_start) -> int:
        slot = self.planner.slot_seconds
        into = (now - night_start).total_seconds() % slot
        return int(max(60, min(3600, slot - into)))

    # --- pace ---------------------------------------------------------------------------------------

    def _clock(self, payload: dict):
        """(CPU seconds left, real seconds left, fair clock?) from the request's wallclock block.

        Fair clock (current platform): the budget is normalized CPU time inside our turns, and
        remaining_real_cpu_seconds converts it to this machine's CPU seconds; a separate real-time cap
        (wall_remaining_seconds) only guards against runaway runs. Older runners count real time only."""
        wall = payload.get("wallclock") or {}
        if "remaining_real_cpu_seconds" in wall:
            return (float(wall["remaining_real_cpu_seconds"]), float(wall.get("wall_remaining_seconds", 1e9)), True)
        remaining = float(wall.get("remaining_seconds", 1e9))
        return remaining, remaining, False

    def _decisions_left(self, now) -> float:
        night_seconds = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in self.planner.nights if end > now)
        return max(1.0, night_seconds / (self.sim_step_ema or 900.0))   # daytime waits cost nothing

    def _pace(self, payload: dict, now) -> None:
        """Pick the search level from the measured cost per decision and the decisions still to come."""
        cpu_left, wall_left, fair = self._clock(payload)
        decisions_left = self._decisions_left(now)
        engine = self.engine_ema or 0.0
        cpu_budget = PACE_SAFETY * cpu_left / decisions_left if fair else 1e9
        wall_budget = PACE_SAFETY * wall_left / decisions_left - engine
        # estimates of levels not used for a while decay, so the agent climbs back up and re-measures them
        self.decisions += 1
        if self.decisions % 50 == 0:
            for k in range(4):
                if k != self.planner.fast_level:
                    self.cost_ema[k] *= 0.85
                    self.wall_ema[k] *= 0.85
        if FIXED_LEVEL >= 0:
            self.planner.fast_level = FIXED_LEVEL
            return
        level = 0
        while level < 3 and (self.cost_ema[level] > cpu_budget or self.wall_ema[level] > wall_budget):
            level += 1
        if min(cpu_left, wall_left) < 15.0:
            level = 4
        if level != self.planner.fast_level:
            log(f"pro: pace level {level} (cpu budget {min(cpu_budget, 99) * 1000:.0f} ms, wall budget {wall_budget * 1000:.0f} ms, "
                f"cpu costs {[round(c * 1000) for c in self.cost_ema]} ms, {decisions_left:.0f} decisions left)")
            self.planner.fast_level = level

    # --- model stages (advisor.py): night plan and fault review at every night start --------------------

    def _night_advice(self, night_start, payload: dict, hours: float) -> None:
        """Night start: rule defaults first, then the two model calls (night plan, fault review)."""
        night_date = (night_start - timedelta(hours=12)).date().isoformat()
        tonight = [n for n in self.forecast_notices if night_date in n.get("nights", [])]
        bulletin = (payload.get("latest_bulletin") or {}).get("notices", [])
        # rule defaults, kept when the model gives no valid answer
        self.planner.bad_forecast = any(n.get("direction") == "ALL" and n.get("event_kind") in BAD_KINDS for n in tonight)
        self.planner.extra_avoid = set()
        self.fault_likely = None
        left = self._clock(payload)[1]
        started = time.monotonic()
        answers = self.advisor.start_night(night_date, tonight, bulletin, self._fault_table(hours), left,
                                           self._model_wait_budget(payload))
        self.model_wait += time.monotonic() - started
        self._apply_advice(*answers)

    def _model_wait_budget(self, payload: dict) -> float:
        """How long a night start may wait for the model. Waiting costs no CPU budget, only real time: use half
        of the real time the planner and the engine will not need, spread over the nights left."""
        _, wall_left, _ = self._clock(payload)
        now = self.last_now
        nights_left = max(1, sum(1 for _, end in self.planner.nights if end > now))
        per_decision = max(self.wall_ema[self.planner.fast_level], 0.05) + (self.engine_ema or 0.02)
        spare = wall_left - 1.5 * self._decisions_left(now) * per_decision - 60.0
        return max(0.0, min(MODEL_WAIT_MAX, 0.5 * spare / nights_left))

    def _apply_advice(self, plan, fault) -> None:
        if plan is not None:
            self.audit["night_plan_accepted"] += 1
            self.planner.bad_forecast = plan["bad_night"]
            self.planner.extra_avoid = set(plan["avoid_directions"])
            log(f"llm night plan {self.advisor.night_date}: accepted")
        if fault is not None:
            self.audit["fault_review_accepted"] += 1
            self.fault_likely = fault["fault_likely"]
            log(f"llm fault review {self.advisor.night_date}: accepted")

    def _scale_ref(self) -> float:
        """Usual clear-sky scale since the last repair: 75th percentile of the hourly medians."""
        values = sorted(sorted(v)[len(v) // 2] for h, v in self.scale_hours.items() if h >= self.ref_from_hours and v)
        return values[(3 * len(values)) // 4] if len(values) >= 4 else 1.0

    def _fault_table(self, hours: float) -> dict:
        """The evidence the fault review reads: the last ~30 observed hours."""
        e_by_hour = {hour: sorted(v)[len(v) // 2] for hour, _, v in self.planner.e_hours if hour >= self.ref_from_hours}
        rows = []
        for hour in sorted(self.scale_hours)[-30:]:
            if hour < self.ref_from_hours:
                continue
            v = sorted(self.scale_hours[hour])
            stamp = (self.start + timedelta(hours=hour)).strftime("%m-%dT%H")
            rows.append([stamp, round(e_by_hour[hour], 2) if hour in e_by_hour else None, round(v[len(v) // 2], 2)])
        notices = sorted({f"{kind} {direction}" for kind, direction in self.planner.notices})
        return {"columns": ["utc_hour", "E", "scale"], "rows": rows, "ref": round(self._scale_ref(), 2),
                "reporting": self.public_reporting,
                "planning_context": {"min_exposure": self.planner.min_exposure,
                    "max_exposure": self.planner.max_exposure,
                    "required": self.public_required,
                    "program": {"bands": self.planner.bands, "multipliers": self.planner.multipliers}},
                "notices_now": notices,
                "hours_since_earthquake_notice_began": None if self.quake_onset_hours < -1e8 else round(hours - self.quake_onset_hours, 1),
                "free_false_reports_left": max(0, self.free_left()), "paid_false_reports_so_far": self.paid_false,
                "correct_reports_so_far": self.correct_reports,
                "hours_since_last_report": None if self.reports == 0 else round(hours - self.last_report_hours, 1)}

    # --- instrument faults ---------------------------------------------------------------------------

    def _maybe_report(self, hours: float, payload: dict):
        """Report (probe) when the quality level stays below what the program bands allow.

        A report costs no time, its answer arrives at once, and the first false reports after each correct
        one are free: spend free probes readily, paid ones only on strong, lasting evidence."""
        explicit = self._report_from_note(hours, payload)
        if explicit is not None:
            return explicit
        if self.false_reports >= MAX_FALSE_REPORTS or hours - self.last_report_hours < MIN_REPORT_SPACING_HOURS:
            return None
        if hours - self.quake_onset_hours < QUAKE_HOLD_HOURS:
            return None   # the earthquake explains the drop; a report would not repair it
        if self.planner.site_closed() or self.planner.all_sky_weather():
            return None
        if self.free_left() <= 0 and (self.paid_false >= MAX_PAID_FALSE or hours - self.last_report_hours < PAID_SPACING_HOURS):
            return None
        if self._fault_verdict(hours, payload) and self._model_agrees(hours, payload):
            self.last_report_hours = hours
            self.reports += 1
            log(f"pro: report at {payload['now_utc']} (quality below what the program bands allow), free left {self.free_left()}")
            return {"action": "report", "reason": "quality level below what the program bands allow"}
        return None

    def _report_from_note(self, hours: float, payload: dict):
        note = self._fresh_instrument_evidence()
        if note is None or self.false_reports >= MAX_FALSE_REPORTS:
            return None
        self.consumed_report_evidence.add(note["id"])
        if self.operations is not None and hasattr(self.operations, "consume_report_evidence"):
            self.operations.consume_report_evidence(note["id"])
        self.audit["operation_report_uses"] += 1
        self.last_report_hours = hours
        self.reports += 1
        log(f"pro: report at {payload['now_utc']} (validated hardware notification)")
        return {"action": "report", "reason": "staff note: persistent hardware fault is due"}

    def _fresh_instrument_evidence(self):
        for note in self.operation_report_evidence:
            identity = str(note.get("id", note.get("source_id", "")))
            if not identity or identity in self.consumed_report_evidence or \
                    note.get("instrument_kind") != "persistent_fault":
                continue
            try:
                start = parse_utc(note.get("report_at_utc", note.get("start_utc")))
                issued = parse_utc(note.get("issued_at", note.get("issued_at_utc")))
            except (KeyError, TypeError, ValueError):
                continue
            if issued <= self.last_now and start <= self.last_now and \
                    (self.correct_report_at is None or start > self.correct_report_at):
                return {**note, "id": identity}
        return None

    def _calibration_feedback(self, result, now):
        pending, self.calibration_pending = self.calibration_pending, None
        if pending is None:
            return
        if not isinstance(result, dict) or result.get("action") != "observe" or \
                result.get("observe_index") != pending["index"] or \
                result.get("assigned_count") != len(pending["ids"]):
            self.audit["feedback_unknown"] += 1
            return
        hits = result.get("hits")
        if not isinstance(hits, list) or result.get("hit_count") != len(hits):
            self.audit["feedback_unknown"] += 1
            return
        seen = set()
        positive = 0
        for hit in hits:
            if not isinstance(hit, dict) or hit.get("target_id") not in pending["ids"] or hit["target_id"] in seen:
                self.audit["feedback_unknown"] += 1
                return
            score = hit.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
                self.audit["feedback_unknown"] += 1
                return
            seen.add(hit["target_id"])
            positive += int(score > 0)
        self.calibrator.label(pending["index"], positive / len(pending["ids"]), now)
        self.gain_calibrator.label(pending["index"], result, now)

    def _freeze_prediction(self, payload, action, now, night_end, night_index, hours):
        index = payload.get("observe_action_index")
        ids = set(action.get("assignments", {}).values())
        if isinstance(index, bool) or not isinstance(index, int) or not ids:
            return
        features = {"direction": ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
                    [int((action["pointing"]["az_deg"] + 22.5) % 360 // 45)],
                    "program": action["program"], "exposure_seconds": action["duration_seconds"],
                    "assignments_count": len(ids)}
        old = {target: self.planner.cur[self.planner.index_of[target]] *
               self.planner.weight[self.planner.index_of[target]] for target in ids}
        token = hashlib.sha256(json.dumps({"now": format_utc(now), "action": action,
            "old_best": old, "revision": self.state_revision}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        end = now + timedelta(seconds=action["duration_seconds"])
        estimate = self.planner.evaluate_action(action, now, night_end, night_index, hours)
        if estimate is None:
            return
        try:
            self.forecasts.freeze_exposure(index, estimate, action, old, now, end)
            self.calibrator.freeze(index, None, features, now, end, input_hash=token)
            self.gain_calibrator.freeze(index, estimate.get("base_science_gain", estimate["science_gain"]),
                features, old, sorted(ids), now, end, input_hash=token)
            self.calibration_pending = {"index": index, "ids": ids}
        except (ValueError, KeyError, TypeError):
            self.audit["feedback_unknown"] += 1

    def _fault_verdict(self, hours: float, payload: dict) -> bool:
        """Hourly E = quality level / band level (planner.e_hours). 1 = consistent; a fault keeps E low."""
        rows = [(hour, night, sorted(v)[len(v) // 2]) for hour, night, v in self.planner.e_hours if hour >= self.ref_from_hours]
        if rows and int(hours) != getattr(self, "_logged_hour", None):
            self._logged_hour = int(hours)
            log(f"pro: E {payload['now_utc']} {rows[-1][2]:.2f} scale {self.planner.scale:.3f} band {self.planner.band_level or 0:.3f}")
        if len(rows) < E_FREE_HOURS or rows[-1][0] < int(hours) - 1:
            return False
        if QUAKE_STEP > 0 and hours - self.quake_last_hours < QUAKE_TAIL_HOURS:
            if len(rows) < 8:
                return False
            last = sorted(e for _, _, e in rows[-3:])[1]
            prev = sorted(e for _, _, e in rows[-12:-3])
            if not (last < QUAKE_STEP * prev[len(prev) // 2] and rows[-3][0] >= int(hours) - 4):
                return False
            if self.episode_blocked and rows[-3][0] <= self.blocked_at_hour:
                return False   # the step that was already probed, not a new one
            self.episode_blocked = False   # a new step is a new episode
        if self.episode_blocked:
            # this low episode was probed already and was not a fault: wait for a recovery first
            last4 = rows[-4:]
            if sum(1 for _, _, e in last4 if e >= E_RECOVER) >= 3 and rows[-1][0] > self.blocked_at_hour:
                self.episode_blocked = False
                log(f"pro: quality recovered at {payload['now_utc']}; probing re-armed")
            else:
                return False
        likely = self.fault_likely
        if MODEL_FREE_PROBE and likely is not None and likely >= MODEL_FAULT_HIGH and self.free_left() > 0 and self._scale_step(hours):
            # off by default: on the practice cards it spent free probes on unannounced weather
            log(f"pro: model-flagged fault (likely {likely:.2f}) and scale below {SCALE_STEP} x ref")
            return True
        if self.free_left() > 0:
            last = rows[-E_FREE_HOURS:]
            threshold = E_LOW_FREE2 if self.free_left() >= 2 else E_LOW_FREE
            low = sum(1 for _, _, e in last if e < threshold)
            return low >= E_FREE_HOURS - 1 and last[-1][2] < threshold and last[-1][0] - last[0][0] <= E_FREE_HOURS + 3
        if self.paid_false >= MAX_PAID_FALSE or hours - self.last_report_hours < PAID_SPACING_HOURS:
            return False
        last = rows[-E_PAID_HOURS:]
        values = sorted(e for _, _, e in last)
        nights = {night for _, night, e in last if e < E_LOW}
        # a fault never goes away on its own: three low nights in a row justify a probe whatever the bar
        by_night: dict = {}
        for _, night, e in rows:
            by_night.setdefault(night, []).append(e)
        nights_seq = sorted(by_night)[-PERSIST_NIGHTS:]
        if (len(nights_seq) == PERSIST_NIGHTS and nights_seq[-1] - nights_seq[0] <= PERSIST_NIGHTS
                and all(len(by_night[n]) >= 3 and sorted(by_night[n])[len(by_night[n]) // 2] < E_LOW for n in nights_seq)
                and hours - self.last_report_hours >= 40.0):
            return True
        if likely is not None and likely <= MODEL_FAULT_LOW:
            return False   # the fault review sees weather, not a fault: only the persistence rule above may report
        # each paid false probe raises the bar for the next one
        paid_low = E_PAID_LOW - E_PAID_STEP * self.paid_false
        return (len(last) == E_PAID_HOURS and values[len(values) // 2] < paid_low and len(nights) >= 2
                and all(e < E_LOW for _, _, e in last[-3:]))

    def _scale_step(self, hours: float) -> bool:
        """The last 3 observed hours all sit below SCALE_STEP x the usual clear-sky scale."""
        recent = [sorted(v)[len(v) // 2] for h, v in sorted(self.scale_hours.items()) if h >= self.ref_from_hours and v][-3:]
        return len(recent) == 3 and max(recent) < SCALE_STEP * self._scale_ref()

    def _model_agrees(self, hours: float, payload: dict) -> bool:
        """Paid probes only: the model looks at the evidence first and may veto. Free probes cost nothing, so they
        never wait for it. No answer in time: the rule's decision stands."""
        if self.free_left() > 0:
            return True
        rows = [(hour, sorted(v)[len(v) // 2]) for hour, _, v in self.planner.e_hours if hour >= self.ref_from_hours][-24:]
        evidence = {"hourly_E_last_24h": [round(e, 2) for _, e in rows], "fault_table": self._fault_table(hours),
                    "paid_false_reports_so_far": self.paid_false, "correct_reports_so_far": self.correct_reports,
                    "reporting": self.public_reporting,
                    "instrument_note_sources": [item.get("source_id", item.get("id")) for item in self.operation_report_evidence]}
        started = time.monotonic()
        left = self._clock(payload)[1]
        verdict = self.advisor.confirm_report(evidence, left, min(30.0, 2.0 * self._model_wait_budget(payload)))
        self.model_wait += time.monotonic() - started
        if verdict is False:
            log(f"pro: model vetoed a paid report at {payload['now_utc']}")
            self.last_report_hours = hours
            return False
        return True

    def free_left(self) -> int:
        return self.free_allowance - self.false_since_correct

    def _on_report_result(self, result: dict, hours: float) -> None:
        if result.get("correct"):
            self.correct_report_at = self.last_now
            log(f"pro: report correct, fault repaired (delta {result.get('score_delta')})")
            self.correct_reports += 1
            self.false_since_correct = 0
            self.planner.forget_quality_history()
            self.forecasts.reset_science()
            self.ref_from_hours = hours
        else:
            self.false_since_correct += 1
            self.false_reports += 1
            self.episode_blocked = True
            self.blocked_at_hour = int(hours)
            if self.false_since_correct > self.free_allowance:
                self.paid_false += 1
            log(f"pro: report false (delta {result.get('score_delta')}); free left {self.free_left()}")


def main() -> int:
    # Official platform injects OPENAI_API_KEY / KIMI_API_KEY (and optional
    # OPENAI_BASE_URL / OPENAI_MODEL). Local .env fills missing names only;
    # keys are never hardcoded and must not be committed.
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    rules_only = model_disabled()
    if rules_only:
        log("pro: OBSERVER_MODEL_DISABLED=1, running rules only (no model calls)")
    elif not api_key():
        # Unlike v4 (hard require_api_key exit), keep the numerical Pro planner
        # available so a missing key never idles out a whole season.
        log("v10: no OPENAI_API_KEY/KIMI_API_KEY; numerical planner + rules only")
        rules_only = True
    else:
        log("v10: LLM enabled via OPENAI_*/KIMI_* env (night_plan/fault_review/"
            "confirm_report/operations; failures fall back to rules)")
    agent = None
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        kind = message.get("message_type")
        if message.get("protocol_version") != PROTOCOL:
            log(f"pro: unexpected protocol {message.get('protocol_version')!r}")
        if kind == "initialize":
            agent = ObserverAgent(message["payload"], rules_only=rules_only)
        elif kind == "decision_request":
            try:
                action = agent.respond(message["payload"])
            except Exception as exc:  # noqa: BLE001 - never crash the run: wait one slot instead
                log(f"pro: error {type(exc).__name__}: {exc}; waiting one slot")
                action = {"action": "wait", "duration_seconds": 900, "reason": "internal error"}
            action.setdefault("decision_source", "rules" if rules_only else "llm-advised")
            print(json.dumps({"protocol_version": PROTOCOL, "message_type": "decision_response",
                              "decision_sequence": message["decision_sequence"], **action}, separators=(",", ":")), flush=True)
        elif kind == "finish":
            payload = message.get("payload", {})
            log(f"pro finished: termination_reason={payload.get('termination_reason')} "
                f"observes={agent.observes if agent else 0} reports={agent.reports if agent else 0}")
            if agent:
                log("v8 audit " + json.dumps({"actions": agent.audit,
                    "model": agent.client.metrics_summary() if agent.client else {"disabled": True, "attempts": 0},
                    "operations": agent.operations.summary() if agent.operations else {"disabled": True},
                    "positive_fraction": agent.calibrator.summary(),
                    "science_gain": agent.gain_calibrator.summary(),
                    "adaptive_forecast": agent.forecasts.summary(),
                    "strategy_proposal": agent.policy_adapter.summary()}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
