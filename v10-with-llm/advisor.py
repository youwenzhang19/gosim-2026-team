"""The model-driven stages of the pro agent. Two calls start at the beginning of every night:

1. night_plan   (natural-language understanding + plan adaptation): reads tonight's forecast and the current
   bulletin and decides whether tonight is a bad night for faint must-observe targets and which compass
   sectors to keep away from. The planner uses both answers for the whole night.
2. fault_review (data parsing + action decision): reads the agent's own hour-by-hour quality table of the last
   nights and judges how likely an unannounced instrument fault is. The answer sets how readily the agent
   reports (probes) a fault tonight.

A third, occasional call confirms a paid fault report before it is sent.

Calls run in the background (llm_client.Call); the agent waits for them only as long as the wall clock allows
and keeps planning otherwise. Every answer is validated; a missing or invalid answer leaves the rule-based
value in place for that night.
"""
from __future__ import annotations

import json
import math
import time
import copy

DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}

NIGHT_PLAN_SYSTEM = (
    "You plan one night of a robotic spectroscopic survey. The input lists tonight's weather forecast notices and the "
    "current bulletin; each notice is an event kind and a compass sector (N, NE, E, SE, S, SW, W, NW) or ALL (the "
    "whole sky). Optional planning_context contains the survey's exposure limits, required threshold, and program "
    "bands or multipliers. Use those supplied values to assess weather impact. Do not assume a target class, exposure "
    "duration, or priority rule that is absent from the input. Decide two things.\n"
    "bad_night: true only when the supplied forecast, bulletin, and planning context support that conclusion for this "
    "night.\n"
    "avoid_directions: the sectors with rain, storm, overcast, haze or cold_snap tonight. Ignore earthquake, "
    "rocket_launch and terrain_obstruction (the scheduler handles those itself). Never list a sector nothing names.\n"
    'Reply with one JSON object only: {"bad_night": true|false, "avoid_directions": ["SW", ...], "reason": "<12 words"}'
)

FAULT_REVIEW_SYSTEM = (
    "You watch the data quality of a robotic telescope. An instrument fault is never announced: it lowers the "
    "instrument efficiency, and so the quality of every exposure, until someone reports it; a correct report "
    "repairs it at once. An earthquake (it appears in the bulletin) also lowers instrument efficiency, and that loss "
    "fades night by night; a report does not repair it. Weather lowers quality too, but it also lowers the program "
    "band, which the instrument does not affect.\n"
    "Columns per hour: E = measured quality / quality the program bands allow (about 1 when healthy; low when the "
    "instrument is the cause; in a very clear sky the bands bound it only loosely, so it can stay near 1), scale = "
    "measured sky quality relative to the clear-sky model, ref = the usual scale since the last repair. "
    "notices_now lists the current bulletin.\n"
    "Signs of a fault: quality that drops and stays down without recovering, E low for many hours across nights, "
    "not explained by announced weather or by a recent earthquake whose effect is fading.\n"
    "A correct report repairs the instrument. Reporting rewards, false-report penalties, and any free allowance are "
    "provided in the input's reporting values; use those values and do not assume fixed point amounts.\n"
    'Reply with one JSON object only: {"fault_likely": <0..1>, "reason": "<15 words"}'
)

CONFIRM_SYSTEM = (
    "You check the evidence for an unannounced instrument fault on a robotic telescope before a report that may carry "
    "a penalty. Reporting rewards and false-report penalties are supplied in the input; use those values and do not "
    "assume fixed point amounts. A correct report repairs the instrument. E per hour = measured quality / "
    "quality the program bands allow: about 1 when healthy, low while the instrument is the cause. Weather lowers both "
    "quality and band; an earthquake lowers instrument efficiency in a way that fades night by night and that a "
    "report does not repair.\n"
    'Reply with one JSON object only: {"report": true|false, "reason": "<15 words"}'
)


_REPORTING_FIELDS = (
    "correct_reward", "correct_report_reward", "report_reward", "correct_report_points",
    "false_penalty", "false_report_penalty", "false_report_cost", "false_report_points",
    "false_report_free_allowance", "free_false_reports_left",
)


def _safe_public_value(value, depth: int = 0):
    if depth > 4:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:80]
    if isinstance(value, (list, tuple)):
        return [_safe_public_value(item, depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:64]:
            if not isinstance(key, str):
                continue
            clean = _safe_public_value(item, depth + 1)
            if clean is not None:
                result[key[:64]] = clean
        return result
    return None


def _reporting_prompt(payload: dict) -> str:
    reporting = payload.get("reporting") if isinstance(payload, dict) else None
    if not isinstance(reporting, dict):
        return "Reporting reward and penalty values are absent from this input; do not assume numeric values."
    values = {}
    for name in _REPORTING_FIELDS:
        value = reporting.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values[name] = value
    if not values:
        return "Reporting reward and penalty values are absent from this input; do not assume numeric values."
    rendered = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return f"Public reporting values from the input: {rendered}. Use these values; do not assume fixed point amounts."


class Advisor:
    def __init__(self, client, log=lambda text: None):
        self.client = client
        self.log = log
        self.plan_call = None
        self.fault_call = None
        self.plan_applied = True
        self.fault_applied = True
        self.night_date = None
        self.announced: set = set()
        self._pending_submissions = {}

    # --- night start ---------------------------------------------------------------------------------

    def start_night(self, night_date: str, tonight: list, bulletin: list, fault_table: dict, wallclock_left: float,
                    wait_seconds: float):
        """Submit both calls; wait up to wait_seconds for them. Returns (plan, fault) answers that are ready."""
        self.night_date = night_date
        self.announced = {n.get("direction") for n in tonight + bulletin if n.get("event_kind") in WEATHER_KINDS}
        self.plan_call = self.fault_call = None
        self.plan_applied = self.fault_applied = False
        self._pending_submissions = {}
        notices = {"night": night_date,
                   "forecast_tonight": [{"event_kind": n.get("event_kind"), "direction": n.get("direction")} for n in tonight],
                   "bulletin_now": [{"event_kind": n.get("event_kind"), "direction": n.get("direction")} for n in bulletin]}
        planning_context = fault_table.get("planning_context") if isinstance(fault_table, dict) else None
        if isinstance(planning_context, dict):
            notices["planning_context"] = _safe_public_value(planning_context)
        if not getattr(self.client, "enabled", True):
            self.plan_applied = self.fault_applied = True
            return None, None
        fault_system = FAULT_REVIEW_SYSTEM + "\n" + _reporting_prompt(fault_table)
        self._pending_submissions = {
            "night_plan": (NIGHT_PLAN_SYSTEM, copy.deepcopy(notices)),
            "fault_review": (fault_system, copy.deepcopy(fault_table)),
        }
        self._submit_pending("night_plan", wallclock_left)
        self._submit_pending("fault_review", wallclock_left)
        deadline = time.monotonic() + max(0.0, wait_seconds)
        for call in (self.plan_call, self.fault_call):
            if call is not None:
                call.wait(deadline - time.monotonic())
        return self.poll()

    def poll(self, wallclock_left: float | None = None):
        """(plan, fault) answers that arrived since the last poll; None for each one not (newly) available."""
        if wallclock_left is not None and getattr(self.client, "enabled", True):
            self._submit_pending("night_plan", wallclock_left)
            self._submit_pending("fault_review", wallclock_left)
        plan = fault = None
        if not self.plan_applied and self.plan_call is not None and self.plan_call.done():
            self.plan_applied = True
            plan = self._valid_plan(self.client.collect(self.plan_call))
        if not self.fault_applied and self.fault_call is not None and self.fault_call.done():
            self.fault_applied = True
            fault = self._valid_fault(self.client.collect(self.fault_call))
        return plan, fault

    def _submit_pending(self, tag: str, wallclock_left: float) -> None:
        request = self._pending_submissions.get(tag)
        if request is None:
            return
        if tag == "night_plan":
            call_attr, applied_attr = "plan_call", "plan_applied"
        else:
            call_attr, applied_attr = "fault_call", "fault_applied"
        if getattr(self, call_attr) is not None or getattr(self, applied_attr):
            self._pending_submissions.pop(tag, None)
            return
        if not getattr(self.client, "enabled", True):
            self._pending_submissions.pop(tag, None)
            setattr(self, applied_attr, True)
            return
        in_flight = getattr(self.client, "in_flight", None)
        if callable(in_flight) and in_flight() >= getattr(self.client, "max_in_flight", 4):
            return
        system, user = request
        call = self.client.submit(tag, system, copy.deepcopy(user), wallclock_left)
        if call is not None:
            setattr(self, call_attr, call)
            setattr(self, applied_attr, False)
            self._pending_submissions.pop(tag, None)

    def _valid_plan(self, answer):
        if not isinstance(answer, dict) or not isinstance(answer.get("bad_night"), bool):
            return None
        avoid = answer.get("avoid_directions", [])
        if not isinstance(avoid, list):
            return None
        # the model may rank announced weather; it may not close sky that nothing announced
        avoid = sorted({str(d).upper() for d in avoid if str(d).upper() in DIRECTIONS and str(d).upper() in self.announced})
        return {"bad_night": answer["bad_night"], "avoid_directions": avoid, "reason": str(answer.get("reason", ""))[:80]}

    @staticmethod
    def _valid_fault(answer):
        if not isinstance(answer, dict):
            return None
        try:
            p = float(answer.get("fault_likely"))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= p <= 1.0:
            return None
        return {"fault_likely": p, "reason": str(answer.get("reason", ""))[:80]}

    # --- paid report confirmation ----------------------------------------------------------------------

    def confirm_report(self, evidence: dict, wallclock_left: float, wait_seconds: float):
        """True / False from the model, or None (no answer in time: the rule decides).

        Prefer v4-style sync ask_json (bounded ~18s) so a hung endpoint cannot
        park the decision loop; fall back to background submit+wait when ask_json
        is unavailable. Missing/invalid answers leave the rule verdict in place.
        """
        if not getattr(self.client, "enabled", True):
            return None
        confirm_system = CONFIRM_SYSTEM + "\n" + _reporting_prompt(evidence)
        ask_json = getattr(self.client, "ask_json", None)
        if callable(ask_json):
            # Cap the sync budget by the agent-provided wait so night start /
            # pace accounting stays consistent with _model_wait_budget.
            budget = min(float(wallclock_left), float(wait_seconds) + 300.0)
            answer = ask_json(confirm_system, evidence, budget)
        else:
            call = self.client.submit("confirm_report", confirm_system, evidence, wallclock_left)
            if call is None:
                return None
            call.wait(wait_seconds)
            answer = self.client.collect(call)
        if isinstance(answer, dict) and isinstance(answer.get("report"), bool):
            return answer["report"]
        return None
