"""Duty sheet types + apply knobs. Decision brain lives in pilot.py (menu pick).

Kept for JointSearch / state field names (duty_mode, protocols, scales).
`classify()` is a thin alias of `pilot.decide` — the old L0–L4 emergency
flowchart is retired as the main brain (see pilot.py docstring).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# Mutual-exclusive mode *labels* (outputs of Pilot menu pick, not a priority automaton).
MODE_NORMAL = "NORMAL"
MODE_PROTECT = "PROTECT"
MODE_RECOVER = "RECOVER"
MODE_CONSERVE = "CONSERVE"

LEVEL_L0 = 0
LEVEL_L1 = 1
LEVEL_L2 = 2
LEVEL_L3 = 3
LEVEL_L4 = 4

PROTO_RQ = "P-RQ"
PROTO_REQ = "P-REQ"
PROTO_Q = "P-Q"
PROTO_RS = "P-RS"

# Engineering thresholds still used when Pilot maps a choice → knobs.
REQUEST_URGENT_HOURS = 12.0
REQUEST_CRITICAL_HOURS = 4.0
QUALITY_LOW = 0.42
QUALITY_MEDIAN_LOW = 0.40
QUALITY_MONTH_LOW = 0.35
DEBT_CRITICAL = 60
DEBT_SOFT_FLOOR = 3
DEBT_SPRINT_NIGHTS = 6
DEBT_FINAL_NIGHTS = 3
RECOVER_WINDOW_HOURS = 48.0
SHORT_EXPOSURE_CAP = 1800
VERY_SHORT_EXPOSURE_CAP = 900
PROTECT_ALWAYS_HOURS = 18.0


@dataclass(frozen=True)
class ModeDecision:
    """Immutable duty sheet handed to JointSearch for one decision step."""

    level: int
    mode: str
    protocols: tuple[str, ...]
    advice_priority: str
    advice_risk: str
    max_exposure_cap: Optional[int]
    forbid_idle_wait: bool
    prefer_programs: tuple[str, ...]
    request_value_scale: float
    required_value_scale: float
    recover_value_scale: float
    request_slot_share: float
    reason: str
    pilot_long: str = ""
    pilot_mid: str = ""
    pilot_short: str = ""
    copilot_reason: str = ""


def _parse_utc(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError, AttributeError):
        return None


def _rolling_quality(state) -> tuple[float, float, float]:
    samples = list(state._all_ratios)
    last = float(state.scale) if state.scale else 1.0
    if samples:
        recent = samples[-5:]
        ordered = sorted(recent)
        median5 = ordered[len(ordered) // 2]
    else:
        median5 = last
    month = float(state.prior_scale) if state.prior_scale else last
    return last, median5, month


def _local_weather_degraded(state) -> bool:
    if state.site_closed():
        return False
    bad_kinds = {"rain", "cloud", "haze", "fog", "dust", "smoke"}
    sector_hits = 0
    for key in state.notices:
        kind, _, direction = key.partition("|")
        if kind not in bad_kinds:
            continue
        if direction == "ALL":
            if kind not in ("rain", "storm"):
                return True
            continue
        sector_hits += 1
        if sector_hits >= 2:
            return True
    return False


def _collect_recover_targets(state) -> set[int]:
    threshold = state.scoring.required_threshold
    out: set[int] = set()
    for i in range(len(state.ids)):
        if state.hmax[i] <= 0:
            continue
        if state.required[i] and state.factor[i] < threshold:
            out.add(i)
            continue
        if state.best_score[i] <= 0 and state.weight[i] >= 1.0:
            out.add(i)
    return out


def apply_to_state(state, decision: ModeDecision) -> None:
    """Write duty decision onto SurveyState knobs JointSearch already reads."""
    state.duty_mode = decision.mode
    state.duty_level = decision.level
    state.duty_protocols = decision.protocols
    state.duty_reason = decision.reason
    state.duty_max_exposure_cap = decision.max_exposure_cap
    state.duty_forbid_idle_wait = decision.forbid_idle_wait
    state.duty_prefer_programs = decision.prefer_programs
    state.duty_request_value_scale = decision.request_value_scale
    state.duty_required_value_scale = decision.required_value_scale
    state.duty_recover_value_scale = decision.recover_value_scale
    state.duty_request_slot_share = decision.request_slot_share
    state.advice_priority = decision.advice_priority
    state.advice_risk = decision.advice_risk
    state.copilot_long = decision.pilot_long
    state.copilot_mid = decision.pilot_mid
    state.copilot_short = decision.pilot_short


def classify(
    state,
    *,
    now: datetime,
    night_index: int,
    hours: float,
    request_views: list,
    last_result: Optional[dict] = None,
) -> ModeDecision:
    """Compatibility entry: delegates to Pilot menu decide (not L0–L4 brain)."""
    from . import pilot as pilot_mod
    return pilot_mod.decide(
        state,
        now=now,
        night_index=night_index,
        hours=hours,
        request_views=request_views,
        last_result=last_result,
    )
