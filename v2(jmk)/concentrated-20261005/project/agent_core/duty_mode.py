"""Duty-mode table: message triage L0–L4 → NORMAL/PROTECT/RECOVER/CONSERVE.

Single decision path only. Classification is O(1) table lookup driven by PIR-style
facts already on SurveyState / the current decision payload. Advisors may suggest
priority/risk elsewhere; hard modes here override those soft knobs. Platform
protocol and scoring remain the highest authority — this module only switches
JointSearch constraints, it never invents extra actions outside jsonl-v4.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

# Mutual-exclusive main modes (strategy table).
MODE_NORMAL = "NORMAL"
MODE_PROTECT = "PROTECT"
MODE_RECOVER = "RECOVER"
MODE_CONSERVE = "CONSERVE"

# Message precedence: smaller number = more urgent.
LEVEL_L0 = 0  # invalidation / state_resync
LEVEL_L1 = 1  # timed request window
LEVEL_L2 = 2  # required debt urgency
LEVEL_L3 = 3  # quality / weather downgrade
LEVEL_L4 = 4  # routine science

PROTO_RQ = "P-RQ"    # request-window continuous pointing
PROTO_REQ = "P-REQ"  # required-debt recovery
PROTO_Q = "P-Q"      # low-quality: short exposures, no long idle
PROTO_RS = "P-RS"    # resync re-observation ledger

# Engineering thresholds (loosened after practice −1.1万 diagnosis 2026-10-05).
# Strict protocol doc remains the intent; these numbers avoid permanent CONSERVE
# when early-season required debt is large but nights remain, and avoid capping
# DARK science on only mildly sub-par quality.
REQUEST_URGENT_HOURS = 12.0
REQUEST_CRITICAL_HOURS = 4.0
QUALITY_LOW = 0.45          # was 0.70 — only truly poor last shot
QUALITY_MEDIAN_LOW = 0.40   # was 0.60
QUALITY_MONTH_LOW = 0.35    # was 0.50
DEBT_URGENT_RATIO = 0.45    # was 0.15 — clear >45% of open debt per remaining night
DEBT_CRITICAL = 60          # was 10 — practice/formal open hundreds early; 10 always fired
DEBT_SOFT_FLOOR = 8         # ignore tiny residual debt for mode switch
# Rough REQUIRED clears per night when JointSearch is debt-aware (multi-fiber).
EST_REQUIRED_PER_NIGHT = 12.0
DEBT_SCHEDULE_SLACK = 0.85  # urgent if nights_needed > nights_left * slack
RECOVER_WINDOW_HOURS = 48.0
SHORT_EXPOSURE_CAP = 1800   # was 600 — still below optimistic 3600 on bad quality
VERY_SHORT_EXPOSURE_CAP = 900  # was 300
# PROTECT only when the open window is actually short / critical (not all season).
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


def _parse_utc(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError, AttributeError):
        return None


def _required_debt(state) -> tuple[int, int]:
    """(open required count D, remaining nights with any required still open)."""
    threshold = state.scoring.required_threshold
    debt = 0
    last_nights = []
    for i, required in enumerate(state.required):
        if required and state.factor[i] < threshold:
            debt += 1
            last_nights.append(state.last_night[i])
    if debt == 0:
        return 0, 0
    # Approximate remaining observable nights for the hardest remaining target.
    nights_left = max(0, max(last_nights) - getattr(state, "_duty_night_index", 0) + 1)
    return debt, nights_left


def _rolling_quality(state) -> tuple[float, float, float]:
    """(last exposure proxy scale, median of last ≤5 samples, monthly-ish prior)."""
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


def _active_request_urgency(request_views, now: datetime) -> tuple[bool, float, bool]:
    """Return (any_open, min_hours_left, critical_window)."""
    if not request_views:
        return False, float("inf"), False
    hours_left = []
    for view in request_views:
        deadline = view.get("deadline")
        if deadline is None:
            continue
        hours_left.append(max(0.0, (deadline - now).total_seconds() / 3600.0))
    if not hours_left:
        return False, float("inf"), False
    remaining = min(hours_left)
    return True, remaining, remaining <= REQUEST_CRITICAL_HOURS


def classify(
    state,
    *,
    now: datetime,
    night_index: int,
    hours: float,
    request_views: list,
    last_result: Optional[dict] = None,
) -> ModeDecision:
    """Map current PIR facts → one main mode + hard constraints (O(1))."""
    state._duty_night_index = night_index
    protocols: list[str] = []
    reason_parts: list[str] = []

    # --- L0 / P-RS: recent invalidation or resync window ---
    recover_until = getattr(state, "recover_until_hours", float("-inf"))
    new_invalidations = False
    if last_result and last_result.get("action") == "observe":
        # Soft signal only; authoritative L0 is state_resync handled in on_messages.
        pass
    if getattr(state, "_resync_just_happened", False):
        new_invalidations = True
        state.recover_until_hours = hours + RECOVER_WINDOW_HOURS
        state._resync_just_happened = False
        state.recover_targets = _collect_recover_targets(state)
    in_recover = hours <= float(getattr(state, "recover_until_hours", float("-inf")))

    # --- L1 / P-RQ: timed request window ---
    has_request, hours_left, critical = _active_request_urgency(request_views, now)

    # --- L2 / P-REQ: required debt ---
    # Early season often has D≫60. Old D/N>0.15 fired forever. Escalate only when
    # estimated nights-to-clear exceeds remaining nights (capacity model), debt is
    # rising, or a short tail remains with large open debt.
    debt, nights_left = _required_debt(state)
    debt_ratio = debt / max(nights_left, 1)  # retained for logs / reason strings
    nights_needed = debt / EST_REQUIRED_PER_NIGHT
    debt_rising = bool(getattr(state, "required_debt_rising", False))
    debt_urgent = debt > DEBT_SOFT_FLOOR and (
        nights_needed > nights_left * DEBT_SCHEDULE_SLACK
        or debt_ratio > DEBT_URGENT_RATIO and nights_left <= 20
        or (debt >= DEBT_CRITICAL and nights_left <= 12)
        or debt_rising
    )
    debt_watch = debt > DEBT_SOFT_FLOOR and not debt_urgent

    # --- L3 / P-Q: quality ---
    last_q, median5, month_q = _rolling_quality(state)
    weather_bad = _local_weather_degraded(state)
    quality_bad = (
        last_q < QUALITY_LOW
        or median5 < QUALITY_MEDIAN_LOW
        or month_q < QUALITY_MONTH_LOW
        or weather_bad
    )
    very_bad = median5 < 0.30 or last_q < 0.30
    # Soft request boost without full PROTECT when the window is still long.
    request_protect = has_request and (
        critical or hours_left <= PROTECT_ALWAYS_HOURS
    )

    # Precedence: L0 > L1 > L2 > L3 > L4
    if new_invalidations or in_recover:
        level = LEVEL_L0
        mode = MODE_RECOVER
        protocols.append(PROTO_RS)
        if quality_bad:
            protocols.append(PROTO_Q)
        if has_request:
            protocols.append(PROTO_RQ)
        if debt_urgent or debt_watch:
            protocols.append(PROTO_REQ)
        reason_parts.append("resync/invalidation recover window")
        cap = SHORT_EXPOSURE_CAP if quality_bad else None
        if very_bad:
            cap = VERY_SHORT_EXPOSURE_CAP
        return ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="required" if debt_urgent or not has_request else "request",
            advice_risk="conservative" if quality_bad else "balanced",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=("BACKUP", "BRIGHT", "DARK") if quality_bad else ("DARK", "BRIGHT", "BACKUP"),
            request_value_scale=1.35 if has_request else 1.0,
            required_value_scale=1.4,
            recover_value_scale=1.5,
            request_slot_share=0.6,
            reason="; ".join(reason_parts) or "RECOVER",
        )

    if request_protect:
        level = LEVEL_L1
        mode = MODE_PROTECT
        protocols.append(PROTO_RQ)
        if debt_urgent or debt_watch:
            protocols.append(PROTO_REQ)
        if quality_bad:
            protocols.append(PROTO_Q)
        reason_parts.append(f"request window {hours_left:.1f}h left")
        share = 0.9 if critical or hours_left <= REQUEST_URGENT_HOURS else 0.7
        cap = SHORT_EXPOSURE_CAP if quality_bad else None
        if very_bad:
            cap = VERY_SHORT_EXPOSURE_CAP
        return ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="request",
            advice_risk="conservative" if quality_bad else "balanced",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=("BACKUP", "BRIGHT", "DARK") if quality_bad else ("DARK", "BRIGHT", "BACKUP"),
            request_value_scale=1.6 if critical else (1.35 if hours_left <= REQUEST_URGENT_HOURS else 1.2),
            required_value_scale=1.25 if debt_urgent else (1.1 if debt_watch else 1.05),
            recover_value_scale=1.0,
            request_slot_share=share,
            reason="; ".join(reason_parts),
        )

    if debt_urgent or (debt > DEBT_SOFT_FLOOR and month_q < QUALITY_MONTH_LOW):
        level = LEVEL_L2
        mode = MODE_CONSERVE
        protocols.append(PROTO_REQ)
        if quality_bad:
            protocols.append(PROTO_Q)
        reason_parts.append(f"required debt D={debt} nights_left≈{nights_left}")
        # Good-sky debt nights: keep DARK uncapped so science tax base survives;
        # only clamp exposure when quality / month prior is actually bad.
        cap = None
        if quality_bad or month_q < QUALITY_MONTH_LOW:
            cap = SHORT_EXPOSURE_CAP
        if very_bad:
            cap = VERY_SHORT_EXPOSURE_CAP
        return ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="required",
            advice_risk="conservative" if quality_bad else "balanced",
            max_exposure_cap=cap,
            forbid_idle_wait=quality_bad or very_bad or debt_rising,
            prefer_programs=("BACKUP", "BRIGHT", "DARK") if quality_bad else ("DARK", "BRIGHT", "BACKUP"),
            request_value_scale=1.15 if has_request else 1.0,
            required_value_scale=1.45 if debt >= DEBT_CRITICAL else 1.25,
            recover_value_scale=1.0,
            request_slot_share=0.15 if has_request else 0.0,
            reason="; ".join(reason_parts),
        )

    if quality_bad:
        level = LEVEL_L3
        mode = MODE_CONSERVE
        protocols.append(PROTO_Q)
        if debt_watch:
            protocols.append(PROTO_REQ)
        reason_parts.append(f"quality last={last_q:.2f} med5={median5:.2f}")
        cap = VERY_SHORT_EXPOSURE_CAP if very_bad else SHORT_EXPOSURE_CAP
        return ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="required" if debt_watch else "balanced",
            advice_risk="conservative",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=("BACKUP", "BRIGHT", "DARK"),
            request_value_scale=1.15 if has_request else 1.0,
            required_value_scale=1.2 if debt_watch else 1.1,
            recover_value_scale=1.0,
            request_slot_share=0.1 if has_request else 0.0,
            reason="; ".join(reason_parts),
        )

    # L4 NORMAL — optional soft request / debt watch without mode lock.
    soft_protocols: list[str] = []
    req_scale = 1.0
    required_scale = 1.0
    slot_share = 0.0
    if has_request:
        soft_protocols.append(PROTO_RQ)
        req_scale = 1.15
        slot_share = 0.25
        reason_parts.append(f"open request {hours_left:.1f}h (soft)")
    if debt_watch:
        soft_protocols.append(PROTO_REQ)
        required_scale = 1.15
        reason_parts.append(f"debt watch D={debt}")
    return ModeDecision(
        level=LEVEL_L4,
        mode=MODE_NORMAL,
        protocols=tuple(soft_protocols),
        advice_priority="request" if has_request and not debt_watch else (
            "required" if debt_watch else "balanced"
        ),
        advice_risk="balanced",
        max_exposure_cap=None,
        forbid_idle_wait=False,
        prefer_programs=("DARK", "BRIGHT", "BACKUP"),
        request_value_scale=req_scale,
        required_value_scale=required_scale,
        recover_value_scale=1.0,
        request_slot_share=slot_share,
        reason="; ".join(reason_parts) if reason_parts else "routine science",
    )


def _local_weather_degraded(state) -> bool:
    """True when bulletin suggests *widespread* bad weather (not one-sector cloud).

    Practice logs showed directional cloud/haze flipping P-Q forever; require
    either all-sky non-storm overcast or ≥2 distinct bad-notice sectors.
    Whole-sky rain/storm still goes through site_closed → wait.
    """
    if state.site_closed():
        return False
    bad_kinds = {"rain", "cloud", "haze", "fog", "dust", "smoke"}
    sector_hits = 0
    for key in state.notices:
        kind, _, direction = key.partition("|")
        if kind not in bad_kinds:
            continue
        if direction == "ALL":
            # All-sky rain/storm → site_closed; residual all-sky cloud/haze → conserve.
            if kind not in ("rain", "storm"):
                return True
            continue
        sector_hits += 1
        if sector_hits >= 2:
            return True
    return False


def _collect_recover_targets(state) -> set[int]:
    """Targets that should be re-tried after invalidation: required, incomplete, high score."""
    threshold = state.scoring.required_threshold
    out: set[int] = set()
    for i in range(len(state.ids)):
        if state.hmax[i] <= 0:
            continue
        if state.required[i] and state.factor[i] < threshold:
            out.add(i)
            continue
        # High science value still incomplete after rollback.
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
    # Hard mode overrides soft advisor knobs (advisor remains best-effort only).
    state.advice_priority = decision.advice_priority
    state.advice_risk = decision.advice_risk
