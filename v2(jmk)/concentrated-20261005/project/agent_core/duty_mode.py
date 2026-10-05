"""Duty-mode table: message triage L0–L4 → NORMAL/PROTECT/RECOVER/CONSERVE.

v3 Pilot sheet. Classification is O(1) table lookup driven by PIR-style facts
already on SurveyState / the current decision payload, plus a thin Copilot menu
(see copilot.py). Advisors may suggest priority/risk elsewhere; hard modes here
override those soft knobs. Platform protocol and scoring remain the highest
authority — this module only switches JointSearch constraints, it never invents
extra actions outside jsonl-v4.

Fiducials (v3):
  - tolerate required debt ≤ 3 (stop hard CONSERVE lock)
  - main debt clear only after ~20% season nights AND capacity shortfall
  - avoid streak ≥ 3 bad-Q nights with debt not falling → force deep / tolerate
  - abandoned timed requests are never re-PROTECTed (no thrash)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import copilot as cp

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

# Engineering thresholds (v3). Fiducial debt/avoid gates live in copilot.py.
REQUEST_URGENT_HOURS = 12.0
REQUEST_CRITICAL_HOURS = cp.REQUEST_CRITICAL_HOURS
QUALITY_LOW = 0.42
QUALITY_MEDIAN_LOW = 0.40
QUALITY_MONTH_LOW = 0.35
DEBT_URGENT_RATIO = 0.40
DEBT_CRITICAL = 60
DEBT_TOLERANCE = cp.DEBT_TOLERANCE          # ≤3 → stop hard CONSERVE
DEBT_CLEAR_AFTER_FRAC = cp.DEBT_CLEAR_AFTER_FRAC  # 0.20
DEBT_SOFT_FLOOR = DEBT_TOLERANCE
EST_REQUIRED_PER_NIGHT = cp.EST_REQUIRED_PER_NIGHT
DEBT_SCHEDULE_SLACK = 0.95
DEBT_SPRINT_NIGHTS = 6
DEBT_FINAL_NIGHTS = 3
RECOVER_WINDOW_HOURS = 48.0
SHORT_EXPOSURE_CAP = 1800
VERY_SHORT_EXPOSURE_CAP = 900
PROTECT_ALWAYS_HOURS = 18.0
AVOID_MAX_STREAK_NIGHTS = cp.AVOID_MAX_STREAK_NIGHTS


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
    # v3 Pilot/Copilot provenance (for logs; JointSearch ignores extras via apply).
    pilot_long: str = ""
    pilot_mid: str = ""
    pilot_short: str = ""
    copilot_reason: str = ""


def _parse_utc(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError, AttributeError):
        return None


def _required_debt(state) -> tuple[int, int]:
    return cp.required_debt_pair(state, getattr(state, "_duty_night_index", 0))


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


def _ensure_v3_state(state) -> None:
    if not hasattr(state, "abandoned_request_ids"):
        state.abandoned_request_ids = set()
    if not hasattr(state, "avoid_streak_nights"):
        state.avoid_streak_nights = 0
    if not hasattr(state, "avoid_last_night_index"):
        state.avoid_last_night_index = -1
    if not hasattr(state, "avoid_debt_anchor"):
        state.avoid_debt_anchor = 0
    if not hasattr(state, "pilot_force_deep"):
        state.pilot_force_deep = False
    if not hasattr(state, "pilot_tolerate_debt"):
        state.pilot_tolerate_debt = False
    if not hasattr(state, "copilot_long"):
        state.copilot_long = ""
        state.copilot_mid = ""
        state.copilot_short = ""


def _update_avoid_streak(state, night_index: int, quality_bad: bool, debt: int) -> int:
    """Count consecutive bad-Q nights where required debt did not fall."""
    last = int(getattr(state, "avoid_last_night_index", -1))
    if night_index != last:
        if quality_bad:
            if last < 0 or night_index == last + 1:
                # Continuing streak: debt must not have dropped since anchor.
                if state.avoid_streak_nights == 0:
                    state.avoid_debt_anchor = debt
                    state.avoid_streak_nights = 1
                else:
                    if debt >= state.avoid_debt_anchor:
                        state.avoid_streak_nights += 1
                    else:
                        state.avoid_debt_anchor = debt
                        state.avoid_streak_nights = 1
            else:
                # Gap in nights → reset.
                state.avoid_debt_anchor = debt
                state.avoid_streak_nights = 1
        else:
            state.avoid_streak_nights = 0
            state.avoid_debt_anchor = debt
        state.avoid_last_night_index = night_index
    return int(state.avoid_streak_nights)


def classify(
    state,
    *,
    now: datetime,
    night_index: int,
    hours: float,
    request_views: list,
    last_result: Optional[dict] = None,
) -> ModeDecision:
    """Map current PIR facts + Copilot brief → one main mode + hard constraints."""
    _ensure_v3_state(state)
    state._duty_night_index = night_index
    protocols: list[str] = []
    reason_parts: list[str] = ["v3"]

    # --- L0 / P-RS: recent invalidation or resync window ---
    new_invalidations = False
    if getattr(state, "_resync_just_happened", False):
        new_invalidations = True
        state.recover_until_hours = hours + RECOVER_WINDOW_HOURS
        state._resync_just_happened = False
        state.recover_targets = _collect_recover_targets(state)
    in_recover = hours <= float(getattr(state, "recover_until_hours", float("-inf")))

    # Filter out permanently abandoned timed requests (no thrash).
    abandoned = set(state.abandoned_request_ids)
    live_views = [v for v in (request_views or []) if str(v.get("id")) not in abandoned]

    # --- quality ---
    last_q, median5, month_q = _rolling_quality(state)
    weather_bad = _local_weather_degraded(state)
    quality_bad = (
        last_q < QUALITY_LOW
        or median5 < QUALITY_MEDIAN_LOW
        or month_q < QUALITY_MONTH_LOW
        or weather_bad
    )
    very_bad = median5 < 0.30 or last_q < 0.30

    debt, nights_left = _required_debt(state)
    avoid_streak = _update_avoid_streak(state, night_index, quality_bad, debt)

    # Copilot menu (table + arithmetic only).
    stamped = cp.stamp_hours_left(live_views, now)
    brief = cp.build_brief(
        state,
        night_index=night_index,
        request_views=stamped,
        quality_bad=quality_bad,
        very_bad=very_bad,
        avoid_streak=avoid_streak,
        abandoned=abandoned,
        in_recover=in_recover or new_invalidations,
    )
    state.copilot_long = brief.long_name
    state.copilot_mid = brief.mid_name
    state.copilot_short = brief.short_name
    state.pilot_force_deep = brief.force_deep
    state.pilot_tolerate_debt = brief.tolerate_debt

    # Pilot adopts Copilot abandon recommendations permanently (no 横跳).
    for rid in brief.abandon_request_ids:
        state.abandoned_request_ids.add(str(rid))
        reason_parts.append(f"abandon_request:{rid}")
    if brief.abandon_request_ids:
        abandoned = set(state.abandoned_request_ids)
        live_views = [v for v in live_views if str(v.get("id")) not in abandoned]
        stamped = cp.stamp_hours_left(live_views, now)

    has_request, hours_left, critical = _active_request_urgency(live_views, now)

    # --- L2 debt urgency (v3: later clear + tolerate ≤3) ---
    debt_rising = bool(getattr(state, "required_debt_rising", False))
    debt_ratio = debt / max(nights_left, 1)
    nights_needed = debt / EST_REQUIRED_PER_NIGHT
    past_gate = brief.past_clear_gate
    capacity_short = brief.capacity_short

    # Tolerate ≤3: never hard-lock CONSERVE for residual debt.
    if debt <= DEBT_TOLERANCE:
        debt_urgent = False
        debt_watch = False
        if debt > 0:
            reason_parts.append(f"tolerate_debt D={debt}")
    else:
        # Main clear only when past ~20% nights AND nights not enough to clear.
        # Keep late sprint / rising as secondary once past the gate.
        debt_urgent = past_gate and (
            capacity_short
            or (debt_ratio > DEBT_URGENT_RATIO and nights_left <= 20)
            or (debt >= DEBT_CRITICAL and nights_left <= 16)
            or (debt_rising and capacity_short)
        )
        debt_watch = not debt_urgent  # soft weight only; no CONSERVE lock

    request_protect = has_request and (
        critical or hours_left <= PROTECT_ALWAYS_HOURS
    )
    # If Copilot/Pilot chose 放弃限时, do not PROTECT those ids (already filtered).

    force_deep = brief.force_deep
    if force_deep:
        reason_parts.append(f"avoid_streak={avoid_streak}→force_deep")

    def _provenance(decision: ModeDecision) -> ModeDecision:
        return ModeDecision(
            level=decision.level,
            mode=decision.mode,
            protocols=decision.protocols,
            advice_priority=decision.advice_priority,
            advice_risk=decision.advice_risk,
            max_exposure_cap=decision.max_exposure_cap,
            forbid_idle_wait=decision.forbid_idle_wait,
            prefer_programs=decision.prefer_programs,
            request_value_scale=decision.request_value_scale,
            required_value_scale=decision.required_value_scale,
            recover_value_scale=decision.recover_value_scale,
            request_slot_share=decision.request_slot_share,
            reason=decision.reason,
            pilot_long=brief.long_name,
            pilot_mid=brief.mid_name,
            pilot_short=brief.short_name,
            copilot_reason=brief.reason,
        )

    # Precedence: L0 > L1 > L2 > L3 > L4
    if new_invalidations or in_recover:
        level = LEVEL_L0
        mode = MODE_RECOVER
        protocols.append(PROTO_RS)
        if quality_bad and not force_deep:
            protocols.append(PROTO_Q)
        if has_request:
            protocols.append(PROTO_RQ)
        if debt_urgent or debt_watch:
            protocols.append(PROTO_REQ)
        reason_parts.append("resync/invalidation recover window")
        cap = None if force_deep else (SHORT_EXPOSURE_CAP if quality_bad else None)
        if very_bad and not force_deep:
            cap = VERY_SHORT_EXPOSURE_CAP
        prefer = ("DARK", "BRIGHT", "BACKUP") if force_deep else (
            ("BACKUP", "BRIGHT", "DARK") if quality_bad else ("DARK", "BRIGHT", "BACKUP")
        )
        return _provenance(ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="required" if debt_urgent or not has_request else "request",
            advice_risk="conservative" if quality_bad and not force_deep else "balanced",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=prefer,
            request_value_scale=1.35 if has_request else 1.0,
            required_value_scale=1.4,
            recover_value_scale=1.5,
            request_slot_share=0.6,
            reason="; ".join(reason_parts) or "RECOVER",
        ))

    if request_protect:
        level = LEVEL_L1
        mode = MODE_PROTECT
        protocols.append(PROTO_RQ)
        if debt_urgent or debt_watch:
            protocols.append(PROTO_REQ)
        if quality_bad and not force_deep:
            protocols.append(PROTO_Q)
        reason_parts.append(f"request window {hours_left:.1f}h left")
        if debt_urgent:
            share = 0.8 if critical or hours_left <= REQUEST_URGENT_HOURS else 0.65
        elif debt_watch:
            share = 0.85 if critical or hours_left <= REQUEST_URGENT_HOURS else 0.7
        else:
            share = 0.9 if critical or hours_left <= REQUEST_URGENT_HOURS else 0.7
        cap = None if force_deep else (SHORT_EXPOSURE_CAP if quality_bad else None)
        if very_bad and not force_deep:
            cap = VERY_SHORT_EXPOSURE_CAP
        req_in_protect = 1.35 if debt_urgent else (1.2 if debt_watch else 1.1)
        rq_scale = 1.6 if critical else (1.35 if hours_left <= REQUEST_URGENT_HOURS else 1.2)
        prefer = ("DARK", "BRIGHT", "BACKUP") if force_deep else (
            ("BACKUP", "BRIGHT", "DARK") if quality_bad else ("DARK", "BRIGHT", "BACKUP")
        )
        return _provenance(ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="request",
            advice_risk="conservative" if quality_bad and not force_deep else "balanced",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=prefer,
            request_value_scale=rq_scale,
            required_value_scale=req_in_protect,
            recover_value_scale=1.0,
            request_slot_share=share,
            reason="; ".join(reason_parts),
        ))

    # L2: real capacity urgency after clear-gate; tolerate ≤3 already stripped.
    if debt_urgent:
        level = LEVEL_L2
        mode = MODE_CONSERVE
        protocols.append(PROTO_REQ)
        if quality_bad and not force_deep and not brief.tolerate_debt:
            protocols.append(PROTO_Q)
        reason_parts.append(
            f"required debt D={debt} nights_left≈{nights_left} "
            f"frac={brief.season_frac:.2f} needed≈{nights_needed:.1f}"
        )
        # Deep-not-wide: force_deep or good/mild sky → uncapped DARK.
        if force_deep:
            cap = None
            prefer = ("DARK", "BRIGHT", "BACKUP")
            reason_parts.append("pilot force_deep")
        elif very_bad:
            cap = VERY_SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
        elif quality_bad and (last_q < QUALITY_LOW or median5 < QUALITY_MEDIAN_LOW):
            cap = None
            prefer = ("BRIGHT", "DARK", "BACKUP")
            reason_parts.append("mild-Q deep debt")
        else:
            cap = None
            prefer = ("DARK", "BRIGHT", "BACKUP")
        req_scale = 1.5 if debt >= DEBT_CRITICAL else 1.3
        if quality_bad and debt > DEBT_SOFT_FLOOR:
            req_scale = max(req_scale, 1.55)
            reason_parts.append("rain debt weight")
        if nights_left <= DEBT_SPRINT_NIGHTS and debt > DEBT_SOFT_FLOOR:
            req_scale = max(req_scale, 1.65)
            reason_parts.append("debt sprint weight")
        if nights_left <= DEBT_FINAL_NIGHTS and debt > DEBT_SOFT_FLOOR:
            req_scale = max(req_scale, 1.8)
            reason_parts.append("debt final nights")
        return _provenance(ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="required",
            advice_risk="conservative" if very_bad and not force_deep else (
                "balanced" if not quality_bad or force_deep else "conservative"
            ),
            max_exposure_cap=cap,
            forbid_idle_wait=very_bad or debt_rising or force_deep,
            prefer_programs=prefer,
            request_value_scale=1.1 if has_request else 1.0,
            required_value_scale=req_scale,
            recover_value_scale=1.0,
            request_slot_share=0.12 if has_request else 0.0,
            reason="; ".join(reason_parts),
        ))

    # After avoid-streak force_deep without debt_urgent: still prefer deep NORMAL.
    if force_deep and quality_bad:
        level = LEVEL_L3
        mode = MODE_NORMAL  # Pilot hard stop: leave shallow CONSERVE
        protocols.append(PROTO_REQ)
        reason_parts.append("pilot stop shallow avoid → deep NORMAL")
        return _provenance(ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="required" if debt_watch else "balanced",
            advice_risk="balanced",
            max_exposure_cap=None,
            forbid_idle_wait=True,
            prefer_programs=("DARK", "BRIGHT", "BACKUP"),
            request_value_scale=1.1 if has_request else 1.0,
            required_value_scale=1.45 if debt_watch else 1.2,
            recover_value_scale=1.0,
            request_slot_share=0.1 if has_request else 0.0,
            reason="; ".join(reason_parts),
        ))

    if quality_bad:
        level = LEVEL_L3
        mode = MODE_CONSERVE
        protocols.append(PROTO_Q)
        if debt_watch:
            protocols.append(PROTO_REQ)
        reason_parts.append(f"quality last={last_q:.2f} med5={median5:.2f}")
        if very_bad:
            cap = VERY_SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
        elif debt_watch:
            cap = None
            prefer = ("DARK", "BRIGHT", "BACKUP")
            reason_parts.append("L3 debt watch deep")
        else:
            cap = SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
        return _provenance(ModeDecision(
            level=level,
            mode=mode,
            protocols=tuple(protocols),
            advice_priority="required" if debt_watch else "balanced",
            advice_risk="conservative",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=prefer,
            request_value_scale=1.1 if has_request else 1.0,
            required_value_scale=1.4 if debt_watch else 1.1,
            recover_value_scale=1.0,
            request_slot_share=0.1 if has_request else 0.0,
            reason="; ".join(reason_parts),
        ))

    # L4 NORMAL — soft request / debt watch; tolerate residual ≤3 stays here.
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
        # Soft early lift only after clear-gate; before gate keep science-first.
        required_scale = 1.25 if past_gate else 1.05
        reason_parts.append(f"debt watch D={debt}")
    elif debt > 0 and debt <= DEBT_TOLERANCE:
        reason_parts.append(f"tolerate residual D={debt}")
    return _provenance(ModeDecision(
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
    ))


def _local_weather_degraded(state) -> bool:
    """True when bulletin suggests *widespread* bad weather (not one-sector cloud)."""
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
    """Targets that should be re-tried after invalidation: required, incomplete, high score."""
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
