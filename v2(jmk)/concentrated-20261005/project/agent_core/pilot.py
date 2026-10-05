"""Pilot (值班长): sole signer — picks from Copilot menu, writes JointSearch knobs.

Decision chain (v3 framework landing):
  facts → Copilot menu+EV → Pilot hard-stop / menu pick → knobs+protocol labels
  → JointSearch executes only.

The old L0–L4 / CONSERVE-first emergency flowchart is NOT the brain here.
Mode names (NORMAL/PROTECT/…) and protocol tags (P-RQ/…) are *outputs* of the
menu choice — labels on the duty sheet — not a competing priority automaton.

PI (pi.py) supplies hard gates only; it never signs actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from . import copilot as cp
from . import pi as pi_gates
from .duty_mode import (
    MODE_CONSERVE,
    MODE_NORMAL,
    MODE_PROTECT,
    MODE_RECOVER,
    ModeDecision,
    PROTO_Q,
    PROTO_REQ,
    PROTO_RQ,
    PROTO_RS,
    QUALITY_LOW,
    QUALITY_MEDIAN_LOW,
    QUALITY_MONTH_LOW,
    RECOVER_WINDOW_HOURS,
    SHORT_EXPOSURE_CAP,
    VERY_SHORT_EXPOSURE_CAP,
    DEBT_CRITICAL,
    DEBT_FINAL_NIGHTS,
    DEBT_SPRINT_NIGHTS,
    DEBT_SOFT_FLOOR,
    REQUEST_CRITICAL_HOURS,
    REQUEST_URGENT_HOURS,
    PROTECT_ALWAYS_HOURS,
    _collect_recover_targets,
    _local_weather_degraded,
    _rolling_quality,
)


@dataclass(frozen=True)
class NightFacts:
    night_index: int
    hours: float
    quality_bad: bool
    very_bad: bool
    last_q: float
    median5: float
    debt: int
    nights_left: int
    avoid_streak: int
    shallow_streak: int
    in_recover: bool
    new_invalidations: bool
    has_request: bool
    hours_left: float
    critical: bool
    live_views: list


def _ensure_state(state) -> None:
    if not hasattr(state, "abandoned_request_ids"):
        state.abandoned_request_ids = set()
    if not hasattr(state, "avoid_streak_nights"):
        state.avoid_streak_nights = 0
    if not hasattr(state, "avoid_last_night_index"):
        state.avoid_last_night_index = -1
    if not hasattr(state, "avoid_debt_anchor"):
        state.avoid_debt_anchor = 0
    if not hasattr(state, "shallow_decision_streak"):
        state.shallow_decision_streak = 0
    if not hasattr(state, "pilot_force_deep"):
        state.pilot_force_deep = False
    if not hasattr(state, "pilot_tolerate_debt"):
        state.pilot_tolerate_debt = False
    if not hasattr(state, "copilot_long"):
        state.copilot_long = ""
        state.copilot_mid = ""
        state.copilot_short = ""
    if not hasattr(state, "pilot_chosen_short"):
        state.pilot_chosen_short = ""


def _update_avoid_streak(state, night_index: int, quality_bad: bool, debt: int) -> int:
    last = int(getattr(state, "avoid_last_night_index", -1))
    shallow_addict = int(getattr(state, "shallow_decision_streak", 0)) >= pi_gates.SHALLOW_DECISION_STREAK
    bad_night = quality_bad or shallow_addict
    if night_index != last:
        if bad_night:
            if last < 0 or night_index == last + 1:
                if state.avoid_streak_nights == 0:
                    state.avoid_debt_anchor = debt
                    state.avoid_streak_nights = 1
                else:
                    if debt >= state.avoid_debt_anchor - 2:
                        state.avoid_streak_nights += 1
                        state.avoid_debt_anchor = min(state.avoid_debt_anchor, debt)
                    else:
                        state.avoid_debt_anchor = debt
                        state.avoid_streak_nights = 1
            elif night_index <= last + 2 and debt >= getattr(state, "avoid_debt_anchor", debt) - 2:
                state.avoid_streak_nights = max(1, int(state.avoid_streak_nights))
                state.avoid_debt_anchor = debt
            else:
                state.avoid_debt_anchor = debt
                state.avoid_streak_nights = 1
        else:
            state.avoid_streak_nights = 0
            state.avoid_debt_anchor = debt
        state.avoid_last_night_index = night_index
    return int(state.avoid_streak_nights)


def _request_urgency(request_views, now: datetime) -> tuple[bool, float, bool]:
    if not request_views:
        return False, float("inf"), False
    hours = []
    for view in request_views:
        deadline = view.get("deadline")
        if deadline is None:
            continue
        hours.append(max(0.0, (deadline - now).total_seconds() / 3600.0))
    if not hours:
        return False, float("inf"), False
    remaining = min(hours)
    return True, remaining, remaining <= REQUEST_CRITICAL_HOURS


def gather_facts(state, *, now: datetime, night_index: int, hours: float, request_views: list) -> NightFacts:
    _ensure_state(state)
    state._duty_night_index = night_index

    new_invalidations = False
    if getattr(state, "_resync_just_happened", False):
        new_invalidations = True
        state.recover_until_hours = hours + RECOVER_WINDOW_HOURS
        state._resync_just_happened = False
        state.recover_targets = _collect_recover_targets(state)
    in_recover = hours <= float(getattr(state, "recover_until_hours", float("-inf")))

    abandoned = set(state.abandoned_request_ids)
    live_views = [v for v in (request_views or []) if str(v.get("id")) not in abandoned]

    last_q, median5, month_q = _rolling_quality(state)
    weather_bad = _local_weather_degraded(state)
    quality_bad = (
        last_q < QUALITY_LOW
        or median5 < QUALITY_MEDIAN_LOW
        or month_q < QUALITY_MONTH_LOW
        or weather_bad
    )
    very_bad = median5 < 0.30 or last_q < 0.30
    debt, nights_left = cp.required_debt_pair(state, night_index)
    avoid_streak = _update_avoid_streak(state, night_index, quality_bad, debt)
    shallow_streak = int(getattr(state, "shallow_decision_streak", 0))
    has_request, hours_left, critical = _request_urgency(live_views, now)

    return NightFacts(
        night_index=night_index,
        hours=hours,
        quality_bad=quality_bad,
        very_bad=very_bad,
        last_q=last_q,
        median5=median5,
        debt=debt,
        nights_left=nights_left,
        avoid_streak=avoid_streak,
        shallow_streak=shallow_streak,
        in_recover=in_recover,
        new_invalidations=new_invalidations,
        has_request=has_request,
        hours_left=hours_left,
        critical=critical,
        live_views=live_views,
    )


def _debt_urgent_from_brief(brief: cp.CopilotBrief, debt_rising: bool) -> bool:
    if brief.debt <= cp.DEBT_TOLERANCE:
        return False
    return brief.past_clear_gate and (
        brief.capacity_short
        or (brief.debt / max(brief.nights_left, 1) > 0.40 and brief.nights_left <= 20)
        or (brief.debt >= DEBT_CRITICAL and brief.nights_left <= 16)
        or (debt_rising and brief.capacity_short)
    )


def _pilot_pick_short(
    brief: cp.CopilotBrief,
    facts: NightFacts,
    *,
    debt_urgent: bool,
    force_deep: bool,
) -> tuple[str, str]:
    """Return (chosen_short, reason). Hard stops beat menu recommendation."""
    reasons = []

    # Hard stop 1: resync / recover → 补课
    if facts.new_invalidations or facts.in_recover:
        return "补课", "hard_stop:resync→补课"

    # Hard stop 2: timed window — must 连指 or 放弃 (never silent waste)
    request_protect = facts.has_request and (
        facts.critical or facts.hours_left <= PROTECT_ALWAYS_HOURS
    )
    if request_protect:
        if brief.short_name == "放弃限时" and brief.abandon_request_ids and not facts.critical:
            return "放弃限时", "pilot:采纳放弃限时EV"
        return "连指", f"hard_stop:限时窗{facts.hours_left:.1f}h→连指"

    # Soft menu row for abandon when window still long (Pilot confirms Copilot EV)
    if brief.short_name == "放弃限时" and brief.abandon_request_ids:
        return "放弃限时", "pilot:采纳放弃限时EV"

    # Hard stop 3: tolerate ≤3 — no hard deep-clear lock
    if brief.tolerate_debt and brief.debt > 0 and not facts.has_request:
        if force_deep:
            reasons.append("tolerate覆盖force_deep浅刮")
        return "容忍欠债", "hard_stop:容忍≤3→停硬锁清债"

    # Hard stop 4: force_deep / 避险上瘾 → 深清债
    if force_deep and not facts.very_bad:
        return "深清债", "hard_stop:避险上瘾→深清债"

    # Hard stop 5: 清债/冲刺 long → never 浅避险 (菜单与手柄一致)
    recommended = brief.short_name
    if brief.long_name in cp.CLEAR_LONGS or debt_urgent:
        if recommended in ("浅避险", "正常排镜") and not facts.very_bad:
            return "深清债", "pilot:清债长程→深清债"
        if recommended == "深清债":
            return "深清债", "menu:深清债"
        if recommended in ("连指", "放弃限时", "补课", "容忍欠债"):
            return recommended, f"menu:{recommended}"
        return "深清债", "pilot:债急默认深清债"

    # Very bad sky without debt clear: 浅避险 allowed
    if facts.very_bad and recommended == "浅避险":
        return "浅避险", "menu:浅避险(very_bad)"

    if recommended in ("浅避险", "深清债", "正常排镜", "容忍欠债", "连指", "补课"):
        return recommended, f"menu:{recommended}"

    if facts.quality_bad and not debt_urgent:
        return "浅避险", "fallback:质量差→浅避险"
    return "正常排镜", "fallback:正常排镜"


def _required_scale(debt: int, nights_left: int, quality_bad: bool) -> float:
    scale = 1.5 if debt >= DEBT_CRITICAL else 1.3
    if quality_bad and debt > DEBT_SOFT_FLOOR:
        scale = max(scale, 1.55)
    if nights_left <= DEBT_SPRINT_NIGHTS and debt > DEBT_SOFT_FLOOR:
        scale = max(scale, 1.65)
    if nights_left <= DEBT_FINAL_NIGHTS and debt > DEBT_SOFT_FLOOR:
        scale = max(scale, 1.8)
    return scale


def _knobs_from_choice(
    choice: str,
    brief: cp.CopilotBrief,
    facts: NightFacts,
    *,
    debt_urgent: bool,
    force_deep: bool,
    pick_reason: str,
) -> ModeDecision:
    """Map Pilot's short choice → duty sheet. Protocols are labels, not drivers."""
    debt = brief.debt
    nights_left = brief.nights_left
    quality_bad = facts.quality_bad
    very_bad = facts.very_bad
    has_request = facts.has_request
    hours_left = facts.hours_left
    critical = facts.critical

    allow_shallow = pi_gates.debt_clear_allows_shallow(very_bad=very_bad)
    reason = f"v3-framework; pick={choice}; {pick_reason}; {brief.reason}"

    if choice == "补课":
        protocols = [PROTO_RS]
        if debt_urgent or debt > cp.DEBT_TOLERANCE:
            protocols.append(PROTO_REQ)
        if has_request:
            protocols.append(PROTO_RQ)
        if very_bad and allow_shallow:
            protocols.append(PROTO_Q)
            cap = VERY_SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
        else:
            cap = None
            prefer = ("DARK", "BRIGHT", "BACKUP")
        return ModeDecision(
            level=0,
            mode=MODE_RECOVER,
            protocols=tuple(protocols),
            advice_priority="required" if debt > cp.DEBT_TOLERANCE else "request",
            advice_risk="conservative" if very_bad else "balanced",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=prefer,
            request_value_scale=1.35 if has_request else 1.0,
            required_value_scale=1.4,
            recover_value_scale=1.5,
            request_slot_share=0.6,
            reason=reason,
            pilot_long=brief.long_name,
            pilot_mid=brief.mid_name,
            pilot_short=choice,
            copilot_reason=brief.reason,
        )

    if choice == "连指":
        protocols = [PROTO_RQ]
        if debt_urgent or (debt > cp.DEBT_TOLERANCE and brief.past_clear_gate):
            protocols.append(PROTO_REQ)
        # 连指保深度：仅 very_bad 才降档短帽
        if very_bad and allow_shallow:
            protocols.append(PROTO_Q)
            cap = VERY_SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
        else:
            cap = None
            prefer = ("DARK", "BRIGHT", "BACKUP")
        share = 0.9 if critical or hours_left <= REQUEST_URGENT_HOURS else 0.7
        if debt_urgent:
            share = min(share, 0.8)
        rq = 1.6 if critical else (1.35 if hours_left <= REQUEST_URGENT_HOURS else 1.2)
        req = 1.35 if debt_urgent else (1.2 if debt > cp.DEBT_TOLERANCE else 1.1)
        return ModeDecision(
            level=1,
            mode=MODE_PROTECT,
            protocols=tuple(protocols),
            advice_priority="request",
            advice_risk="conservative" if very_bad else "balanced",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=prefer,
            request_value_scale=rq,
            required_value_scale=req,
            recover_value_scale=1.0,
            request_slot_share=share,
            reason=reason,
            pilot_long=brief.long_name,
            pilot_mid=brief.mid_name,
            pilot_short=choice,
            copilot_reason=brief.reason,
        )

    if choice == "深清债":
        protocols = [PROTO_REQ]
        # 还债少而深：禁止绑浅避险，除非天真烂
        if very_bad and allow_shallow:
            protocols.append(PROTO_Q)
            cap = VERY_SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
            reason += "; very_bad允许浅"
        else:
            cap = None
            prefer = ("DARK", "BRIGHT", "BACKUP")
            reason += "; deep_debt_clear"
        if force_deep:
            reason += "; force_deep"
        return ModeDecision(
            level=2,
            mode=MODE_CONSERVE,
            protocols=tuple(protocols),
            advice_priority="required",
            advice_risk="balanced" if not very_bad else "conservative",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=prefer,
            request_value_scale=1.1 if has_request else 1.0,
            required_value_scale=_required_scale(debt, nights_left, quality_bad),
            recover_value_scale=1.0,
            request_slot_share=0.12 if has_request else 0.0,
            reason=reason,
            pilot_long=brief.long_name,
            pilot_mid=brief.mid_name,
            pilot_short=choice,
            copilot_reason=brief.reason,
        )

    if choice == "浅避险":
        # Only reachable when NOT clear-long (Pilot already blocked that)
        protocols = [PROTO_Q]
        if debt > cp.DEBT_TOLERANCE and brief.past_clear_gate:
            protocols.append(PROTO_REQ)
        if very_bad:
            cap = VERY_SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
        else:
            cap = SHORT_EXPOSURE_CAP
            prefer = ("BACKUP", "BRIGHT", "DARK")
        return ModeDecision(
            level=3,
            mode=MODE_CONSERVE,
            protocols=tuple(protocols),
            advice_priority="balanced",
            advice_risk="conservative",
            max_exposure_cap=cap,
            forbid_idle_wait=True,
            prefer_programs=prefer,
            request_value_scale=1.1 if has_request else 1.0,
            required_value_scale=1.25 if debt > cp.DEBT_TOLERANCE else 1.1,
            recover_value_scale=1.0,
            request_slot_share=0.1 if has_request else 0.0,
            reason=reason,
            pilot_long=brief.long_name,
            pilot_mid=brief.mid_name,
            pilot_short=choice,
            copilot_reason=brief.reason,
        )

    if choice == "容忍欠债":
        protocols = []
        if has_request:
            protocols.append(PROTO_RQ)
        return ModeDecision(
            level=4,
            mode=MODE_NORMAL,
            protocols=tuple(protocols),
            advice_priority="balanced",
            advice_risk="balanced",
            max_exposure_cap=None,
            forbid_idle_wait=False,
            prefer_programs=("DARK", "BRIGHT", "BACKUP"),
            request_value_scale=1.15 if has_request else 1.0,
            required_value_scale=1.0,
            recover_value_scale=1.0,
            request_slot_share=0.25 if has_request else 0.0,
            reason=reason + "; tolerate≤3",
            pilot_long=brief.long_name,
            pilot_mid=brief.mid_name,
            pilot_short=choice,
            copilot_reason=brief.reason,
        )

    # 正常排镜
    protocols = []
    req_scale = 1.0
    required_scale = 1.0
    slot = 0.0
    if has_request:
        protocols.append(PROTO_RQ)
        req_scale = 1.15
        slot = 0.25
    if debt > cp.DEBT_TOLERANCE and brief.past_clear_gate:
        protocols.append(PROTO_REQ)
        required_scale = 1.25
    elif debt > cp.DEBT_TOLERANCE:
        required_scale = 1.05
    return ModeDecision(
        level=4,
        mode=MODE_NORMAL,
        protocols=tuple(protocols),
        advice_priority="request" if has_request else (
            "required" if debt > cp.DEBT_TOLERANCE and brief.past_clear_gate else "balanced"
        ),
        advice_risk="balanced",
        max_exposure_cap=None,
        forbid_idle_wait=False,
        prefer_programs=("DARK", "BRIGHT", "BACKUP"),
        request_value_scale=req_scale,
        required_value_scale=required_scale,
        recover_value_scale=1.0,
        request_slot_share=slot,
        reason=reason,
        pilot_long=brief.long_name,
        pilot_mid=brief.mid_name,
        pilot_short=choice,
        copilot_reason=brief.reason,
    )


def decide(
    state,
    *,
    now: datetime,
    night_index: int,
    hours: float,
    request_views: list,
    last_result: Optional[dict] = None,
) -> ModeDecision:
    """Framework Pilot step: menu → pick → knobs. Sole signer for duty sheet."""
    facts = gather_facts(
        state, now=now, night_index=night_index, hours=hours, request_views=request_views
    )
    stamped = cp.stamp_hours_left(facts.live_views, now)

    # Preliminary brief for long/capacity (force_deep refined after debt_urgent)
    pre = cp.build_brief(
        state,
        night_index=night_index,
        request_views=stamped,
        quality_bad=facts.quality_bad,
        very_bad=facts.very_bad,
        avoid_streak=facts.avoid_streak,
        abandoned=set(state.abandoned_request_ids),
        in_recover=facts.in_recover or facts.new_invalidations,
    )
    debt_rising = bool(getattr(state, "required_debt_rising", False))
    debt_urgent = _debt_urgent_from_brief(pre, debt_rising)

    force_deep = pi_gates.should_force_deep(
        avoid_streak=facts.avoid_streak,
        avoid_max=cp.AVOID_MAX_STREAK_NIGHTS,
        debt=facts.debt,
        debt_tolerance=cp.DEBT_TOLERANCE,
        debt_urgent=debt_urgent,
        quality_bad=facts.quality_bad,
        shallow_streak=facts.shallow_streak,
    )

    brief = cp.build_brief(
        state,
        night_index=night_index,
        request_views=stamped,
        quality_bad=facts.quality_bad,
        very_bad=facts.very_bad,
        avoid_streak=facts.avoid_streak,
        abandoned=set(state.abandoned_request_ids),
        in_recover=facts.in_recover or facts.new_invalidations,
        force_deep=force_deep,
        debt_urgent=debt_urgent,
    )

    state.copilot_long = brief.long_name
    state.copilot_mid = brief.mid_name
    state.copilot_short = brief.short_name
    state.pilot_force_deep = force_deep
    state.pilot_tolerate_debt = brief.tolerate_debt

    choice, pick_reason = _pilot_pick_short(
        brief, facts, debt_urgent=debt_urgent, force_deep=force_deep
    )

    # Pilot confirms abandon (menu EV) — only when chosen, then re-pick main short
    if choice == "放弃限时" and brief.abandon_request_ids:
        for rid in brief.abandon_request_ids:
            state.abandoned_request_ids.add(str(rid))
        # After abandon, re-gather without those requests and pick again
        live = [v for v in facts.live_views if str(v.get("id")) not in state.abandoned_request_ids]
        stamped2 = cp.stamp_hours_left(live, now)
        brief2 = cp.build_brief(
            state,
            night_index=night_index,
            request_views=stamped2,
            quality_bad=facts.quality_bad,
            very_bad=facts.very_bad,
            avoid_streak=facts.avoid_streak,
            abandoned=set(state.abandoned_request_ids),
            in_recover=facts.in_recover or facts.new_invalidations,
            force_deep=force_deep,
            debt_urgent=debt_urgent,
        )
        # Rebuild facts-like urgency without abandoned
        has_req, hours_left, critical = _request_urgency(live, now)
        facts = NightFacts(
            night_index=facts.night_index,
            hours=facts.hours,
            quality_bad=facts.quality_bad,
            very_bad=facts.very_bad,
            last_q=facts.last_q,
            median5=facts.median5,
            debt=facts.debt,
            nights_left=facts.nights_left,
            avoid_streak=facts.avoid_streak,
            shallow_streak=facts.shallow_streak,
            in_recover=facts.in_recover,
            new_invalidations=facts.new_invalidations,
            has_request=has_req,
            hours_left=hours_left,
            critical=critical,
            live_views=live,
        )
        brief = brief2
        state.copilot_short = brief.short_name
        choice, pick_reason = _pilot_pick_short(
            brief, facts, debt_urgent=debt_urgent, force_deep=force_deep
        )
        pick_reason = "abandon后→" + pick_reason

    state.pilot_chosen_short = choice
    return _knobs_from_choice(
        choice,
        brief,
        facts,
        debt_urgent=debt_urgent,
        force_deep=force_deep,
        pick_reason=pick_reason,
    )
