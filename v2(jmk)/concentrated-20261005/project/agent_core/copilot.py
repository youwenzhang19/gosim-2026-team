"""Copilot (副手): O(1) table + arithmetic menu / EV. Never signs actions.

v3 fiducials (友文 2026-10-05):
  1. DEBT_TOLERANCE ≤ 3 — stop hard-lock CONSERVE on residual required debt
  2. DEBT_CLEAR_AFTER_FRAC = 0.20 — main debt clear only after first 20% nights
     AND remaining nights are not enough to clear
  3. AVOID_MAX_STREAK_NIGHTS = 3 — consecutive bad-Q nights with debt not falling
     → recommend Pilot force-deep or tolerate
  4. Abandoned timed requests stay abandoned (no thrash / 横跳)

Pilot alone selects; this module only scores options and flags risks.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Fiducials (authoritative numbers for v3) ---
DEBT_TOLERANCE = 3
DEBT_CLEAR_AFTER_FRAC = 0.20
AVOID_MAX_STREAK_NIGHTS = 3
EST_REQUIRED_PER_NIGHT = 11.0
REQUEST_CRITICAL_HOURS = 4.0
EV_REQUIRED_UNIT = 50.0
EV_SCIENCE_UNIT = 1.0
SHALLOW_TAX = 0.15
IDLE_TAX = 0.10


@dataclass(frozen=True)
class MenuOption:
    name: str
    layer: str  # long | mid | short
    ev: float
    risk: str
    confidence: float
    recommend: bool


@dataclass(frozen=True)
class CopilotBrief:
    """Thin menu handed to Pilot for one decide step."""

    long_name: str
    mid_name: str
    short_name: str
    options: tuple[MenuOption, ...]
    abandon_request_ids: tuple[str, ...]
    force_deep: bool
    tolerate_debt: bool
    season_frac: float
    debt: int
    nights_left: int
    capacity_short: bool
    past_clear_gate: bool
    reason: str


def season_progress(state, night_index: int) -> float:
    total = max(1, len(getattr(state, "nights", []) or []))
    return max(0.0, min(1.0, (night_index + 1) / total))


def required_debt_pair(state, night_index: int) -> tuple[int, int]:
    threshold = state.scoring.required_threshold
    debt = 0
    last_nights = []
    for i, required in enumerate(state.required):
        if required and state.factor[i] < threshold:
            debt += 1
            last_nights.append(state.last_night[i])
    if debt == 0:
        return 0, 0
    nights_left = max(0, max(last_nights) - night_index + 1)
    return debt, nights_left


def capacity_shortfall(debt: int, nights_left: int) -> bool:
    if debt <= DEBT_TOLERANCE:
        return False
    nights_needed = debt / EST_REQUIRED_PER_NIGHT
    return nights_needed > nights_left * 0.95


def stamp_hours_left(request_views: list, now) -> list:
    """Return copies of views with hours_left for Copilot arithmetic."""
    out = []
    for view in request_views or []:
        stamped = dict(view)
        deadline = view.get("deadline")
        if deadline is not None and now is not None:
            stamped["hours_left"] = max(0.0, (deadline - now).total_seconds() / 3600.0)
        out.append(stamped)
    return out


def build_brief(
    state,
    *,
    night_index: int,
    request_views: list,
    quality_bad: bool,
    very_bad: bool,
    avoid_streak: int,
    abandoned: set,
    in_recover: bool = False,
) -> CopilotBrief:
    """Score long/mid/short options; flag abandon / force-deep / tolerate."""
    debt, nights_left = required_debt_pair(state, night_index)
    frac = season_progress(state, night_index)
    past_gate = frac >= DEBT_CLEAR_AFTER_FRAC
    short = capacity_shortfall(debt, nights_left)
    tolerate = debt <= DEBT_TOLERANCE
    force_deep = avoid_streak >= AVOID_MAX_STREAK_NIGHTS and debt > DEBT_TOLERANCE

    if frac >= 0.85 or (past_gate and short and nights_left <= 3):
        long_name = "冲刺收官"
        long_ev = _ev_sprint(debt, tolerate)
    elif past_gate and short:
        long_name = "清债优先"
        long_ev = _ev_debt_clear(debt, force_deep)
    elif frac < DEBT_CLEAR_AFTER_FRAC:
        long_name = "开拓为主"
        long_ev = _ev_explore(debt)
    else:
        long_name = "均衡推进"
        long_ev = _ev_balanced(debt, quality_bad)

    if in_recover:
        mid_name = "失效后恢复段"
        mid_ev = long_ev - 5.0
    elif very_bad:
        mid_name = "雨季/质量谷段"
        mid_ev = long_ev - SHALLOW_TAX * 40 - (0 if force_deep else 10)
    elif quality_bad:
        mid_name = "波动段"
        mid_ev = long_ev - 5.0
    else:
        mid_name = "晴资本段"
        mid_ev = long_ev + 5.0

    open_views = [v for v in (request_views or []) if str(v.get("id")) not in abandoned]
    hours_left, critical = _min_hours(open_views)
    abandon_ids: list[str] = []
    req_protect_ev = 0.0
    req_abandon_ev = 0.0

    if open_views:
        reward = sum(float(v.get("reward", 100.0)) for v in open_views)
        remaining_tgts = sum(int(v.get("remaining", 1)) for v in open_views)
        req_protect_ev = reward - (remaining_tgts * 4.0)
        # Mild conflict tax only — do not let large D always veto timed rewards.
        if short and debt > DEBT_TOLERANCE:
            req_protect_ev -= min(40.0, 8.0 * max(1, remaining_tgts // 3))
        # v4: missing a timed request usually does not penalize — forfeit reward only.
        req_abandon_ev = EV_REQUIRED_UNIT * min(debt, 40) * 0.02 - IDLE_TAX * 5
        # Abandon only when window still long (> urgent) AND abandon clearly wins.
        # Never abandon inside the default 连指 window (≤12h) or critical (≤4h).
        if (
            short
            and debt > DEBT_TOLERANCE
            and hours_left > 12.0
            and req_abandon_ev > req_protect_ev + 40.0
        ):
            # Drop the single greediest (largest remaining) request, not the whole set.
            worst = max(open_views, key=lambda v: int(v.get("remaining", 1)))
            abandon_ids = [str(worst["id"])]
            short_name = "放弃限时"
            short_ev = req_abandon_ev
        elif hours_left <= 12.0 or critical:
            short_name = "连指"
            short_ev = req_protect_ev
        else:
            short_name = "正常排镜"
            short_ev = mid_ev + req_protect_ev * 0.05
    elif force_deep:
        short_name = "深清债"
        short_ev = long_ev + 8.0
    elif tolerate and debt > 0 and past_gate:
        short_name = "容忍欠债"
        short_ev = long_ev + 4.0
    elif quality_bad and not force_deep:
        short_name = "浅避险"
        short_ev = mid_ev - (SHALLOW_TAX * 20 if avoid_streak >= 2 else 0)
    else:
        short_name = "正常排镜"
        short_ev = mid_ev

    if force_deep and short_name not in ("连指", "放弃限时"):
        short_name = "深清债"
        short_ev = max(short_ev, long_ev + 8.0)

    options = (
        MenuOption(long_name, "long", long_ev, _risk_long(long_name, debt), 0.7, True),
        MenuOption(mid_name, "mid", mid_ev,
                   "广而浅" if quality_bad and not force_deep else "均衡", 0.65, True),
        MenuOption(short_name, "short", short_ev, _risk_short(short_name), 0.7, True),
    )
    if open_views:
        options = options + (
            MenuOption("连指", "short", req_protect_ev, "限时落空/伤必做", 0.6,
                       short_name == "连指"),
            MenuOption("放弃限时", "short", req_abandon_ev, "弃奖", 0.65,
                       short_name == "放弃限时"),
        )

    reason_parts = [
        f"frac={frac:.2f}",
        f"D={debt}",
        f"nights_left≈{nights_left}",
        f"past20%={int(past_gate)}",
        f"cap_short={int(short)}",
        f"avoid_streak={avoid_streak}",
    ]
    if abandon_ids:
        reason_parts.append(f"abandon={','.join(abandon_ids)}")
    if force_deep:
        reason_parts.append("force_deep")
    if tolerate:
        reason_parts.append("tolerate≤3")

    return CopilotBrief(
        long_name=long_name,
        mid_name=mid_name,
        short_name=short_name,
        options=options,
        abandon_request_ids=tuple(abandon_ids),
        force_deep=force_deep,
        tolerate_debt=tolerate,
        season_frac=frac,
        debt=debt,
        nights_left=nights_left,
        capacity_short=short,
        past_clear_gate=past_gate,
        reason="; ".join(reason_parts),
    )


def _min_hours(views) -> tuple[float, bool]:
    if not views:
        return float("inf"), False
    hours = [float(v.get("hours_left", 12.0)) for v in views]
    remaining = min(hours)
    return remaining, remaining <= REQUEST_CRITICAL_HOURS


def _ev_explore(debt: int) -> float:
    return 40.0 + EV_SCIENCE_UNIT * 20 - EV_REQUIRED_UNIT * max(0, debt - 80) * 0.01


def _ev_balanced(debt: int, quality_bad: bool) -> float:
    return 35.0 - EV_REQUIRED_UNIT * max(0, debt - DEBT_TOLERANCE) * 0.02 - (
        SHALLOW_TAX * 15 if quality_bad else 0
    )


def _ev_debt_clear(debt: int, force_deep: bool) -> float:
    base = 30.0 - EV_REQUIRED_UNIT * max(0, debt - DEBT_TOLERANCE) * 0.03
    if force_deep:
        return base + 10.0
    return base - SHALLOW_TAX * 25


def _ev_sprint(debt: int, tolerate: bool) -> float:
    if tolerate:
        return 38.0
    return 28.0 - EV_REQUIRED_UNIT * max(0, debt - DEBT_TOLERANCE) * 0.04


def _risk_long(name: str, debt: int) -> str:
    if name == "清债优先" and debt > 100:
        return "广而浅/欠债爆炸"
    if name == "开拓为主" and debt > 200:
        return "尾段债爆"
    return "均衡"


def _risk_short(name: str) -> str:
    return {
        "连指": "限时落空",
        "放弃限时": "弃奖",
        "深清债": "少场",
        "浅避险": "广而浅",
        "容忍欠债": "个位必做罚",
        "正常排镜": "无",
    }.get(name, "无")
