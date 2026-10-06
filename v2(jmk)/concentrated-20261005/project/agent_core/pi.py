"""PI (首席/编制) hard gates — not a second Pilot.

Pure arithmetic flags that Pilot.classify must respect. Never signs observe/wait.
Copilot may read the same gates for menu/EV alignment.
"""
from __future__ import annotations

# Rolling shallow-scan sentinel (decisions, not nights).
SHALLOW_DECISION_STREAK = 8
SHORT_EXPOSURE_SEC = 400


def debt_clear_allows_shallow(*, very_bad: bool) -> bool:
    """债急清债时：仅天真烂才允许浅避险叠乘。"""
    return bool(very_bad)


def should_force_deep(
    *,
    avoid_streak: int,
    avoid_max: int,
    debt: int,
    debt_tolerance: int,
    debt_urgent: bool,
    quality_bad: bool,
    shallow_streak: int,
) -> bool:
    """避险/浅刮上瘾 → 强制少而深。任一哨兵即可。"""
    if debt <= debt_tolerance:
        return False
    if avoid_streak >= avoid_max:
        return True
    # 债急 + 质量差：不要等满 3 夜才改深
    if debt_urgent and quality_bad:
        return True
    # 近段决策已浅刮成习
    if shallow_streak >= SHALLOW_DECISION_STREAK:
        return True
    return False


def update_shallow_streak(state, *, duration_seconds: float | None, program: str | None, debt: int, debt_tolerance: int) -> int:
    """Call after an observe when debt is live; resets on deep DARK."""
    if not hasattr(state, "shallow_decision_streak"):
        state.shallow_decision_streak = 0
    if debt <= debt_tolerance:
        state.shallow_decision_streak = 0
        return 0
    prog = (program or "").upper()
    dur = float(duration_seconds or 0)
    shallow = prog == "BACKUP" or (dur > 0 and dur <= SHORT_EXPOSURE_SEC)
    deep = prog == "DARK" and dur >= 1200
    if deep:
        state.shallow_decision_streak = 0
    elif shallow:
        state.shallow_decision_streak = int(state.shallow_decision_streak) + 1
    else:
        # mild: decay slowly
        state.shallow_decision_streak = max(0, int(state.shallow_decision_streak) - 1)
    return int(state.shallow_decision_streak)
