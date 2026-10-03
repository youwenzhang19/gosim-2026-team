#!/usr/bin/env python3
"""Generate multiple SYNTHETIC alpha practice-card variants for local smoke.

SYNTHETIC / 非官方真值，分数无官方意义。

Copies recoverable files from practice-cards/alpha/ (never modifies that tree),
trims the night calendar, and synthesizes runner products. Does not touch
gosim-observer-examples/.
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

WS = Path("/Users/zhangyouwen/我的云端硬盘/WORKSPACE_PROJECT/巡天智能体黑客松")
ALPHA = WS / "practice-cards" / "alpha"
OUT_ROOT = WS / "practice-cards"
SLOT_SECONDS = 900


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Variant:
    name: str
    title: str
    nights: int  # how many nights to keep from the start of alpha calendar
    night_offset: int = 0  # skip first N nights of alpha calendar
    weather_mode: str = "mild"  # mild | storm | faults | pressure | forecast_lie | short
    close_fraction: float = 0.05
    fault_nights: int = 0
    forecast_mode: str = "honest"  # honest | false_positive | miss | mixed
    required_boost: int = 0  # extra targets to mark required
    wallclock_seconds: int = 600
    seed: int = 20261003
    notes: list[str] = field(default_factory=list)


VARIANTS: list[Variant] = [
    Variant(
        name="alpha-synth-mild",
        title="温和基线",
        nights=12,
        weather_mode="mild",
        close_fraction=0.04,
        forecast_mode="honest",
        wallclock_seconds=600,
        seed=101,
        notes=["裁剪 α 夜历前 12 夜", "少量关穹", "预报与真值大致一致"],
    ),
    Variant(
        name="alpha-synth-storm",
        title="长期坏天气/高关穹",
        nights=12,
        night_offset=4,
        weather_mode="storm",
        close_fraction=0.55,
        forecast_mode="honest",
        wallclock_seconds=600,
        seed=202,
        notes=["裁剪 α 夜历第 5–16 夜", "约一半以上 slots force_close", "多段 rainy/cloudy"],
    ),
    Variant(
        name="alpha-synth-forecast-lie",
        title="预报假阳+漏报",
        nights=12,
        night_offset=8,
        weather_mode="mild",
        close_fraction=0.18,
        forecast_mode="mixed",
        wallclock_seconds=600,
        seed=303,
        notes=["裁剪 α 夜历第 9–20 夜", "预报大量假阳雨夜 + 对真实雨夜漏报"],
    ),
    Variant(
        name="alpha-synth-faults",
        title="仪器故障密集",
        nights=12,
        night_offset=10,
        weather_mode="faults",
        close_fraction=0.08,
        fault_nights=8,
        forecast_mode="honest",
        wallclock_seconds=600,
        seed=404,
        notes=["裁剪 α 夜历第 11–22 夜", "多段不重叠 instrument_fault + earthquake", "故障窗不得重叠（引擎硬约束）"],
    ),
    Variant(
        name="alpha-synth-pressure",
        title="强 required / 均匀性压力",
        nights=14,
        night_offset=2,
        weather_mode="pressure",
        close_fraction=0.35,
        required_boost=1500,
        forecast_mode="honest",
        wallclock_seconds=600,
        seed=505,
        notes=[
            "裁剪 α 夜历第 3–16 夜",
            "额外标记 ~1500 个 required",
            "分带关穹制造 RA 覆盖不均",
        ],
    ),
    Variant(
        name="alpha-synth-short",
        title="SHORT 极速 smoke",
        nights=5,
        weather_mode="short",
        close_fraction=0.02,
        forecast_mode="honest",
        wallclock_seconds=300,
        seed=606,
        notes=["SHORT：仅 α 夜历前 5 夜", "几乎全晴，专供 <1min 通路检查"],
    ),
]


def load_full_calendar() -> list[dict]:
    return list(csv.DictReader((ALPHA / "public" / "v4_night_calendar.csv").open(encoding="utf-8")))


def trim_calendar(full: list[dict], nights: int, offset: int) -> list[dict]:
    sliced = full[offset : offset + nights]
    if len(sliced) < nights:
        raise SystemExit(f"calendar slice too short: offset={offset} nights={nights}")
    return sliced


def build_slots(calendar: list[dict]) -> list[dict]:
    slots: list[dict] = []
    for night in calendar:
        night_id = night["night_id"]
        start = parse_utc(night["observing_start_utc"])
        end = parse_utc(night["observing_end_utc"])
        expected = int(night["slot_count"])
        t = start
        idx = 1
        while t < end:
            slots.append(
                {
                    "slot_id": f"{night_id}-S{idx:03d}",
                    "night_id": night_id,
                    "timestamp_utc": fmt_utc(t),
                    "duration_seconds": str(SLOT_SECONDS),
                    "_start": t,
                    "_end": t + timedelta(seconds=SLOT_SECONDS),
                    "night_date": night["night_date"],
                }
            )
            t += timedelta(seconds=SLOT_SECONDS)
            idx += 1
        actual = sum(1 for s in slots if s["night_id"] == night_id)
        if actual != expected:
            raise SystemExit(f"{night_id}: expected {expected} slots, got {actual}")
    return slots


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_close_intervals(slots: list[dict], calendar: list[dict], v: Variant, rng: random.Random):
    """Return list of (start, end) force-close intervals plus event rows."""
    events: list[dict] = []
    intervals: list[tuple[datetime, datetime]] = []
    survey_start, survey_end = slots[0]["_start"], slots[-1]["_end"]

    # Terrain always present (directional zero_score)
    for eid, az0, az1, alt_max in (
        ("SYNTH_EV_T0", 220.0, 270.0, 40.0),
        ("SYNTH_EV_T1", 100.0, 140.0, 35.0),
    ):
        events.append(
            {
                "event_id": eid,
                "event_type": "terrain_obstruction",
                "actual_start_utc": fmt_utc(survey_start),
                "actual_end_utc": fmt_utc(survey_end),
                "scope_type": "HORIZON_SECTOR",
                "azimuth_start_deg": f"{az0:.6f}",
                "azimuth_end_deg": f"{az1:.6f}",
                "min_altitude_deg": "0.000000",
                "max_altitude_deg": f"{alt_max:.6f}",
                "magnitude": "",
                "opacity": "1.000000",
                "force_close": "false",
                "zero_score": "true",
                "seeing_multiplier": "1.000000",
                "transparency_multiplier": "1.000000",
                "sky_quality_multiplier": "1.000000",
                "instrument_efficiency_multiplier": "1.000000",
            }
        )

    def night_slots(night_date: str) -> list[dict]:
        return [s for s in slots if s["night_date"] == night_date]

    dates = [n["night_date"] for n in calendar]

    if v.weather_mode == "storm":
        # Close large contiguous chunks most nights
        for i, nd in enumerate(dates):
            ns = night_slots(nd)
            if not ns:
                continue
            if i % 5 == 0:
                # whole night closed
                a, b = ns[0], ns[-1]
            else:
                # close middle 50–70%
                lo = max(1, int(len(ns) * 0.15))
                hi = min(len(ns) - 1, int(len(ns) * (0.15 + v.close_fraction)))
                if hi <= lo:
                    hi = min(len(ns) - 1, lo + max(3, len(ns) // 3))
                a, b = ns[lo], ns[hi]
            intervals.append((a["_start"], b["_end"]))
            events.append(
                {
                    "event_id": f"SYNTH_EV_R{i:02d}",
                    "event_type": "rainy" if i % 2 == 0 else "cloudy",
                    "actual_start_utc": a["timestamp_utc"],
                    "actual_end_utc": fmt_utc(b["_end"]),
                    "scope_type": "ALL" if i % 2 == 0 else "HORIZON_SECTOR",
                    "azimuth_start_deg": "" if i % 2 == 0 else "110.000000",
                    "azimuth_end_deg": "" if i % 2 == 0 else "230.000000",
                    "min_altitude_deg": "" if i % 2 == 0 else "0.000000",
                    "max_altitude_deg": "" if i % 2 == 0 else "70.000000",
                    "magnitude": "",
                    "opacity": "0.850000",
                    "force_close": "true" if i % 2 == 0 else "false",
                    "zero_score": "false",
                    "seeing_multiplier": "1.600000",
                    "transparency_multiplier": "0.250000",
                    "sky_quality_multiplier": "0.350000",
                    "instrument_efficiency_multiplier": "1.000000",
                }
            )
            if i % 2 != 0:
                # cloudy not force_close; also add a rainy force_close chunk
                lo2 = int(len(ns) * 0.4)
                hi2 = min(len(ns) - 1, int(len(ns) * 0.85))
                a2, b2 = ns[lo2], ns[hi2]
                intervals.append((a2["_start"], b2["_end"]))
                events.append(
                    {
                        "event_id": f"SYNTH_EV_S{i:02d}",
                        "event_type": "rainy",
                        "actual_start_utc": a2["timestamp_utc"],
                        "actual_end_utc": fmt_utc(b2["_end"]),
                        "scope_type": "ALL",
                        "azimuth_start_deg": "",
                        "azimuth_end_deg": "",
                        "min_altitude_deg": "",
                        "max_altitude_deg": "",
                        "magnitude": "",
                        "opacity": "0.950000",
                        "force_close": "true",
                        "zero_score": "false",
                        "seeing_multiplier": "1.700000",
                        "transparency_multiplier": "0.200000",
                        "sky_quality_multiplier": "0.300000",
                        "instrument_efficiency_multiplier": "1.000000",
                    }
                )
    elif v.weather_mode == "pressure":
        # Close early half of even nights and late half of odd nights → uneven opportunity
        for i, nd in enumerate(dates):
            ns = night_slots(nd)
            if len(ns) < 8:
                continue
            if i % 2 == 0:
                a, b = ns[0], ns[len(ns) // 2]
            else:
                a, b = ns[len(ns) // 2], ns[-1]
            intervals.append((a["_start"], b["_end"]))
            events.append(
                {
                    "event_id": f"SYNTH_EV_P{i:02d}",
                    "event_type": "rainy",
                    "actual_start_utc": a["timestamp_utc"],
                    "actual_end_utc": fmt_utc(b["_end"]),
                    "scope_type": "ALL",
                    "azimuth_start_deg": "",
                    "azimuth_end_deg": "",
                    "min_altitude_deg": "",
                    "max_altitude_deg": "",
                    "magnitude": "",
                    "opacity": "0.900000",
                    "force_close": "true",
                    "zero_score": "false",
                    "seeing_multiplier": "1.500000",
                    "transparency_multiplier": "0.300000",
                    "sky_quality_multiplier": "0.400000",
                    "instrument_efficiency_multiplier": "1.000000",
                }
            )
        # Broad directional cloudy on remaining open windows
        mid = dates[len(dates) // 2]
        ns = night_slots(mid)
        if ns:
            events.append(
                {
                    "event_id": "SYNTH_EV_PC",
                    "event_type": "cloudy",
                    "actual_start_utc": ns[0]["timestamp_utc"],
                    "actual_end_utc": fmt_utc(ns[-1]["_end"]),
                    "scope_type": "HORIZON_SECTOR",
                    "azimuth_start_deg": "90.000000",
                    "azimuth_end_deg": "180.000000",
                    "min_altitude_deg": "0.000000",
                    "max_altitude_deg": "80.000000",
                    "magnitude": "",
                    "opacity": "0.650000",
                    "force_close": "false",
                    "zero_score": "false",
                    "seeing_multiplier": "1.200000",
                    "transparency_multiplier": "0.650000",
                    "sky_quality_multiplier": "0.700000",
                    "instrument_efficiency_multiplier": "1.000000",
                }
            )
    else:
        # mild / faults / short / forecast_lie: sparse rain
        n_rain = max(1, int(len(dates) * (0.15 if v.weather_mode != "short" else 0.0)))
        if v.close_fraction > 0 and v.weather_mode != "short":
            n_rain = max(n_rain, 1)
        rain_idxs = sorted(rng.sample(range(len(dates)), k=min(n_rain, len(dates)))) if n_rain else []
        # For forecast_lie ensure at least 2 real rain nights
        if v.forecast_mode in ("miss", "mixed") and len(rain_idxs) < 2 and len(dates) >= 2:
            rain_idxs = sorted({0, len(dates) // 2, len(dates) - 1})[:3]
        for j, i in enumerate(rain_idxs):
            ns = night_slots(dates[i])
            if len(ns) < 6:
                continue
            span = max(3, int(len(ns) * max(0.15, v.close_fraction)))
            lo = rng.randint(1, max(1, len(ns) - span - 1))
            hi = min(len(ns) - 1, lo + span)
            a, b = ns[lo], ns[hi]
            intervals.append((a["_start"], b["_end"]))
            events.append(
                {
                    "event_id": f"SYNTH_EV_R{j:02d}",
                    "event_type": "rainy",
                    "actual_start_utc": a["timestamp_utc"],
                    "actual_end_utc": fmt_utc(b["_end"]),
                    "scope_type": "ALL",
                    "azimuth_start_deg": "",
                    "azimuth_end_deg": "",
                    "min_altitude_deg": "",
                    "max_altitude_deg": "",
                    "magnitude": "",
                    "opacity": "0.900000",
                    "force_close": "true",
                    "zero_score": "false",
                    "seeing_multiplier": "1.500000",
                    "transparency_multiplier": "0.300000",
                    "sky_quality_multiplier": "0.400000",
                    "instrument_efficiency_multiplier": "1.000000",
                }
            )

    # Instrument faults — must NOT overlap (WeatherTruth enforces this).
    # Dense variant: many abutting/short repair cycles across nights.
    if v.weather_mode == "faults" or v.fault_nights > 0:
        n_fault = v.fault_nights or max(3, len(dates) // 2)
        # Place sequential non-overlapping windows: each covers ~half a night,
        # next starts at or after previous end (abut OK if end==start of next? use gap).
        cursor: datetime | None = None
        placed = 0
        for j, nd in enumerate(dates):
            if placed >= n_fault:
                break
            ns = night_slots(nd)
            if len(ns) < 6:
                continue
            # two fault pulses per night when dense, else one
            pulses = 2 if v.weather_mode == "faults" else 1
            limit = (n_fault * 2) if v.weather_mode == "faults" else n_fault
            for p in range(pulses):
                if placed >= limit:
                    break
                lo = 2 + p * (len(ns) // 2)
                hi = min(len(ns) - 1, lo + max(4, len(ns) // 3))
                if lo >= len(ns):
                    break
                start = ns[lo]["_start"]
                end = ns[hi]["_end"]
                if cursor is not None and start < cursor:
                    # skip this pulse if it would overlap; try next night
                    continue
                if end <= start:
                    continue
                eff = 0.35 + 0.07 * (placed % 5)
                events.append(
                    {
                        "event_id": f"SYNTH_EV_F{placed:02d}",
                        "event_type": "instrument_fault",
                        "actual_start_utc": fmt_utc(start),
                        "actual_end_utc": fmt_utc(end),
                        "scope_type": "ALL",
                        "azimuth_start_deg": "",
                        "azimuth_end_deg": "",
                        "min_altitude_deg": "",
                        "max_altitude_deg": "",
                        "magnitude": "",
                        "opacity": "1.000000",
                        "force_close": "false",
                        "zero_score": "false",
                        "seeing_multiplier": "1.000000",
                        "transparency_multiplier": "1.000000",
                        "sky_quality_multiplier": "1.000000",
                        "instrument_efficiency_multiplier": f"{eff:.6f}",
                    }
                )
                cursor = end  # next fault must start >= this (abut OK)
                placed += 1
        # earthquake pulse on a night without overlapping fault start issues
        # (earthquake is not FAULT_TYPE; OK to overlap faults temporally)
        ns = night_slots(dates[min(2, len(dates) - 1)])
        if ns:
            eq = ns[len(ns) // 2]
            events.append(
                {
                    "event_id": "SYNTH_EV_EQ",
                    "event_type": "earthquake",
                    "actual_start_utc": eq["timestamp_utc"],
                    "actual_end_utc": fmt_utc(eq["_end"]),
                    "scope_type": "ALL",
                    "azimuth_start_deg": "",
                    "azimuth_end_deg": "",
                    "min_altitude_deg": "",
                    "max_altitude_deg": "",
                    "magnitude": "6.200",
                    "opacity": "1.000000",
                    "force_close": "false",
                    "zero_score": "false",
                    "seeing_multiplier": "1.000000",
                    "transparency_multiplier": "1.000000",
                    "sky_quality_multiplier": "1.000000",
                    "instrument_efficiency_multiplier": "1.000000",
                }
            )

    events.sort(key=lambda e: e["actual_start_utc"])
    return events, intervals


def slot_closed(s: dict, intervals: list[tuple[datetime, datetime]]) -> bool:
    for a, b in intervals:
        if s["_start"] < b and s["_end"] > a:
            return True
    return False


def build_weather(slots: list[dict], intervals, rng: random.Random, mode: str) -> list[dict]:
    rows = []
    for i, s in enumerate(slots):
        closed = slot_closed(s, intervals)
        # faults mode: also degrade efficiency on open slots (events handle multipliers;
        # keep weather truth mostly open with lower baseline efficiency)
        if closed:
            rows.append(
                {
                    "slot_id": s["slot_id"],
                    "night_id": s["night_id"],
                    "timestamp_utc": s["timestamp_utc"],
                    "duration_seconds": str(SLOT_SECONDS),
                    "is_observable": "false",
                    "seeing_arcsec": "",
                    "transparency": "",
                    "sky_quality": "",
                    "instrument_efficiency": "",
                }
            )
            continue
        phase = math.sin(i * 0.17)
        seeing = 1.05 + 0.25 * (0.5 + 0.5 * phase) + rng.uniform(-0.05, 0.05)
        transp = 0.88 + 0.08 * (0.5 - 0.5 * phase) + rng.uniform(-0.02, 0.02)
        sky = 0.90 + 0.06 * (0.5 + 0.5 * math.cos(i * 0.11)) + rng.uniform(-0.02, 0.02)
        eff = 0.92 + 0.06 * (0.5 + 0.5 * math.sin(i * 0.09)) + rng.uniform(-0.02, 0.02)
        if mode == "faults":
            eff *= 0.55 + 0.1 * math.sin(i * 0.03)
            seeing += 0.3
        if mode == "storm":
            transp *= 0.7
            sky *= 0.75
            seeing += 0.4
        if mode == "pressure":
            # worse quality on "late" slots of night to hurt uniformity indirectly
            slot_num = int(s["slot_id"].rsplit("-S", 1)[-1])
            if slot_num % 2 == 0:
                transp *= 0.75
                sky *= 0.8
        rows.append(
            {
                "slot_id": s["slot_id"],
                "night_id": s["night_id"],
                "timestamp_utc": s["timestamp_utc"],
                "duration_seconds": str(SLOT_SECONDS),
                "is_observable": "true",
                "seeing_arcsec": f"{seeing:.6f}",
                "transparency": f"{max(0.2, min(1.0, transp)):.6f}",
                "sky_quality": f"{max(0.2, min(1.0, sky)):.6f}",
                "instrument_efficiency": f"{max(0.15, min(1.0, eff)):.6f}",
            }
        )
    return rows


def build_earthquake_effects(slots: list[dict], events: list[dict]) -> list[dict]:
    eq = next((e for e in events if e["event_type"] == "earthquake"), None)
    if not eq:
        return []
    eq_start = parse_utc(eq["actual_start_utc"])
    nights: list[str] = []
    seen: set[str] = set()
    for s in slots:
        if s["_start"] >= eq_start and s["night_id"] not in seen:
            seen.add(s["night_id"])
            nights.append(s["night_id"])
        if len(nights) >= 8:
            break
    rows = []
    deg = 0.6
    for nid in nights:
        rows.append(
            {
                "event_id": eq["event_id"],
                "night_id": nid,
                "magnitude": eq.get("magnitude") or "6.000",
                "degradation": f"{deg:.6f}",
                "seeing_multiplier": "1.000000",
                "transparency_multiplier": "1.000000",
                "sky_quality_multiplier": "1.000000",
                "instrument_efficiency_multiplier": f"{max(0.2, 1.0 - deg):.6f}",
            }
        )
        deg *= 0.8
    return rows


def build_bulletins(slots: list[dict], events: list[dict]) -> list[dict]:
    rainish = [
        (parse_utc(e["actual_start_utc"]), parse_utc(e["actual_end_utc"]), e["event_type"])
        for e in events
        if e["event_type"] in ("rainy", "cloudy", "rocket_launch")
    ]
    out = []
    prev = None
    for s in slots:
        notices = []
        initial = s["night_id"] != prev
        if initial:
            notices.extend(
                [
                    {"event_kind": "terrain_obstruction", "direction": "SW"},
                    {"event_kind": "terrain_obstruction", "direction": "SE"},
                ]
            )
            prev = s["night_id"]
        for a, b, et in rainish:
            if s["_start"] < b and s["_end"] > a:
                kind = {"rainy": "rain", "cloudy": "overcast", "rocket_launch": "rocket_launch"}[et]
                notices.append({"event_kind": kind, "direction": "ALL" if et == "rainy" else "S"})
        out.append(
            {
                "record_type": "bulletin",
                "slot_id": s["slot_id"],
                "night_id": s["night_id"],
                "issued_at_utc": s["timestamp_utc"],
                "initial": initial,
                "notices": notices,
            }
        )
    return out


def build_forecasts(
    slots: list[dict],
    calendar: list[dict],
    events: list[dict],
    mode: str,
    rng: random.Random,
) -> list[dict]:
    starts = [parse_utc(n["observing_start_utc"]) for n in calendar]
    dates = [n["night_date"] for n in calendar]
    real_rain_nights = set()
    for e in events:
        if e["event_type"] != "rainy":
            continue
        a, b = parse_utc(e["actual_start_utc"]), parse_utc(e["actual_end_utc"])
        for s in slots:
            if s["_start"] < b and s["_end"] > a:
                real_rain_nights.add(s["night_date"])

    forecasts = []
    step = 4 if len(calendar) >= 8 else max(2, len(calendar) // 2)
    for i in range(0, len(calendar), step):
        issued = starts[i]
        cov_end = starts[i + step] if i + step < len(starts) else slots[-1]["_end"]
        window_dates = dates[i : i + step]
        notices = []
        if mode == "honest":
            for nd in window_dates:
                if nd in real_rain_nights:
                    notices.append({"event_kind": "rain", "direction": "ALL", "nights": [nd]})
        elif mode == "false_positive":
            # claim rain on clear nights
            clear = [nd for nd in window_dates if nd not in real_rain_nights]
            for nd in clear[: max(1, len(clear))]:
                notices.append({"event_kind": "rain", "direction": "ALL", "nights": [nd]})
            notices.append({"event_kind": "overcast", "direction": "S", "nights": clear[:2] or window_dates[:1]})
        elif mode == "miss":
            # omit real rain; maybe mention haze elsewhere
            for nd in window_dates:
                if nd not in real_rain_nights and rng.random() < 0.3:
                    notices.append({"event_kind": "haze", "direction": "ALL", "nights": [nd]})
        elif mode == "mixed":
            # false positives on clear + miss all real rain
            clear = [nd for nd in window_dates if nd not in real_rain_nights]
            for nd in clear:
                notices.append({"event_kind": "rain", "direction": "ALL", "nights": [nd]})
            if clear:
                notices.append(
                    {
                        "event_kind": "cold_snap",
                        "direction": "ALL",
                        "nights": clear[: min(3, len(clear))],
                    }
                )
            # deliberate miss: do not add real_rain_nights
        forecasts.append(
            {
                "record_type": "forecast",
                "issued_at_utc": fmt_utc(issued),
                "coverage_start_utc": fmt_utc(issued),
                "coverage_end_utc": fmt_utc(cov_end),
                "notices": notices,
            }
        )
    return forecasts


def build_requests(slots: list[dict], calendar: list[dict], target_ids: list[str]) -> list[dict]:
    if len(calendar) < 4 or len(target_ids) < 8:
        return []
    survey_start, survey_end = slots[0]["_start"], slots[-1]["_end"]
    reqs = []
    specs = [
        (0, 1, 2, target_ids[0:8], "SYNTH_RQ0001"),
        (max(0, len(calendar) - 3), len(calendar) - 2, len(calendar) - 1, target_ids[8:16], "SYNTH_RQ0002"),
    ]
    for issued_i, _mid, deadline_i, ids, rid in specs:
        if len(ids) < 4:
            continue
        issued = parse_utc(calendar[issued_i]["observing_start_utc"])
        deadline = parse_utc(calendar[deadline_i]["observing_end_utc"])
        if not (survey_start <= issued < deadline <= survey_end):
            continue
        reqs.append(
            {
                "completion_factor_threshold": 0.5,
                "completion_reward": 100.0,
                "deadline_utc": fmt_utc(deadline),
                "issued_at_utc": fmt_utc(issued),
                "minimum_completed": min(6, len(ids)),
                "reason": "SYNTHETIC time-critical follow-up",
                "record_type": "observation_request",
                "request_id": rid,
                "schema_version": "v4-observation-request-v1",
                "target_ids": ids,
            }
        )
    return reqs


def boost_required(src_targets: Path, dst_targets: Path, extra: int, seed: int) -> dict:
    rng = random.Random(seed)
    rows = list(csv.DictReader(src_targets.open(encoding="utf-8")))
    already = [r for r in rows if r["required"].strip().lower() == "true"]
    optional = [r for r in rows if r["required"].strip().lower() != "true"]
    # Prefer boosting targets clustered in few RA bands for uniformity stress
    optional.sort(key=lambda r: float(r["ra_deg"]))
    # take from first 60 deg and last 60 deg heavily
    n = min(extra, len(optional))
    low = [r for r in optional if float(r["ra_deg"]) < 60]
    high = [r for r in optional if float(r["ra_deg"]) >= 300]
    mid = [r for r in optional if 60 <= float(r["ra_deg"]) < 300]
    pick: list[dict] = []
    for pool, share in ((low, 0.45), (high, 0.45), (mid, 0.10)):
        k = min(len(pool), int(n * share))
        pick.extend(rng.sample(pool, k) if k else [])
    # fill remainder
    chosen_ids = {r["target_id"] for r in pick}
    remain = [r for r in optional if r["target_id"] not in chosen_ids]
    while len(pick) < n and remain:
        pick.append(remain.pop(rng.randrange(len(remain))))
    boost_ids = {r["target_id"] for r in pick}
    out_rows = []
    for r in rows:
        rr = dict(r)
        if rr["target_id"] in boost_ids:
            rr["required"] = "true"
        out_rows.append(rr)
    write_csv(
        dst_targets,
        ["target_id", "ra_deg", "dec_deg", "target_class", "feature_flux", "science_weight", "required"],
        out_rows,
    )
    return {
        "original_required": len(already),
        "boosted": len(boost_ids),
        "final_required": len(already) + len(boost_ids),
    }


def materialize(v: Variant, full_cal: list[dict]) -> dict:
    root = OUT_ROOT / v.name
    if root.exists():
        shutil.rmtree(root)
    (root / "config").mkdir(parents=True)
    (root / "public").mkdir()
    (root / "truth").mkdir()

    # copy base files
    for name in ("v4_fiber_config.json", "v4_score_config.json", "v4_scenario.json"):
        shutil.copy2(ALPHA / "config" / name, root / "config" / name)
    shutil.copy2(ALPHA / "public" / "footprint.csv", root / "public" / "footprint.csv")
    if (ALPHA / "card.md").exists():
        shutil.copy2(ALPHA / "card.md", root / "card.md")
    if (ALPHA / "card.json").exists():
        shutil.copy2(ALPHA / "card.json", root / "card.json")

    calendar = trim_calendar(full_cal, v.nights, v.night_offset)
    write_csv(
        root / "public" / "v4_night_calendar.csv",
        list(calendar[0].keys()),
        calendar,
    )

    # targets
    req_info = {"original_required": 500, "boosted": 0, "final_required": 500}
    if v.required_boost > 0:
        req_info = boost_required(
            ALPHA / "public" / "targets.csv",
            root / "public" / "targets.csv",
            v.required_boost,
            v.seed,
        )
    else:
        shutil.copy2(ALPHA / "public" / "targets.csv", root / "public" / "targets.csv")

    # scenario config
    cfg_path = root / "config" / "v4_scenario.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["name"] = f"v4-practice-{v.name}"
    cfg["task_card"] = {
        "card_id": v.name,
        "scenario_slug": f"v4-practice-{v.name}",
        "phase": "practice-projects-synthetic",
    }
    cfg["limits"] = {"global_wallclock_seconds": int(v.wallclock_seconds)}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    rng = random.Random(v.seed)
    slots = build_slots(calendar)
    events, intervals = pick_close_intervals(slots, calendar, v, rng)
    weather = build_weather(slots, intervals, rng, v.weather_mode)
    eq_effects = build_earthquake_effects(slots, events)
    bulletins = build_bulletins(slots, events)
    forecasts = build_forecasts(slots, calendar, events, v.forecast_mode, rng)

    targets = list(csv.DictReader((root / "public" / "targets.csv").open(encoding="utf-8")))
    required_ids = [t["target_id"] for t in targets if t["required"].strip().lower() == "true"]
    pool = required_ids if len(required_ids) >= 16 else [t["target_id"] for t in targets]
    requests = build_requests(slots, calendar, pool)

    write_csv(
        root / "truth" / "v4_slots.csv",
        ["slot_id", "night_id", "timestamp_utc", "duration_seconds"],
        slots,
    )
    write_csv(
        root / "truth" / "v4_weather_truth.csv",
        [
            "slot_id",
            "night_id",
            "timestamp_utc",
            "duration_seconds",
            "is_observable",
            "seeing_arcsec",
            "transparency",
            "sky_quality",
            "instrument_efficiency",
        ],
        weather,
    )
    event_fields = [
        "event_id",
        "event_type",
        "actual_start_utc",
        "actual_end_utc",
        "scope_type",
        "azimuth_start_deg",
        "azimuth_end_deg",
        "min_altitude_deg",
        "max_altitude_deg",
        "magnitude",
        "opacity",
        "force_close",
        "zero_score",
        "seeing_multiplier",
        "transparency_multiplier",
        "sky_quality_multiplier",
        "instrument_efficiency_multiplier",
    ]
    write_csv(root / "truth" / "v4_events.csv", event_fields, events)
    write_csv(
        root / "truth" / "v4_earthquake_effects.csv",
        [
            "event_id",
            "night_id",
            "magnitude",
            "degradation",
            "seeing_multiplier",
            "transparency_multiplier",
            "sky_quality_multiplier",
            "instrument_efficiency_multiplier",
        ],
        eq_effects,
    )
    write_jsonl(root / "public" / "v4_bulletins.jsonl", bulletins)
    write_jsonl(root / "public" / "v4_forecasts.jsonl", forecasts)
    write_jsonl(root / "truth" / "v4_observation_requests.jsonl", requests)

    closed = sum(1 for r in weather if r["is_observable"] == "false")
    cal_note = (
        f"从完整 α 夜历（38 夜，N20261004–N20261110）裁剪："
        f"offset={v.night_offset}, nights={v.nights} → "
        f"{calendar[0]['night_id']} … {calendar[-1]['night_id']}"
    )
    summary = {
        "label": "SYNTHETIC / 非官方真值，分数无官方意义",
        "name": v.name,
        "title": v.title,
        "calendar_trim": cal_note,
        "notes": v.notes,
        "nights": len(calendar),
        "slots": len(slots),
        "weather_closed_slots": closed,
        "close_rate": round(closed / max(1, len(slots)), 4),
        "events": len(events),
        "event_types": sorted({e["event_type"] for e in events}),
        "bulletins": len(bulletins),
        "forecasts": len(forecasts),
        "forecast_mode": v.forecast_mode,
        "observation_requests": len(requests),
        "earthquake_effect_rows": len(eq_effects),
        "required_targets": req_info,
        "wallclock_seconds_in_card": v.wallclock_seconds,
        "survey_start": slots[0]["timestamp_utc"],
        "survey_end": fmt_utc(slots[-1]["_end"]),
    }
    (root / "SYNTHETIC_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for sub in ("", "public", "truth"):
        (root / sub / "SYNTHETIC.txt" if sub else root / "SYNTHETIC.txt").write_text(
            "SYNTHETIC / 非官方真值，分数无官方意义\n", encoding="utf-8"
        )
    readme = f"""# {v.name} — {v.title}

**SYNTHETIC / 非官方真值，分数无官方意义。**

- 源：`practice-cards/alpha/`（只读拷贝）；**未修改** examples。
- 夜历：{cal_note}
- 说明：{'; '.join(v.notes)}
- 卡内 `limits.global_wallclock_seconds` = {v.wallclock_seconds}（仅本合成卡；未改 examples）
- 摘要：`SYNTHETIC_SUMMARY.json`
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    if (root / "card.md").exists():
        text = (root / "card.md").read_text(encoding="utf-8")
        banner = f"> **SYNTHETIC / 非官方真值，分数无官方意义。** 变体：`{v.name}`（{v.title}）。\n\n"
        if "SYNTHETIC" not in text[:240]:
            (root / "card.md").write_text(banner + text, encoding="utf-8")
    return summary


def main() -> None:
    full = load_full_calendar()
    all_sum = []
    for v in VARIANTS:
        s = materialize(v, full)
        all_sum.append(s)
        print(f"OK {v.name}: nights={s['nights']} slots={s['slots']} closed={s['close_rate']}")
    index = {
        "label": "SYNTHETIC / 非官方真值，分数无官方意义",
        "variants": all_sum,
        "generator": str(Path(__file__).resolve()),
    }
    (OUT_ROOT / "alpha-synth-variants-index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
