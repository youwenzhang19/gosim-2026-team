"""Prequential weather-notice and selected-science forecasts for v8 Pro."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone


DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
PROGRAMS = ("DARK", "BRIGHT", "BACKUP")
WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}
MIN_SUPPORT = 8
MAX_MIXTURE_WEIGHT = 0.35
FACTOR_MIN = 0.8
FACTOR_MAX = 1.2
MAX_HISTORY = 256
MAX_WEATHER_WINDOWS = 256
MAX_WEATHER_ROWS = MAX_WEATHER_WINDOWS * len(DIRECTIONS)
MAX_SEEN_SNAPSHOTS = 4096
MAX_REQUESTS = 64
MAX_NEW_MESSAGES = 128
MAX_NOTICES = 64
MAX_WAIT_SECONDS = 8.0
RETRY_BASE_SECONDS = 15.0
RETRY_MAX_SECONDS = 300.0


SYSTEM_PROMPT = (
    "Estimate public-notice forecasts for a robotic survey. The target is whether the FIRST newly issued public "
    "bulletin snapshot in each supplied fixed time window contains a weather notice for each compass sector or ALL. "
    "This is a forecast of a public notice, not a claim about physical weather. Use the supplied rule probabilities "
    "as the deterministic persistence/announced-notice baseline. Also estimate selected-action science residual "
    "multipliers by program only; they are not weather measurements and cannot infer unselected-action effects. "
    "Return JSON only, with exactly this shape: {\"windows\":[{\"window_id\":\"<supplied id>\","
    "\"probabilities\":{\"N\":0.0,\"NE\":0.0,\"E\":0.0,\"SE\":0.0,\"S\":0.0,"
    "\"SW\":0.0,\"W\":0.0,\"NW\":0.0}}],\"science_factors\":{\"DARK\":1.0,"
    "\"BRIGHT\":1.0,\"BACKUP\":1.0}}. Include every supplied window exactly once, use only its supplied ID, and "
    "include exactly the eight probability keys shown. Probabilities must be finite in [0,1]. Science factors must be "
    "finite in [0.8,1.2]. Do not add keys, prose, timestamps, coordinates, targets, rules, or actions."
)


def _parse_time(value):
    """Parse an aware datetime, ISO timestamp, or Unix timestamp into epoch seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    result = parsed.timestamp()
    return result if math.isfinite(result) else None


def _time_text(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value, *, low=None, high=None):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    if low is not None and result < low:
        return None
    if high is not None and result > high:
        return None
    return result


def _target_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    result = str(value)
    return result if result else None


def _direction(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value < len(DIRECTIONS):
        return DIRECTIONS[value]
    if isinstance(value, str):
        text = value.strip().upper()
        return text if text in DIRECTIONS else None
    return None


def _compass_from_azimuth(value):
    azimuth = _finite(value)
    if azimuth is None:
        return None
    return DIRECTIONS[int(((azimuth % 360.0) + 22.5) // 45.0) % 8]


def _stable_hash(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_reason(exc):
    return type(exc).__name__[:40]


class AdaptiveForecasts:
    """Collect shared-client forecasts and score only timestamp-matched public evidence."""

    def __init__(self, client=None, log=lambda text: None):
        self.client = client
        self.log = log
        self._epoch = 0
        self._science_epoch = 0
        self._last_now = None
        self._pending_call = None
        self._pending_meta = None
        self._requests = OrderedDict()
        self._submitted_nights = set()
        self._retry_after = {}
        self._retry_count = defaultdict(int)
        self._weather_windows = OrderedDict()
        self._weather_rows = OrderedDict()
        self._snapshot_seen = OrderedDict()
        self._weather_stats_cache = {}
        self._science_stats_cache = {}
        self._science_exposures = OrderedDict()
        self._active_science_packet = None
        self._latest_packet = None
        self._metrics = {
            "submissions": 0,
            "queue_rejections": 0,
            "accepted_packets": 0,
            "invalid_packets": 0,
            "late_windows_discarded": 0,
            "weather_labels": 0,
            "weather_unknown_windows": 0,
            "science_labels": 0,
            "invalid_feedback": 0,
            "duplicate_feedback": 0,
        }

    # --- calls and nightly request freezing ---------------------------------------------------------

    def update(self, payload, planner, wallclock_left) -> bool:
        """Collect replies, grade public labels, then submit/retry the current night's frozen request."""
        now = _parse_time(payload.get("now_utc")) if isinstance(payload, dict) else None
        if now is not None:
            self._last_now = now
        self._collect(now if now is not None else self._last_now)
        if now is None or not isinstance(payload, dict):
            return False
        self._ingest_snapshots(payload, now)
        self._finalize_weather_windows(now)
        night = self._current_night(planner, now)
        if night is None or self.client is None or not getattr(self.client, "enabled", True):
            return False
        night_index, night_start, night_end = night
        night_key = self._night_key(night_index, night_start)
        if night_key in self._submitted_nights:
            return False
        request = self._requests.get(night_key)
        if request is None:
            request = self._build_request(payload, planner, now, night_index, night_start, night_end)
            if request is None:
                return False
            self._remember_request(night_key, request)
        if self._pending_call is not None:
            return False
        retry_at = self._retry_after.get(night_key, 0.0)
        if time.monotonic() < retry_at:
            return False
        try:
            call = self.client.submit("adaptive_forecast", SYSTEM_PROMPT, copy.deepcopy(request["input"]),
                                      _finite(wallclock_left, low=0.0) or 0.0)
        except Exception as exc:
            call = None
            self._log(f"adaptive_forecast: submit unavailable ({_safe_reason(exc)})")
        if call is None:
            self._metrics["queue_rejections"] += 1
            attempt = self._retry_count[night_key]
            delay = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** min(attempt, 5)))
            self._retry_count[night_key] = attempt + 1
            self._retry_after[night_key] = time.monotonic() + delay
            return False
        self._pending_call = call
        self._pending_meta = {"night_key": night_key, "request": request, "epoch": self._epoch}
        self._submitted_nights.add(night_key)
        self._metrics["submissions"] += 1
        self._retry_after.pop(night_key, None)
        self._retry_count.pop(night_key, None)
        return True

    def wait(self, seconds):
        call = self._pending_call
        wait = getattr(call, "wait", None) if call is not None else None
        budget = _finite(seconds, low=0.0)
        if callable(wait) and budget:
            try:
                wait(min(MAX_WAIT_SECONDS, budget))
            except Exception as exc:
                self._log(f"adaptive_forecast: pending call unavailable ({_safe_reason(exc)})")
        return None

    def _collect(self, now):
        call, meta = self._pending_call, self._pending_meta
        if call is None or meta is None or meta["epoch"] != self._epoch:
            return False
        done = getattr(call, "done", None)
        try:
            if not callable(done) or not done():
                return False
            answer = self.client.collect(call)
        except Exception as exc:
            answer = None
            self._log(f"adaptive_forecast: reply unavailable ({_safe_reason(exc)})")
        self._pending_call = None
        self._pending_meta = None
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                answer = None
        valid = self._validate_answer(answer, meta["request"])
        if valid is None:
            self._metrics["invalid_packets"] += 1
            return True
        if now is None:
            return True
        request = meta["request"]
        packet_id = request["input"]["packet_id"]
        packet = {
            "packet_id": packet_id,
            "night_index": request["input"]["night_index"],
            "accepted_at": now,
            "expires_at": request["expires_at"],
            "science_factors": (valid["science_factors"] if request["science_epoch"] == self._science_epoch else {}),
            "input_hash": request["input"]["input_hash"],
            "epoch": self._epoch,
        }
        self._latest_packet = packet
        for spec in request["windows"]:
            start, end = spec["start"], spec["end"]
            if now >= start:
                self._metrics["late_windows_discarded"] += 1
                continue
            window_id = spec["window_id"]
            if window_id in self._weather_windows:
                continue
            probabilities = valid["windows"][window_id]
            self._weather_windows[window_id] = {
                "window_id": window_id,
                "packet_id": packet_id,
                "start": start,
                "end": end,
                "start_utc": spec["start_utc"],
                "end_utc": spec["end_utc"],
                "issued_at": request["issued_at"],
                "accepted_at": now,
                "probabilities": probabilities,
                "rule_probabilities": dict(spec["rule_probabilities"]),
                "slot_seconds": spec["slot_seconds"],
                "finalized": False,
                "candidate": None,
                "candidate_conflict": False,
            }
            for direction in DIRECTIONS:
                self._weather_rows[(window_id, direction)] = {
                    "window_id": window_id,
                    "direction": direction,
                    "packet_id": packet_id,
                    "agent_probability": probabilities[direction],
                    "rule_probability": spec["rule_probabilities"][direction],
                    "label": None,
                    "received_at": None,
                }
            while len(self._weather_windows) > MAX_WEATHER_WINDOWS:
                self._weather_windows.popitem(last=False)
            while len(self._weather_rows) > MAX_WEATHER_ROWS:
                self._weather_rows.popitem(last=False)
        if now < packet["expires_at"] and (self._active_science_packet is None
                                            or packet["night_index"] >= self._active_science_packet["night_index"]):
            self._active_science_packet = packet
        self._metrics["accepted_packets"] += 1
        return True

    @staticmethod
    def _validate_answer(answer, request):
        if not isinstance(answer, dict) or set(answer) != {"windows", "science_factors"}:
            return None
        windows = answer.get("windows")
        factors = answer.get("science_factors")
        expected = {item["window_id"] for item in request["windows"]}
        if not isinstance(windows, list) or len(windows) != len(expected):
            return None
        parsed_windows = {}
        for item in windows:
            if not isinstance(item, dict) or set(item) != {"window_id", "probabilities"}:
                return None
            window_id = item.get("window_id")
            probabilities = item.get("probabilities")
            if not isinstance(window_id, str) or window_id not in expected or window_id in parsed_windows:
                return None
            if not isinstance(probabilities, dict) or set(probabilities) != set(DIRECTIONS):
                return None
            clean = {}
            for direction in DIRECTIONS:
                value = _finite(probabilities[direction], low=0.0, high=1.0)
                if value is None:
                    return None
                clean[direction] = value
            parsed_windows[window_id] = clean
        if set(parsed_windows) != expected:
            return None
        if not isinstance(factors, dict) or set(factors) != set(PROGRAMS):
            return None
        parsed_factors = {}
        for program in PROGRAMS:
            value = _finite(factors[program], low=FACTOR_MIN, high=FACTOR_MAX)
            if value is None:
                return None
            parsed_factors[program] = value
        return {"windows": parsed_windows, "science_factors": parsed_factors}

    def _build_request(self, payload, planner, now, night_index, night_start, night_end):
        slot = _finite(getattr(planner, "slot_seconds", 900), low=1.0) or 900.0
        windows = []
        current_start = now + 2.0 * slot
        if current_start < night_end:
            current_end = min(night_end, current_start + slot)
            if current_end > current_start:
                windows.append((night_index, "current_slot", current_start, current_end))
        nights = getattr(planner, "nights", ())
        for offset in (1, 2):
            index = night_index + offset
            if index >= len(nights):
                continue
            start, end = nights[index]
            start_epoch, end_epoch = _parse_time(start), _parse_time(end)
            if start_epoch is None or end_epoch is None or end_epoch <= start_epoch or start_epoch <= now:
                continue
            windows.append((index, "first_hour", start_epoch, min(end_epoch, start_epoch + 3600.0)))
        if not windows:
            return None
        packet_number = self._metrics["submissions"] + len(self._requests) + 1
        packet_id = f"e{self._epoch}-n{night_index}-p{packet_number}"
        normalized_notices = self._announced_notices(payload, planner, night_index, now)
        specs = []
        for target_night, window_kind, start, end in windows[:3]:
            window_id = f"e{self._epoch}-n{target_night}-{window_kind}-{int(start)}-{int(end)}"
            applies = [notice for notice in normalized_notices
                       if self._notice_applies(notice, target_night, start, end, night_index, night_start)]
            rule = {
                direction: 1.0 if any(notice["direction"] in {direction, "ALL"} for notice in applies) else 0.0
                for direction in DIRECTIONS
            }
            specs.append({
                "window_id": window_id,
                "target_night": target_night,
                "kind": window_kind,
                "start": start,
                "end": end,
                "slot_seconds": slot,
                "start_utc": _time_text(start),
                "end_utc": _time_text(end),
                "rule_probabilities": rule,
                "announced_notices": [{"event_kind": item["event_kind"], "direction": item["direction"]}
                                      for item in applies[:MAX_NOTICES]],
            })
        issued = _time_text(now)
        expires_text = _time_text(night_end)
        input_obj = {
            "epoch": self._epoch,
            "packet_id": packet_id,
            "night_index": night_index,
            "issued_at_utc": issued,
            "expires_at_utc": expires_text,
            "target": "next_public_weather_notice",
            "windows": [{key: value for key, value in spec.items()
                         if key not in {"start", "end"}} for spec in specs],
            "science": {
                "programs": list(PROGRAMS),
                "prior_condition_support": self._science_support_summary(),
                "support_policy": "selected-action science score increments by program and compass sector",
            },
            "diagnostics": self._diagnostics_context(payload, planner, now, normalized_notices),
        }
        input_obj["input_hash"] = _stable_hash(input_obj)
        return {
            "input": input_obj,
            "windows": specs,
            "issued_at": now,
            "expires_at": night_end,
            "science_epoch": self._science_epoch,
        }

    def _remember_request(self, night_key, request):
        self._requests[night_key] = request
        self._requests.move_to_end(night_key)
        while len(self._requests) > MAX_REQUESTS:
            old_key, _ = self._requests.popitem(last=False)
            self._retry_after.pop(old_key, None)
            self._retry_count.pop(old_key, None)

    @staticmethod
    def _current_night(planner, now):
        current = getattr(planner, "current_night", None)
        try:
            found = current(datetime.fromtimestamp(now, tz=timezone.utc)) if callable(current) else None
        except Exception:
            found = None
        if isinstance(found, (tuple, list)) and len(found) >= 3:
            index = found[0]
            start, end = _parse_time(found[1]), _parse_time(found[2])
            if isinstance(index, int) and not isinstance(index, bool) and start is not None and end is not None:
                if start <= now < end:
                    return index, start, end
        nights = getattr(planner, "nights", ())
        for index, pair in enumerate(nights):
            if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                continue
            start, end = _parse_time(pair[0]), _parse_time(pair[1])
            if start is not None and end is not None and start <= now < end:
                return index, start, end
        return None

    def _night_key(self, index, start):
        return f"{self._epoch}:{index}:{int(start)}"

    # --- frozen deterministic forecast inputs --------------------------------------------------------

    def _announced_notices(self, payload, planner, current_night_index, now):
        notices = []

        def add_rows(rows, source, issued=None):
            if not isinstance(rows, list):
                return
            for row in rows[:MAX_NOTICES]:
                if not isinstance(row, dict):
                    continue
                kind = row.get("event_kind", row.get("kind"))
                direction = str(row.get("direction", "")).strip().upper()
                if not isinstance(kind, str) or kind.lower() not in WEATHER_KINDS:
                    continue
                if direction not in (*DIRECTIONS, "ALL"):
                    continue
                clean = {"event_kind": kind.lower(), "direction": direction, "source": source}
                if issued is not None:
                    clean["issued_at"] = issued
                for field in ("nights", "night_index", "start_utc", "end_utc"):
                    value = row.get(field)
                    if isinstance(value, (list, str, int, float)) and not isinstance(value, bool):
                        clean[field] = copy.deepcopy(value)
                notices.append(clean)

        bulletin = payload.get("latest_bulletin")
        if isinstance(bulletin, dict):
            add_rows(bulletin.get("notices"), "latest_bulletin")
        for kind, direction in getattr(planner, "notices", ()) or ():
            if isinstance(kind, str) and kind.lower() in WEATHER_KINDS and str(direction).upper() in (*DIRECTIONS, "ALL"):
                notices.append({"event_kind": kind.lower(), "direction": str(direction).upper(),
                                "source": "planner_current"})
        forecast_sources = []
        stored_notices = payload.get("forecast_notices")
        if isinstance(stored_notices, list):
            forecast_sources.append(stored_notices)
        latest_forecast = payload.get("latest_forecast")
        if isinstance(latest_forecast, dict) and isinstance(latest_forecast.get("notices"), list):
            forecast_sources.append(latest_forecast["notices"])
        elif isinstance(latest_forecast, list):
            forecast_sources.append(latest_forecast)
        planner_forecasts = getattr(planner, "forecast_notices", None)
        if isinstance(planner_forecasts, list):
            forecast_sources.append(planner_forecasts)
        for rows in forecast_sources:
            add_rows(rows, "announced_forecast")
        messages = payload.get("new_messages")
        if isinstance(messages, list):
            for message in messages[:MAX_NEW_MESSAGES]:
                if not isinstance(message, dict):
                    continue
                tag = self._message_tag(message)
                if tag not in {"forecast", "weather_forecast", "forecast_notice"}:
                    continue
                issued = _parse_time(message.get("issued_at_utc"))
                if issued is None or issued > now:
                    continue
                nested = message.get("forecast")
                rows = nested.get("notices") if isinstance(nested, dict) else message.get("notices")
                add_rows(rows, "announced_forecast", issued)
        return notices[:MAX_NOTICES]

    @staticmethod
    def _message_tag(message):
        values = [message.get(field) for field in ("record_type", "message_type", "type", "kind")]
        normalized = {str(value).strip().lower().replace("-", "_").replace(" ", "_")
                      for value in values if isinstance(value, str)}
        if message.get("public_snapshot") is True or "public_snapshot" in normalized or "bulletin" in normalized:
            return "public_snapshot"
        for tag in ("forecast", "weather_forecast", "forecast_notice"):
            if tag in normalized:
                return tag
        return ""

    @staticmethod
    def _notice_applies(notice, target_night, start, end, current_night, current_start):
        explicit_index = notice.get("night_index")
        if isinstance(explicit_index, int) and not isinstance(explicit_index, bool):
            return explicit_index == target_night
        nights = notice.get("nights")
        if isinstance(nights, list):
            names = {str(item) for item in nights if isinstance(item, (str, int)) and not isinstance(item, bool)}
            target_date = datetime.fromtimestamp(start - 12 * 3600, tz=timezone.utc).date().isoformat()
            if str(target_night) in names or target_date in names:
                return True
            if names:
                return False
        elif isinstance(nights, str):
            target_date = datetime.fromtimestamp(start - 12 * 3600, tz=timezone.utc).date().isoformat()
            return nights == str(target_night) or nights == target_date
        notice_start = _parse_time(notice.get("start_utc"))
        notice_end = _parse_time(notice.get("end_utc"))
        if notice_start is not None or notice_end is not None:
            return (notice_start is None or notice_start < end) and (notice_end is None or notice_end > start)
        return notice.get("source") in {"latest_bulletin", "planner_current"} and target_night == current_night \
            and start >= current_start

    # --- weather snapshot labels ----------------------------------------------------------------------

    def _ingest_snapshots(self, payload, now):
        messages = payload.get("new_messages")
        sources = []
        if isinstance(messages, list) and len(messages) <= MAX_NEW_MESSAGES:
            sources.extend(message for message in messages if isinstance(message, dict)
                           and self._message_tag(message) == "public_snapshot")
        latest = payload.get("latest_bulletin")
        if isinstance(latest, dict) and latest.get("issued_at_utc") is not None:
            sources.append({**latest, "record_type": "public_snapshot"})
        for message in sources:
            issued = _parse_time(message.get("issued_at_utc"))
            if issued is None or issued > now:
                continue
            snapshot = self._parse_snapshot(message)
            bulletin_id = next((message.get(key) for key in ("bulletin_id", "snapshot_id", "message_id", "id")
                                if isinstance(message.get(key), (str, int))
                                and not isinstance(message.get(key), bool)), None)
            signature = (f"id:{bulletin_id}" if bulletin_id is not None else
                         _stable_hash({"issued_at_utc": issued, "snapshot": snapshot}))
            if signature in self._snapshot_seen:
                continue
            self._snapshot_seen[signature] = None
            self._snapshot_seen.move_to_end(signature)
            while len(self._snapshot_seen) > MAX_SEEN_SNAPSHOTS:
                self._snapshot_seen.popitem(last=False)
            for meta in self._weather_windows.values():
                if meta["finalized"] or not (meta["start"] <= issued < meta["end"]):
                    continue
                if issued <= meta["issued_at"]:
                    continue
                max_age = max(meta["end"] - meta["start"] + self._slot_for_meta(meta),
                              2.0 * self._slot_for_meta(meta))
                candidate = {
                    "issued_at": issued,
                    "signature": signature,
                    "snapshot": snapshot,
                    "stale": now - issued > max_age,
                }
                old = meta["candidate"]
                if old is None or issued < old["issued_at"]:
                    meta["candidate"] = candidate
                    meta["candidate_conflict"] = False
                elif issued == old["issued_at"] and signature != old["signature"]:
                    meta["candidate_conflict"] = True

    @staticmethod
    def _parse_snapshot(message):
        snapshot = message.get("snapshot")
        if not isinstance(snapshot, dict):
            snapshot = message.get("public_snapshot")
        if isinstance(snapshot, dict):
            bulletin = snapshot.get("bulletin")
            if isinstance(bulletin, dict):
                rows = bulletin.get("notices")
            else:
                rows = snapshot.get("notices")
        else:
            bulletin = message.get("bulletin")
            rows = bulletin.get("notices") if isinstance(bulletin, dict) else message.get("notices")
        if not isinstance(rows, list) or len(rows) > MAX_NOTICES:
            return None
        result = []
        for row in rows[:MAX_NOTICES]:
            if not isinstance(row, dict):
                return None
            kind = row.get("event_kind", row.get("kind"))
            direction = row.get("direction")
            if not isinstance(kind, str) or not kind.strip():
                return None
            normalized = str(direction).strip().upper() if direction is not None else ""
            if normalized not in (*DIRECTIONS, "ALL"):
                if kind.lower() in WEATHER_KINDS or normalized:
                    return None
                normalized = ""
            result.append((kind.lower(), normalized))
        return result

    @staticmethod
    def _slot_for_meta(meta):
        return meta.get("slot_seconds", 900.0)

    def _finalize_weather_windows(self, now):
        for window_id, meta in self._weather_windows.items():
            if meta["finalized"] or now < meta["end"]:
                continue
            meta["finalized"] = True
            candidate = meta["candidate"]
            if candidate is None or candidate["stale"] or candidate["snapshot"] is None or meta["candidate_conflict"]:
                self._metrics["weather_unknown_windows"] += 1
                continue
            notices = candidate["snapshot"]
            for direction in DIRECTIONS:
                row = self._weather_rows.get((window_id, direction))
                if row is None or row["label"] is not None:
                    continue
                row["label"] = 1.0 if any(kind in WEATHER_KINDS and named in {direction, "ALL"}
                                          for kind, named in notices) else 0.0
                row["received_at"] = now
                self._metrics["weather_labels"] += 1
                self._weather_stats_cache.pop(direction, None)

    # --- selected-exposure science feedback -----------------------------------------------------------

    def freeze_exposure(self, index, estimate, action, old_best_scores, issued_at, end_time):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in self._science_exposures:
            return False
        if not isinstance(estimate, dict) or not isinstance(action, dict):
            return False
        assignments = action.get("assignments")
        if not isinstance(assignments, dict) or not assignments:
            return False
        target_ids = [_target_id(value) for value in assignments.values()]
        if any(value is None for value in target_ids) or len(set(target_ids)) != len(target_ids):
            return False
        if not isinstance(old_best_scores, dict):
            return False
        old = {}
        for key, value in old_best_scores.items():
            normalized = _target_id(key)
            score = _finite(value, low=0.0)
            if normalized is None or score is None or normalized in old:
                return False
            old[normalized] = score
        if any(target not in old for target in target_ids):
            return False
        base_gain = _finite(estimate.get("base_science_gain", estimate.get("base_gain")), low=0.0)
        issued = _parse_time(issued_at)
        end = _parse_time(end_time)
        if base_gain is None or issued is None or end is None or end <= issued:
            return False
        program = str(action.get("program", estimate.get("program", ""))).upper()
        if program not in PROGRAMS:
            program = None
        direction = self._exposure_direction(estimate, action)
        model_factor, packet_id = self._science_forecast(program, issued)
        if program is not None and direction is not None and model_factor is not None:
            stats = self._science_stats((program, direction), before=issued)
            shadow_weight = self._mixture_weight(stats["support"])
            agent_prediction = base_gain * (1.0 + shadow_weight * (model_factor - 1.0))
        else:
            shadow_weight = 0.0
            agent_prediction = base_gain
        record = {
            "index": index,
            "program": program,
            "direction": direction,
            "base_gain": base_gain,
            "base_prediction": base_gain,
            "agent_prediction": agent_prediction,
            "model_factor": model_factor,
            "mixture_weight_at_issue": shadow_weight,
            "packet_id": packet_id,
            "assigned_ids": tuple(target_ids),
            "old_best_scores": old,
            "issued_at": issued,
            "end_time": end,
            "label": None,
            "received_at": None,
            "science_epoch": self._science_epoch,
        }
        self._science_exposures[index] = record
        self._science_exposures.move_to_end(index)
        while len(self._science_exposures) > MAX_HISTORY:
            self._science_exposures.popitem(last=False)
        return True

    def observe_result(self, result, now):
        if not isinstance(result, dict):
            self._metrics["invalid_feedback"] += 1
            return False
        index = self._result_index(result)
        record = self._science_exposures.get(index) if index is not None else None
        if record is None or record["science_epoch"] != self._science_epoch:
            self._metrics["invalid_feedback"] += 1
            return False
        if record["label"] is not None:
            self._metrics["duplicate_feedback"] += 1
            return False
        received = _parse_time(now)
        hits = result.get("hits")
        count = result.get("hit_count")
        if (received is None or received < record["end_time"] or result.get("action") != "observe"
                or not isinstance(hits, list) or isinstance(count, bool)
                or not isinstance(count, int) or count != len(hits)):
            self._metrics["invalid_feedback"] += 1
            return False
        assigned_count = result.get("assigned_count")
        if (isinstance(assigned_count, bool) or not isinstance(assigned_count, int)
                or assigned_count != len(record["assigned_ids"])):
            self._metrics["invalid_feedback"] += 1
            return False
        allowed = set(record["assigned_ids"])
        hit_scores = {}
        for hit in hits:
            if not isinstance(hit, dict) or "target_id" not in hit or "score" not in hit:
                self._metrics["invalid_feedback"] += 1
                return False
            target = _target_id(hit["target_id"])
            score = _finite(hit["score"], low=0.0)
            if target is None or target not in allowed or target in hit_scores or score is None:
                self._metrics["invalid_feedback"] += 1
                return False
            hit_scores[target] = score
        observed = sum(max(0.0, hit_scores.get(target, 0.0) - record["old_best_scores"][target])
                       for target in record["assigned_ids"])
        if not math.isfinite(observed):
            self._metrics["invalid_feedback"] += 1
            return False
        record["label"] = observed
        record["received_at"] = received
        self._metrics["science_labels"] += 1
        if record["program"] is not None and record["direction"] is not None:
            self._science_stats_cache.pop((record["program"], record["direction"]), None)
        return True

    @staticmethod
    def _result_index(result):
        if result.get("action") != "observe":
            return None
        value = result.get("observe_index")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    @staticmethod
    def _exposure_direction(estimate, action):
        for source in (estimate, action):
            for field in ("direction", "sector", "direction_bucket", "compass_direction"):
                candidate = _direction(source.get(field))
                if candidate is not None:
                    return candidate
            for field in ("azimuth_deg", "az_deg", "center_az_deg", "pointing_az_deg", "azimuth", "az"):
                candidate = _compass_from_azimuth(source.get(field))
                if candidate is not None:
                    return candidate
        return None

    def reset_science(self):
        self._science_epoch += 1
        self._science_exposures.clear()
        self._science_stats_cache.clear()
        self._metrics["science_labels"] = 0
        if self._active_science_packet is not None:
            packet = dict(self._active_science_packet)
            packet["science_factors"] = {}
            self._active_science_packet = packet
        return None

    # --- independent trust gates ----------------------------------------------------------------------

    def factor(self, program, direction, now) -> float:
        condition = (str(program).upper(), _direction(direction))
        current = _parse_time(now)
        packet = self._active_science_packet
        if (condition[0] not in PROGRAMS or condition[1] is None or current is None or packet is None
                or packet["epoch"] != self._epoch or packet["accepted_at"] > current
                or current >= packet["expires_at"]):
            return 1.0
        raw = packet["science_factors"].get(condition[0])
        if raw is None:
            return 1.0
        stats = self._science_stats(condition, before=current)
        if not self._science_gate(stats):
            return 1.0
        weight = self._mixture_weight(stats["support"])
        return min(FACTOR_MAX, max(FACTOR_MIN, 1.0 + weight * (raw - 1.0)))

    def risk(self, direction, now) -> float:
        direction = _direction(direction)
        current = _parse_time(now)
        if direction is None or current is None:
            return 0.0
        active = [meta for meta in self._weather_windows.values()
                  if not meta["finalized"] and meta["start"] <= current < meta["end"]]
        if not active:
            return 0.0
        meta = max(active, key=lambda item: (item["accepted_at"], item["start"]))
        stats = self._weather_stats(direction, before=current)
        if not self._weather_gate(stats):
            return 0.0
        row = self._weather_rows.get((meta["window_id"], direction))
        if row is None:
            return 0.0
        weight = self._mixture_weight(stats["support"])
        result = row["rule_probability"] + weight * (row["agent_probability"] - row["rule_probability"])
        return min(1.0, max(0.0, result))

    @staticmethod
    def _mixture_weight(support):
        if support <= 0:
            return 0.0
        return min(MAX_MIXTURE_WEIGHT, MAX_MIXTURE_WEIGHT * support / (support + MIN_SUPPORT))

    def _science_forecast(self, program, issued):
        packet = self._active_science_packet
        if (packet is None or packet["epoch"] != self._epoch or issued >= packet["expires_at"]
                or packet["accepted_at"] > issued or program not in PROGRAMS):
            return None, None
        value = packet["science_factors"].get(program)
        if value is None:
            return None, None
        return value, packet["packet_id"]

    def _science_stats(self, condition, before=None):
        if before is None and condition in self._science_stats_cache:
            return self._science_stats_cache[condition]
        program, direction = condition
        rows = [row for row in self._science_exposures.values()
                if row["science_epoch"] == self._science_epoch and row["label"] is not None
                and row["program"] == program and row["direction"] == direction
                and row["base_gain"] > 1e-6
                and row["model_factor"] is not None
                and (before is None or row["received_at"] <= before)]
        rows.sort(key=lambda row: (row["received_at"], row["issued_at"], row["index"]))
        result = self._loss_stats(rows, science=True)
        if before is None:
            self._science_stats_cache[condition] = result
        return result

    def _weather_stats(self, direction, before=None):
        if before is None and direction in self._weather_stats_cache:
            return self._weather_stats_cache[direction]
        rows = [row for row in self._weather_rows.values()
                if row["direction"] == direction and row["label"] is not None
                and (before is None or row["received_at"] <= before)]
        rows.sort(key=lambda row: (row["received_at"], row["window_id"]))
        result = self._loss_stats(rows, science=False)
        if before is None:
            self._weather_stats_cache[direction] = result
        return result

    @staticmethod
    def _loss_stats(rows, *, science):
        if not rows:
            return {"support": 0, "base_loss": None, "agent_loss": None,
                    "recent_base_loss": None, "recent_agent_loss": None, "recent_support": 0}
        recent = rows[-MIN_SUPPORT:]
        if science:
            def loss(row, field):
                denominator = max(1.0, row["base_gain"])
                return ((row[field] - row["label"]) / denominator) ** 2
        else:
            def loss(row, field):
                return (row[field] - row["label"]) ** 2
        base = [loss(row, "base_prediction" if science else "rule_probability") for row in rows]
        agent = [loss(row, "agent_prediction" if science else "agent_probability") for row in rows]
        recent_base = [loss(row, "base_prediction" if science else "rule_probability") for row in recent]
        recent_agent = [loss(row, "agent_prediction" if science else "agent_probability") for row in recent]
        return {
            "support": len(rows),
            "base_loss": sum(base) / len(base),
            "agent_loss": sum(agent) / len(agent),
            "recent_base_loss": sum(recent_base) / len(recent_base),
            "recent_agent_loss": sum(recent_agent) / len(recent_agent),
            "recent_support": len(recent),
        }

    @staticmethod
    def _science_gate(stats):
        return bool(stats["support"] >= MIN_SUPPORT and stats["base_loss"] is not None
                    and stats["agent_loss"] < stats["base_loss"]
                    and stats["recent_agent_loss"] <= stats["recent_base_loss"])

    @staticmethod
    def _weather_gate(stats):
        return AdaptiveForecasts._science_gate(stats)

    # --- diagnostics and epoch invalidation ------------------------------------------------------------

    def _science_support_summary(self):
        groups = {}
        for row in self._science_exposures.values():
            if row["label"] is None or row["model_factor"] is None or row["program"] is None or row["direction"] is None:
                continue
            if row["base_gain"] <= 1e-6:
                continue
            key = (row["program"], row["direction"])
            groups[key] = groups.get(key, 0) + 1
        return [{"program": program, "direction": direction, "support": count}
                for (program, direction), count in sorted(groups.items())[:32]]

    def _diagnostics_context(self, payload, planner, now, normalized_notices):
        weather_errors = []
        for direction in DIRECTIONS:
            stats = self._weather_stats(direction)
            if stats["support"]:
                weather_errors.append({
                    "direction": direction,
                    "support": stats["support"],
                    "base_brier": stats["base_loss"],
                    "agent_brier": stats["agent_loss"],
                    "recent_base_brier": stats["recent_base_loss"],
                    "recent_agent_brier": stats["recent_agent_loss"],
                })

        science_errors = []
        for item in self._science_support_summary():
            program, direction = item["program"], item["direction"]
            stats = self._science_stats((program, direction))
            science_errors.append({
                "program": program,
                "direction": direction,
                "support": stats["support"],
                "base_normalized_squared_error": stats["base_loss"],
                "agent_normalized_squared_error": stats["agent_loss"],
                "recent_base_normalized_squared_error": stats["recent_base_loss"],
                "recent_agent_normalized_squared_error": stats["recent_agent_loss"],
            })

        mature = [row for row in self._science_exposures.values()
                  if row["label"] is not None and row["received_at"] is not None
                  and row["received_at"] <= now and row["program"] in PROGRAMS
                  and row["direction"] in DIRECTIONS]
        mature.sort(key=lambda row: (row["received_at"], row["issued_at"], row["index"]))
        recent_exposures = [{
            "program": row["program"],
            "direction": row["direction"],
            "base_gain": row["base_gain"],
            "observed_gain": row["label"],
        } for row in mature[-8:]]

        bulletin = payload.get("latest_bulletin")
        bulletin_issued = _parse_time(bulletin.get("issued_at_utc")) if isinstance(bulletin, dict) else None
        current_notices = []
        seen_notices = set()
        for item in normalized_notices:
            if item["source"] not in {"latest_bulletin", "planner_current"}:
                continue
            notice = (item["event_kind"], item["direction"])
            if notice in seen_notices:
                continue
            seen_notices.add(notice)
            current_notices.append({"event_kind": notice[0], "direction": notice[1]})

        scale = _finite(getattr(planner, "scale", None), low=0.0)
        band_level = getattr(planner, "band_level", None)
        if isinstance(band_level, bool) or not isinstance(band_level, (str, int, float)):
            band_level = None
        elif isinstance(band_level, str):
            band_level = band_level[:64]
        else:
            band_level = _finite(band_level)

        return {
            "planner": {"scale": scale, "band_level": band_level},
            "current_bulletin": {
                "issued_at_utc": _time_text(bulletin_issued) if bulletin_issued is not None else None,
                "notices": current_notices[:MAX_NOTICES],
            },
            "weather_errors_by_direction": weather_errors,
            "science_errors_by_condition": science_errors,
            "recent_exposures": recent_exposures,
        }

    def context(self):
        now = self._last_now
        windows = []
        for meta in self._weather_windows.values():
            if meta["end"] <= (now if now is not None else float("-inf")) or meta["finalized"]:
                continue
            windows.append({
                "window_id": meta["window_id"],
                "start_utc": meta["start_utc"],
                "end_utc": meta["end_utc"],
                "probabilities": dict(meta["probabilities"]),
                "rule_probabilities": dict(meta["rule_probabilities"]),
            })
        windows = windows[-6:]
        packet = self._active_science_packet
        factors = {}
        if packet is not None and packet["epoch"] == self._epoch and now is not None and now < packet["expires_at"]:
            factors = dict(packet["science_factors"])
        latest = self._latest_packet or {}
        return {
            "epoch": self._epoch,
            "packet_id": latest.get("packet_id"),
            "night_index": latest.get("night_index"),
            "expires_at": _time_text(latest["expires_at"]) if latest.get("expires_at") is not None else None,
            "weather": {"target": "next_public_weather_notice", "windows": windows},
            "science_factors": factors,
            "history": self._history_context(),
        }

    def _history_context(self):
        weather = []
        for direction in DIRECTIONS:
            stats = self._weather_stats(direction)
            if stats["support"]:
                weather.append({
                    "direction": direction,
                    "support": stats["support"],
                    "rule_brier": stats["base_loss"],
                    "agent_brier": stats["agent_loss"],
                    "recent_rule_brier": stats["recent_base_loss"],
                    "recent_agent_brier": stats["recent_agent_loss"],
                    "trusted": self._weather_gate(stats),
                    "mixture_weight": self._mixture_weight(stats["support"])
                    if self._weather_gate(stats) else 0.0,
                })
        science = []
        for item in self._science_support_summary():
            program, direction, support = item["program"], item["direction"], item["support"]
            stats = self._science_stats((program, direction))
            science.append({
                "program": program,
                "direction": direction,
                "support": support,
                "base_normalized_squared_error": stats["base_loss"],
                "agent_normalized_squared_error": stats["agent_loss"],
                "recent_base_normalized_squared_error": stats["recent_base_loss"],
                "recent_agent_normalized_squared_error": stats["recent_agent_loss"],
                "trusted": self._science_gate(stats),
            })
        return {"weather_by_direction": weather[:8], "science_by_condition": science[:32]}

    def summary(self):
        result = {
            "epoch": self._epoch,
            "science_epoch": self._science_epoch,
            "pending": self._pending_call is not None,
            "packet_id": (self._latest_packet or {}).get("packet_id"),
            "metrics": dict(self._metrics),
            "weather": {
                "target": "next_public_weather_notice",
                "window_count": len(self._weather_windows),
                "label_count": self._metrics["weather_labels"],
                "unknown_window_count": self._metrics["weather_unknown_windows"],
                "groups": self._history_context()["weather_by_direction"],
                "support_set": "direction x fixed public-snapshot window; first issued-at-matched snapshot only",
            },
            "science": {
                "frozen_exposure_count": len(self._science_exposures),
                "label_count": self._metrics["science_labels"],
                "groups": self._history_context()["science_by_condition"],
                "minimum_condition_support": MIN_SUPPORT,
                "maximum_agent_mixture_weight": MAX_MIXTURE_WEIGHT,
                "factor_bounds": [FACTOR_MIN, FACTOR_MAX],
                "support_set": "executed selected exposures only, matched by program and compass sector",
            },
        }
        if self._latest_packet is not None:
            result["expires_at"] = _time_text(self._latest_packet["expires_at"])
        return result

    def reset(self):
        self._epoch += 1
        self._science_epoch += 1
        self._last_now = None
        self._pending_call = None
        self._pending_meta = None
        self._requests.clear()
        self._submitted_nights.clear()
        self._retry_after.clear()
        self._retry_count.clear()
        self._weather_windows.clear()
        self._weather_rows.clear()
        self._snapshot_seen.clear()
        self._weather_stats_cache.clear()
        self._science_stats_cache.clear()
        self._science_exposures.clear()
        self._active_science_packet = None
        self._latest_packet = None
        for key in self._metrics:
            self._metrics[key] = 0
        return None

    def _log(self, message):
        try:
            self.log(message)
        except Exception:
            pass
