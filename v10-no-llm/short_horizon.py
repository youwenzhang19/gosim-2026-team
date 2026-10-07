"""Finite two-step pressure-scenario reranking for the official pro planner.

This module evaluates existing planner actions only. It never executes a future
action, reads private card data, or calls an external model.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta

from skymath import altaz_to_radec, local_sidereal_deg, radec_to_altaz


SCENARIOS = (("persistent", 1.0), ("adverse", 0.8), ("favorable", 1.1))
MAX_CANDIDATES = 4
IMMEDIATE_UTILITY_FLOOR = 0.98


class _CPUDeadline(Exception):
    pass


def _finite(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _expired(deadline):
    try:
        return time.process_time() >= float(deadline)
    except (TypeError, ValueError, OverflowError):
        return False


def _check_deadline(deadline):
    if _expired(deadline):
        raise _CPUDeadline


def _action(candidate):
    if not isinstance(candidate, dict):
        return None
    nested = candidate.get("action")
    if isinstance(nested, dict):
        return nested
    return candidate


def _signature(action):
    if not isinstance(action, dict):
        return (id(action),)
    pointing = action.get("pointing") or {}
    if isinstance(pointing, dict):
        point = (_finite(pointing.get("alt_deg"), math.nan),
                 _finite(pointing.get("az_deg"), math.nan))
    else:
        point = tuple(_finite(value, math.nan) for value in pointing[:2])
    assignments = action.get("assignments") or {}
    if isinstance(assignments, dict):
        assigned = tuple(sorted((str(fiber), str(target)) for fiber, target in assignments.items()))
    else:
        assigned = tuple(sorted(map(str, assignments)))
    return (tuple(round(value, 4) for value in point),
            int(_finite(action.get("duration_seconds"))), str(action.get("program", "")), assigned)


def _candidate_actions(planner, base_action):
    base = _action(base_action)
    if base is None:
        return []
    base_signature = _signature(base)
    supplied = getattr(planner, "last_candidates", ())
    supplied = supplied(MAX_CANDIDATES) if callable(supplied) else supplied
    supplied = supplied or []
    selected = []
    seen = set()
    for candidate in supplied:
        action = _action(candidate)
        if action is None:
            continue
        signature = _signature(action)
        if signature not in seen:
            selected.append(action)
            seen.add(signature)
        if len(selected) >= MAX_CANDIDATES:
            break
    if base_signature not in seen:
        selected = [base] + selected[:MAX_CANDIDATES - 1]
    else:
        selected = [base] + [action for action in selected if _signature(action) != base_signature]
        selected = selected[:MAX_CANDIDATES]
    return selected


def _evaluate(planner, action, now, night_end, night_index, hours, scenario_scale,
              cur_override, factor_override, deadline):
    _check_deadline(deadline)
    candidate = planner.evaluate_action(
        action, now, night_end, night_index, hours,
        scale_override=planner.scale * scenario_scale,
        cur_override=cur_override,
        factor_override=factor_override,
    )
    _check_deadline(deadline)
    return candidate


def _future_template(planner, action, now, future_now):
    """Keep the same equatorial field centre as the sky rotates between steps."""
    if not all(hasattr(planner, name) for name in ("lat", "lon", "offset")):
        return action
    try:
        latitude = float(planner.lat)
        longitude = float(planner.lon)
        offset_alt, offset_az = map(float, planner.offset)
        pointing = action["pointing"]
        command_alt = float(pointing["alt_deg"])
        command_az = float(pointing["az_deg"])
        actual_alt = command_alt + offset_alt
        actual_az = (command_az + offset_az) % 360.0
        if not all(math.isfinite(value) for value in
                   (latitude, longitude, offset_alt, offset_az, actual_alt, actual_az)):
            return None
        lst_now = local_sidereal_deg(now, longitude)
        ra, dec = altaz_to_radec(actual_alt, actual_az, lst_now, latitude)
        lst_future = local_sidereal_deg(future_now, longitude)
        future_alt, future_az = radec_to_altaz(ra, dec, lst_future, latitude)
        future_command_alt = future_alt - offset_alt
        if (not all(math.isfinite(value) for value in (ra, dec, future_alt, future_az))
                or not 0.0 <= future_command_alt <= 90.0):
            return None
        future_action = dict(action)
        future_pointing = dict(pointing)
        future_pointing["alt_deg"] = round(future_command_alt, 4)
        future_pointing["az_deg"] = round((future_az - offset_az) % 360.0, 4)
        future_action["pointing"] = future_pointing
        return future_action
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


def _index(planner, target_id):
    try:
        return planner.index_of.get(target_id, planner.index_of.get(str(target_id)))
    except AttributeError:
        return None


def _projected_overrides(planner, candidate, cur_override, factor_override):
    projected_cur = dict(cur_override or {})
    projected_factor = dict(factor_override or {})
    predictions = candidate.get("predictions") or {}
    for target_id, prediction in predictions.items():
        if not isinstance(prediction, dict):
            continue
        index = _index(planner, target_id)
        if index is None:
            continue
        target_id = planner.ids[index]
        weight = max(1e-12, _finite(planner.weight[index], 1.0))
        current_cur = projected_cur.get(target_id, planner.cur[index])
        current_factor = projected_factor.get(target_id, planner.factor[index])
        score = _finite(prediction.get("expected_score")) / weight
        factor = _finite(prediction.get("factor"))
        projected_cur[target_id] = max(_finite(current_cur), score)
        projected_factor[target_id] = max(_finite(current_factor), factor)
    return projected_cur, projected_factor


def _nonrequest_utility(candidate):
    return (_finite(candidate.get("utility")) - _finite(candidate.get("request_gain"))
            - _finite(candidate.get("required_gain")))


def _required_gain_sequence(candidates):
    per_target = []
    complete = True
    aggregate = []
    for candidate in candidates:
        aggregate.append(max(0.0, _finite(candidate.get("required_gain"))))
        predictions = candidate.get("predictions") or {}
        if not predictions and aggregate[-1] > 0.0:
            complete = False
        gains = {}
        for target_id, prediction in predictions.items():
            if not isinstance(prediction, dict) or "required_gain" not in prediction:
                complete = False
                continue
            gains[target_id] = max(0.0, _finite(prediction.get("required_gain")))
        per_target.append(gains)
    if not complete:
        return max(aggregate, default=0.0)
    if not per_target:
        return 0.0
    first = per_target[0]
    total = sum(first.values())
    for future in per_target[1:]:
        total += sum(max(0.0, gain - first.get(target_id, 0.0))
                     for target_id, gain in future.items())
    return total


def _moment(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _request_reward(planner, steps, deadline):
    reward = 0.0
    for request in getattr(planner, "requests", ()):
        _check_deadline(deadline)
        target_indices = set(request.get("target_indices", ()))
        remaining = int(request.get("remaining_count", 0))
        threshold = _finite(request.get("threshold"))
        request_reward = max(0.0, _finite(request.get("reward")))
        request_deadline = _moment(request.get("deadline"))
        if not target_indices or remaining <= 0 or request_deadline is None:
            continue
        qualified = set()
        for start, end, candidate in steps:
            _check_deadline(deadline)
            if end > request_deadline:
                continue
            for target_id, prediction in (candidate.get("predictions") or {}).items():
                _check_deadline(deadline)
                index = _index(planner, target_id)
                request_factor = _finite(prediction.get(
                    "request_factor", prediction.get("planning_factor", prediction.get("factor"))
                ))
                if index in target_indices and request_factor >= threshold:
                    qualified.add(index)
        if len(qualified) >= remaining:
            reward += request_reward
    return reward


def _fallback(base_action, reason, **details):
    summary = {
        "module": "official-pro-short-horizon-v1",
        "used": False,
        "fallback": True,
        "reason": reason,
        "scenarios": [{"name": name, "scale": factor} for name, factor in SCENARIOS],
        "scenario_role": "frozen pressure cases, not weather probabilities",
    }
    summary.update(details)
    return base_action, summary


def rerank(planner, base_action, now, night_end, night_index, hours, deadline):
    """Compare up to four current actions and one future continuation.

    ``deadline`` is an absolute ``time.process_time()`` value. The winning
    first action is committed through ``record_action`` only when it differs
    from the planner's already-pending base action.
    """
    if base_action is None:
        return _fallback(base_action, "no_base_action")
    if _expired(deadline):
        return _fallback(base_action, "cpu_deadline")

    try:
        actions = _candidate_actions(planner, base_action)
        if not actions:
            return _fallback(base_action, "empty_candidates")
        signatures = [_signature(action) for action in actions]
        base_signature = _signature(_action(base_action))
        sequence_options = []
        future_rejections = 0

        for first_action, first_signature in zip(actions, signatures):
            _check_deadline(deadline)
            first_duration = int(_finite(first_action.get("duration_seconds")))
            if first_duration <= 0:
                continue
            scenario_first = {}
            for scenario_name, scenario_scale in SCENARIOS:
                first = _evaluate(planner, first_action, now, night_end, night_index, hours,
                                  scenario_scale, None, None, deadline)
                if first is None:
                    scenario_first = {}
                    break
                projected_cur, projected_factor = _projected_overrides(planner, first, None, None)
                scenario_first[scenario_name] = {
                    "evaluation": first,
                    "cur": projected_cur,
                    "factor": projected_factor,
                }
            if len(scenario_first) != len(SCENARIOS):
                continue
            first_end = now + timedelta(seconds=first_duration)
            one_step_values = {}
            for scenario_name, _scenario_scale in SCENARIOS:
                first = scenario_first[scenario_name]["evaluation"]
                one_step_values[scenario_name] = (
                    _nonrequest_utility(first) + _required_gain_sequence([first])
                    + _request_reward(planner, [(now, first_end, first)], deadline)
                )
            first_immediate = {name: _finite(data["evaluation"].get("utility"))
                               for name, data in scenario_first.items()}
            sequence_options.append({
                "first": first_action,
                "signature": first_signature,
                "values": one_step_values,
                "immediate": first_immediate,
                "steps": 1,
            })

            future_now = first_end
            future_hours = hours + first_duration / 3600.0
            for continuation, continuation_signature in zip(actions, signatures):
                if continuation_signature == first_signature:
                    continue
                _check_deadline(deadline)
                future_action = _future_template(planner, continuation, now, future_now)
                if future_action is None:
                    future_rejections += 1
                    continue
                scenario_future = {}
                for scenario_name, scenario_scale in SCENARIOS:
                    prior = scenario_first[scenario_name]
                    future = _evaluate(planner, future_action, future_now, night_end, night_index,
                                       future_hours, scenario_scale, prior["cur"], prior["factor"], deadline)
                    if future is None:
                        scenario_future = {}
                        break
                    scenario_future[scenario_name] = future
                if len(scenario_future) != len(SCENARIOS):
                    future_rejections += 1
                    continue
                pair_values = {}
                for scenario_name, _scenario_scale in SCENARIOS:
                    first = scenario_first[scenario_name]["evaluation"]
                    future = scenario_future[scenario_name]
                    pair_values[scenario_name] = (
                        _nonrequest_utility(first) + _nonrequest_utility(future)
                        + _required_gain_sequence([first, future])
                        + _request_reward(planner, [(now, first_end, first),
                                                    (future_now, future_now + timedelta(
                                                        seconds=int(_finite(future_action.get("duration_seconds")))), future)],
                                            deadline)
                    )
                sequence_options.append({
                    "first": first_action,
                    "signature": first_signature,
                    "values": pair_values,
                    "immediate": first_immediate,
                    "steps": 2,
                })

        if not sequence_options:
            return _fallback(base_action, "no_valid_sequence", future_rejections=future_rejections)
        base_rows = [row for row in sequence_options if row["signature"] == base_signature]
        if not base_rows:
            return _fallback(base_action, "base_comparator_missing", future_rejections=future_rejections)
        base_immediate = base_rows[0]["immediate"]["persistent"]

        def rank_key(row):
            worst = min(row["values"].values())
            persistent = row["values"]["persistent"]
            immediate = row["immediate"]["persistent"]
            is_base = row["signature"] == base_signature
            return (worst, persistent, immediate, is_base, -row["steps"])

        selected = max(sequence_options, key=rank_key)
        selected_immediate = selected["immediate"]["persistent"]
        if selected_immediate + 1e-9 < base_immediate * IMMEDIATE_UTILITY_FLOOR:
            return _fallback(base_action, "immediate_utility_floor",
                             candidate_count=len(actions), sequence_count=len(sequence_options),
                             base_immediate_utility=round(base_immediate, 6),
                             selected_immediate_utility=round(selected_immediate, 6),
                             future_rejections=future_rejections)

        if selected["signature"] == base_signature:
            return _fallback(base_action, "base_selected",
                             candidate_count=len(actions), sequence_count=len(sequence_options),
                             scenario_utility={name: round(value, 6)
                                               for name, value in selected["values"].items()},
                             base_immediate_utility=round(base_immediate, 6),
                             selected_immediate_utility=round(selected_immediate, 6),
                             selected_steps=selected["steps"], future_rejections=future_rejections)

        if _expired(deadline):
            return _fallback(base_action, "cpu_deadline", future_rejections=future_rejections)

        return selected["first"], {
            "module": "official-pro-short-horizon-v1",
            "used": True,
            "fallback": False,
            "reason": "worst_scenario_utility",
            "scenarios": [{"name": name, "scale": factor} for name, factor in SCENARIOS],
            "scenario_role": "frozen pressure cases, not weather probabilities",
            "candidate_count": len(actions),
            "sequence_count": len(sequence_options),
            "selected_steps": selected["steps"],
            "scenario_utility": {name: round(value, 6) for name, value in selected["values"].items()},
            "worst_scenario_utility": round(min(selected["values"].values()), 6),
            "base_immediate_utility": round(base_immediate, 6),
            "selected_immediate_utility": round(selected_immediate, 6),
            "immediate_floor": IMMEDIATE_UTILITY_FLOOR,
            "future_rejections": future_rejections,
        }
    except _CPUDeadline:
        return _fallback(base_action, "cpu_deadline")
    except Exception as exc:
        return _fallback(base_action, "evaluation_error", error_type=type(exc).__name__)
