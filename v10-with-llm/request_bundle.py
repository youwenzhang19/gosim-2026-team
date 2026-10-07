"""Bounded, current-night planning for complete public observation requests.

The returned planning values are search proxies. Request completion reward is
counted once, only after a complete request is feasible before its deadline.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from short_horizon import (
    _action,
    _candidate_actions,
    _future_template,
    _moment,
    _projected_overrides,
    _signature,
)
from planner import EDGE_MARGIN_DEG, NEIGHBOUR_RADIUS_DEG
from skymath import local_sidereal_deg, radec_to_altaz, shift_altaz, tangent_offsets


MODULE = "official-pro-request-bundle-v1"
MAX_TARGETS = 8
MAX_STEPS = 8
BEAM_WIDTH = 4
MAX_PRO_CANDIDATES = 4
MAX_ANCHOR_NEIGHBORS = 64
MAX_EVALUATIONS = 640


class _CPUDeadline(Exception):
    pass


class _EvaluationBudget(Exception):
    pass


@dataclass
class _Node:
    now: datetime
    hours: float
    steps: tuple
    cur: dict
    factor: dict
    assigned: frozenset
    qualified: frozenset
    planning_value: float


def _finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _expired(deadline):
    try:
        return time.process_time() >= float(deadline)
    except (TypeError, ValueError, OverflowError):
        return False


def _check_deadline(deadline):
    if _expired(deadline):
        raise _CPUDeadline


def _fallback(base_action, reason, **details):
    audit = {
        "module": MODULE,
        "used": False,
        "fallback": True,
        "reason": reason,
        "scope": "current_night",
        "planning_value_is_auxiliary": True,
        "terminal_reward_is_prospective": True,
        "full_pro_rollout_estimated": False,
        "early_stop": False,
    }
    audit.update(details)
    return base_action, audit


def _index(planner, target_id):
    try:
        return planner.index_of.get(target_id, planner.index_of.get(str(target_id)))
    except AttributeError:
        return None


def _action_indices(planner, action):
    assignments = action.get("assignments") or {}
    values = assignments.values() if isinstance(assignments, dict) else assignments
    result = set()
    for target_id in values:
        index = _index(planner, target_id)
        if index is not None:
            result.add(index)
    return result


def _without_assigned(planner, action, assigned):
    """Clone an action while removing targets already observed in this bundle."""
    if not isinstance(action, dict):
        return None
    assignments = action.get("assignments") or {}
    if not isinstance(assignments, dict):
        return None
    kept = {}
    for fiber, target_id in assignments.items():
        index = _index(planner, target_id)
        if index is None or index not in assigned:
            kept[str(fiber)] = target_id
    if not kept:
        return None
    result = dict(action)
    result["pointing"] = dict(action.get("pointing") or {})
    result["assignments"] = kept
    return result


def _request_rows(planner, base_action):
    rows = []
    candidate_indices = []
    seen_candidate_indices = set()
    for action in _candidate_actions(planner, base_action)[:MAX_PRO_CANDIDATES]:
        assignments = action.get("assignments") or {}
        values = assignments.values() if isinstance(assignments, dict) else assignments
        for target_id in values:
            index = _index(planner, target_id)
            if index is not None and index not in seen_candidate_indices:
                candidate_indices.append(index)
                seen_candidate_indices.add(index)
    candidate_order = {index: rank for rank, index in enumerate(candidate_indices)}

    for ordinal, raw in enumerate(getattr(planner, "requests", ())):
        if not isinstance(raw, dict):
            continue
        try:
            remaining = int(raw.get("remaining_count", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        deadline = _moment(raw.get("deadline"))
        threshold = _finite(raw.get("threshold"), math.nan)
        reward = _finite(raw.get("reward"), math.nan)
        target_indices = []
        for item in raw.get("target_indices", ()):
            if isinstance(item, int):
                index = item
            else:
                index = _index(planner, item)
            if (index is not None and 0 <= index < len(getattr(planner, "ids", ()))
                    and index not in target_indices):
                target_indices.append(index)
        if (remaining <= 0 or not target_indices or deadline is None
                or not math.isfinite(threshold) or not math.isfinite(reward) or reward <= 0.0):
            continue

        target_indices.sort(key=lambda index: (candidate_order.get(index, 10**9), index))
        target_indices = target_indices[:MAX_TARGETS]
        if remaining > min(MAX_TARGETS, len(target_indices)):
            continue
        rows.append({
            "ordinal": ordinal,
            "target_indices": tuple(target_indices),
            "target_set": frozenset(target_indices),
            "remaining_count": remaining,
            "threshold": threshold,
            "reward": max(0.0, reward),
            "deadline": deadline,
        })
    return rows


def _anchor_action(planner, base_action, target_index, at, blocked, deadline):
    """Build one executable field centred so the request target lands on a fiber."""
    try:
        target_id = planner.ids[target_index]
        lst = local_sidereal_deg(at, float(planner.lon))
        target_alt, target_az = radec_to_altaz(
            float(planner.ra[target_index]), float(planner.dec[target_index]), lst, float(planner.lat)
        )
        if target_alt < float(planner.min_alt):
            return None

        grid = planner.grid
        central = min(range(grid.n), key=lambda fiber: (
            sum(value * value for value in grid.fiber_center(fiber)), fiber
        ))
        north, east = grid.fiber_center(central)
        center_alt, center_az = shift_altaz(target_alt, target_az, -north, -east)
        offset_alt, offset_az = map(float, planner.offset)
        command_alt = center_alt - offset_alt
        command_az = (center_az - offset_az) % 360.0
        if not 0.0 <= command_alt <= 90.0:
            return None

        offsets = tangent_offsets(target_alt, target_az, center_alt, center_az)
        if offsets is None:
            return None
        target_fiber, _margin = grid.classify(*offsets)
        if target_fiber is None:
            return None

        assignments = {str(target_fiber): target_id}
        neighbors = getattr(planner, "neighbours", None)
        if callable(neighbors):
            nearby = neighbors(float(planner.ra[target_index]), float(planner.dec[target_index]),
                               max(NEIGHBOUR_RADIUS_DEG, float(grid.fov)))
            ranked = []
            for index in nearby[:MAX_ANCHOR_NEIGHBORS]:
                if index == target_index or index in blocked or not 0 <= index < len(planner.ids):
                    continue
                _check_deadline(deadline)
                alt, az = radec_to_altaz(float(planner.ra[index]), float(planner.dec[index]),
                                         lst, float(planner.lat))
                neighbor_offsets = tangent_offsets(alt, az, center_alt, center_az)
                if neighbor_offsets is None:
                    continue
                fiber, margin = grid.classify(*neighbor_offsets)
                misses = getattr(planner, "misses", ())
                minimum_margin = (EDGE_MARGIN_DEG * (0.5 + 1.5 * misses[index])
                                  if index < len(misses) else EDGE_MARGIN_DEG * 0.5)
                if fiber is None or margin < minimum_margin:
                    continue
                density = (_finite(planner.weight[index]) * _finite(planner.flux[index]))
                ranked.append((density, -index, fiber, index))
            for _density, _neg_index, fiber, index in sorted(ranked, reverse=True):
                if str(fiber) not in assignments:
                    assignments[str(fiber)] = planner.ids[index]

        action = {
            "action": "observe",
            "pointing": {"alt_deg": round(command_alt, 4), "az_deg": round(command_az, 4)},
            "assignments": assignments,
            "duration_seconds": int(base_action["duration_seconds"]),
            "program": base_action["program"],
        }
        return _without_assigned(planner, action, blocked)
    except _CPUDeadline:
        raise
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None


def _options(planner, base_action, pro_actions, request, node, origin, deadline):
    options = []
    seen = set()
    for template in pro_actions:
        _check_deadline(deadline)
        if node.now == origin:
            at_time = template
        else:
            at_time = _future_template(planner, template, origin, node.now)
        action = _without_assigned(planner, at_time, node.assigned) if at_time else None
        if action is None:
            continue
        signature = _signature(action)
        if signature not in seen:
            options.append(action)
            seen.add(signature)
        if len(options) >= MAX_PRO_CANDIDATES:
            break

    for target_index in request["target_indices"]:
        _check_deadline(deadline)
        if target_index in node.assigned:
            continue
        anchor = _anchor_action(planner, base_action, target_index, node.now,
                                node.assigned, deadline)
        if anchor is None:
            continue
        signature = _signature(anchor)
        if signature not in seen:
            options.append(anchor)
            seen.add(signature)
        if len(options) >= MAX_PRO_CANDIDATES + MAX_TARGETS:
            break
    return options


def _request_factor(prediction):
    if not isinstance(prediction, dict):
        return None
    value = prediction.get("request_factor", prediction.get("planning_factor"))
    if value is None:
        return None
    result = _finite(value, math.nan)
    return result if math.isfinite(result) else None


def _qualified_targets(planner, request, start, end, evaluation):
    if end > request["deadline"]:
        return set()
    result = set()
    for target_id, prediction in (evaluation.get("predictions") or {}).items():
        index = _index(planner, target_id)
        factor = _request_factor(prediction)
        if index in request["target_set"] and factor is not None and factor >= request["threshold"]:
            result.add(index)
    return result


def _terminal_reward(request, steps):
    qualified = set()
    for step in steps:
        if step["end"] > request["deadline"]:
            continue
        qualified.update(step["qualified"])
    return request["reward"] if len(qualified) >= request["remaining_count"] else 0.0


def _planning_value(evaluation):
    """Reconstruct Pro's value while keeping request settlement separate."""
    estimates = evaluation.get("estimated_scores") or {}
    if "planning_science_gain" in estimates:
        return (
            _finite(estimates.get("planning_science_gain"))
            + _finite(estimates.get("required_gain", evaluation.get("required_gain")))
            - _finite(estimates.get("time_cost"))
            + _finite(estimates.get("science_gain", evaluation.get("science_gain")))
            - _finite(evaluation.get("base_science_gain"))
        )
    if "planning_science_gain" in evaluation:
        return (
            _finite(evaluation.get("planning_science_gain"))
            + _finite(evaluation.get("required_gain"))
            - _finite(evaluation.get("time_cost"))
            + _finite(evaluation.get("science_gain"))
            - _finite(evaluation.get("base_science_gain"))
        )
    return _finite(evaluation.get("utility")) - _finite(evaluation.get("request_gain"))


def _baseline_choice_value(evaluation):
    """Use Pro's request-aware objective for choice, without accruing its proxy."""
    estimates = evaluation.get("estimated_scores") or {}
    if ("planning_request_gain" in evaluation
            or "planning_request_gain" in estimates):
        return _finite(evaluation.get("utility"), _planning_value(evaluation))
    return _planning_value(evaluation)


def _evaluate(planner, action, now, night_end, night_index, hours, cur, factor,
              deadline, budget):
    _check_deadline(deadline)
    if budget[0] >= MAX_EVALUATIONS:
        raise _EvaluationBudget
    budget[0] += 1
    evaluation = planner.evaluate_action(
        action, now, night_end, night_index, hours,
        cur_override=cur,
        factor_override=factor,
    )
    _check_deadline(deadline)
    return evaluation


def _expand(planner, action, request, node, night_end, night_index,
            deadline, budget):
    duration = int(_finite(action.get("duration_seconds")))
    if duration <= 0:
        return None
    end = node.now + timedelta(seconds=duration)
    if end > night_end:
        return None
    evaluation = _evaluate(planner, action, node.now, night_end, night_index, node.hours,
                           node.cur, node.factor, deadline, budget)
    if evaluation is None:
        return None
    _check_deadline(deadline)
    assigned = _action_indices(planner, action)
    qualified = _qualified_targets(planner, request, node.now, end, evaluation)
    cur, factor = _projected_overrides(planner, evaluation, node.cur, node.factor)
    step = {
        "start": node.now,
        "end": end,
        "action": action,
        "evaluation": evaluation,
        "assigned": frozenset(assigned),
        "qualified": frozenset(qualified),
        "planning_value_auxiliary": _planning_value(evaluation),
    }
    return _Node(
        now=end,
        hours=node.hours + duration / 3600.0,
        steps=node.steps + (step,),
        cur=cur,
        factor=factor,
        assigned=node.assigned | assigned,
        qualified=node.qualified | qualified,
        planning_value=node.planning_value + step["planning_value_auxiliary"],
    )


def _prune(nodes):
    nodes.sort(key=lambda node: (len(node.qualified), node.planning_value,
                                 -len(node.steps)), reverse=True)
    selected = []
    seen_coverage = set()
    for node in nodes:
        coverage = tuple(sorted(node.qualified))
        if coverage in seen_coverage:
            continue
        selected.append(node)
        seen_coverage.add(coverage)
        if len(selected) >= BEAM_WIDTH:
            return selected
    if len(selected) < BEAM_WIDTH:
        for node in nodes:
            if node not in selected:
                selected.append(node)
            if len(selected) >= BEAM_WIDTH:
                break
    return selected


def _baseline(planner, request, durations, now, night_end, night_index,
              hours, pro_actions, opportunity_rate, deadline, budget):
    current = now
    current_hours = hours
    cur = {}
    factor = {}
    steps = []
    greedy_value = 0.0
    for duration in durations:
        _check_deadline(deadline)
        best = None
        for template in pro_actions:
            if current == now:
                action = dict(template)
                action["pointing"] = dict(template.get("pointing") or {})
                action["assignments"] = dict(template.get("assignments") or {})
            else:
                action = _future_template(planner, template, now, current)
                if action is None:
                    continue
                action = dict(action)
            action["duration_seconds"] = duration
            end = current + timedelta(seconds=duration)
            if end > night_end:
                continue
            evaluation = _evaluate(planner, action, current, night_end, night_index,
                                   current_hours, cur, factor, deadline, budget)
            if evaluation is None:
                continue
            step_value = _planning_value(evaluation)
            choice_value = _baseline_choice_value(evaluation)
            if best is None or choice_value > best["choice_value"]:
                best = {"action": action, "evaluation": evaluation,
                        "planning_value_auxiliary": step_value,
                        "choice_value": choice_value, "end": end}
        if best is None:
            return None
        evaluation = best["evaluation"]
        end = best["end"]
        step_value = best["planning_value_auxiliary"]
        steps.append({
            "start": current,
            "end": end,
            "evaluation": evaluation,
            "qualified": frozenset(_qualified_targets(planner, request, current, end, evaluation)),
            "planning_value_auxiliary": step_value,
        })
        greedy_value += step_value
        cur, factor = _projected_overrides(planner, evaluation, cur, factor)
        current = end
        current_hours += duration / 3600.0
    terminal_reward = _terminal_reward(request, steps)
    total_duration = sum(durations)
    opportunity_floor = opportunity_rate * total_duration
    greedy_total = greedy_value + terminal_reward
    floor_total = opportunity_floor + terminal_reward
    return {
        "steps": tuple(steps),
        "planning_value_auxiliary": greedy_value,
        "opportunity_rate_auxiliary": opportunity_rate,
        "opportunity_floor_auxiliary": opportunity_floor,
        "terminal_reward": terminal_reward,
        "total_auxiliary": max(greedy_total, floor_total),
        "method": "greedy_frozen_pro_pool_plus_stationary_rate_floor",
    }


def _initial_opportunity_rate(planner, pro_actions, now, night_end, night_index,
                              hours, deadline, budget):
    cached = {}
    supplied = getattr(planner, "last_candidates", ())
    supplied = supplied(MAX_PRO_CANDIDATES) if callable(supplied) else supplied
    for row in supplied or ():
        action = _action(row)
        if (action is not None and isinstance(row, dict)
                and "planning_science_gain" in (row.get("estimated_scores") or {})
                and "base_science_gain" in row):
            cached[_signature(action)] = row

    best_rate = None
    for action in pro_actions:
        _check_deadline(deadline)
        duration = int(_finite(action.get("duration_seconds")))
        if duration <= 0:
            continue
        evaluation = cached.get(_signature(action))
        if evaluation is None:
            evaluation = _evaluate(planner, action, now, night_end, night_index,
                                   hours, {}, {}, deadline, budget)
        if evaluation is None:
            continue
        rate = _planning_value(evaluation) / duration
        if best_rate is None or rate > best_rate:
            best_rate = rate
    return best_rate


def _search_request(planner, base_action, pro_actions, request, origin, night_end,
                   night_index, hours, opportunity_rate, deadline, budget,
                   baseline_cache):
    initial = _Node(origin, hours, (), {}, {}, frozenset(), frozenset(), 0.0)
    frontier = [initial]
    complete = []
    rejected = 0

    for depth in range(MAX_STEPS):
        _check_deadline(deadline)
        expanded = []
        for node in frontier:
            _check_deadline(deadline)
            if node.now >= request["deadline"]:
                continue
            options = _options(planner, base_action, pro_actions, request, node, origin, deadline)
            for action in options:
                _check_deadline(deadline)
                candidate = _expand(planner, action, request, node, night_end,
                                    night_index, deadline, budget)
                if candidate is None:
                    rejected += 1
                    continue
                if len(candidate.qualified) >= request["remaining_count"]:
                    durations = tuple(int(step["action"]["duration_seconds"])
                                      for step in candidate.steps)
                    key = (request["ordinal"], durations)
                    if key not in baseline_cache:
                        baseline_cache[key] = _baseline(
                            planner, request, durations, origin, night_end,
                            night_index, hours, pro_actions, opportunity_rate,
                            deadline, budget,
                        )
                    baseline = baseline_cache[key]
                    if baseline is None:
                        rejected += 1
                        continue
                    request_reward = _terminal_reward(request, candidate.steps)
                    if request_reward <= 0.0:
                        rejected += 1
                        continue
                    baseline_reward = _terminal_reward(request, baseline["steps"])
                    selected_total = candidate.planning_value + request_reward
                    baseline_total = max(
                        baseline["planning_value_auxiliary"] + baseline_reward,
                        baseline["opportunity_floor_auxiliary"] + baseline_reward,
                    )
                    proof = {
                        "request": request,
                        "node": candidate,
                        "baseline": baseline,
                        "request_reward": request_reward,
                        "baseline_request_reward": baseline_reward,
                        "selected_total": selected_total,
                        "baseline_total": baseline_total,
                        "advantage": selected_total - baseline_total,
                    }
                    complete.append(proof)
                    if proof["advantage"] > 1e-9:
                        return [proof], rejected, True
                    continue

                needed = request["remaining_count"] - len(candidate.qualified)
                available = len(request["target_set"] - candidate.assigned)
                if (needed > available or len(candidate.steps) >= MAX_STEPS
                        or candidate.now >= request["deadline"]):
                    continue
                expanded.append(candidate)

        if not expanded:
            break
        frontier = _prune(expanded)
    return complete, rejected, False


def _success(proof, pro_actions, budget, rejected, *, early_stop):
    node = proof["node"]
    baseline = proof["baseline"]
    return node.steps[0]["action"], {
        "module": MODULE,
        "used": True,
        "fallback": False,
        "reason": "complete_request_beats_same_duration_pro_fallback",
        "scope": "current_night",
        "request_target_count": len(proof["request"]["target_set"]),
        "request_remaining_count": proof["request"]["remaining_count"],
        "qualified_target_count": len(node.qualified),
        "selected_steps": len(node.steps),
        "bundle_duration_seconds": sum(
            int(step["action"]["duration_seconds"]) for step in node.steps
        ),
        "first_step_planning_value_auxiliary": round(
            node.steps[0]["planning_value_auxiliary"], 6
        ),
        "terminal_request_reward": round(proof["request_reward"], 6),
        "baseline_terminal_request_reward": round(proof["baseline_request_reward"], 6),
        "planning_value_auxiliary": round(node.planning_value, 6),
        "same_duration_base_value_auxiliary": round(
            baseline["planning_value_auxiliary"], 6
        ),
        "stationary_opportunity_floor_auxiliary": round(
            baseline["opportunity_floor_auxiliary"], 6
        ),
        "stationary_opportunity_rate_auxiliary": round(
            baseline["opportunity_rate_auxiliary"], 9
        ),
        "bundle_value_auxiliary": round(proof["selected_total"], 6),
        "same_duration_base_total_auxiliary": round(proof["baseline_total"], 6),
        "net_advantage_auxiliary": round(proof["advantage"], 6),
        "baseline_method": baseline["method"],
        "full_pro_rollout_estimated": False,
        "early_stop": early_stop,
        "planning_value_is_auxiliary": True,
        "terminal_reward_is_prospective": True,
        "candidate_count": len(pro_actions),
        "evaluation_count": budget[0],
        "rejected_sequences": rejected,
    }


def select_bundle(planner, base_action, now, night_end, night_index, hours, deadline):
    """Return a first action only when a complete request bundle beats Pro fallback.

    ``deadline`` is an absolute ``time.process_time()`` value. The search is
    current-night only and stateless; the caller commits a changed action and
    invokes this function again on the next observation round.
    """
    if base_action is None or _action(base_action) is None:
        return _fallback(base_action, "no_base_action")
    if _expired(deadline):
        return _fallback(base_action, "cpu_deadline", evaluation_count=0)
    try:
        requests = _request_rows(planner, base_action)
    except Exception as exc:
        return _fallback(base_action, "request_contract_error", evaluation_count=0,
                         error_type=type(exc).__name__)
    if not requests:
        return _fallback(base_action, "no_requests", evaluation_count=0)

    base = _action(base_action)
    budget = [0]
    baseline_cache = {}
    rejected = 0
    complete = []
    try:
        pro_actions = _candidate_actions(planner, base)[:MAX_PRO_CANDIDATES]
        if not pro_actions:
            return _fallback(base_action, "empty_candidates", evaluation_count=budget[0])
        opportunity_rate = _initial_opportunity_rate(
            planner, pro_actions, now, night_end, night_index, hours, deadline, budget
        )
        if opportunity_rate is None:
            return _fallback(base_action, "no_initial_pro_rate",
                             candidate_count=len(pro_actions), evaluation_count=budget[0])
        for request in requests:
            _check_deadline(deadline)
            found, request_rejected, early_stop = _search_request(
                planner, base, pro_actions, request, now, night_end,
                night_index, hours, opportunity_rate, deadline, budget, baseline_cache,
            )
            complete.extend(found)
            rejected += request_rejected
            if early_stop:
                return _success(found[0], pro_actions, budget, rejected, early_stop=True)

        if not complete:
            return _fallback(base_action, "no_feasible_complete_request",
                             candidate_count=len(pro_actions), evaluation_count=budget[0],
                             rejected_sequences=rejected)

        selected = max(complete, key=lambda item: (
            item["advantage"], item["selected_total"], -len(item["node"].steps),
            -item["request"]["ordinal"],
        ))
        if selected["advantage"] <= 1e-9:
            return _fallback(base_action, "nonpositive_net_value",
                             candidate_count=len(pro_actions), evaluation_count=budget[0],
                             rejected_sequences=rejected,
                             best_net_advantage_auxiliary=round(selected["advantage"], 6),
                             best_bundle_value_auxiliary=round(selected["selected_total"], 6),
                             same_duration_base_value_auxiliary=round(selected["baseline_total"], 6),
                             best_bundle_duration_seconds=sum(
                                 int(step["action"]["duration_seconds"])
                                 for step in selected["node"].steps
                             ),
                             best_request_terminal_reward=round(
                                 selected["request_reward"], 6
                             ),
                             stationary_opportunity_rate_auxiliary=round(
                                 selected["baseline"]["opportunity_rate_auxiliary"], 9
                             ),
                             stationary_opportunity_floor_auxiliary=round(
                                 selected["baseline"]["opportunity_floor_auxiliary"], 6
                             ),
                             baseline_method=selected["baseline"]["method"],
                             full_pro_rollout_estimated=False)
        _check_deadline(deadline)
        return _success(selected, pro_actions, budget, rejected, early_stop=False)
    except _CPUDeadline:
        return _fallback(base_action, "cpu_deadline", evaluation_count=budget[0],
                         rejected_sequences=rejected)
    except _EvaluationBudget:
        return _fallback(base_action, "evaluation_budget", evaluation_count=budget[0],
                         rejected_sequences=rejected)
    except Exception as exc:
        return _fallback(base_action, "evaluation_error", evaluation_count=budget[0],
                         error_type=type(exc).__name__)
