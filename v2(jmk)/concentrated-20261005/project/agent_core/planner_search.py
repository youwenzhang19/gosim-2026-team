"""Bounded joint science/program/exposure search with deadline-aware requests.

Only public geometry, public messages and the agent's own feedback are used.
Completion predictions never update real progress; only a real result can do so.
"""
from __future__ import annotations

import heapq
import math
from datetime import timedelta

from .geometry import (Moon, SIDEREAL_DEG_PER_SECOND, altaz_to_radec,
                       local_sidereal_deg, lunar_factor, max_hour_angle_deg,
                       parse_utc, radec_to_altaz, shift_altaz, tangent_offsets, wrap180)
from .state import PendingPrediction


class JointSearch:
    def _request_views(self, requests, now):
        state = self.state
        views = []
        for request in requests:
            try:
                issued = parse_utc(request["issued_at_utc"])
                deadline = parse_utc(request["deadline_utc"])
                minimum = int(request["minimum_completed"])
                threshold = float(request["completion_factor_threshold"])
                reward = max(0.0, float(request["completion_reward"]))
            except (KeyError, ValueError, TypeError):
                continue
            completed = set(map(str, request.get("completed_target_ids") or []))
            if now < issued or now >= deadline or len(completed) >= minimum or int(request.get("remaining_count", 1)) <= 0:
                continue
            needed = {state.index_of[str(t)] for t in request.get("target_ids", [])
                      if str(t) in state.index_of and str(t) not in completed}
            remaining = max(0, minimum - len(completed))
            if not 0 < threshold <= 1 or remaining > len(needed) or not remaining:
                continue
            views.append({"id": str(request.get("request_id", len(views))), "issued": issued,
                          "deadline": deadline, "needed": needed, "remaining": remaining,
                          "threshold": threshold, "reward": reward})
        return views

    def _request_thresholds(self, active_requests):
        thresholds = {}
        for view in self._request_views_now:
            for i in view["needed"]:
                thresholds[i] = min(thresholds.get(i, 1.0), view["threshold"])
        return thresholds

    def _request_bonuses(self, active_requests, now):
        bonuses = {}
        for view in self._request_views_now:
            seconds_left = max(1, (view["deadline"] - now).total_seconds())
            value = view["reward"] / view["remaining"] * (1 + min(2.0, 21600 / seconds_left))
            for i in view["needed"]:
                bonuses[i] = max(bonuses.get(i, 0), value)
        return bonuses  # candidate recall only; never added to final action utility

    def _uniformity_penalty(self, observed):
        totals = self._uniformity_totals
        if not totals or self.state.scoring.uniformity_weight <= 0 or self.state.scoring.uniformity_band_width_deg <= 0:
            return 0.0
        ratios = [observed.get(b, 0) / n for b, n in totals.items()]
        squares = sum(r * r for r in ratios)
        jain = sum(ratios)**2 / (len(ratios) * squares) if squares else 0.0
        return self.state.scoring.uniformity_weight * (1 - jain)

    def _prepare_uniformity(self):
        state = self.state
        if self._uniformity_version == state.progress_version:
            return
        self._uniformity_version = state.progress_version
        self._uniformity_totals = state.band_totals
        self._uniformity_observed = state.band_observed
        self._uniformity_penalty_now = self._uniformity_penalty(state.band_observed)
        self._uniformity_gain_cache.clear()

    def _uniformity_gain(self, i):
        state = self.state
        if state.factor[i] >= state.scoring.uniformity_threshold:
            return 0.0
        band = state.ra_band[i]
        if band not in self._uniformity_gain_cache:
            observed = dict(self._uniformity_observed)
            observed[band] = min(self._uniformity_totals.get(band, 0), observed.get(band, 0) + 1)
            self._uniformity_gain_cache[band] = max(0.0, self._uniformity_penalty_now - self._uniformity_penalty(observed))
        return self._uniformity_gain_cache[band]

    def _uniformity_group_gain(self, items, duration):
        state = self.state
        observed = dict(self._uniformity_observed)
        for item in items.values():
            i = item["i"]
            if state.factor[i] < state.scoring.uniformity_threshold <= item["low_k"] * duration:
                b = state.ra_band[i]
                observed[b] = min(state.band_totals[b], observed.get(b, 0) + 1)
        return max(0.0, self._uniformity_penalty_now - self._uniformity_penalty(observed))

    def _request_hits(self, items, duration, now, view):
        if now < view["issued"] or now + timedelta(seconds=duration) > view["deadline"]:
            return set()
        return {x["i"] for x in items.values() if x["i"] in view["needed"] and
                x["up"] >= duration and min(1, x["low_k"] * duration) >= view["threshold"]}

    def _request_reward_for_plan(self, info, duration, now):
        reward = 0.0
        for view in self._request_views_now:
            if len(self._request_hits(info, duration, now, view)) >= view["remaining"]:
                reward += view["reward"]
        return reward, reward > 0

    def _value(self, i):
        state = self.state
        value = max(0, state.weight[i] * self._top_multiplier - state.best_score[i])
        if state.required[i] and state.factor[i] < state.scoring.required_threshold:
            value += max(0, state.scoring.required_penalty) * getattr(state, "duty_required_value_scale", 1.0)
        request_scale = {"request": 1.1, "science": 0.85}.get(state.advice_priority, 1.0)
        request_scale *= getattr(state, "duty_request_value_scale", 1.0)
        value += self._request_bonus_now.get(i, 0) * request_scale
        if i in getattr(state, "recover_targets", ()):
            value *= getattr(state, "duty_recover_value_scale", 1.0)
        value += self._uniformity_gain(i)
        return value * max(0.1, 0.65 ** min(5, state.misses[i]))

    def _item(self, i, altaz, lst, moon, seconds_left, night_index):
        state = self.state
        alt, az = altaz(i)
        model = state.scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], state.scoring.lunar_model))
        bucket = int((az + 22.5) % 360 / 45)
        scales = self._quality_cache.get(bucket)
        if scales is None:
            scales = state.quality_scales(az)
            if state.advice_risk == "conservative":
                scales = (scales[0] * 0.9, scales[1], scales[2])
            self._quality_cache[bucket] = scales
        risk = self._risk_cache.get(bucket)
        if risk is None:
            risk = state.invalidation_probability(az)
            self._risk_cache[bucket] = risk
        ha = wrap180(lst - state.ra[i])
        up = (state.hmax[i] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[i] < 180 else 1e9
        direction = self._direction_factor(alt, az)
        k = state.flux[i] * model / state.scoring.f0t0
        return {"i": i, "alt": alt, "az": az, "model": model,
                "up": min(up, seconds_left), "ks": tuple(k * s for s in scales),
                "band_qs": tuple(model * s / 0.95 for s in scales),
                "low_k": k * scales[0] * min(1.0, direction) * 0.98,
                "valid": 1 - risk, "direction": direction,
                "required_value": self.required_calendar.value(i, night_index)}

    def _science_gain(self, item, duration, program):
        state = self.state
        key = (duration, program)
        cache = item.setdefault("science_cache", {})
        if key in cache:
            return cache[key]
        i = item["i"]
        # Expectation of the positive improvement, not positive improvement at
        # the mean quality; this matters near saturation / best-score boundaries.
        gain = 0.0
        for probability, k, q in zip((0.2, 0.6, 0.2), item["ks"], item["band_qs"]):
            score = state.weight[i] * min(1, k * duration * item["direction"]) * state.scoring.program_multiplier(program, state.scoring.program_band(q))
            gain += probability * max(0, score - state.best_score[i])
        cache[key] = gain * item["valid"]
        return cache[key]

    def _target_gain(self, item, duration, program):
        gain = self._science_gain(item, duration, program)
        if item["low_k"] * duration >= self.state.scoring.required_threshold:
            gain += item["required_value"] * item["valid"]
        return gain

    def _duration_candidates(self, groups, cap, now):
        state = self.state
        if state.duty_max_exposure_cap is not None:
            cap = min(cap, state.duty_max_exposure_cap)
        cap = int(min(state.max_exposure, cap))
        if cap < state.min_exposure:
            return []
        durations = {state.min_exposure, cap}
        bases = (300, 600, 900, 1500, 2400, 3600)
        if state.duty_max_exposure_cap is not None:
            # CONSERVE / P-Q: keep shorter candidates; drop optimistic 3600 DARK.
            bases = tuple(d for d in (120, 180, 300, 450, 600, 900, 1200, 1500, 1800) if d <= cap)
        durations.update(d for d in bases if state.min_exposure <= d <= cap)
        boundaries = []
        for items in groups.values():
            for item in items:
                i, k = item["i"], item["low_k"]
                if k <= 0:
                    continue
                thresholds = []
                if item["required_value"] > 0:
                    thresholds.append(self.state.scoring.required_threshold)
                if i in self._request_thresholds_now:
                    thresholds.append(self._request_thresholds_now[i])
                for threshold in thresholds:
                    d = math.ceil(threshold / k) + 1
                    if state.min_exposure <= d <= min(cap, item["up"]):
                        boundaries.append((item["required_value"] + self._request_bonus_now.get(i, 0), d))
                # Science saturation at the expected sky level.
                d = math.ceil(1 / max(1e-9, item["ks"][1] * item["direction"]))
                if state.min_exposure <= d <= min(cap, item["up"]):
                    boundaries.append((state.weight[i], d))
        durations.update(d for _, d in heapq.nlargest(8, boundaries))
        for view in self._request_views_now:
            d = int((view["deadline"] - now).total_seconds())
            if state.min_exposure <= d <= cap:
                durations.add(d)
        return sorted(durations)

    def _joint_pointing(self, pointing, now, lst, seconds_left, night_index):
        state = self.state
        _, c_alt, c_az, groups = pointing
        c_ra, c_dec = altaz_to_radec(c_alt, c_az, lst, state.lat)
        h = max_hour_angle_deg(c_dec, state.lat, state.min_alt + 0.3)
        ha = wrap180(lst - c_ra)
        up = (h - ha) / SIDEREAL_DEG_PER_SECOND if h < 180 else 1e9
        plans = []
        programs = (state.force_program,) if state.force_program else tuple(state.duty_prefer_programs or ("DARK", "BRIGHT", "BACKUP"))
        for duration in self._duration_candidates(groups, min(seconds_left, up), now):
            for program in programs:
                chosen = {}
                for fiber, options in groups.items():
                    eligible = [item for item in options if item["up"] >= duration and item["direction"] > 0]
                    if not eligible:
                        continue
                    # Request partial utility is only a tie/recall aid here.
                    # A completed-request alternative is evaluated below exactly.
                    item = max(eligible, key=lambda x: self._target_gain(x, duration, program) +
                               (self._request_bonus_now.get(x["i"], 0) * 0.05 if x["low_k"] * duration >= self._request_thresholds_now.get(x["i"], 2) else 0))
                    chosen[fiber] = item
                if not chosen:
                    continue
                variants = [chosen]
                # For each request, prefer its completable targets in occupied
                # fibers and compare the actual lost science with one reward.
                for view in self._request_views_now:
                    if now + timedelta(seconds=duration) > view["deadline"]:
                        continue
                    request_choice = dict(chosen)
                    replacements = []
                    for fiber, options in groups.items():
                        eligible = [x for x in options if x["i"] in view["needed"] and x["up"] >= duration and
                                    x["direction"] > 0 and x["low_k"] * duration >= view["threshold"]]
                        if eligible:
                            item = max(eligible, key=lambda x: self._target_gain(x, duration, program))
                            old_gain = self._target_gain(chosen[fiber], duration, program) if fiber in chosen else 0
                            loss = old_gain - self._target_gain(item, duration, program)
                            replacements.append((loss, fiber, item))
                    replacements.sort(key=lambda x: x[0])
                    # Keep this partial layout for bounded two-step planning even
                    # when one field cannot supply the entire remaining count.
                    for _, fiber, item in replacements[:view["remaining"]]:
                        request_choice[fiber] = item
                    if replacements:
                        variants.append(request_choice)
                for selected in variants:
                    science = sum(self._science_gain(item, duration, program) for item in selected.values())
                    utility = sum(self._target_gain(item, duration, program) for item in selected.values())
                    reward, completed = self._request_reward_for_plan(selected, duration, now)
                    # Joint validity is bounded conservatively by the weakest
                    # direction; no multiplied independent per-fiber fiction.
                    valid = min(x["valid"] for x in selected.values())
                    uniformity = self._uniformity_group_gain(selected, duration) * valid
                    utility += reward * valid + uniformity
                    if utility <= 0:
                        continue
                    plans.append({"utility": utility, "science": science, "duration": duration,
                                  "program": program, "items": selected, "pointing": (c_alt, c_az),
                                  "rate": utility / duration, "reward": reward,
                                  "valid": valid, "uniformity": uniformity,
                                  "start": now, "completed": completed})
        return heapq.nlargest(6, plans, key=lambda x: x["rate"])

    def _two_step(self, plans, best, now, night_end, night_index, hours):
        if not self._request_views_now or self.state.fast_level >= 2 or self.clock.compute_left() < 5:
            return best
        state = self.state
        baseline_rate = max((x["science"] / x["duration"] for x in plans), default=0)
        contenders = []
        for view in self._request_views_now:
            partial = [p for p in plans if 0 < len(self._request_hits(p["items"], p["duration"], now, view)) < view["remaining"]]
            contenders.extend(heapq.nlargest(4, partial, key=lambda p: len(self._request_hits(p["items"], p["duration"], now, view)) / p["duration"]))
        # At most four first steps, each with four fresh second-step field plans.
        for first in sorted(contenders, key=lambda p: -p["rate"])[:4]:
            later = now + timedelta(seconds=first["duration"])
            horizon = min(night_end, min(v["deadline"] for v in self._request_views_now))
            if (horizon - later).total_seconds() < state.min_exposure:
                continue
            second_plans = self._search(later, horizon, night_index, hours + first["duration"] / 3600, lookahead=True)
            for second in heapq.nlargest(4, second_plans, key=lambda p: p["rate"]):
                reward = 0
                for view in self._request_views_now:
                    one = self._request_hits(first["items"], first["duration"], now, view)
                    two = self._request_hits(second["items"], second["duration"], later, view)
                    if len(one | two) >= view["remaining"]:
                        reward += view["reward"]
                if not reward:
                    continue
                # Conservatively exclude overlapping second-step gain. This is
                # a bounded lookahead estimate, not an exact virtual ledger.
                overlap = {x["i"] for x in first["items"].values()} & {x["i"] for x in second["items"].values()}
                duplicate = sum(self._target_gain(x, second["duration"], second["program"])
                                for x in second["items"].values() if x["i"] in overlap)
                combined = (first["utility"] + second["utility"] - first["reward"] * first["valid"] -
                            second["reward"] * second["valid"] - duplicate - second["uniformity"])
                combined += reward * max(0, first["valid"] + second["valid"] - 1)
                duration = first["duration"] + second["duration"]
                # R2: explicit opportunity cost, R3: replan after first real result.
                net = combined - baseline_rate * duration
                if net > 0 and combined / duration > best["rate"]:
                    best = dict(first, rate=combined / duration, two_step=True)
        return best

    def _search(self, now, night_end, night_index, hours, lookahead=False):
        state = self.state
        lst = local_sidereal_deg(now, state.lon)
        seconds_left = (min(night_end, state.survey_end) - now).total_seconds()
        if seconds_left < state.min_exposure:
            return []
        min_visible = state.min_exposure * SIDEREAL_DEG_PER_SECOND
        candidates, still_active = [], []
        for i in state.active:
            # cheap upper-bound value; celestial/quality work only for shortlist
            value = self._value(i)
            if value <= 1e-8:
                continue
            still_active.append(i)
            ha, h = wrap180(lst - state.ra[i]), state.hmax[i]
            if h <= 0 or not -h <= ha <= h - min_visible:
                continue
            if lookahead and i not in self._request_thresholds_now:
                continue
            nights_left = max(1, state.last_night[i] - night_index + 1)
            setting = 1 + 0.5 * max(0, ha / h) if h < 180 else 1
            candidates.append((value * (1 + 2 / nights_left) * setting, i))
        if not lookahead:
            state.active = still_active
        if not candidates:
            return []
        visible = {i for _, i in candidates}
        pool = heapq.nlargest(160 if state.fast_level >= 2 else 240, candidates)
        moon = Moon(now + timedelta(seconds=min(450, seconds_left / 2)), lst, state.lat)
        altaz_cache, item_cache, reach_cache = {}, {}, {}
        def altaz(i):
            if i not in altaz_cache:
                altaz_cache[i] = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
            return altaz_cache[i]
        def item(i):
            if i not in item_cache:
                item_cache[i] = self._item(i, altaz, lst, moon, seconds_left, night_index)
            return item_cache[i]
        def achievable(i):
            if i not in reach_cache:
                x = item(i)
                d = min(state.max_exposure, x["up"])
                value = max(self._target_gain(x, d, p) for p in ("DARK", "BRIGHT", "BACKUP"))
                if x["low_k"] * d >= self._request_thresholds_now.get(i, 2):
                    value += self._request_bonus_now.get(i, 0)
                value += self._uniformity_gain(i) if x["low_k"] * d >= state.scoring.uniformity_threshold else 0
                reach_cache[i] = value * min(1, x["direction"]) * max(0.1, 0.65 ** min(5, state.misses[i]))
            return reach_cache[i]
        anchors = heapq.nlargest(8, ((achievable(i) * priority / max(1e-9, self._value(i)), i) for priority, i in pool))
        n_anchors = 2 if lookahead else (2 if state.fast_level >= 1 else 4)
        if state.fast_level == 0 and self.grid.n <= 25:
            fibers = tuple(range(self.grid.n))
        else:
            middle = self.grid.representative_fibers()
            corners = (0, self.grid.side - 1, self.grid.n - self.grid.side, self.grid.n - 1)
            fibers = middle if state.fast_level >= 2 else tuple(dict.fromkeys(middle + corners))
        # Radius covers anchor-to-any-fiber distance across the complete field.
        radius = min(25.0, math.degrees(math.atan(math.radians(self.grid.fov * math.sqrt(2)))))
        pointings = []
        for _, anchor in anchors[:n_anchors]:
            a_alt, a_az = altaz(anchor)
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], radius) if j in visible]
            near_values = {j: achievable(j) for j in near}
            for fiber in fibers:
                dn, de = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -dn, -de)
                if not state.min_alt + 1 <= c_alt <= 89.0:
                    continue
                c_alt, c_az = round(c_alt, 4), round(c_az, 4) % 360
                groups = {}
                for j, v in near_values.items():
                    if v <= 0:
                        continue
                    alt, az = altaz(j)
                    offsets = tangent_offsets(alt, az, c_alt, c_az)
                    if offsets is None:
                        continue
                    fib, margin = self.grid.classify(*offsets)
                    if fib is None:
                        continue
                    edge = min(0.08, self.grid.glass * 0.12) * (1 + min(2, state.misses[j]))
                    score = v * (1 if margin >= edge else 0.4)
                    groups.setdefault(fib, []).append((score, j))
                if not groups:
                    continue
                selected = {fib: [item(j) for _, j in heapq.nlargest(3, values)] for fib, values in groups.items()}
                heuristic = sum(max(v for v, _ in values) for values in groups.values())
                pointings.append((heuristic, c_alt, c_az, selected))
        plans = []
        for pointing in heapq.nlargest(2 if state.fast_level >= 2 else 3, pointings, key=lambda x: x[0]):
            plans.extend(self._joint_pointing(pointing, now, lst, seconds_left, night_index))
        return plans

    def plan(self, now, night_end, night_index, hours):
        state = self.state
        state.update_scale(hours)
        self._prepare_uniformity()
        self._quality_cache, self._risk_cache = {}, {}
        self.required_calendar.builds_left = 2 if state.fast_level >= 2 else 8
        plans = self._search(now, night_end, night_index, hours)
        if not plans:
            return None
        best = max(plans, key=lambda x: x["rate"])
        # PROTECT / P-RQ: when a plan completes an open request this step, prefer it
        # even if the pure rate is slightly lower (window discipline > science rate).
        if state.duty_mode == "PROTECT" and self._request_views_now:
            completers = [p for p in plans if p.get("completed")]
            if completers:
                best = max(completers, key=lambda x: (x["reward"], x["rate"]))
            else:
                # Prefer any plan that hits at least one request target.
                partial = []
                for p in plans:
                    hits = 0
                    for view in self._request_views_now:
                        hits += len(self._request_hits(p["items"], p["duration"], now, view))
                    if hits:
                        partial.append((hits, p))
                if partial:
                    best = max(partial, key=lambda x: (x[0], x[1]["rate"]))[1]
        best = self._two_step(plans, best, now, night_end, night_index, hours)
        c_alt, c_az = best["pointing"]
        duration = best["duration"]
        state.pending.clear()
        state.pending_action_index = self._current_action_index
        state.pending_start, state.pending_end = now, now + timedelta(seconds=duration)
        for item in best["items"].values():
            state.pending[state.ids[item["i"]]] = PendingPrediction(
                model=item["model"], band_model=item["model"] / 0.95,
                alt=item["alt"], az=item["az"], clean=not state.all_sky_notice() and item["direction"] >= 1)
        state.pending_program, state.pending_duration, state.pending_night = best["program"], duration, night_index
        self.exposure_ema = 0.9 * self.exposure_ema + 0.1 * duration
        if best.get("two_step"):
            self.trace.write({"event": "request_two_step", "night": night_index, "executed_steps": 1})
        return {"action": "observe", "pointing": {"alt_deg": c_alt, "az_deg": c_az},
                "assignments": {str(f): state.ids[x["i"]] for f, x in best["items"].items()},
                "duration_seconds": duration, "program": best["program"],
                "decision_source": "joint-planner"}
