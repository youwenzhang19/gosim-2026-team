"""Effective exposure quality, without diagnosing weather or instrument faults.

Feedback scores alone may not distinguish a matched program from a mismatch.
Keep that ambiguity as an interval; a saturated hit contributes only a lower
bound. Bounds are evidence constraints, not statistical confidence intervals.
One record represents one exposure, independent of its fibre count/order.
"""
from __future__ import annotations

import math
from collections import deque
from statistics import median


def score_interval(score, weight, flux, duration, model, declared, mismatch):
    """Return (lower, upper-or-None) consistent with either legal multiplier.

No assumption about instrument efficiency enters program selection. The small
score tolerance accounts for rounded public feedback, not physical saturation.
"""
    values = (score, weight, flux, duration, model, declared, mismatch)
    if not all(math.isfinite(float(v)) for v in values):
        return None
    if score < 0 or min(weight, flux, duration, model, declared, mismatch) <= 0:
        return None
    seen = score / weight
    tolerance = 0.000001 / weight
    # Units here exclude the public flux*time zero point, applied by the caller.
    conversion = 1.0 / (flux * duration * model)
    intervals = []
    for multiplier in sorted({declared, mismatch}):
        if seen > multiplier + tolerance:
            continue  # this multiplier would require factor > 1
        lower = max(0.0, seen - tolerance) / multiplier * conversion
        upper = None if seen >= multiplier - tolerance else (seen + tolerance) / multiplier * conversion
        intervals.append((lower, upper))
    if not intervals:
        return None
    lower = min(lo for lo, _ in intervals)
    upper = None if any(hi is None for _, hi in intervals) else max(hi for _, hi in intervals)
    return lower, upper


class ScaleEstimator:
    def __init__(self, memory_hours=2.0):
        self.memory_hours = float(memory_hours)
        self.samples = deque()  # time-pruned exposure summaries; no fibre-count cap
        self.priors = deque(maxlen=64)  # finite estimates only, one vote per exposure
        self.reset()

    def reset(self):
        self.samples.clear()
        self.priors.clear()
        self.point = 1.0
        self.prior = 1.0
        self.lower = 0.0
        self.upper = None
        self.status = 'unmeasured'
        self.last_finite_hour = None
        self.last_result_hour = None
        self.last_duration = 300
        self.needs_probe = False
        self.rejected_directional = 0
        self.support = 0

    def advance(self, hours):
        while self.samples and self.samples[0]['hour'] < hours - self.memory_hours:
            self.samples.popleft()
        if self.last_result_hour is not None and self.last_result_hour < hours - self.memory_hours:
            # Keep the last measurement as a planning fallback, but do not
            # silently relabel a long-run median or weak bound as current truth.
            self.status = 'stale'
            self.lower, self.upper = 0.0, None
            self.needs_probe = True

    def observe(self, hours, intervals, duration, rejected_directional=0):
        self.advance(hours)
        self.rejected_directional = rejected_directional
        self.support = len(intervals)
        self.last_duration = duration
        if not intervals:
            self.status = 'no_clean_feedback'
            self.needs_probe = True
            return
        self.last_result_hour = hours
        finite = [(lo, hi) for lo, hi in intervals if hi is not None]
        bounds = [lo for lo, hi in intervals if hi is None]
        if finite:
            # A contemporaneous exposure replaces older weather evidence.
            # Intersect when all targets admit a common scale. If direction
            # effects/model approximation make them inconsistent, retain a
            # robust within-exposure envelope and advertise the disagreement.
            lo = max(x[0] for x in intervals)
            hi = min(x[1] for x in finite)
            coherent = lo <= hi
            if not coherent:
                lo = median(x[0] for x in finite)
                hi = median(x[1] for x in finite)
            if lo <= self.point <= hi:
                point = self.point
            else:
                point = math.sqrt(lo * hi) if lo > 0 else (lo + hi) / 2
            self.point = point  # no physical 0.05/0.01 floor
            self.lower, self.upper = lo, hi
            self.status = 'bounded' if coherent else 'heterogeneous'
            self.last_finite_hour = hours
            self.needs_probe = not coherent
            self.priors.append(point)
            self.prior = median(self.priors)
        else:
            # A lower bound may raise an estimate but must never drag it down.
            lo = max(bounds)
            contradicted = self.upper is not None and lo > self.upper
            self.point = max(self.point, lo)
            self.lower, self.upper = lo, None
            self.status = 'lower_bound_only'
            self.needs_probe = (contradicted or self.last_finite_hour is None
                                or hours - self.last_finite_hour >= self.memory_hours / 2)
        self.samples.append({'hour': hours, 'lower': self.lower, 'upper': self.upper,
                             'estimate': self.point, 'status': self.status,
                             'finite_count': len(finite), 'censored_count': len(bounds),
                             'rejected_directional': rejected_directional})

    def diagnostics(self):
        return {'estimate': self.point, 'lower': self.lower, 'upper': self.upper,
                'status': self.status, 'last_finite_hour': self.last_finite_hour,
                'last_result_hour': self.last_result_hour, 'support': self.support,
                'rejected_directional': self.rejected_directional,
                'needs_probe': self.needs_probe, 'exposures_retained': len(self.samples)}
