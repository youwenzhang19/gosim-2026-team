"""Lazy public REQUIRED geometry/lunar calendar; future weather remains unknown."""
from __future__ import annotations

from .geometry import Moon, local_sidereal_deg, lunar_factor, radec_to_altaz, wrap180


class RequiredCalendar:
    def __init__(self, state):
        self.state = state
        self.suffix = {}
        self.builds_left = 0
        self.moons = []
        self.spans = []
        for start, end in state.nights:
            mid = start + (end - start) / 2
            self.moons.append(Moon(mid, local_sidereal_deg(mid, state.lon), state.lat))
            self.spans.append((local_sidereal_deg(start, state.lon), (end - start).total_seconds() * 360.98564736629 / 86400))

    def _build(self, i):
        state = self.state
        opportunities = []
        for k, (lst, span) in enumerate(self.spans):
            ha = wrap180(lst - state.ra[i])
            opportunity = False
            for shift in (-360, 0, 360):
                h0 = ha + shift
                low, high = max(h0, -state.hmax[i]), min(h0 + span, state.hmax[i])
                if high <= low:
                    continue
                duration = (high - low) * 86400 / 360.98564736629
                best_ha = max(low, min(high, 0))
                alt, _ = radec_to_altaz(state.ra[i], state.dec[i], state.ra[i] + best_ha, state.lat)
                lunar = lunar_factor(self.moons[k], state.ra[i], state.dec[i], state.scoring.lunar_model)
                quality = state.scoring.quality_model(alt, lunar)
                achievable = state.flux[i] * quality * 0.65 * min(duration, state.max_exposure) / state.scoring.f0t0
                opportunity |= duration >= state.min_exposure and achievable >= state.scoring.required_threshold
            opportunities.append(int(opportunity))
        suffix = [0] * (len(opportunities) + 1)
        for k in range(len(opportunities) - 1, -1, -1):
            suffix[k] = suffix[k + 1] + opportunities[k]
        self.suffix[i] = suffix

    def value(self, i, night_index):
        state = self.state
        if not state.required[i] or state.factor[i] >= state.scoring.required_threshold:
            return 0.0
        if i not in self.suffix and self.builds_left > 0:
            self.builds_left -= 1
            self._build(i)
        future = (self.suffix[i][min(len(state.nights), night_index + 1)] if i in self.suffix else
                  max(0, state.last_night[i] - night_index))
        # Approximate deferral loss under finite capacity and unknown weather,
        # never a calibrated probability or a hidden-event forecast.
        miss_defer = 0.35 + 0.65 / (1 + 0.08 * future)
        value = max(0.0, state.scoring.required_penalty) * miss_defer
        if state.advice_priority == "required":
            value = min(state.scoring.required_penalty, value * 1.1)
        value *= getattr(state, "duty_required_value_scale", 1.0)
        return value
