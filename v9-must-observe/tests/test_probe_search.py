"""Planner regressions on the public L1 catalogue; no hidden truth inputs."""
import copy
import os
from pathlib import Path
import statistics
import sys
import unittest
from unittest.mock import patch
from datetime import timedelta

PROJECT = Path(__file__).resolve().parents[1]
KIT = Path(os.environ.get('OBSERVER_KIT_ROOT',
    'C:/Users/HP/Documents/Harry/gosim-2026-team-main/training/official-examples/gosim-observer-examples'))
sys.path[:0] = [str(PROJECT), str(KIT / 'runner')]
from planner import Planner, CALIBRATION_POOL, CALIBRATION_ANCHORS
from skymath import local_sidereal_deg, radec_to_altaz, altaz_to_radec, shift_altaz
from challenge.v4_workflow import V4Workflow


class ProbeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init = V4Workflow(KIT / 'local-cards/L1').initialize_payload(900)

    def fresh(self, level=2):
        p = Planner(copy.deepcopy(self.init))
        p.fast_level = level
        p.log = lambda *_: None
        return p

    def plan(self, p, hour=0):
        start, end = p.nights[0]
        return p.plan(start + timedelta(hours=hour), end, 0, hour)

    def verify(self, p, action, hour=0):
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'observe')
        self.assertGreaterEqual(action['duration_seconds'], p.min_exposure)
        self.assertLessEqual(action['duration_seconds'], p.max_exposure)
        self.assertEqual(set(action['assignments'].values()), set(p.pending))
        self.assertEqual(len(set(action['assignments'].values())), len(action['assignments']))
        start, end = p.nights[0]
        self.assertIsNotNone(p.evaluate_action(action, start + timedelta(hours=hour), end, 0, hour,
                                               scale_override=max(.1, p.scale)))
        diag = p.last_calibration_search
        self.assertLessEqual(diag['new_exact'], CALIBRATION_POOL[p.fast_level])
        self.assertLessEqual(diag['anchors'], CALIBRATION_ANCHORS[p.fast_level])
        self.assertLessEqual(diag['reserved_fibers'], 2)

    def split_geometry(self, p):
        start, _ = p.nights[0]
        lst = local_sidereal_deg(start, p.lon)
        coords = [radec_to_altaz(ra, dec, lst, p.lat) for ra, dec in zip(p.ra, p.dec)]
        split = statistics.median(z for a, z in coords if a >= p.min_alt + 2)
        return coords, split

    def make_dirty_gain(self, p):
        coords, split = self.split_geometry(p)
        p._direction_factor = lambda alt, az: .5 if az < split else 1.
        p.scale_estimator.point = .1
        p.required = [False] * len(p.ids)
        p.factor = [.01 if az < split else .9 for _, az in coords]
        p.cur = [factor * 1.2 for factor in p.factor]
        return coords, split

    def test_wait_results_do_not_reset_reprobe_clock(self):
        # REPROBE_WAIT_HOURS default is 0.35; wait below that must not force a wait-probe.
        for hour, expected in ((.25, False), (.35, True)):
            p = self.fresh()
            p.scale_estimator.point = .8
            p._probe_night = 0
            p._last_observe_hour = 0.
            p._last_scale_probe_hour = 0.
            logs = []
            p.log = logs.append
            start, _ = p.nights[0]
            for step in range(1, int(hour * 4) + 1):
                p.on_result({'action': 'wait'}, start + timedelta(hours=step / 4), step / 4)
            action = self.plan(p, hour)
            self.verify(p, action, hour)
            self.assertEqual(any('probe: wait ' in s for s in logs), expected)
            self.assertEqual(p._last_observe_hour, 0.)
            if expected:
                self.assertEqual(action['duration_seconds'], 300)

    def test_zero_science_gain_probe_still_executes(self):
        p = self.fresh()
        p.scale_estimator.point = 0.
        p.cur = [.5] * len(p.ids)
        p.factor = [.4] * len(p.ids)
        self.verify(p, self.plan(p))

    def test_positive_dirty_gain_does_not_hide_clean_calibrators(self):
        p = self.fresh()
        self.make_dirty_gain(p)
        action = self.plan(p)
        self.verify(p, action)
        self.assertTrue(p.last_calibration_search['triggered'])
        self.assertGreater(p.last_calibration_search['usable'], 0)
        self.assertTrue(any(row['dir_clean'] for row in p.pending.values()))

    def test_completed_targets_are_independent_calibrators(self):
        p = self.fresh()
        p.factor = [1.] * len(p.ids)
        p.cur = [1.2] * len(p.ids)
        p.active = []
        p.scale_estimator.needs_probe = True
        action = self.plan(p)
        self.verify(p, action)
        self.assertGreater(p.last_calibration_search['usable'], 0)
        self.assertLessEqual(len(action['assignments']), 2)
        self.assertEqual(p.active, [])

    def test_no_clean_direction_has_bounded_fallback(self):
        p = self.fresh()
        p._direction_factor = lambda *_: .5
        action = self.plan(p)
        self.verify(p, action)
        self.assertEqual(p.last_calibration_search['usable'], 0)
        self.assertIsNone(p._calibration_guard)

    def test_blocked_directions_never_force_illegal_probe(self):
        p = self.fresh()
        p._direction_factor = lambda *_: 0.
        self.assertIsNone(self.plan(p))
        self.assertLessEqual(p.last_calibration_search['new_exact'], CALIBRATION_POOL[2])

    def test_empty_science_pool_still_tries_legal_clean_targets(self):
        p = self.fresh()
        coords, split = self.split_geometry(p)
        p.value = lambda i: 1000. if coords[i][1] < split else 1.
        p._direction_factor = lambda alt, az: 0. if az < split else 1.
        action = self.plan(p)
        self.verify(p, action)
        self.assertTrue(all(row['dir_clean'] for row in p.pending.values()))

    def test_search_budget_at_all_levels(self):
        for level in range(5):
            p = self.fresh(level)
            self.make_dirty_gain(p)
            self.verify(p, self.plan(p))
            self.assertGreater(p.last_calibration_search['usable'], 0)

    def test_normally_useful_probe_does_not_expand(self):
        p = self.fresh()
        self.verify(p, self.plan(p))
        self.assertFalse(p.last_calibration_search['triggered'])
        self.assertEqual(p.last_calibration_search['new_exact'], 0)

    def test_no_probe_keeps_calibration_search_off(self):
        p = self.fresh()
        p._probe_night = 0
        p._last_scale_probe_hour = 0.
        p.scale_estimator.needs_probe = False
        self.verify(p, self.plan(p))
        self.assertFalse(p.last_calibration_search['triggered'])

    def test_probe_commit_guard_preserves_pending_and_expires(self):
        p = self.fresh()
        self.make_dirty_gain(p)
        action = self.plan(p)
        self.verify(p, action)
        self.assertIsNotNone(p._calibration_guard)
        before = copy.deepcopy(p.pending)
        changed = copy.deepcopy(action)
        changed['duration_seconds'] += 60
        start, end = p.nights[0]
        self.assertFalse(p.record_action(changed, start, end, 0, 0.))
        self.assertEqual(p.pending, before)
        action['reason'] = 'added by agent after planning'
        self.assertTrue(p.record_action(action, start, end, 0, 0.))
        self.assertEqual(set(p.pending), set(before))
        p.scale_estimator.needs_probe = False
        self.plan(p, .1)
        self.assertIsNone(p._calibration_guard)

    def test_short_remaining_night_respects_minimum(self):
        p = self.fresh()
        start, _ = p.nights[0]
        self.assertIsNone(p.plan(start, start + timedelta(seconds=p.min_exposure - 1), 0, 0.))

    def saturation_fixture(self, faint=True):
        init = copy.deepcopy(self.init)
        original = self.fresh()
        start, _ = original.nights[0]
        lst = local_sidereal_deg(start, original.lon)
        columns = init['targets']['columns']
        rows = []
        for fiber in range(original.grid.n):
            dn, de = original.grid.fiber_center(fiber)
            alt, az = shift_altaz(60., 180., dn, de)
            ra, dec = altaz_to_radec(alt, az, lst, original.lat)
            for kind, flux, weight in [('science', original.f0t0 / 10, 10.)] + (
                    [('cal', original.f0t0 / 900, 1.)] if faint else []):
                row = list(init['targets']['rows'][0])
                for key, value in {'target_id': f'{kind}-{fiber}', 'ra_deg': ra,
                        'dec_deg': dec, 'feature_flux': flux, 'science_weight': weight,
                        'required': False}.items():
                    row[columns.index(key)] = value
                rows.append(row)
        init['targets']['rows'] = rows
        p = Planner(init)
        p.fast_level = 2
        p.log = lambda *_: None
        for i, tid in enumerate(p.ids):
            if tid.startswith('cal-'):
                p.factor[i], p.cur[i] = 1., 1.2
        return p

    def test_same_field_reserves_only_two_fibers_and_preserves_science(self):
        baseline = self.saturation_fixture()
        with patch('planner.CALIBRATION_MAX_REACH', 100.):
            original = self.plan(baseline)
        p = self.saturation_fixture()
        action = self.plan(p)
        self.verify(p, action)
        self.assertEqual(action['pointing'], original['pointing'])
        self.assertEqual(set(action['assignments']), set(original['assignments']))
        changed = [f for f in action['assignments']
                   if action['assignments'][f] != original['assignments'][f]]
        self.assertGreaterEqual(len(changed), 1)
        self.assertLessEqual(len(changed), 2)
        for fiber in changed:
            self.assertTrue(action['assignments'][fiber].startswith('cal-'))
        self.assertEqual(p.last_calibration_search['anchors'], 0)
        self.assertGreater(p.last_calibration_search['usable'], 0)

    def test_all_saturated_is_not_promised_finite(self):
        p = self.saturation_fixture(faint=False)
        action = self.plan(p)
        self.verify(p, action)
        self.assertEqual(p.last_calibration_search['usable'], 0)
        self.assertIsNone(p._calibration_guard)


if __name__ == '__main__':
    unittest.main(verbosity=2)
