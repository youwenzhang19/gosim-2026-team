"""Required-sprint, dynamic anchors, and plan determinism regressions."""
import copy
import os
import sys
import unittest
from datetime import timedelta
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
KIT = Path(os.environ.get(
    'OBSERVER_KIT_ROOT',
    '/Users/zhangyouwen/我的云端硬盘/WORKSPACE_PROJECT/巡天智能体黑客松/gosim-observer-examples',
))
sys.path[:0] = [str(PROJECT), str(KIT / 'runner')]

from planner import (
    Planner,
    REQUIRED_SPRINT_SCALE,
    _dynamic_n_anchors,
    _dynamic_calibration_anchors,
)
from challenge.v4_workflow import V4Workflow


class RequiredSprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init = V4Workflow(KIT / 'local-cards/L1').initialize_payload(900)

    def fresh(self, level=0):
        p = Planner(copy.deepcopy(self.init))
        p.fast_level = level
        p.log = lambda *_: None
        return p

    def plan(self, p, hour=0):
        start, end = p.nights[0]
        return p.plan(start + timedelta(hours=hour), end, 0, hour)

    def test_dynamic_anchors_scale_with_backlog(self):
        # a1~1500 / b1~2500: more unfinished REQUIRED → more full-speed anchors.
        a1 = _dynamic_n_anchors(1500, 0)
        b1 = _dynamic_n_anchors(2500, 0)
        self.assertGreaterEqual(a1, 8)
        self.assertGreater(b1, a1)
        self.assertLessEqual(b1, 28)
        self.assertEqual(_dynamic_n_anchors(0, 2), 1)
        self.assertGreaterEqual(_dynamic_calibration_anchors(2500, 0), 8)

    def test_required_sprint_active_on_high_scale(self):
        p = self.fresh()
        p.scale = REQUIRED_SPRINT_SCALE
        pending = p.pending_required_count()
        self.assertGreater(pending, 0)
        self.assertTrue(p.required_sprint_active())
        p.scale = 0.1
        p._sprint_latched = False
        self.assertFalse(p.required_sprint_active())

    def test_sprint_prefers_required_over_science_in_gain(self):
        p = self.fresh(level=0)
        p.scale = 0.9
        p.scale_estimator.point = 0.9
        # Finish ordinary science so only REQUIRED backlog competes; sprint must
        # still fire and put unfinished REQUIRED on most fibres.
        for i in range(len(p.ids)):
            if not p.required[i]:
                p.factor[i] = 1.0
                p.cur[i] = 1.2
            else:
                p.factor[i] = 0.0
                p.cur[i] = 0.0
        p.vcache = None
        logs = []
        p.log = logs.append
        action = self.plan(p, 1.0)
        self.assertIsNotNone(action)
        self.assertTrue(any('required-sprint' in line for line in logs))
        assigned = [p.index_of[tid] for tid in action['assignments'].values()]
        required_share = sum(
            1 for i in assigned
            if p.required[i] and p.factor[i] < p.required_threshold
        ) / max(1, len(assigned))
        self.assertGreaterEqual(required_share, 0.75)
        # Utility path: required_gain should dominate discounted science.
        start, end = p.nights[0]
        estimate = p.evaluate_action(action, start + timedelta(hours=1.0), end, 0, 1.0)
        self.assertIsNotNone(estimate)
        self.assertGreater(estimate['required_gain'], estimate['planning_science_gain'])

    def test_plan_is_deterministic_across_repeated_calls(self):
        """Same planner state → identical observe (probe target included)."""
        actions = []
        for _ in range(3):
            p = self.fresh(level=2)
            p.scale = 0.2
            p.scale_estimator.point = 0.2
            p.terrain = {'N', 'SE', 'W'}
            p.extra_avoid = {'E', 'NW'}
            p.notices = {('haze', 'NE'), ('rain', 'S'), ('rocket_launch', 'SW')}
            action = self.plan(p, 0.5)
            self.assertIsNotNone(action)
            actions.append({
                'pointing': action['pointing'],
                'assignments': dict(sorted(action['assignments'].items())),
                'duration_seconds': action['duration_seconds'],
                'program': action['program'],
            })
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(actions[1], actions[2])

    def test_must_observe_still_forces_probe_when_gains_zero(self):
        p = self.fresh(level=2)
        p.scale_estimator.point = 0.0
        p.scale = 0.0
        p.cur = [0.5] * len(p.ids)
        p.factor = [0.4] * len(p.ids)
        action = self.plan(p)
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'observe')
        self.assertEqual(action['duration_seconds'], 300)


if __name__ == '__main__':
    unittest.main()
