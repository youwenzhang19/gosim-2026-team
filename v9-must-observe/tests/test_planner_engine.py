"""Regression tests against actual Planner plus public engine generated feedback."""
import unittest,sys,json,copy,collections,os
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent if (ROOT.parent/'planner.py').is_file() else next((ROOT/'fix3').rglob('planner.py')).parent
KIT=Path(os.environ.get('OBSERVER_KIT_ROOT','C:/Users/HP/Documents/Harry/gosim-2026-team-main/training/official-examples/gosim-observer-examples'))
sys.path[:0]=[str(PROJECT),str(KIT/'runner')]
from scale_estimator import ScaleEstimator,score_interval
from planner import Planner
from challenge.v4_scorer import SlotTruth,WeatherTruth,score_target_exposure,program_band,program_multiplier
from challenge.v4_workflow import V4Workflow

class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init=V4Workflow(KIT/'local-cards/L1').initialize_payload(900)
        cls.config=copy.deepcopy(cls.init['scoring']);cls.config.pop('lunar_model');cls.config['q0']=1.

    def fresh(self):
        p=Planner(self.init)
        p.pending_cmd=None
        return p

    def exposure(self,p,hour,sky=1.,eff=1.,T=300,n=40,flux=.6,dirty=False,reverse=False,program=None):
        now=datetime(2027,1,1,tzinfo=timezone.utc)+timedelta(hours=hour)
        end=now+timedelta(seconds=T)
        slot=SlotTruth('test',now,end,True,1.,sky,1.,eff)
        weather=WeatherTruth([slot],None)
        program=program or program_band(sky,self.config)
        multiplier=program_multiplier(program,program_band(sky,self.config),self.config)
        ids=p.ids[:n];ids=ids[::-1] if reverse else ids
        hits=[];p.pending={}
        for tid in ids:
            i=p.index_of[tid];p.flux[i]=flux;p.weight[i]=1.
            p.pending[tid]={'model':1.,'band_model':1/.95,'dir_clean':not dirty,'alt':90.,'az':0.,'pred':1.}
            result=score_target_exposure({'target_id':tid,'feature_flux':flux,'science_weight':1.},[(T,slot)],[],weather,T,multiplier,self.config,now,lambda _:(90.,0.))
            hits.append({'target_id':tid,'score':result.score})
        p.pending_program=program;p.pending_duration=T
        p.on_result({'action':'observe','hits':hits},end,hour+T/3600)
        return p.scale_estimator.diagnostics()

    def test_40_target_probe_survives_saturation(self):
        p=self.fresh();self.exposure(p,0)
        before=p.scale
        self.exposure(p,.1,T=3000)
        self.assertAlmostEqual(p.scale,before)
        self.assertEqual(len(p.scale_estimator.samples),2)

    def test_960_targets_do_not_evict_probe(self):
        p=self.fresh();self.exposure(p,0)
        before=p.scale
        for k in range(1,24):self.exposure(p,k/12,flux=2.)
        self.assertAlmostEqual(p.scale,before)
        self.assertEqual(len(p.scale_estimator.samples),24)
        self.assertTrue(p.scale_estimator.needs_probe)

    def test_latest_bad_weather_replaces_old_good(self):
        p=self.fresh();self.exposure(p,0)
        self.exposure(p,.1,sky=.2)
        self.assertLess(abs(p.scale/.2-1),.1)

    def test_improvement_bounds_raise_and_request_probe(self):
        p=self.fresh();self.exposure(p,0,sky=.2)
        diag=self.exposure(p,.1,sky=1.,T=3000)
        self.assertGreaterEqual(p.scale,.249)
        self.assertTrue(diag['needs_probe'])
        self.assertIsNone(diag['upper'])
        self.exposure(p,1.,sky=1.,T=300)
        self.assertLess(abs(p.scale-1),.1)

    def test_three_finite_not_diluted_by_bounds(self):
        s=ScaleEstimator()
        s.observe(0,[(.99,1.01)]*3+[(.2425,None)]*21,3000)
        self.assertEqual(s.point,1.)

    def test_sample_order_invariance(self):
        p=self.fresh();q=self.fresh()
        self.exposure(p,0,sky=.4);self.exposure(q,0,sky=.4,reverse=True)
        self.assertEqual(p.scale_estimator.diagnostics(),q.scale_estimator.diagnostics())

    def test_no_floor_during_severe_fault(self):
        p=self.fresh();d=self.exposure(p,0,eff=.001,program='DARK')
        self.assertLess(p.scale,.002)
        self.assertLessEqual(d['lower'],.001)
        self.assertGreaterEqual(d['upper'],.001)

    def test_multiplier_ambiguity_not_mislabelled_exact(self):
        p=self.fresh();d=self.exposure(p,0,eff=.5,program='DARK')
        self.assertLessEqual(d['lower'],.5)
        self.assertGreater(d['upper'],.5)
        self.assertLess(p.scale,.6)

    def test_directional_feedback_does_not_poison_global(self):
        p=self.fresh();self.exposure(p,0);before=p.scale
        d=self.exposure(p,.1,sky=.01,dirty=True)
        self.assertEqual(p.scale,before)
        self.assertEqual(d['status'],'no_clean_feedback')
        self.assertEqual(d['rejected_directional'],40)

    def test_expired_evidence_is_marked_stale_not_median(self):
        p=self.fresh();self.exposure(p,0);before=p.scale
        p.update_scale(3)
        self.assertEqual(p.scale,before)
        self.assertEqual(p.scale_estimator.status,'stale')
        self.assertEqual(len(p.scale_estimator.samples),0)

    def test_reset_and_resync_clear_evidence(self):
        p=self.fresh();self.exposure(p,0,eff=.01)
        p.forget_quality_history()
        self.assertEqual(p.scale,1.)
        self.assertEqual(p.scale_estimator.status,'unmeasured')
        p._resync({'best_scores':[]})
        self.assertEqual(p.scale_estimator.status,'unmeasured')

    def test_adaptive_probe_respects_instrument(self):
        p=self.fresh();p.fast_level=2;p._probe_night=0
        p.scale_estimator.needs_probe=True;p.scale_estimator.last_duration=300
        start,end=p.nights[0]
        action=p.plan(start,end,0,0)
        self.assertIsNotNone(action)
        self.assertEqual(action['duration_seconds'],max(p.min_exposure,150))

    def test_invalid_or_zero_scores(self):
        self.assertIsNone(score_interval(float('nan'),1,1,300,1,1.2,1))
        p=self.fresh();d=self.exposure(p,0,sky=0.,program='BACKUP')
        self.assertLess(p.scale,.0001)
        self.assertEqual(d['lower'],0.)

if __name__=='__main__':unittest.main(verbosity=2)
