"""Full agent, unchanged public engine; truth is used only for post-run metrics.

Example (Python 3.12):
  python tests/run_offline_comparison.py --kit KIT --card CARD --out NEW_DIR
Optional --agent-root DIRECTORY_OR_ZIP selects an immutable baseline.
No API client is constructed; this is not the official fair-clock transport.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import zipfile

os.environ.update(OBSERVER_MODEL_DISABLED='1', PRO_FIXED_LEVEL='2',
                  V8_FORECAST_ENABLED='0', V8_POLICY_ENABLED='0',
                  PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kit', required=True, type=Path)
parser.add_argument('--card', required=True, type=Path)
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--agent-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--version', default='v9.1-calibration')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
if args.agent_root.is_file():
    with zipfile.ZipFile(args.agent_root) as archive:
        planner_path = next(n for n in archive.namelist() if n.endswith('planner.py'))
        prefix = planner_path[:-len('planner.py')]
        planner_bytes = archive.read(planner_path)
        import_path = args.agent_root.as_posix() + '/' + prefix.rstrip('/')
else:
    planner_bytes = (args.agent_root / 'planner.py').read_bytes()
    import_path = str(args.agent_root)
sys.path[:0] = [import_path, str(args.kit / 'runner')]
from challenge.v4_workflow import V4Workflow
from challenge.v4_runner import run_scenario
from agent import ObserverAgent

workflow = V4Workflow(args.card)
init = workflow.initialize_payload(900, 1.)
trace, decisions = [], []
searches = []
cpu = [0.]
started = time.monotonic()
with (args.out / 'agent.log').open('w', encoding='utf-8') as stream, contextlib.redirect_stderr(stream):
    observer = ObserverAgent(init, rules_only=True)
    assert observer.client is None
    planner = observer.planner
    old_result, old_plan = planner.on_result, planner.plan

    def on_result(result, now, hours):
        if result and result.get('action') == 'observe' and planner.pending:
            row = {'time': now.isoformat(), 'scale_before': planner.scale,
                   'duration': planner.pending_duration, 'pending': planner.pending.copy(),
                   'program': planner.pending_program}
            old_result(result, now, hours)
            row.update(scale_after=planner.scale, evidence=planner.scale_estimator.diagnostics())
            trace.append(row)
        else:
            old_result(result, now, hours)

    def plan(*a, **kw):
        begin = time.process_time()
        result = old_plan(*a, **kw)
        cpu[0] += time.process_time() - begin
        diag = getattr(planner, 'last_calibration_search', None)
        if diag is not None and diag['triggered']:
            searches.append(dict(diag))
        return result

    planner.on_result, planner.plan = on_result, plan

    def decide(snapshot):
        payload = {**snapshot, 'wallclock': {'remaining_seconds': 900,
                    'remaining_real_cpu_seconds': 900, 'wall_remaining_seconds': 1800}}
        result = observer.respond(payload)
        decisions.append(result['action'])
        return {k: v for k, v in result.items() if k in (
            'action', 'pointing', 'assignments', 'duration_seconds', 'program', 'until_utc')}

    report = run_scenario(workflow.scenario_path, lambda _: decide, args.out)
assert report['termination']['reason'] == 'survey_complete', report['termination']
assert (args.out / 'score_report.json').is_file()
(args.out / 'scale_trace.json').write_text(json.dumps(trace), encoding='utf-8')
observations = {}
with (args.out / 'observations.csv').open(newline='', encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        observations.setdefault(int(row['observe_index']), []).append(row)
metrics = []
for index, t in enumerate(trace):
    # Match the historical comparison metric; also report direction-clean only.
    pairs = [(float(r['quality']) / t['pending'][r['target_id']]['model'],
              t['pending'][r['target_id']]['dir_clean'])
             for r in observations.get(index, []) if r['target_id'] in t['pending']]
    if not pairs:
        continue
    truth = statistics.median(v for v, _ in pairs)
    clean = [v for v, valid in pairs if valid]
    clean_truth = statistics.median(clean) if clean else None
    metrics.append({'observe_index': index, 'time': t['time'],
        'true_effective_scale': truth, 'estimated_scale': t['scale_after'],
        'absolute_error': abs(t['scale_after'] - truth),
        'relative_error': abs(t['scale_after'] / truth - 1) if truth > 0 else None,
        'clean_true_effective_scale': clean_truth,
        'clean_absolute_error': abs(t['scale_after'] - clean_truth) if clean_truth is not None else None,
        'clean_relative_error': abs(t['scale_after'] / clean_truth - 1) if clean_truth and clean_truth > 0 else None,
        'estimator_status': t['evidence']['status']})
assert metrics
with (args.out / 'scale_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)

def stats(rows):
    positive = [r for r in rows if r['true_effective_scale'] > .02]
    return {'n': len(rows),
        'mae': statistics.mean(r['absolute_error'] for r in rows) if rows else None,
        'median_relative_error_positive': statistics.median(r['relative_error'] for r in positive) if positive else None,
        'within_10pct_positive': sum(r['relative_error'] <= .1 for r in positive) / len(positive) if positive else None}

with (args.out / 'decisions.csv').open(newline='', encoding='utf-8-sig') as stream:
    decision_rows = list(csv.DictReader(stream))
log = (args.out / 'agent.log').read_text(encoding='utf-8')
summary = {'version': args.version, 'card': str(args.card),
    'elapsed_seconds': time.monotonic() - started, 'plan_cpu_seconds': cpu[0],
    'termination': report['termination'], 'components': report['components'],
    'counts': report['counts'], 'score': sum(report['components'].values()),
    'overall': stats(metrics),
    'good_scale_ge_0_7': stats([r for r in metrics if r['true_effective_scale'] >= .7]),
    'bad_scale_lt_0_4': stats([r for r in metrics if r['true_effective_scale'] < .4]),
    'probe_counts': {kind: len(re.findall(r'probe: ' + kind + r' ', log))
                     for kind in ('nightly', 'adaptive', 'wait')},
    'wait_hours_including_daytime': sum(float(r['duration_seconds']) for r in decision_rows if r['action'] == 'wait') / 3600,
    'calibration_searches': {'count': len(searches),
        'with_usable_target': sum(s.get('usable', 0) > 0 for s in searches),
        'max_new_exact': max((s['new_exact'] for s in searches), default=0),
        'max_anchors': max((s['anchors'] for s in searches), default=0)},
    'limitations': 'Public direct-engine path, rules_only, fixed level 2, constant advertised clock. Not official fair-clock or hidden d1. Post-feedback effective-scale metric, not next-exposure prediction. Truth never enters agent inputs. Final exposure may not receive a subsequent callback.',
    'planner_sha256': hashlib.sha256(planner_bytes).hexdigest(),
    'engine_sha256': {name: hashlib.sha256((args.kit / 'runner/challenge' / name).read_bytes()).hexdigest()
                      for name in ('v4_runner.py', 'v4_scorer.py', 'v4_workflow.py')}}
(args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary), flush=True)
