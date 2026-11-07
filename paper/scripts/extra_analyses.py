"""Two analyses on the saved main_v5 profiles, using the campaign's own code.

1. Mechanism: median dimension scores (fixed prespecified mapping) per flaw and dose.
2. Post hoc sensitivity: matched-cost acceptance when the score-mapping temperature
   changes (prespecified value 3). Re-derives every decision; nothing is refit.
"""
import json, glob, sys, pathlib, statistics
from dataclasses import replace
_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = _ROOT / 'paper' / 'generated'
DATA = _ROOT / 'results' / 'main_v5' / 'analysis_tables'
PROFILES = _ROOT / 'results' / 'main_v5' / 'profiles' / 'jobs'
sys.path.insert(0, str(_ROOT / 'campaign'))
from syntrustbench_campaign.config import ExperimentConfig, GATE_DIMENSIONS
from syntrustbench_campaign.calibration import CalibratedEvaluator
from syntrustbench_campaign.judges import MeanJudge, GeometricJudge, HarmonicJudge, matched_acceptance, dimension_scores

P = PROFILES
model = json.load(open(glob.glob(str(P / 'calibrate__model/*/empirical_maxT.json'))[0]))
profiles = [json.load(open(f)) for f in glob.glob(str(P / '*/*/profile.json'))]
audit = set(model['audit_ids'])
def primary(lab):
    if lab['kind'] == 'control': return lab.get('control') == 'copy'
    return lab['kind'] == 'injection' and lab.get('dose', 0) > 0 and lab.get('baseline_kind') == 'positive'
bad = [p for p in profiles if primary(p['labels'])]
honest = [p for p in profiles if p['labels']['kind'] == 'null' and p['labels']['repeat'] in audit]
positive = [p for p in profiles if p['labels']['kind'] == 'control' and p['labels'].get('control') == 'positive']
assert len(bad) == 39 and len(honest) == 50 and len(positive) == 3, (len(bad), len(honest))
out = {}

# 1. Mechanism table (prespecified temperature).
cfg = ExperimentConfig()
def scores(p, c):
    s, _ = dimension_scores(p["v04"], model, c.judges)
    return s
groups = {}
for p in bad:
    lab = p['labels']
    key = ('copy', 'all') if lab['kind'] == 'control' else (lab['family'], lab['dose'])
    groups.setdefault(key, []).append(scores(p, cfg))
groups[('honest', 'audit')] = [scores(p, cfg) for p in honest]
groups[('positive', 'control')] = [scores(p, cfg) for p in positive]
mech = {}
for key, rows in groups.items():
    mech['|'.join(map(str, key))] = {d: statistics.median(r[d] for r in rows if r[d] is not None) for d in GATE_DIMENSIONS}
out['mechanism'] = mech

# 2. Temperature sensitivity at matched cost (alpha 0.05).
sens = {}
for T in (1., 2., 3., 5., 10.):
    c = replace(cfg, judges=replace(cfg.judges, temperature=T))
    row = {}
    for judge in (MeanJudge(), GeometricJudge(), HarmonicJudge()):
        acc_bad = [matched_acceptance(judge, p['v04'], p['sdmetrics'], model, c, .05)[0] for p in bad]
        acc_hon = [matched_acceptance(judge, p['v04'], p['sdmetrics'], model, c, .05)[0] for p in honest]
        assert None not in acc_bad and None not in acc_hon
        row[judge.name] = {'bad': sum(acc_bad), 'honest': sum(acc_hon)}
    sens[str(T)] = row
gate_bad = sum(CalibratedEvaluator(model).apply(p['v04'], .05)['gate'] == 'Pass' for p in bad)
gate_hon = sum(CalibratedEvaluator(model).apply(p['v04'], .05)['gate'] == 'Pass' for p in honest)
out['sensitivity'] = sens; out['gate'] = {'bad': gate_bad, 'honest': gate_hon}
json.dump(out, open(HERE / 'extra_numbers.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
