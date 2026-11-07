"""Four reanalyses on the saved main_v5 evidence (no new runs, no refits).

1. Percentile score mapping for the compensatory judges (reviewer #7).
2. Gate and judges recalibrated without the closer-than-real privacy signal (reviewer #8).
3. Gaussian Copula secondary known-bad set (reviewer #9).
4. Calibration diagnostics: audit conformal p-values and rejection rates by alpha.
Uses the campaign's own calibration and judge code; decisions are re-derived, nothing is refit.
"""
import copy, glob, json, pathlib, sys
import numpy as np
_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = _ROOT / 'paper' / 'generated'
DATA = _ROOT / 'results' / 'main_v5' / 'analysis_tables'
PROFILES = _ROOT / 'results' / 'main_v5' / 'profiles' / 'jobs'
sys.path.insert(0, str(_ROOT / 'campaign'))
from syntrustbench_campaign.config import ExperimentConfig, GATE_DIMENSIONS
from syntrustbench_campaign.calibration import CalibratedEvaluator, rejection_rank, standardize
from syntrustbench_campaign.judges import (MeanJudge, GeometricJudge, HarmonicJudge, SDMetricsJudge,
                                           matched_acceptance, _gate_family)

P = PROFILES
MODEL = json.load(open(glob.glob(str(P / 'calibrate__model/*/empirical_maxT.json'))[0]))
PROFILES = [json.load(open(f)) for f in glob.glob(str(P / '*/*/profile.json'))]
CFG = ExperimentConfig()
ALPHAS = (.025, .05, .10)
audit = set(MODEL['audit_ids'])
L = lambda p: p['labels']
honest = [p for p in PROFILES if L(p)['kind'] == 'null' and L(p)['repeat'] in audit]
def is_primary(l):
    if l['kind'] == 'control': return l.get('control') == 'copy'
    return l['kind'] == 'injection' and l.get('dose', 0) > 0 and l.get('baseline_kind') == 'positive'
bad = [p for p in PROFILES if is_primary(L(p))]
gc_bad = [p for p in PROFILES if L(p)['kind'] == 'injection' and L(p).get('baseline_kind') == 'GaussianCopula' and L(p).get('dose', 0) > 0]
gc_base = [p for p in PROFILES if L(p)['kind'] == 'injection' and L(p).get('baseline_kind') == 'GaussianCopula' and L(p).get('dose', 0) == 0]
positive = [p for p in PROFILES if L(p)['kind'] == 'control' and L(p).get('control') == 'positive']
clean = [p for p in PROFILES if L(p)['kind'] == 'clean']
assert (len(honest), len(bad), len(gc_bad), len(gc_base), len(positive), len(clean)) == (50, 39, 36, 12, 3, 12)
MEANS = (MeanJudge(), GeometricJudge(), HarmonicJudge())
out = {}

def gate_pass(model, p, a):
    return CalibratedEvaluator(model).apply(p['v04'], a)['gate'] == 'Pass'

def tally(model, tables, a, judge=None):
    acc = [gate_pass(model, p, a) if judge is None else matched_acceptance(judge, p['v04'], p['sdmetrics'], model, CFG, a)[0]
           for p in tables]
    return {'accepted': sum(bool(x) for x in acc), 'evaluable': sum(x is not None for x in acc), 'n': len(tables)}

# Sanity: the saved decisions are reproduced before anything is changed.
assert tally(MODEL, bad, .05)['accepted'] == 9 and tally(MODEL, honest, .05)['accepted'] == 48
assert tally(MODEL, bad, .05, MeanJudge())['accepted'] == 17

# 1. Percentile mapping: s_k = share of the n+1 pooled runs whose dimension statistic is at least the run's own.
def percentile_acceptance(judge, p, a):
    ev = p['v04']
    if any(ev['dimensions'][d]['status'] == 'NotEvaluated' or ev['dimensions'][d].get('missing_required_evidence')
           for d in GATE_DIMENSIONS):
        return None
    fam = _gate_family(MODEL, ev)
    if fam is None: return None
    cols, fit, cand = fam
    z = standardize(np.vstack([fit, cand]), MODEL['scale_floor'])
    stat = np.column_stack([z[:, [j for j, c in enumerate(cols) if c['dimension'] == d]].max(axis=1) for d in GATE_DIMENSIONS])
    s = np.column_stack([(stat[:, k][None, :] >= stat[:, k][:, None]).mean(axis=1) for k in range(3)])
    combined = np.array([judge.combine(list(r)) for r in s])
    r = rejection_rank(len(combined) - 1, a)
    return bool(combined[-1] >= np.sort(combined[:-1])[r - 1])

pct = {}
for a in ALPHAS:
    row = {'Gate': {'bad': tally(MODEL, bad, a)['accepted'], 'honest': tally(MODEL, honest, a)['accepted']}}
    for j in MEANS:
        b = [percentile_acceptance(j, p, a) for p in bad]; h = [percentile_acceptance(j, p, a) for p in honest]
        assert None not in b and None not in h
        row[j.name] = {'bad': sum(b), 'honest': sum(h)}
    pct[str(a)] = row
out['percentile_mapping'] = pct

# 2. Without the closer-than-real share.
DROP = 'share_closer_than_real_real_floor'
M2 = copy.deepcopy(MODEL)
M2['columns'] = [c for c in M2['columns'] if c['metric'] != DROP]
assert len(M2['columns']) == len(MODEL['columns']) - 1
noprox = {'audit': {}, 'matched': {}, 'controls_generators': {}}
for a in ALPHAS:
    applied = [CalibratedEvaluator(M2).apply(p['v04'], a) for p in honest]
    noprox['audit'][str(a)] = {d: sum(x['dimensions'][d]['status'] != 'Pass' for x in applied) for d in GATE_DIMENSIONS}
    noprox['audit'][str(a)]['gate'] = sum(x['gate'] != 'Pass' for x in applied)
    row = {'Gate': {'bad': tally(M2, bad, a)['accepted'], 'honest': tally(M2, honest, a)['accepted']}}
    for j in MEANS + (SDMetricsJudge(),):
        name = getattr(j, 'name', 'SDMetrics')
        row[name] = {'bad': tally(M2, bad, a, j)['accepted'], 'honest': tally(M2, honest, a, j)['accepted']}
    noprox['matched'][str(a)] = row
for p in positive + clean + [q for q in PROFILES if L(q)['kind'] == 'control' and L(q).get('control') in ('copy', 'negative')]:
    l = L(p); key = f"{l['kind']}|{l.get('control') or l.get('generator')}|{l.get('seed', l.get('replicate', ''))}"
    before = CalibratedEvaluator(MODEL).apply(p['v04'], .05); after = CalibratedEvaluator(M2).apply(p['v04'], .05)
    noprox['controls_generators'][key] = {
        'gate_before': before['gate'], 'gate_after': after['gate'],
        'privacy_before': before['dimensions']['privacy']['status'], 'privacy_after': after['dimensions']['privacy']['status'],
        'fidelity_after': after['dimensions']['fidelity']['status'], 'utility_after': after['dimensions']['utility']['status']}
# Which primary tables change verdict without the signal?
noprox['primary_changes'] = [f"{L(p).get('family', L(p).get('control'))}|{L(p).get('dose', '')}"
                             for p in bad if gate_pass(MODEL, p, .05) != gate_pass(M2, p, .05)]
out['without_proximity'] = noprox

# 3. Gaussian Copula secondary set.
gc = {}
for name, tables in (('injected', gc_bad), ('dose0_baseline', gc_base)):
    gc[name] = {'Gate': tally(MODEL, tables, .05)}
    for j in MEANS + (SDMetricsJudge(),):
        gc[name][getattr(j, 'name', 'SDMetrics')] = tally(MODEL, tables, .05, j)
    dims = [CalibratedEvaluator(MODEL).apply(p['v04'], .05)['dimensions'] for p in tables]
    gc[name]['failing_dimensions'] = {d: sum(x[d]['status'] != 'Pass' for x in dims) for d in GATE_DIMENSIONS}
out['gaussian_copula'] = gc

# 4. Calibration diagnostics on the 50 audit runs.
pv = [CalibratedEvaluator(MODEL).apply(p['v04'], .05)['gate_family_pvalue'] for p in honest]
rates = {}
for a in ALPHAS:
    applied = [CalibratedEvaluator(MODEL).apply(p['v04'], a) for p in honest]
    r = rejection_rank(50, a)
    rates[str(a)] = {'nominal': r / 51, 'gate': sum(x['gate'] != 'Pass' for x in applied),
                     **{d: sum(x['dimensions'][d]['status'] != 'Pass' for x in applied) for d in GATE_DIMENSIONS}}
from scipy.stats import kstest
out['calibration_diagnostics'] = {'audit_gate_pvalues': pv, 'rates': rates,
                                  'ks_vs_uniform_p': float(kstest(pv, 'uniform').pvalue),
                                  'share_p_below_0.1': float(np.mean(np.array(pv) <= .1)),
                                  'deciles': np.histogram(pv, bins=10, range=(0, 1))[0].tolist()}
json.dump(out, open(HERE / 'reanalysis.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.3, 1.6))
ax.hist(pv, bins=10, range=(0, 1), color='#1f6f8b', edgecolor='white')
ax.axhline(5, color='#2b2b2b', lw=.6, ls=(0, (2, 2)))
ax.set_xlabel('conformal $p$-value (gate family)'); ax.set_ylabel('audit runs')
ax.text(.02, 5.4, 'uniform: 5 per bin', ha='left', va='bottom', fontsize=6)
fig.savefig(HERE / 'fig' / 'fig_pvalues.pdf', bbox_inches='tight'); fig.savefig(HERE / 'fig' / 'fig_pvalues.png', dpi=200, bbox_inches='tight')
print(json.dumps({k: v for k, v in out.items() if k != 'calibration_diagnostics'}, indent=1))
print(json.dumps({k: v for k, v in out['calibration_diagnostics'].items() if k != 'audit_gate_pvalues'}, indent=1))
print(sorted(round(x, 3) for x in pv))
