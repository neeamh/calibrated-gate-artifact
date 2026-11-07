"""Calibration validation by reshuffled fit/audit splits (no new runs, no new evidence).

The frozen model used one random 50/50 split of the 100 null runs. The conformal guarantee is
marginal over that split, so this analysis redraws it many times: for each draw it refits the
calibration on 50 runs with the campaign's own NullCalibrator and scores the other 50 with the
campaign's CalibratedEvaluator. Under exchangeability the average audit rejection rate is at most
r/(n+1). The frozen model and every paper number are unchanged.
"""
import glob, json, pathlib, sys, time
import numpy as np
_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = _ROOT / 'paper' / 'generated'
PROFILES = _ROOT / 'results' / 'main_v5' / 'profiles' / 'jobs'
sys.path.insert(0, str(_ROOT / 'campaign'))
from syntrustbench_campaign.config import ExperimentConfig, GATE_DIMENSIONS
from syntrustbench_campaign.calibration import NullCalibrator, CalibratedEvaluator, rejection_rank

SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ALPHAS = (.025, .05, .10)
cfg = ExperimentConfig()
nulls = {}
for f in glob.glob(str(PROFILES / 'null__profile__*/*/profile.json')):
    p = json.load(open(f)); nulls[p['labels']['repeat']] = p
assert sorted(nulls) == list(range(100))
frozen = json.load(open(glob.glob(str(PROFILES / 'calibrate__model/*/empirical_maxT.json'))[0]))
ids = np.arange(100)

def audit_rates(fit_ids, audit_ids):
    calibrator = NullCalibrator(cfg.calibration)
    calibrator.assignment = lambda smoke=False: (list(fit_ids), list(audit_ids))
    model = calibrator.fit({r: nulls[r]['v04'] for r in fit_ids}, False,
                           sdmetrics_scores={r: nulls[r]['sdmetrics'].get('score') for r in fit_ids})
    evaluator = CalibratedEvaluator(model)
    out = {}
    for a in ALPHAS:
        applied = [evaluator.apply(nulls[r]['v04'], a) for r in audit_ids]
        out[str(a)] = {'gate': sum(x['gate'] != 'Pass' for x in applied),
                       **{d: sum(x['dimensions'][d]['status'] != 'Pass' for x in applied) for d in GATE_DIMENSIONS}}
    return out

# Sanity: the frozen split reproduces the paper's audit counts.
original = audit_rates(frozen['fit_ids'], frozen['audit_ids'])
assert original['0.05']['gate'] == 2 and original['0.025']['gate'] == 1 and original['0.1']['gate'] == 3, original
rng = np.random.default_rng(20261107)
draws, started = [], time.time()
for b in range(SPLITS):
    perm = rng.permutation(ids)
    draws.append(audit_rates(sorted(perm[:50].tolist()), sorted(perm[50:].tolist())))
summary = {'splits': SPLITS, 'seed': 20261107, 'original_split': original, 'by_alpha': {}}
for a in ALPHAS:
    r = rejection_rank(50, a); nominal = r / 51
    for d in ('gate',) + GATE_DIMENSIONS:
        counts = np.array([x[str(a)][d] for x in draws])
        summary['by_alpha'].setdefault(str(a), {})[d] = {
            'nominal_rate': nominal, 'mean_rate': float(counts.mean() / 50), 'sd_rate': float(counts.std(ddof=1) / 50),
            'q05': float(np.quantile(counts, .05) / 50), 'q95': float(np.quantile(counts, .95) / 50),
            'share_of_splits_above_nominal_plus_5pt': float(np.mean(counts / 50 > nominal + .05)),
            'mc_se_mean_rate': float(counts.std(ddof=1) / 50 / np.sqrt(SPLITS))}
seconds = time.time() - started
json.dump(summary, open(HERE / 'resplit_validation.json', 'w'), indent=1)
open(HERE / 'resplit_validation.json', 'a').write('\n')
for a, rows in summary['by_alpha'].items():
    g = rows['gate']
    print(f"alpha={a}: nominal {g['nominal_rate']:.3f}; gate mean {g['mean_rate']:.4f} (MC se {g['mc_se_mean_rate']:.4f}), "
          f"5-95% of splits {g['q05']:.2f}-{g['q95']:.2f}; " +
          ', '.join(f"{d} {rows[d]['mean_rate']:.4f}" for d in GATE_DIMENSIONS))
print('seconds', round(seconds))
