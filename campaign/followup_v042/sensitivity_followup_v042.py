"""Sensitivity analyses of the v0.4.2 follow-up, recomputed on the 50 bases from saved evidence.

Post hoc (requested in review after the follow-up was scored). Nothing is rerun or refit: every
decision is re-derived from the evidence saved in each profile, with the frozen calibration model
and the campaign's own judge code. Three variations of the matched-cost comparison:

  1. Score temperature t in {1, 2, 3, 5, 10} for the exponential mapping s_k = exp(-max(0, z_k)/t).
  2. Empirical percentile mapping: s_k = share of the n+1 pooled runs whose dimension statistic is
     at least the run's own.
  3. Gate and judges recalibrated without the closer-than-real privacy signal (column dropped
     from the frozen model; no refit).

For each variation and alpha: acceptance of the 650 single-flaw tables, the 300 dense tables, the
50 honest audit runs (and, for variation 3, the 150 supplementary runs), and the per-base
difference D_J (rule minus gate) with a 95% bootstrap interval and an exact sign test.
Writes runs/followup_v042/analysis/sensitivity.json and sensitivity_per_condition.csv.

Usage (from the folder that contains runs/):  python sensitivity_followup_v042.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ALPHAS = (0.025, 0.05, 0.1)
TEMPERATURES = (1., 2., 3., 5., 10.)
DROP = 'share_closer_than_real_real_floor'
BOOTSTRAP = 10000


def sign_test(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(positive, negative) + 1)) / 2 ** n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/followup_v042')
    parser.add_argument('--run-dir', default='runs/main_v5/full_489b297b38e6f526')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(manifest['library']).parent))
    import numpy as np
    import followup_v042 as F
    import syntrustbench_campaign
    if Path(syntrustbench_campaign.__file__).resolve().parent != Path(manifest['library']).resolve():
        raise SystemExit(f'Imported the wrong library: {syntrustbench_campaign.__file__}')
    from syntrustbench_campaign.calibration import CalibratedEvaluator, rejection_rank, standardize
    from syntrustbench_campaign.config import GATE_DIMENSIONS, from_mapping
    from syntrustbench_campaign.judges import (GeometricJudge, HarmonicJudge, MeanJudge, SDMetricsJudge,
                                               _gate_family, matched_acceptance)

    run = F.FrozenRun(Path(args.run_dir))
    cfg = from_mapping(json.loads((run.dir / 'configuration.json').read_text()))
    if F.sha256(manifest['model_path']) != manifest['model_sha256']:
        raise SystemExit('Frozen model changed since the follow-up started.')
    model = json.loads(Path(manifest['model_path']).read_text())
    model_noprox = copy.deepcopy(model)
    model_noprox['columns'] = [c for c in model['columns'] if c['metric'] != DROP]
    if len(model_noprox['columns']) != len(model['columns']) - 1:
        raise SystemExit(f'Signal {DROP} not found exactly once in the frozen model.')
    means = (MeanJudge(), GeometricJudge(), HarmonicJudge())
    names = {'Mean': 'Mean', 'Geometric': 'Geometric', 'Harmonic': 'Harmonic'}

    # Load every table: (group, base, condition, v04, sdmetrics, saved decisions).
    tables = []
    for base in manifest['bases']:
        for c in manifest['conditions']:
            p = json.loads((out / 'profiles' / f'base{base:02d}' / (c['name'].replace('/', '__') + '.json')).read_text())
            if p['protocol'] != F.PROTOCOL or p['model_sha256'] != manifest['model_sha256']:
                raise SystemExit(f'Profile {base} {c["name"]} is not from this follow-up.')
            group = 'dense' if c['design'] == 'dense' else 'single'
            tables.append((group, base, c['name'], p['v04'], p['sdmetrics'], p['decisions']))
    for base in manifest['audit_order']:
        result, hashes = run.result(f'null/profile/{base:02d}')
        run.verify(result['profile_path'], hashes.get(result['profile_path']))
        p = json.loads(Path(result['profile_path']).read_text())
        tables.append(('honest', base, 'honest', p['v04'], p['sdmetrics'], None))
    for path in sorted((out / 'supplementary').glob('null_*.json')):
        p = json.loads(path.read_text())
        tables.append(('supplementary', int(path.stem.split('_')[1]), 'supplementary', p['v04'], p['sdmetrics'], p['decisions']))
    counts = {g: sum(t[0] == g for t in tables) for g in ('single', 'dense', 'honest', 'supplementary')}
    if counts != {'single': 650, 'dense': 300, 'honest': 50, 'supplementary': 150}:
        raise SystemExit(f'Unexpected table counts {counts}')

    def gate(m, v04, alpha):
        return CalibratedEvaluator(m).apply(v04, alpha)['gate'] == 'Pass'

    def percentile(judge, m, v04, alpha):
        if any(v04['dimensions'][d]['status'] == 'NotEvaluated' or v04['dimensions'][d].get('missing_required_evidence')
               for d in GATE_DIMENSIONS):
            return None
        family = _gate_family(m, v04)
        if family is None:
            return None
        cols, fit, cand = family
        z = standardize(np.vstack([fit, cand]), m['scale_floor'])
        stat = np.column_stack([z[:, [j for j, c in enumerate(cols) if c['dimension'] == d]].max(axis=1) for d in GATE_DIMENSIONS])
        s = np.column_stack([(stat[:, k][None, :] >= stat[:, k][:, None]).mean(axis=1) for k in range(stat.shape[1])])
        combined = np.array([judge.combine(list(row)) for row in s])
        r = rejection_rank(len(combined) - 1, alpha)
        return bool(combined[-1] >= np.sort(combined[:-1])[r - 1])

    # Every variant maps (rule, table, alpha) -> accepted (True/False/None).
    variants = {}
    for t in TEMPERATURES:
        c_t = replace(cfg, judges=replace(cfg.judges, temperature=t))
        variants[f'temperature_{t:g}'] = (model, lambda j, v, s, a, c_t=c_t: matched_acceptance(j, v, s, model, c_t, a)[0])
    variants['percentile'] = (model, lambda j, v, s, a: percentile(j, model, v, a))
    variants['without_proximity'] = (model_noprox, lambda j, v, s, a: matched_acceptance(j, v, s, model_noprox, cfg, a)[0])

    decisions = {}  # (variant, alpha) -> list of dicts per table
    for vname, (m, accept) in variants.items():
        for alpha in ALPHAS:
            rows = []
            for group, base, cond, v04, sdm, _ in tables:
                row = {'Gate': gate(m, v04, alpha)}
                for j in means:
                    row[names[j.name]] = accept(j, v04, sdm, alpha)
                row['SDMetrics'] = matched_acceptance(SDMetricsJudge(), v04, sdm, m, cfg, alpha)[0]
                rows.append(row)
            decisions[(vname, alpha)] = rows

    # Reproduction check: the prespecified variant (t = 3, full model) equals the saved decisions.
    mismatches = 0
    for alpha in ALPHAS:
        for (group, base, cond, v04, sdm, saved), row in zip(tables, decisions[('temperature_3', alpha)]):
            if saved is None:
                continue
            for rule, value in saved['by_alpha'][str(alpha)]['accepted_matched'].items():
                mismatches += value != row[rule]
    if mismatches:
        raise SystemExit(f'{mismatches} decisions differ from the saved follow-up decisions; stopping.')

    rules = ('Gate', 'Harmonic', 'Geometric', 'Mean', 'SDMetrics')
    rng = np.random.default_rng(20261107)
    bases = manifest['bases']
    summary, per_condition = {'reproduction_check': 'passed', 'drop_signal': DROP, 'variants': {}}, []
    for (vname, alpha), rows in decisions.items():
        entry = {}
        for group in ('single', 'dense', 'honest', 'supplementary'):
            idx = [i for i, t in enumerate(tables) if t[0] == group]
            entry[group] = {r: {'accepted': sum(bool(rows[i][r]) for i in idx),
                                'not_scorable': sum(rows[i][r] is None for i in idx), 'n': len(idx)} for r in rules}
        per_base = {r: np.array([sum(bool(rows[i][r]) for i, t in enumerate(tables) if t[0] == 'single' and t[1] == b)
                                 for b in bases]) for r in rules}
        entry['per_base'] = {}
        for r in rules[1:]:
            d = per_base[r] - per_base['Gate']
            boots = rng.choice(d, size=(BOOTSTRAP, len(d)), replace=True).mean(axis=1)
            pos, neg = int((d > 0).sum()), int((d < 0).sum())
            entry['per_base'][r] = {'mean_difference': float(d.mean()),
                                    'bootstrap_95': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                                    'bases_rule_accepts_more': pos, 'bases_equal': int((d == 0).sum()),
                                    'bases_gate_accepts_more': neg, 'sign_test_p': sign_test(pos, neg)}
        summary['variants'].setdefault(vname, {})[str(alpha)] = entry
        for cond in [c['name'] for c in manifest['conditions']]:
            idx = [i for i, t in enumerate(tables) if t[2] == cond]
            per_condition.append({'variant': vname, 'alpha': alpha, 'condition': cond,
                                  **{r: sum(bool(rows[i][r]) for i in idx) for r in rules}})

    analysis = out / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / 'sensitivity.json').write_text(json.dumps(summary, indent=1) + '\n')
    with (analysis / 'sensitivity_per_condition.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_condition[0]))
        writer.writeheader()
        writer.writerows(per_condition)

    print('Reproduction check passed: t = 3 with the full model equals the saved follow-up decisions.')
    print('Single-flaw tables accepted of 650 (honest audit runs accepted of 50); per-base mean D_J [95% CI]')
    for vname in variants:
        for alpha in ALPHAS:
            e = summary['variants'][vname][str(alpha)]
            if vname.startswith('temperature') and alpha != 0.05:
                continue
            cells = '  '.join(f"{r[:4]} {e['single'][r]['accepted']:3d} ({e['honest'][r]['accepted']})" for r in rules)
            diffs = '  '.join(f"{r[:4]} {e['per_base'][r]['mean_difference']:+.2f} "
                              f"[{e['per_base'][r]['bootstrap_95'][0]:+.2f},{e['per_base'][r]['bootstrap_95'][1]:+.2f}]"
                              for r in rules[1:4])
            print(f'{vname:18s} a={alpha:<5} {cells} | {diffs}')
    e = summary['variants']['without_proximity']
    print('Without proximity, supplementary runs gate-rejected: ' +
          ', '.join(f"a={a}: {150 - e[a]['supplementary']['Gate']['accepted']}/150" for a in e))
    print(f'Wrote {analysis / "sensitivity.json"}')


if __name__ == '__main__':
    main()
