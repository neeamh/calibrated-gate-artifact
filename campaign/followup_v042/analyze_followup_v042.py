"""Prespecified analysis of the v0.4.2 follow-up (see PREREGISTRATION_v0.4.2.md).

Reads the follow-up profiles, the 50 main_v5 audit profiles (honest acceptance at matched cost) and
the supplementary null runs, and writes runs/followup_v042/analysis/:
  summary.json            every number below, for the paper scripts
  per_condition.csv       acceptance of each condition by each rule, with Wilson intervals
  per_base_differences.csv  D_J per base (rule J minus gate, single-flaw conditions)
Run it at any time; it reports how many bases are complete and analyses complete bases only.

Usage (from the folder that contains runs/):  python analyze_followup_v042.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

RULES = ('Gate', 'Harmonic', 'Geometric', 'Mean', 'SDMetrics')
MEANS = ('Harmonic', 'Geometric', 'Mean')
ALPHA = '0.05'
BOOTSTRAP = 10000


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float('nan'), float('nan')
    z = 1.959963984540054
    p, den = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., mid - half), min(1., mid + half)


def sign_test(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.
    tail = sum(math.comb(n, i) for i in range(min(positive, negative) + 1)) / 2 ** n
    return min(1., 2 * tail)


def upper_bound(k: int, n: int, level: float = .95) -> float:
    from scipy.stats import beta
    return 1. if k >= n else float(beta.ppf(level, k + 1, n - k))


def accepted(profile: dict, rule: str, alpha: str = ALPHA) -> bool:
    """A table a rule cannot score (None) counts as not accepted; such cases are counted separately."""
    return bool(profile['decisions']['by_alpha'][alpha]['accepted_matched'][rule])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/followup_v042')
    parser.add_argument('--run-dir', default='runs/main_v5/full_489b297b38e6f526')
    parser.add_argument('--allow-missing-honest', action='store_true', help=argparse.SUPPRESS)  # testing only
    args = parser.parse_args()
    out = Path(args.out).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    # The frozen library must win over any older syntrustbench_campaign folder next to this script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(manifest['library']).parent))
    import numpy as np
    import followup_v042 as F
    import syntrustbench_campaign
    from syntrustbench_campaign.config import from_mapping
    loaded = Path(syntrustbench_campaign.__file__).resolve().parent
    if loaded != Path(manifest['library']).resolve():
        raise SystemExit(f'Imported the wrong library: {loaded}')

    run = F.FrozenRun(Path(args.run_dir))
    cfg = from_mapping(json.loads((run.dir / 'configuration.json').read_text()))
    model = json.loads(Path(manifest['model_path']).read_text())
    if F.sha256(manifest['model_path']) != manifest['model_sha256']:
        raise SystemExit('Frozen model changed since the follow-up started.')
    names = [c['name'] for c in manifest['conditions']]
    singles = [c['name'] for c in manifest['conditions'] if c['design'] in ('single', 'copy_control')]
    dense = [c['name'] for c in manifest['conditions'] if c['design'] == 'dense']

    # Follow-up profiles; only complete bases (every condition present) are analysed.
    profiles, incomplete = {}, []
    for base in manifest['bases']:
        folder = out / 'profiles' / f'base{base:02d}'
        found = {}
        for name in names:
            path = folder / (name.replace('/', '__') + '.json')
            if path.is_file():
                p = json.loads(path.read_text())
                if p['protocol'] == F.PROTOCOL and p['model_sha256'] == manifest['model_sha256']:
                    found[name] = p
        if len(found) == len(names):
            profiles[base] = found
        else:
            incomplete.append(base)
    bases = [b for b in manifest['bases'] if b in profiles]
    if not bases:
        raise SystemExit('No complete base yet.')

    # Honest acceptance at matched cost: the same 50 audit runs, rescored by the same decision code.
    honest = {}
    for base in manifest['audit_order']:
        try:
            result, hashes = run.result(f'null/profile/{base:02d}')
        except SystemExit:
            if args.allow_missing_honest:
                continue
            raise
        run.verify(result['profile_path'], hashes.get(result['profile_path']))
        saved = json.loads(Path(result['profile_path']).read_text())
        honest[base] = {'decisions': F.decisions(saved['v04'], saved['sdmetrics'], model, cfg)}

    summary = {'bases_complete': len(bases), 'bases_incomplete': incomplete, 'alpha': float(ALPHA),
               'honest_audit_accepted': {r: sum(accepted(honest[b], r) for b in honest) for r in RULES},
               'honest_audit_runs': len(honest), 'conditions': {}, 'not_scorable': {}}
    if len(honest) != 50 and not args.allow_missing_honest:
        raise SystemExit('Expected 50 honest audit profiles.')
    rows = []
    for name in names:
        summary['conditions'][name] = {}
        for rule in RULES:
            k = sum(accepted(profiles[b][name], rule) for b in bases)
            lo, hi = wilson(k, len(bases))
            summary['conditions'][name][rule] = {'accepted': k, 'n': len(bases), 'wilson': [lo, hi]}
            rows.append({'condition': name, 'rule': rule, 'accepted': k, 'n': len(bases),
                         'share': k / len(bases), 'wilson_low': lo, 'wilson_high': hi})
    for rule in RULES:
        summary['not_scorable'][rule] = sum(
            profiles[b][n]['decisions']['by_alpha'][ALPHA]['accepted_matched'][rule] is None for b in bases for n in names)

    # Primary analysis 1: D_J per base over the 13 single-flaw conditions.
    rng = np.random.default_rng(20261107)
    diff_rows, summary['replication'] = [], {}
    gate_counts = np.array([sum(accepted(profiles[b][n], 'Gate') for n in singles) for b in bases])
    for rule in MEANS + ('SDMetrics',):
        counts = np.array([sum(accepted(profiles[b][n], rule) for n in singles) for b in bases])
        d = counts - gate_counts
        boots = rng.choice(d, size=(BOOTSTRAP, len(d)), replace=True).mean(axis=1)
        positive, zero, negative = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
        summary['replication'][rule] = {
            'mean_difference': float(d.mean()), 'bootstrap_95': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            'bases_rule_accepts_more': positive, 'bases_equal': zero, 'bases_gate_accepts_more': negative,
            'sign_test_p': sign_test(positive, negative),
            'total_accepted': int(counts.sum()), 'gate_total_accepted': int(gate_counts.sum()),
            'tables': len(bases) * len(singles)}
        for b, value, g, c in zip(bases, d, gate_counts, counts):
            diff_rows.append({'rule': rule, 'base': b, 'gate_accepted': int(g), 'rule_accepted': int(c), 'difference': int(value)})

    # Primary analysis 3: dense map (F alone is the single-flaw shuffle@0.5 cell).
    dense_map = {'F': 'single/feature_shuffle@0.5', **{n.split('/')[1]: n for n in dense}}
    summary['dense'] = {label: {rule: summary['conditions'][name][rule]['accepted'] for rule in RULES}
                        for label, name in dense_map.items() if name in summary['conditions']}

    # Primary analysis 4b: supplementary audit.
    supplementary = [json.loads(p.read_text()) for p in sorted((out / 'supplementary').glob('null_*.json'))]
    supplementary = [p for p in supplementary if p['protocol'] == F.PROTOCOL and p['model_sha256'] == manifest['model_sha256']]
    summary['supplementary_audit'] = {}
    for alpha in ('0.025', '0.05', '0.1'):
        entry = {}
        for scope, items in (('supplementary', supplementary),
                             ('pooled_with_original_audit', supplementary + [honest[b] for b in honest])):
            n = len(items)
            gate_not = sum(not accepted(p, 'Gate', alpha) for p in items)
            dims = {d: sum(p['decisions']['by_alpha'][alpha]['dimensions'][d] != 'Pass' for p in items)
                    for d in ('fidelity', 'utility', 'privacy')}
            entry[scope] = {'runs': n, 'gate_not_accepted': gate_not,
                            'gate_upper_95': upper_bound(gate_not, n) if n else None, 'dimensions_not_passed': dims}
        summary['supplementary_audit'][alpha] = entry

    analysis = out / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    for filename, data in (('per_condition.csv', rows), ('per_base_differences.csv', diff_rows)):
        with (analysis / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)

    print(f'Complete bases: {len(bases)} of {len(manifest["bases"])}; supplementary null runs: {len(supplementary)}')
    print(f'Honest audit runs accepted at alpha {ALPHA}: ' + ', '.join(f'{r} {summary["honest_audit_accepted"][r]}/{len(honest)}' for r in RULES))
    for rule, r in summary['replication'].items():
        print(f'{rule:9s} minus gate, single flaws: mean {r["mean_difference"]:+.2f} per base '
              f'[{r["bootstrap_95"][0]:+.2f}, {r["bootstrap_95"][1]:+.2f}]; bases more/equal/fewer '
              f'{r["bases_rule_accepts_more"]}/{r["bases_equal"]}/{r["bases_gate_accepts_more"]}; sign test p={r["sign_test_p"]:.3g}')
    print('Dense map (tables accepted of ' + str(len(bases)) + '):')
    for label, counts in summary['dense'].items():
        print(f'  {label:4s} ' + '  '.join(f'{r} {counts[r]}' for r in RULES))
    s = summary['supplementary_audit']['0.05']
    for scope, e in s.items():
        if e['runs']:
            print(f'Audit ({scope}, alpha 0.05): gate rejected {e["gate_not_accepted"]}/{e["runs"]}, '
                  f'one-sided 95% upper bound {e["gate_upper_95"]:.3f}')
    print(f'Wrote {analysis}')


if __name__ == '__main__':
    main()
