"""Flatten the saved v0.4.2 follow-up profiles into two long tables used by make_followup.py.

decisions_long.csv         one row per table and alpha: matched-cost acceptance of every rule and the
                           calibrated status of every dimension (950 flawed tables + 150 fresh nulls)
dimension_scores_long.csv  one row per flawed table: the three gating dimension scores s_k (t = 3)
Both are read directly from results/followup_v042/profiles/ and supplementary/; nothing is recomputed.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'followup_v042'
RULES = ('Gate', 'Harmonic', 'Geometric', 'Mean', 'SDMetrics')


def main():
    m = json.loads((OUT / 'manifest.json').read_text())
    decisions, scores = [], []

    def add(base, condition, design, p):
        for a, d in p['decisions']['by_alpha'].items():
            row = {'base': base, 'condition': condition, 'design': design, 'alpha': a}
            row.update({f'acc_{r}': d['accepted_matched'][r] for r in RULES})
            row.update({f'dim_{k}': v for k, v in d['dimensions'].items()})
            decisions.append(row)

    for b in m['bases']:
        for c in m['conditions']:
            p = json.loads((OUT / 'profiles' / f'base{b:02d}' / (c['name'].replace('/', '__') + '.json')).read_text())
            add(b, c['name'], c['design'], p)
            s = p['decisions']['dimension_scores'] or {}
            scores.append({'base': b, 'condition': c['name'],
                           **{f's_{k}': s.get(k) for k in ('fidelity', 'utility', 'privacy')}, 'sdmetrics_score': None})
    for path in sorted((OUT / 'supplementary').glob('null_*.json')):
        add(path.stem.split('_')[-1], 'supplementary_null', 'null', json.loads(path.read_text()))

    for name, rows in (('decisions_long.csv', decisions), ('dimension_scores_long.csv', scores)):
        with (OUT / 'analysis' / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'{len(decisions)} decision rows, {len(scores)} score rows')


if __name__ == '__main__':
    main()
