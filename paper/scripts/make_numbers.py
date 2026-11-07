"""Derive every number used in the paper from the saved campaign tables.

Writes numbers.json (for auditing) and numbers.tex (LaTeX macros the paper uses),
so that no result is typed by hand.
"""
import json, math, pathlib
import pandas as pd
from scipy.stats import binom, norm, beta

_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = _ROOT / 'paper' / 'generated'
DATA = _ROOT / 'results' / 'main_v5' / 'analysis_tables'
PROFILES = _ROOT / 'results' / 'main_v5' / 'profiles' / 'jobs'
D = DATA
OUT = HERE
N = {}


def wilson(k, n):
    z = norm.ppf(.975); p = k / n; den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., mid - half), min(1., mid + half)


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


# Cohort and roles.
cohort = pd.read_csv(D / 'cohort_split_summary.csv')
get = lambda label: int(cohort.loc[cohort.label == label, 'rows'].iloc[0])
N['cohort'] = get('frozen earliest-encounter cohort')
N['train'] = get('real_train'); N['pool'] = get('calibration_pool'); N['test'] = get('real_test')
N['tableRows'] = get('floor')
N['prevalencePct'] = round(100 * float(cohort.loc[cohort.label == 'real_train', 'prevalence'].iloc[0]), 1)

# Admission table (held-out audit, primary alpha).
adm = pd.read_csv(D / 'gate_admission.csv')
for _, r in adm.iterrows():
    key = f"{r.variant}_{r.dimension}"
    k, n = int(r.honest_runs_not_accepted), int(r.audit_runs)
    lo, hi = wilson(k, n)
    ub = float(beta.ppf(.95, k + 1, n - k)) if k < n else 1.0   # exact one-sided 95% upper bound
    N[f'adm_{key}'] = {'k': k, 'n': n,
                       'notEval': int(r.not_evaluable_runs), 'p': float(r.binomial_excess_pvalue),
                       'meets': bool(r.meets_admission_rule), 'lo': lo, 'hi': hi, 'ub': ub}

# Matched and tau* acceptance.
ja = pd.read_csv(D / 'judge_acceptance.csv'); ha = pd.read_csv(D / 'honest_data_acceptance.csv')
JUDGES = ['Gate', 'Harmonic', 'Geometric', 'Mean', 'SDMetrics']
for scheme, alpha, offset in [('matched', .025, 0), ('matched', .05, 0), ('matched', .1, 0), ('tau_star', .05, 0)]:
    for j in JUDGES:
        sel = lambda t, prefix: t[(t.scheme == scheme) & (t.alpha == alpha) & (t.offset == offset)
                                   & (t.judge == j) & t.set.str.startswith(prefix)].iloc[0]
        b, h = sel(ja, 'primary'), sel(ha, 'held-out')
        N[f'{scheme}_{alpha}_{j}'] = {'bad': int(b.accepted), 'badN': int(b.tables_evaluable),
                                      'honest': int(h.accepted), 'honestN': int(h.tables_evaluable)}

# Paired comparison on the 39 primary tables (matched, alpha 0.05).
rows = json.load(open(D / 'judge_evidence.json'))['rows']
rows = [r for r in rows if r['variant'] == 'calibrated' and r['alpha'] == .05]
def primary(r):
    if r['kind'] == 'control':
        return r.get('control') == 'copy'
    return r['kind'] == 'injection' and r.get('dose', 0) > 0 and r.get('baseline_kind') == 'positive'
acc = {}
for r in rows:
    if primary(r):
        acc.setdefault(r['judge'], {})[r['table']] = r['accepted_matched']
assert len(acc['Gate']) == 39 and all(v is not None for j in acc for v in acc[j].values())
for j in JUDGES[1:]:
    only_j = sum(acc[j][t] and not acc['Gate'][t] for t in acc['Gate'])
    only_g = sum(acc['Gate'][t] and not acc[j][t] for t in acc['Gate'])
    N[f'mcnemar_{j}'] = {'onlyJudge': only_j, 'onlyGate': only_g, 'p': exact_mcnemar(only_j, only_g)}
N['gateSubsetOfAll'] = all(all(acc[j][t] for t in acc['Gate'] if acc['Gate'][t]) for j in JUDGES[1:])

# Per-family acceptance (matched, alpha 0.05).
fam = pd.read_csv(D / 'acceptance_by_injection.csv')
fam = fam[(fam.scheme == 'matched') & (fam.alpha == .05)]
N['byInjection'] = {f'{r.family}|{r.dose:g}|{r.judge}': int(r.accepted) for r in fam.itertuples()}

# Gate verdicts on controls and generators.
ds = pd.read_csv(D / 'dimension_status.csv')
for v in ('official_v03', 'provisional', 'calibrated'):
    s = ds[(ds.variant == v) & (ds.alpha == .05)]
    N[f'verdict_{v}'] = {f"{r.kind}|{r.control if r.kind == 'control' else r.generator}|{int(r.seed)}": r.gate
                         for r in s[s.kind.isin(['control', 'clean'])].itertuples()}
json.dump(N, open(OUT / 'numbers.json', 'w'), indent=1, default=float)

# LaTeX macros.
def pct(k, n, d=0):
    return f'{100 * k / n:.{d}f}\\%'
mac = []
def m(name, value):
    assert name.isalpha(), name
    mac.append(f'\\newcommand{{\\{name}}}{{{value}}}')
m('NCohort', f"{N['cohort']:,}"); m('NTrain', f"{N['train']:,}"); m('NPool', f"{N['pool']:,}")
m('NTest', f"{N['test']:,}"); m('NTable', f"{N['tableRows']:,}"); m('Prev', f"{N['prevalencePct']}\\%")
words = {'official_v03': 'Vthree', 'provisional': 'Uncal', 'calibrated': 'Cal'}
dims = {'fidelity': 'Fid', 'utility': 'Util', 'privacy': 'Priv', 'equity': 'Eq', 'robustness': 'Rob', 'gate': 'Gate'}
for v, vw in words.items():
    for d, dw in dims.items():
        a = N[f'adm_{v}_{d}']
        m(f'Adm{vw}{dw}', f"{a['k']}/{a['n']}")
        m(f'Adm{vw}{dw}Pct', pct(a['k'], a['n']))
        m(f'Adm{vw}{dw}CI', f"{100 * a['lo']:.1f}--{100 * a['hi']:.1f}\\%")
        m(f'Adm{vw}{dw}NE', str(a['notEval']))
        m(f'Adm{vw}{dw}UB', f"{100 * a['ub']:.1f}\\%")
        m(f'Adm{vw}{dw}P', f"{a['p']:.2f}" if a['p'] >= .01 else '<0.01')
        m(f'Adm{vw}{dw}Meets', 'yes' if a['meets'] else 'no')
jw = {'Gate': 'Gate', 'Harmonic': 'Harm', 'Geometric': 'Geo', 'Mean': 'Mean', 'SDMetrics': 'SDM'}
aw = {.025: 'A', .05: 'B', .1: 'C'}
for j, w in jw.items():
    for a, al in aw.items():
        x = N[f'matched_{a}_{j}']
        m(f'Bad{w}{al}', f"{x['bad']}/{x['badN']}"); m(f'Bad{w}{al}Pct', pct(x['bad'], x['badN']))
        m(f'Hon{w}{al}', f"{x['honest']}/{x['honestN']}"); m(f'Hon{w}{al}Pct', pct(x['honest'], x['honestN']))
    t = N[f'tau_star_0.05_{j}']
    m(f'TauBad{w}', f"{t['bad']}/{t['badN']}"); m(f'TauBad{w}Pct', pct(t['bad'], t['badN']))
    m(f'TauHon{w}', f"{t['honest']}/{t['honestN']}"); m(f'TauHon{w}Pct', pct(t['honest'], t['honestN']))
    if j != 'Gate':
        mc = N[f'mcnemar_{j}']
        m(f'Only{w}', str(mc['onlyJudge'])); m(f'OnlyGateVs{w}', str(mc['onlyGate']))
        m(f'McN{w}', f"{mc['p']:.3f}" if mc['p'] >= .001 else '<0.001')
(OUT / 'numbers.tex').write_text('% Generated by make_numbers.py from runs/main_v5; do not edit by hand.\n' + '\n'.join(mac) + '\n')
print('macros:', len(mac)); print(json.dumps({k: N[k] for k in N if k.startswith('mcnemar') or k == 'gateSubsetOfAll'}, indent=1))
print({k: N[k] for k in ('cohort', 'train', 'pool', 'test', 'tableRows', 'prevalencePct')})

# Table bodies generated from the same numbers.
# Table 4: accepted flawed tables as k/3; shaded where a rule accepted more than the gate.
fam_names = {'exact_copy': 'Copied training rows', 'feature_shuffle': 'Shuffled predictor',
             'mcar': 'Missing values', 'subgroup_suppression': 'Suppressed subgroup'}
order = [('exact_copy', [.001, .01, .05]), ('feature_shuffle', [.1, .5, .9]),
         ('mcar', [.05, .15, .3]), ('subgroup_suppression', [.1, .5, .9])]
def cells(counts, n):
    g = counts[0]
    out = [f'\\textbf{{{g}/{n}}}']
    for c in counts[1:]:
        out.append(f'\\miss{{{c}/{n}}}' if c > g else f'{c}/{n}')
    return out
lines = []
for fam_key, doses in order:
    for i, d in enumerate(doses):
        counts = [N['byInjection'][f'{fam_key}|{d:g}|{j}'] for j in JUDGES]
        label = fam_names[fam_key] if i == 0 else ''
        lines.append(f'{label} & {100 * d:g}\\% & ' + ' & '.join(cells(counts, 3)) + r' \\')
    lines.append(r'\addlinespace[2pt]')
copy_counts = [int(ja[(ja.scheme == 'matched') & (ja.alpha == .05) & (ja.judge == j) & (ja.set == 'copy_control')].accepted.iloc[0]) for j in JUDGES]
lines.append(r'Copy control & 100\% & ' + ' & '.join(cells(copy_counts, 3)) + r' \\')
lines.append(r'\midrule')
tot = [N[f'matched_0.05_{j}']['bad'] for j in JUDGES]
lines.append(r'\textit{All flawed tables} & & ' + ' & '.join(cells(tot, N['matched_0.05_Gate']['badN'])) + r' \\')
hon = [N[f'matched_0.05_{j}']['honest'] for j in JUDGES]
lines.append(r'\textit{Honest runs accepted} & & ' + ' & '.join(f'{h}/{N["matched_0.05_Gate"]["honestN"]}' for h in hon) + r' \\')
(OUT / 'table_injection.tex').write_text('\n'.join(lines) + '\n\\bottomrule\n')

# Table 3: runs that did not pass, split from runs that could not be evaluated.
labels = {'fidelity': 'Fidelity', 'utility': 'Utility', 'privacy': 'Privacy',
          'equity': 'Equity', 'robustness': 'Robustness', 'gate': 'Gate'}
def adm_row(d):
    cs = []
    for v in ('official_v03', 'provisional', 'calibrated'):
        a = N[f'adm_{v}_{d}']
        cs += [str(a['k'] - a['notEval']), str(a['notEval']) if a['notEval'] else '--']
    a = N[f'adm_calibrated_{d}']
    return f'{labels[d]} & ' + ' & '.join(cs) + ' & ' + ('Yes' if a['meets'] else 'No') + r' \\'
rowsA = [r'\multicolumn{8}{@{}l}{\textit{Gating dimensions}} \\']
rowsA += [adm_row(d) for d in ('fidelity', 'utility', 'privacy')]
rowsA += [r'\addlinespace[2pt]', r'\multicolumn{8}{@{}l}{\textit{Diagnostic dimensions}} \\']
rowsA += [adm_row(d) for d in ('equity', 'robustness')]
rowsA += [r'\midrule', adm_row('gate').replace('Gate &', r'\textbf{Gate} &', 1)]
(OUT / 'table_admission.tex').write_text('\n'.join(rowsA) + '\n\\bottomrule\n')
print(open(OUT / 'table_injection.tex').read()); print(open(OUT / 'table_admission.tex').read())
