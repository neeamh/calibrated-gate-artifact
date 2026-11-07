"""Paper numbers, tables and the results figure for the v0.4.2 follow-up (the primary analysis).

Inputs: results/followup_v042/manifest.json and analysis/ (summary.json, sensitivity.json and the
long tables written by extract_followup_tables.py), paper/generated/resplit_validation.json and
paper/generated/numbers.json (honest-audit counts of the three gate versions for panel a).
Outputs in paper/generated/: numbers_followup.tex, table_followup.tex,
table_followup_sensitivity.tex, table_followup_mechanism.tex, numbers_followup.json, fig/fig1.pdf.
"""
import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT / 'paper' / 'generated'
D = ROOT / 'results' / 'followup_v042' / 'analysis'
S = json.loads((D / 'summary.json').read_text())
M = json.loads((D.parent / 'manifest.json').read_text())
R = json.loads((HERE / 'resplit_validation.json').read_text())
N = json.loads((HERE / 'numbers.json').read_text())
dec = pd.read_csv(D / 'decisions_long.csv', dtype={'alpha': str}, keep_default_na=False)
scores = pd.read_csv(D / 'dimension_scores_long.csv')

RULES = ['Gate', 'Harmonic', 'Geometric', 'Mean', 'SDMetrics']
SHORT = {'Gate': 'Gate', 'Harmonic': 'Harm', 'Geometric': 'Geo', 'Mean': 'Mean', 'SDMetrics': 'SDM'}
GATING = ['fidelity', 'utility', 'privacy']
for r in RULES:
    dec[f'acc_{r}'] = dec[f'acc_{r}'].astype(str).eq('True')
flawed = dec[dec.design != 'null']
single = flawed[flawed.design.isin(['single', 'copy_control'])]
dense = flawed[flawed.design == 'dense']
supp = dec[dec.design == 'null']
n_bases = S['bases_complete']
assert n_bases == 50 and len(single) == 3 * 650 and len(dense) == 3 * 300 and len(supp) == 3 * 150

macros = {}


def put(name, value):
    macros[name] = value


def pct(x, digits=0):
    return f'{100 * x:.{digits}f}\\%'


def signed(x):
    if abs(x) < 0.005:
        return '0.00'
    return f'{x:+.2f}'.replace('-', '$-$').replace('+', '$+$')


def pval(p):
    if p < 1e-3:
        mant, exp = f'{p:.1e}'.split('e')
        return f'{mant}\\times10^{{{int(exp)}}}'
    return f'{p:.2f}'


put('FUBases', n_bases)
put('FUConditions', len(M['conditions']))
put('FUSingleN', 650)
put('FUDenseN', 300)
put('FUFlawedN', 950)
put('FUSuppN', len(supp) // 3)

# Primary analysis 1: matched comparison over single flaws, alpha 0.05.
s05 = single[single.alpha == '0.05']
for r in RULES:
    k = int(s05[f'acc_{r}'].sum())
    put(f'FUAcc{SHORT[r]}', k)
    put(f'FUAcc{SHORT[r]}Pct', pct(k / 650))
assert macros['FUAccGate'] == S['replication']['Mean']['gate_total_accepted']
for r in ('Harmonic', 'Geometric', 'Mean', 'SDMetrics'):
    rep, key = S['replication'][r], SHORT[r]
    assert rep['total_accepted'] == macros[f'FUAcc{key}']
    put(f'FUDiff{key}', signed(rep['mean_difference']))
    lo, hi = rep['bootstrap_95']
    put(f'FUDiff{key}CI', f'{signed(lo)} to {signed(hi)}')
    put(f'FUMore{key}', rep['bases_rule_accepts_more'])
    put(f'FUEqual{key}', rep['bases_equal'])
    put(f'FUFewer{key}', rep['bases_gate_accepts_more'])
    put(f'FUSignP{key}', pval(rep['sign_test_p']))
    # Paired discordance across all single-flaw tables.
    put(f'FUOnly{key}', int((s05[f'acc_{r}'] & ~s05['acc_Gate']).sum()))
    put(f'FUOnlyGateVs{key}', int((s05['acc_Gate'] & ~s05[f'acc_{r}']).sum()))
    d05 = dense[dense.alpha == '0.05']
    put(f'FUDenseOnly{key}', int((d05[f'acc_{r}'] & ~d05['acc_Gate']).sum()))
    put(f'FUDenseOnlyGateVs{key}', int((d05['acc_Gate'] & ~d05[f'acc_{r}']).sum()))
for r in RULES:
    put(f'FUHon{SHORT[r]}', S['honest_audit_accepted'][r])

# Ordering at the other calibration levels.
for alpha, tag in (('0.025', 'A'), ('0.1', 'C')):
    sa = single[single.alpha == alpha]
    for r in RULES:
        put(f'FUAcc{SHORT[r]}{tag}', int(sa[f'acc_{r}'].sum()))

# Copy control and the single copied row.
cc = s05[s05.condition == 'single/copy_control']
put('FUCopyCtrlSDM', int(cc['acc_SDMetrics'].sum()))
p05 = flawed[(flawed.condition == 'dense/P') & (flawed.alpha == '0.05')]
for r in RULES:
    put(f'FUP{SHORT[r]}', int(p05[f'acc_{r}'].sum()))

# Mechanism: how many gating dimensions failed among single-flaw tables the gate rejected.
rej = s05[~s05.acc_Gate]
nfail = (rej[[f'dim_{d}' for d in GATING]] != 'Pass').sum(axis=1)
put('FUNRejected', len(rej))
put('FUNOneDim', int((nfail == 1).sum()))
put('FUNTwoDim', int((nfail == 2).sum()))
put('FUNThreeDim', int((nfail == 3).sum()))
assert len(rej) == int(nfail.ge(1).sum())

# Calibration validation: supplementary audit and reshuffled splits.
for alpha, tag in (('0.025', 'A'), ('0.05', 'B'), ('0.1', 'C')):
    e = S['supplementary_audit'][alpha]
    sp, po = e['supplementary'], e['pooled_with_original_audit']
    put(f'FUSuppGate{tag}', f"{sp['gate_not_accepted']}/{sp['runs']}")
    put(f'FUSuppGate{tag}UB', pct(sp['gate_upper_95'], 1))
    put(f'FUPoolGate{tag}', f"{po['gate_not_accepted']}/{po['runs']}")
    put(f'FUPoolGate{tag}Pct', pct(po['gate_not_accepted'] / po['runs'], 1))
    put(f'FUPoolGate{tag}UB', pct(po['gate_upper_95'], 1))
    for dname in GATING:
        put(f'FUPool{dname.capitalize()}{tag}', po['dimensions_not_passed'][dname])
    g = R['by_alpha'][alpha]['gate']
    put(f'FUResplit{tag}', pct(g['mean_rate'], 2))
    put(f'FUResplitNominal{tag}', pct(g['nominal_rate'], 2))
    put(f'FUResplitRange{tag}', f"{100 * g['q05']:.0f}--{100 * g['q95']:.0f}\\%")
put('FUResplits', f"{R['splits']:,}")

# Per-condition table (counts of 50 accepted at alpha 0.05).
LABELS = {
    'exact_copy': 'Copied training rows', 'feature_shuffle': 'Shuffled predictor',
    'mcar': 'Missing values', 'subgroup_suppression': 'Suppressed subgroup'}
DENSE_LABELS = {'U': 'U: outcome rotated, 10\\%', 'P': 'P: one copied training row',
                'FU': 'F + U', 'FP': 'F + P', 'UP': 'U + P', 'FUP': 'F + U + P'}
counts = flawed[flawed.alpha == '0.05'].groupby('condition')[[f'acc_{r}' for r in RULES]].sum().astype(int)


def cells(name):
    row = counts.loc[name]
    gate = row['acc_Gate']
    out = [f'\\textbf{{{gate}}}']
    for r in RULES[1:]:
        v = row[f'acc_{r}']
        out.append(f'\\miss{{{v}}}' if v > gate else str(v))
    return ' & '.join(out)


lines, previous = [], None
for c in M['conditions']:
    if c['design'] == 'dense':
        continue
    name = c['name']
    if name == 'single/copy_control':
        lines.append('\\addlinespace[2pt]')
        lines.append(f'Copy control & 100\\% & {cells(name)} \\\\')
        continue
    family, dose = name.split('/')[1].split('@')
    label = LABELS[family] if family != previous else ''
    if family != previous and previous is not None:
        lines.append('\\addlinespace[2pt]')
    previous = family
    lines.append(f'{label} & {float(dose) * 100:g}\\% & {cells(name)} \\\\')
tot = s05[[f'acc_{r}' for r in RULES]].sum().astype(int)
lines.append('\\midrule')
lines.append('\\textit{All single flaws} & & ' + ' & '.join(
    [f'\\textbf{{{tot["acc_Gate"]}/650}}'] + [f'{tot[f"acc_{r}"]}/650' for r in RULES[1:]]) + ' \\\\')
lines.append('\\textit{Per base, rule minus gate} & & -- & ' + ' & '.join(
    f"{signed(S['replication'][r]['mean_difference'])}" for r in RULES[1:]) + ' \\\\')
lines.append('\\textit{\\quad 95\\% bootstrap interval} & & & ' + ' & '.join(
    f"[{signed(S['replication'][r]['bootstrap_95'][0])}, {signed(S['replication'][r]['bootstrap_95'][1])}]"
    for r in RULES[1:]) + ' \\\\')
lines.append('\\midrule')
lines.append('\\multicolumn{7}{@{}l}{\\textit{Mild components alone and combined (F: shuffled predictor, 50\\%, row above)}} \\\\')
for c in M['conditions']:
    if c['design'] == 'dense':
        lines.append(f"{DENSE_LABELS[c['dense_set']]} & & {cells(c['name'])} \\\\")
lines.append('\\midrule')
lines.append('\\textit{Honest bases accepted} & & ' + ' & '.join(
    f"{S['honest_audit_accepted'][r]}/50" for r in RULES) + ' \\\\')
(HERE / 'table_followup.tex').write_text('\n'.join(lines) + '\n')

# Mechanism table: median dimension score and passes of the calibrated check, of 50.
MECH = [('Copied rows, 0.1\\%', 'single/exact_copy@0.001'), ('One copied row (P)', 'dense/P'),
        ('Shuffled predictor, 90\\%', 'single/feature_shuffle@0.9'),
        ('Missing values, 5\\%', 'single/mcar@0.05'), 
        ('Suppressed subgroup, 90\\%', 'single/subgroup_suppression@0.9'),
        ('Copy control', 'single/copy_control')]
mlines = []
for label, name in MECH:
    sc = scores[scores.condition == name]
    dd = s05[s05.condition == name] if name.startswith('single') else flawed[(flawed.condition == name) & (flawed.alpha == '0.05')]
    out = []
    for dname in GATING:
        med = float(np.median(sc[f's_{dname}']))
        passed = int((dd[f'dim_{dname}'] == 'Pass').sum())
        txt = '$<$0.01' if med < 0.005 else f'{med:.2f}'
        cell = f'{txt}\\,({passed})'
        out.append(f'\\miss{{{cell}}}' if passed < 25 else cell)
    mlines.append(f'{label} & ' + ' & '.join(out) + ' \\\\')
(HERE / 'table_followup_mechanism.tex').write_text('\n'.join(mlines) + '\n')

# Figure 1.
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .6, 'xtick.major.width': .6,
                     'ytick.major.width': .6, 'pdf.fonttype': 42})
INK, ACCENT, MUTED, MID = '#2b2b2b', '#1f6f8b', '#b9bec5', '#7d8590'
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw={'width_ratios': [1, 1.3, 1.0], 'wspace': .55})

gates = [('Frozen v0.3 gate\n(fixed thresholds)', 'official_v03'),
         ('Reference-relative,\nuncalibrated', 'provisional'),
         ('Calibrated gate\n(this work)', 'calibrated')]
vals = [1 - N[f'adm_{v}_gate']['k'] / N[f'adm_{v}_gate']['n'] for _, v in gates]
ys = list(range(len(gates)))[::-1]
a.barh(ys, vals, color=[MUTED, MUTED, ACCENT], height=.6)
for y, v, (_, key) in zip(ys, vals, gates):
    a.text(v + .02, y, f"{N[f'adm_{key}_gate']['n'] - N[f'adm_{key}_gate']['k']}/{N[f'adm_{key}_gate']['n']}", va='center', color=INK)
pooled = S['supplementary_audit']['0.05']['pooled_with_original_audit']
acc200 = 1 - pooled['gate_not_accepted'] / pooled['runs']
a.plot([acc200], [ys[-1] - .45], marker='v', color=INK, ms=3.5, ls='none')
a.text(acc200 - .02, ys[-1] - .45, f"{pooled['runs'] - pooled['gate_not_accepted']}/{pooled['runs']} incl. fresh runs",
       ha='right', va='center', fontsize=5.6, color=INK)
target = 1 - 2 / 51
a.axvline(target, color=INK, lw=.6, ls=(0, (2, 2)))
a.set_yticks(ys, [g for g, _ in gates], fontsize=6.3)
a.set_xlim(0, 1.15); a.set_ylim(-.95, len(gates) - .5); a.set_xticks([0, .5, 1], ['0', '50%', '100%'])
a.set_title('(a) Honest real data accepted', loc='left', fontsize=7.5)

# (b) single-flaw acceptance with base-clustered bootstrap intervals.
rng = np.random.default_rng(20261107)
per_base = s05.groupby('base')[[f'acc_{r}' for r in RULES]].sum()
boot_idx = rng.integers(0, len(per_base), size=(10000, len(per_base)))
xs = np.arange(len(RULES))
shares, lows, highs = [], [], []
for r in RULES:
    v = per_base[f'acc_{r}'].to_numpy()
    share = v.sum() / 650
    bs = v[boot_idx].sum(axis=1) / 650
    shares.append(share); lows.append(share - np.quantile(bs, .025)); highs.append(np.quantile(bs, .975) - share)
b.bar(xs, shares, color=[ACCENT] + [MUTED] * 4, width=.62, yerr=[lows, highs], ecolor=INK, error_kw={'lw': .6, 'capsize': 1.5})
for x, r, sh, hi in zip(xs, RULES, shares, highs):
    b.text(x, sh + hi + .03, f"{int(round(sh * 650))}", ha='center', color=INK, fontsize=6.5)
names = ['Gate', 'Harm.', 'Geom.', 'Arith.', 'SDM.']
b.set_xticks(xs, [f"{nm}\n{S['honest_audit_accepted'][r]}/50" for nm, r in zip(names, RULES)], fontsize=5.9)
b.set_ylim(0, 1.12); b.set_yticks([0, .5, 1], ['0', '50%', '100%'])
b.set_title(f'(b) 650 single-flaw tables accepted', loc='left', fontsize=7.5)

# (c) dense map.
order = ['P', 'UP', 'FP', 'FUP']
styles = {'Gate': (ACCENT, '-', 'o'), 'Harmonic': (MID, '--', 's'), 'Mean': (INK, ':', '^')}
for r, (col, ls, mk) in styles.items():
    yv = [S['dense'][k][r] for k in order]
    c.plot(range(len(order)), yv, color=col, ls=ls, marker=mk, ms=3, lw=1, label={'Mean': 'Arithmetic'}.get(r, r))
c.set_xticks(range(len(order)), ['P', 'U+P', 'F+P', 'F+U+P'], fontsize=6.5)
c.set_ylim(30, 50); c.set_yticks([30, 40, 50])
c.set_ylabel('accepted, of 50', fontsize=6.5)
c.legend(frameon=False, fontsize=6, loc='upper right', handlelength=2.2)
c.set_title('(c) One copied row, plus mild flaws', loc='left', fontsize=7.5)
fig.savefig(HERE / 'fig' / 'fig1.pdf', bbox_inches='tight')
fig.savefig(HERE / 'fig' / 'fig1.png', dpi=220, bbox_inches='tight')

# Sensitivity of the matched comparison on the 50 bases (post hoc, from saved evidence).
SENS = json.loads((D / 'sensitivity.json').read_text())
assert SENS['reproduction_check'] == 'passed'
V = SENS['variants']
base = V['temperature_3']['0.05']
assert base['single']['Gate']['accepted'] == macros['FUAccGate'] and base['single']['Mean']['accepted'] == macros['FUAccMean']


def dcell(entry, rule):
    e = entry['per_base'][rule]
    lo, hi = e['bootstrap_95']
    mark = '' if lo <= 0 <= hi else '$^\\dagger$'
    return f"{signed(e['mean_difference'])}{mark}"


srows = []
ROWS = [(f'$t={t}$' + ('$^\\ast$' if t == '3' else ''), f'temperature_{t}') for t in ('1', '2', '3', '5', '10')]
ROWS += [('Percentile', 'percentile'), ('No proximity', 'without_proximity')]
for label, key in ROWS:
    cells = []
    for a in ('0.025', '0.05', '0.1'):
        cells += [dcell(V[key][a], r) for r in ('Harmonic', 'Geometric', 'Mean')]
    srows.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    if key == 'temperature_10':
        srows.append('\\addlinespace[2pt]')
gate_counts = [V['temperature_3'][a]['single']['Gate']['accepted'] for a in ('0.025', '0.05', '0.1')]
for key, _ in [(k, None) for _, k in ROWS]:
    assert [V[key][a]['single']['Gate']['accepted'] for a in ('0.025', '0.05', '0.1')] == gate_counts
(HERE / 'table_followup_sensitivity.tex').write_text('\n'.join(srows) + '\n')
put('SensGateA', gate_counts[0]); put('SensGateB', gate_counts[1]); put('SensGateC', gate_counts[2])
all_mean = [V[k][a]['per_base']['Mean'] for _, k in ROWS for a in ('0.025', '0.05', '0.1')]
put('SensSettings', len(all_mean))
put('SensMeanPositive', sum(e['mean_difference'] > 0 for e in all_mean))
put('SensMeanExcl', sum(e['bootstrap_95'][0] > 0 for e in all_mean))
for t in ('1', '10'):
    e = V[f'temperature_{t}']['0.05']['per_base']['Mean']
    tw = {'1': 'one', '10': 'ten'}[t]
    put(f'SensMeanT{tw}', signed(e['mean_difference']))
    put(f'SensMeanT{tw}CI', f"{signed(e['bootstrap_95'][0])} to {signed(e['bootstrap_95'][1])}")
e = V['temperature_10']['0.025']['per_base']['Mean']
put('SensMeanTtenA', signed(e['mean_difference'])); put('SensMeanTtenACI', f"{signed(e['bootstrap_95'][0])} to {signed(e['bootstrap_95'][1])}")
for t in ('1', '3', '10'):
    put(f'SensMeanAccT{["", "one", "", "three"][int(t)] if t != "10" else "ten"}', V[f'temperature_{t}']['0.05']['single']['Mean']['accepted'])
e = V['temperature_3']['0.025']['per_base']['Harmonic']
put('SensHarmA', signed(e['mean_difference'])); put('SensHarmACI', f"{signed(e['bootstrap_95'][0])} to {signed(e['bootstrap_95'][1])}")
e = V['temperature_3']['0.1']['per_base']['Harmonic']
put('SensHarmC', signed(e['mean_difference'])); put('SensHarmCCI', f"{signed(e['bootstrap_95'][0])} to {signed(e['bootstrap_95'][1])}")
e = V['temperature_3']['0.1']['per_base']['Geometric']
put('SensGeoC', signed(e['mean_difference'])); put('SensGeoCCI', f"{signed(e['bootstrap_95'][0])} to {signed(e['bootstrap_95'][1])}")
pc = V['percentile']['0.05']['per_base']
put('SensPctHarm', signed(pc['Harmonic']['mean_difference'])); put('SensPctMean', signed(pc['Mean']['mean_difference']))
npx = V['without_proximity']['0.05']
put('SensNPMean', signed(npx['per_base']['Mean']['mean_difference']))
put('SensNPMeanCI', f"{signed(npx['per_base']['Mean']['bootstrap_95'][0])} to {signed(npx['per_base']['Mean']['bootstrap_95'][1])}")
put('SensNPHarm', signed(npx['per_base']['Harmonic']['mean_difference']))
put('SensNPHonGate', npx['honest']['Gate']['accepted'])
put('SensNPSuppGate', f"{150 - npx['supplementary']['Gate']['accepted']}/150")
put('FUPGateRejected', 50 - macros['FUPGate']); put('FUPHarmRejected', 50 - macros['FUPHarm'])

with (HERE / 'numbers_followup.tex').open('w') as f:
    f.write('% Generated by make_followup.py from runs/followup_v042/analysis; do not edit.\n')
    for k, v in macros.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
(HERE / 'numbers_followup.json').write_text(json.dumps(macros, indent=1, default=str))
print(len(macros), 'macros')
for k in ('FUAccGate', 'FUAccHarm', 'FUAccGeo', 'FUAccMean', 'FUAccSDM', 'FUNRejected', 'FUNOneDim', 'FUNTwoDim',
          'FUNThreeDim', 'FUOnlyHarm', 'FUOnlyGateVsHarm', 'FUDenseOnlyHarm', 'FUDenseOnlyGateVsHarm',
          'FUAccGateA', 'FUAccMeanA', 'FUAccGateC', 'FUAccMeanC', 'FUAccHarmA', 'FUAccHarmC', 'FUAccGeoA', 'FUAccGeoC',
          'FUCopyCtrlSDM', 'FUPoolGateB', 'FUPoolGateBUB', 'FUResplitB', 'FUSignPMean', 'FUSignPSDM'):
    print(k, macros[k])
