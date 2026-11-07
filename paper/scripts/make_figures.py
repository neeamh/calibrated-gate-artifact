"""Figure 1, drawn only from numbers.json (itself derived from the campaign tables)."""
import json, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = _ROOT / 'paper' / 'generated'
DATA = _ROOT / 'results' / 'main_v5' / 'analysis_tables'
PROFILES = _ROOT / 'results' / 'main_v5' / 'profiles' / 'jobs'
N = json.load(open(HERE / 'numbers.json'))
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .6, 'xtick.major.width': .6,
                     'ytick.major.width': .6, 'pdf.fonttype': 42})
INK, ACCENT, MUTED = '#2b2b2b', '#1f6f8b', '#b9bec5'

fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 1.95), gridspec_kw={'width_ratios': [1, 1.25], 'wspace': .55})

# (a) Honest real-vs-real audit runs accepted by each version of the gate.
gates = [('Frozen v0.3 gate\n(fixed thresholds)', 'official_v03'),
         ('Reference-relative,\nuncalibrated', 'provisional'),
         ('Calibrated gate\n(this work)', 'calibrated')]
vals = [1 - N[f'adm_{v}_gate']['k'] / N[f'adm_{v}_gate']['n'] for _, v in gates]
ys = range(len(gates))[::-1]
a.barh(list(ys), vals, color=[MUTED, MUTED, ACCENT], height=.6)
for y, v, (_, key) in zip(ys, vals, gates):
    k, n = N[f'adm_{key}_gate']['n'] - N[f'adm_{key}_gate']['k'], N[f'adm_{key}_gate']['n']
    a.text(v + .02, y, f'{k}/{n}', va='center', color=INK)
target = 1 - 2 / 51
a.axvline(target, color=INK, lw=.6, ls=(0, (2, 2)))
a.text(target - .02, -.62, 'target 96%', ha='right', va='center', fontsize=6, color=INK)
a.set_yticks(list(ys), [g for g, _ in gates]); a.set_xlim(0, 1.12); a.set_ylim(-.9, len(gates) - .5); a.set_xticks([0, .5, 1], ['0', '50%', '100%'])
a.set_title('(a) Honest real data accepted (50 held-out runs)', loc='left', fontsize=7.5)

# (b) Real tables with one serious flaw accepted at matched honest-data cost.
judges = ['Gate', 'Harmonic', 'Geometric', 'Mean', 'SDMetrics']
bad = [N[f'matched_0.05_{j}']['bad'] for j in judges]
n_bad = N['matched_0.05_Gate']['badN']
hon = [N[f'matched_0.05_{j}']['honest'] for j in judges]
xs = range(len(judges))
b.bar(list(xs), [v / n_bad for v in bad], color=[ACCENT] + [MUTED] * 4, width=.62)
for x, v, h in zip(xs, bad, hon):
    b.text(x, v / n_bad + .03, f'{v}/{n_bad}', ha='center', color=INK)
names = ['Gate', 'Harmonic', 'Geometric', 'Mean', 'SDMetrics']
b.set_xticks(list(xs), [f'{nm}\n{h}/50' for nm, h in zip(names, hon)], fontsize=6.5)
b.set_xlabel('rule (honest audit runs it accepts)', fontsize=6.5)
b.set_ylim(0, 1.12); b.set_yticks([0, .5, 1], ['0', '50%', '100%'])
b.set_title('(b) Flawed real tables accepted, equal honest-data cost', loc='left', fontsize=7.5)
fig.savefig(HERE / 'fig' / 'fig1_original.pdf', bbox_inches='tight')
fig.savefig(HERE / 'fig' / 'fig1_original.png', dpi=220, bbox_inches='tight')
print('ok', vals, bad, hon)
