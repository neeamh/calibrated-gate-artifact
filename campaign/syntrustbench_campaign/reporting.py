"""Tables and publication figures derived only from saved, verified evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calibration import CalibratedEvaluator, binomial_admission
from .config import DIMENSIONS, GATE_DIMENSIONS, DIAGNOSTIC_DIMENSIONS, PROTOCOL, ExperimentConfig
from .core import JobLedger, UnavailableAnalysis, available_number, read_json, save_frame, write_json
from .judges import COMPENSATORY, JUDGE_NAMES, JUDGE_PROVENANCE, score_table


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    z = norm.ppf(.975); p = k/n; den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return float(max(0., mid-half)), float(min(1., mid+half))


def known_bad_group(row: dict) -> str | None:
    """Which known-bad family a table belongs to; None for tables expected to pass."""
    if row.get('kind') == 'control':
        return None if row.get('control') == 'positive' else f"{row['control']}_control"
    if row.get('kind') == 'injection' and row.get('dose', 0) > 0:
        return 'real_with_one_injected_flaw' if row.get('baseline_kind') == 'positive' else 'generator_with_injected_flaw'
    return None


# The headline known-bad set: real patient data with exactly one serious, known flaw.
PRIMARY_BAD = ('real_with_one_injected_flaw', 'copy_control')
BAD_GROUPS = PRIMARY_BAD + ('negative_control', 'generator_with_injected_flaw')


def collect_profiles(ledger: JobLedger, descriptors: list[dict]) -> list[dict]:
    records = []
    for descriptor in descriptors:
        key = descriptor['profile_key']
        state = ledger.state(key)
        if state['state'] == 'completed':
            record = ledger.result(key)
            records.append(read_json(record['profile_path']))
        else:
            records.append({'labels': descriptor['labels'], 'availability': 'Unavailable',
                            'reason': f'{key}: {state["state"]}; {state.get("reason", "missing completed evidence")}'})
    return records


def judge_artifact(cfg: ExperimentConfig, profiles: list[dict], model: dict, folder: Path) -> dict:
    """Score every planned table and every held-out null audit run with every judge."""
    rows = []
    for profile in profiles:
        if profile.get('availability') == 'Unavailable':
            for alpha in cfg.calibration.target_rates:
                for variant in ('provisional', 'calibrated'):
                    for judge in JUDGE_NAMES:
                        rows.append({**profile['labels'], 'alpha': alpha, 'variant': variant, 'judge': judge,
                                     'score': None, 'status': 'NotEvaluated', 'reason': profile['reason']})
        else:
            rows.extend(score_table(profile, model, cfg))
    expected_reps = 1 if cfg.execution.smoke_mode else len(cfg.positive_replicates)
    thresholds = []
    for judge in COMPENSATORY:
        positives = [r for r in rows if r['kind'] == 'control' and r.get('control') == 'positive'
                     and r['judge'] == judge and r['alpha'] == cfg.calibration.primary_rate and r['variant'] == 'calibrated']
        values = [r['score'] for r in positives]
        valid = len(values) == expected_reps and all(available_number(v) for v in values)
        tau = min(values) if valid else None
        thresholds.append({'judge': judge, 'tau_star': tau, 'positive_replicates_expected': expected_reps,
                           'positive_replicates_available': sum(available_number(v) for v in values),
                           'status': 'Available' if valid else 'Unavailable',
                           'reason': '' if valid else 'Every planned positive-control score is required.'})
    for row in rows:
        row['accepted_by_gate'] = row['status'] == 'Pass' if row['judge'] == 'Gate' and row['status'] != 'NotEvaluated' else None
    payload = {'rows': rows, 'thresholds': thresholds,
               'tau_definition': ('minimum compensatory score over the positive-control replicates (real held-out '
                                  'patients), i.e. the most lenient threshold that still accepts all real data; '
                                  'also applied at offsets -0.05 and -0.10'),
               'positive_replicates_note': 'Fixed P pool reused; seeded subsampling/order, not independent patient replicates.'}
    path = write_json(folder / 'judge_evidence.json', payload)
    save_frame(pd.DataFrame(thresholds), folder / 'positive_control_thresholds.csv')
    return {'judge_path': path}


@dataclass
class Analysis:
    cfg: ExperimentConfig
    profiles: list[dict]
    null_profiles: list[dict]
    model: dict
    judges: dict

    def evaluated(self, p: dict, variant: str, alpha: float | None = None) -> tuple[dict, str]:
        """Statuses and gate of one profile under one variant; missing evidence never passes."""
        if p.get('availability') == 'Unavailable':
            return {d: 'NotEvaluated' for d in DIMENSIONS}, 'NotEvaluated'
        if variant == 'official_v03':
            return dict(p['official']['dimension_status']), p['official']['benchmark_gate']
        evidence = p['v04'] if variant == 'provisional' else CalibratedEvaluator(self.model).apply(
            p['v04'], self.cfg.calibration.primary_rate if alpha is None else alpha)
        return {d: evidence['dimensions'][d]['status'] for d in DIMENSIONS}, evidence['gate']

    def status_rows(self) -> list[dict]:
        rows = []
        for p in self.profiles:
            for variant in ('official_v03', 'provisional', 'calibrated'):
                for alpha in (self.cfg.calibration.target_rates if variant == 'calibrated' else (self.cfg.calibration.primary_rate,)):
                    statuses, gate = self.evaluated(p, variant, alpha)
                    rows.append({**p['labels'], 'variant': variant, 'alpha': alpha, **statuses, 'gate': gate,
                                 'gate_rule': ('frozen v0.3: all five dimensions' if variant == 'official_v03'
                                               else 'v0.4.1: fidelity, utility, privacy; equity and robustness diagnostic'),
                                 'availability': p.get('availability', 'Available'), 'reason': p.get('reason', '')})
        return rows

    def control_rows(self) -> list[dict]:
        thresholds = {r['judge']: r['tau_star'] for r in self.judges['thresholds']}
        rows = []
        for r in self.judges['rows']:
            if r['kind'] != 'control': continue
            expected = r['control'] == 'positive'
            for offset in self.cfg.judges.threshold_offsets:
                tau = thresholds.get(r['judge'])
                accepted = r.get('accepted_by_gate') if r['judge'] == 'Gate' else (
                    r['score'] >= tau+offset if available_number(r['score']) and tau is not None else None)
                rows.append({**r, 'threshold_offset': offset, 'threshold': tau+offset if tau is not None else None,
                             'expected_overall_acceptance': expected, 'accepted': accepted,
                             'matches_truth': accepted == expected if accepted is not None else None,
                             'truth_note': 'Copy should pass fidelity/utility but fail privacy.' if r['control'] == 'copy' else r['truth']})
        return rows

    def injections(self) -> tuple[list, list, list]:
        lookup = {p['labels']['table']: p for p in self.profiles}
        by_baseline = {}
        for p in self.profiles:
            label = p['labels']
            if label['kind'] == 'injection':
                by_baseline.setdefault((label['baseline'], label['family']), []).append(p)
        minimum, responses, selectivity = [], [], []
        judge_lookup = {r['table']: r.get('dimension_scores') for r in self.judges['rows']
                        if r['judge'] == 'Mean' and r['variant'] == 'calibrated' and r['alpha'] == self.cfg.calibration.primary_rate}
        intended_map = dict(self.cfg.injections.intended)
        for (baseline, family), members in by_baseline.items():
            base = lookup[baseline]
            members = sorted(members, key=lambda p: p['labels']['dose'])
            intended = intended_map[family]
            for variant in ('provisional', 'calibrated'):
                base_statuses, base_gate = self.evaluated(base, variant)
                before = base_statuses | {'gate': base_gate}
                observed_all = [(p['labels']['dose'], self.evaluated(p, variant)) for p in members if p['labels']['dose'] > 0]
                for d in DIMENSIONS + ('gate',):
                    observed = [(dose, statuses[d] if d != 'gate' else gate) for dose, (statuses, gate) in observed_all]
                    missing = before[d] == 'NotEvaluated' or any(s == 'NotEvaluated' for _, s in observed)
                    detected = [dose for dose, status in observed if status == 'Fail']
                    reason = 'Baseline already fails; incident failure not identifiable.' if before[d] == 'Fail' else (
                        'Required dose/baseline evidence missing.' if missing else ('No failure in tested doses.' if not detected else ''))
                    minimum.append({'baseline': baseline, 'family': family, 'variant': variant, 'dimension': d,
                                    'role': 'gate' if d in GATE_DIMENSIONS + ('gate',) else 'diagnostic',
                                    'intended': d in intended, 'baseline_status': before[d],
                                    'minimum_detected_dose': None if missing or before[d] == 'Fail' else min(detected, default=None),
                                    'minimum_observed_absolute_failure_dose': 0. if before[d] == 'Fail' else min(detected, default=None),
                                    'status': 'NotEvaluated' if missing or before[d] == 'Fail' else 'Available', 'reason': reason})
            base_scores = judge_lookup.get(baseline) or {}
            for p in members:
                label = p['labels']
                if label['dose'] == 0: continue
                scores = judge_lookup.get(label['table']) or {}
                delta = {d: base_scores[d]-scores[d] if available_number(base_scores.get(d)) and available_number(scores.get(d)) else None
                         for d in GATE_DIMENSIONS}
                responses.extend({**label, 'dimension': d, 'response': delta[d], 'intended': d in intended,
                                  'status': 'Available' if delta[d] is not None else 'NotEvaluated',
                                  'reason': '' if delta[d] is not None else 'Baseline or injected dimension score unavailable.'}
                                 for d in GATE_DIMENSIONS)
                off_target = [d for d in GATE_DIMENSIONS if d not in intended]
                available = all(v is not None for v in delta.values())
                score = (max(delta[d] for d in intended)-(max(delta[d] for d in off_target) if off_target else 0.)) if available else None
                selectivity.append({**label, 'selectivity': score, 'status': 'Available' if available else 'NotEvaluated',
                                    'definition': 'maximum intended score decrease minus maximum off-target score decrease (gating dimensions)'})
        return minimum, responses, selectivity

    def audit_runs(self) -> list[dict]:
        audit_ids = set(self.model['audit_ids'])
        return [p for p in self.null_profiles if p['labels']['repeat'] in audit_ids]

    def false_alarms(self) -> list[dict]:
        """Honest-data rejection on held-out audit runs, per dimension and for each gate."""
        rows = []
        audits = self.audit_runs(); expected = len(self.model['audit_ids'])
        plans = [('official_v03', self.cfg.calibration.primary_rate), ('provisional', self.cfg.calibration.primary_rate)]
        plans += [('calibrated', a) for a in self.cfg.calibration.target_rates]
        for variant, alpha in plans:
            evaluated = [self.evaluated(p, variant, alpha) for p in audits]
            for d in DIMENSIONS + ('gate',):
                statuses = [gate if d == 'gate' else s[d] for s, gate in evaluated]
                evaluable = [s for s in statuses if s != 'NotEvaluated']
                k, n = evaluable.count('Fail'), len(evaluable)
                low, high = wilson(k, n)
                complete = n == expected
                rows.append({'variant': variant, 'alpha': alpha, 'dimension': d,
                             'role': 'gate' if d in GATE_DIMENSIONS + ('gate',) else 'diagnostic',
                             'audit_runs_expected': expected, 'evaluable_runs': n,
                             'missing_runs': expected-n, 'failures': k,
                             'false_alarm_rate': k/n if complete and n else None,
                             'observed_rate_among_evaluable': k/n if n else None, 'ci_low': low, 'ci_high': high,
                             'not_accepted_rate': sum(s != 'Pass' for s in statuses)/len(statuses) if statuses else None,
                             'status': 'Available' if complete else 'Unavailable',
                             'reason': '' if complete else 'Some audit runs were not evaluable; subset rate explicitly labeled.'})
        return rows

    def admission(self) -> list[dict]:
        """Prespecified rule: a dimension may gate only if, on every held-out honest audit run,
        it is evaluable and its non-acceptance shows no significant excess over the target rate."""
        c = self.cfg.calibration
        audits = self.audit_runs(); expected = len(self.model['audit_ids'])
        rows = []
        for variant in ('official_v03', 'provisional', 'calibrated'):
            evaluated = [self.evaluated(p, variant, c.primary_rate) for p in audits]
            for d in DIMENSIONS + ('gate',):
                statuses = [gate if d == 'gate' else s[d] for s, gate in evaluated]
                n = len(statuses)
                not_evaluable = sum(s == 'NotEvaluated' for s in statuses)
                not_accepted = sum(s != 'Pass' for s in statuses)
                pvalue, no_excess = binomial_admission(not_accepted, n, c.primary_rate, c.admission_test_alpha) if n else (None, False)
                low, high = wilson(not_accepted, n)
                complete = n == expected and n > 0
                meets = bool(complete and not_evaluable == 0 and no_excess)
                rows.append({'variant': variant, 'dimension': d,
                             'declared_role': ('gate' if d in GATE_DIMENSIONS + ('gate',) else 'diagnostic') if variant != 'official_v03' else 'gate (v0.3)',
                             'audit_runs': n, 'audit_runs_expected': expected, 'not_evaluable_runs': not_evaluable,
                             'honest_runs_not_accepted': not_accepted,
                             'not_accepted_rate': not_accepted/n if n else None, 'ci_low': low, 'ci_high': high,
                             'target_rate': c.primary_rate, 'binomial_excess_pvalue': pvalue,
                             'meets_admission_rule': meets,
                             'rule': (f'every audit run evaluable and one-sided exact binomial p >= {c.admission_test_alpha} '
                                      f'for excess non-acceptance over {c.primary_rate}')})
        return rows

    def comparison(self) -> tuple[list, list, list, list]:
        """Known-bad and honest-data acceptance per judge under two prespecified schemes.

        matched (primary): every judge is calibrated on the same fit null runs to the same
        honest-data rejection level, so known-bad acceptance is compared at equal cost.
        tau_star (secondary): compensatory threshold = lowest positive-control score, at offsets.
        """
        thresholds = {r['judge']: r['tau_star'] for r in self.judges['thresholds']}
        primary_alpha = self.cfg.calibration.primary_rate

        def decide(r, scheme, offset):
            if scheme == 'matched':
                return r.get('accepted_matched')
            if r['judge'] == 'Gate':
                return r.get('accepted_by_gate')
            tau = thresholds.get(r['judge'])
            return r['score'] >= tau+offset if available_number(r['score']) and tau is not None else None

        def summary(selected, judge, scheme, alpha, offset, label):
            values = [decide(r, scheme, offset) for r in selected]
            usable = [v for v in values if v is not None]
            complete = len(usable) == len(values) and bool(values)
            low, high = wilson(sum(usable), len(usable))
            return {'scheme': scheme, 'alpha': alpha, 'offset': offset, 'judge': judge, 'set': label,
                    'tau_star': thresholds.get(judge) if scheme == 'tau_star' else None,
                    'tables_expected': len(values), 'tables_evaluable': len(usable), 'accepted': sum(usable),
                    'acceptance_rate': sum(usable)/len(usable) if complete else None, 'ci_low': low, 'ci_high': high,
                    'status': 'Available' if complete else 'NotEvaluated',
                    'reason': '' if complete else 'Missing decision/threshold or no tables in this set.'}

        plans = [('matched', a, 0.) for a in self.cfg.calibration.target_rates]
        plans += [('tau_star', primary_alpha, o) for o in self.cfg.judges.threshold_offsets]
        acceptance, honest, by_family, rankings = [], [], [], []
        for scheme, alpha, offset in plans:
            rows = [r for r in self.judges['rows'] if r['variant'] == 'calibrated' and r['alpha'] == alpha]
            for judge in JUDGE_NAMES:
                selected = [r for r in rows if r['judge'] == judge]
                bad = {g: [r for r in selected if known_bad_group(r) == g] for g in BAD_GROUPS}
                primary = [r for g in PRIMARY_BAD for r in bad[g]]
                every_bad = [r for g in BAD_GROUPS for r in bad[g]]
                acceptance.append(summary(primary, judge, scheme, alpha, offset,
                                          'primary: real data with one serious flaw (injections on real + copy control)'))
                for g in BAD_GROUPS:
                    acceptance.append(summary(bad[g], judge, scheme, alpha, offset, g))
                acceptance.append(summary(every_bad, judge, scheme, alpha, offset, 'all known-bad tables'))
                honest.append(summary([r for r in selected if r['kind'] == 'null'], judge, scheme, alpha, offset,
                                      'held-out honest real-vs-real audit runs'))
                honest.append(summary([r for r in selected if r['kind'] == 'control' and r.get('control') == 'positive'],
                                      judge, scheme, alpha, offset, 'positive controls'))
                for family, doses in self.cfg.injections.doses:
                    for dose in doses:
                        members = [r for r in bad['real_with_one_injected_flaw'] if r['family'] == family and r['dose'] == dose]
                        by_family.append({**summary(members, judge, scheme, alpha, offset, 'real_with_one_injected_flaw'),
                                          'family': family, 'dose': dose})
        rows = [r for r in self.judges['rows'] if r['variant'] == 'calibrated' and r['alpha'] == primary_alpha]
        for judge in JUDGE_NAMES:
            selected = [r for r in rows if r['judge'] == judge]
            every_bad = [r for r in selected if known_bad_group(r) is not None]
            clean = [r for r in selected if r['kind'] == 'clean']
            pairs = [(a['score'], b['score']) for a in every_bad for b in clean]
            valid = [(a, b) for a, b in pairs if available_number(a) and available_number(b)]
            complete = len(valid) == len(pairs) and bool(pairs) and judge != 'Gate'
            inversions = sum(a > b for a, b in valid)
            rankings.append({'judge': judge, 'pairs_expected': len(pairs), 'pairs_evaluable': len(valid),
                             'inversions': inversions, 'inversion_fraction': inversions/len(valid) if complete else None,
                             'definition': 'known-bad table scored strictly above a clean generator table',
                             'status': 'Available' if complete else 'NotEvaluated',
                             'reason': 'Gate has no scalar ranking.' if judge == 'Gate' else '' if complete else 'Missing pair evidence; no omission.'})
        return acceptance, honest, by_family, rankings

    def sensitivity(self) -> list[dict]:
        rows = []
        for p in self.profiles:
            evaluated = [self.evaluated(p, 'calibrated', a) for a in self.cfg.calibration.target_rates]
            for d in DIMENSIONS + ('gate',):
                values = [gate if d == 'gate' else s[d] for s, gate in evaluated]
                baseline = values[list(self.cfg.calibration.target_rates).index(self.cfg.calibration.primary_rate)]
                available = 'NotEvaluated' not in values
                rows.append({**p['labels'], 'dimension': d, 'statuses_by_alpha': dict(zip(self.cfg.calibration.target_rates, values)),
                             'fraction_changed_from_primary': sum(v != baseline for v in values)/len(values) if available else None,
                             'any_change': len(set(values)) > 1 if available else None,
                             'status': 'Available' if available else 'NotEvaluated'})
        return rows

    def build(self, cohort: dict, folder: Path, roles: dict | None = None) -> dict:
        minimum, responses, selectivity = self.injections()
        acceptance, honest, by_family, rankings = self.comparison()
        status = self.status_rows()
        formal = []
        for profile in self.profiles:
            if profile['labels']['kind'] != 'clean':
                continue
            privacy = profile.get('generator_provenance', {}).get('formal_privacy',
                {'status':'Unavailable', 'reason':profile.get('reason','Generator provenance missing.')})
            formal.append({**profile['labels'], **privacy})
        tables = {'gate_admission': self.admission(),
                  'judge_acceptance': acceptance, 'honest_data_acceptance': honest,
                  'acceptance_by_injection': by_family, 'ranking_inversions': rankings,
                  'judge_provenance': [{'judge': j, 'represents': JUDGE_PROVENANCE[j]} for j in JUDGE_NAMES],
                  'generator_formal_privacy': formal,
                  'dimension_status': status, 'clean_profiles': [r for r in status if r['kind'] == 'clean'],
                  'controls_by_judge': self.control_rows(), 'injection_minimum_dose': minimum,
                  'injection_responses': responses, 'injection_selectivity': selectivity,
                  'audit_false_positive_rates': self.false_alarms(), 'threshold_sensitivity': self.sensitivity()}
        metrics = []
        for profile in self.profiles:
            if profile.get('availability') == 'Unavailable': continue
            for variant, dimensions in [('official_v03',profile['official']['dimensions']),('provisional',profile['v04']['dimensions'])]:
                for d, value in dimensions.items():
                    metrics.extend({**profile['labels'], 'variant':variant, 'dimension':d, 'metric':name, **m}
                                   for name,m in value['metrics'].items())
        tables['metric_evidence'] = metrics
        tables['calibrated_metric_thresholds'] = self.model['raw_cutoffs']
        tables['null_role_sizes'] = [{**p['labels'], **p['rows']} for p in self.null_profiles if p.get('availability') != 'Unavailable']
        paths = {name: save_frame(pd.DataFrame(rows), folder / f'{name}.csv') for name, rows in tables.items()}
        for name in ('cohort_split_summary', 'missingness', 'subgroup_counts'):
            source = Path(cohort['folder']) / f'{name}.csv'
            shutil.copyfile(source, folder / source.name); paths[name] = str(folder / source.name)
        if roles is not None:
            cohort_table = pd.read_csv(paths['cohort_split_summary'])
            extra = pd.DataFrame([{'section':'v04_role', 'label':k, 'rows':v}
                                  for k,v in roles['role_counts'].items()])
            save_frame(pd.concat([cohort_table,extra], ignore_index=True), paths['cohort_split_summary'])
        evidence = write_json(folder / 'analysis_tables.json', tables)
        return {'tables': paths, 'evidence': evidence, 'source': 'verified saved profiles only; no fitting in analysis'}


@dataclass
class Reporter:
    cfg: ExperimentConfig

    @staticmethod
    def _pct(value) -> str:
        return 'NA' if not available_number(value) else f'{100*value:.1f}%'

    def headline(self, tables: dict) -> list[str]:
        """Plain-language results computed from the saved tables; no hand-written numbers."""
        c = self.cfg.calibration
        lines = ['## Headline results (computed from the tables below)', '']
        lines += ['### 1. Which dimensions may gate (prespecified admission rule)', '',
                  '| Variant | Dimension | Declared role | Honest audit runs not accepted | 95% CI | Not evaluable | Meets rule |',
                  '|---|---|---|---|---|---|---|']
        for r in tables['gate_admission']:
            lines.append(f"| {r['variant']} | {r['dimension']} | {r['declared_role']} | {r['honest_runs_not_accepted']}/{r['audit_runs']} "
                         f"({self._pct(r['not_accepted_rate'])}) | {self._pct(r['ci_low'])}–{self._pct(r['ci_high'])} | "
                         f"{r['not_evaluable_runs']} | {'yes' if r['meets_admission_rule'] else 'NO'} |")
        declared = [r for r in tables['gate_admission'] if r['variant'] == 'calibrated' and r['declared_role'] == 'gate']
        ok = bool(declared) and all(r['meets_admission_rule'] for r in declared)
        lines += ['', ('All declared gating dimensions and the calibrated gate meet the admission rule.' if ok else
                       '**WARNING: at least one declared gating dimension or the calibrated gate fails the admission rule. '
                       'The declared three-dimension gate is not supported as specified; report this, do not re-tune.**'), '']
        get = lambda rows, judge, scheme, alpha, offset, prefix: next((r for r in rows if r['judge'] == judge and r['scheme'] == scheme
                                                and r['alpha'] == alpha and r['offset'] == offset and r['set'].startswith(prefix)), None)
        def cell(r):
            return 'NA' if r is None or r['acceptance_rate'] is None else f"{r['accepted']}/{r['tables_evaluable']} ({self._pct(r['acceptance_rate'])})"
        for scheme, offset, title, note in (
                ('matched', 0., f'### 2. PRIMARY: every judge calibrated to the same honest-data rate (alpha {c.primary_rate}, same fit runs)',
                 'All judges share the same full-conformal calibration on the same fit runs, so honest-data rejection is matched and known-bad acceptance is compared at equal cost.'),
                ('tau_star', 0., '### 3. SECONDARY: compensatory threshold tau* = lowest positive-control score',
                 'tau* accepts all positive controls by construction; it leaves each judge at a different honest-data rejection rate, so this comparison is not at matched cost.')):
            lines += [title, '', '| Judge | Real data with one serious flaw accepted | All known-bad accepted | Honest audit runs accepted | Positive controls accepted |',
                      '|---|---|---|---|---|']
            for judge in JUDGE_NAMES:
                cells = [cell(get(tables['judge_acceptance'], judge, scheme, c.primary_rate, offset, 'primary')),
                         cell(get(tables['judge_acceptance'], judge, scheme, c.primary_rate, offset, 'all known-bad')),
                         cell(get(tables['honest_data_acceptance'], judge, scheme, c.primary_rate, offset, 'held-out')),
                         cell(get(tables['honest_data_acceptance'], judge, scheme, c.primary_rate, offset, 'positive'))]
                lines.append(f'| {judge} | ' + ' | '.join(cells) + ' |')
            lines += ['', note, '']
        return lines

    def build(self, analysis: dict, folder: Path) -> dict:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch, Rectangle
        tables = read_json(analysis['evidence'])
        figures = {}
        columns = list(GATE_DIMENSIONS) + ['gate'] + list(DIAGNOSTIC_DIMENSIONS)
        labels_x = list(GATE_DIMENSIONS) + ['GATE'] + [f'{d}\n(diagnostic)' for d in DIAGNOSTIC_DIMENSIONS]
        statuses = [r for r in tables['dimension_status'] if r['kind'] in ('clean', 'control')
                    and r['variant'] == 'calibrated' and r['alpha'] == self.cfg.calibration.primary_rate]
        if statuses:
            codes = {'NotEvaluated': -1, 'Pass': 0, 'Conditional': 1, 'Fail': 2}
            matrix = np.array([[codes[r[d]] for d in columns] for r in statuses])
            fig, ax = plt.subplots(figsize=(10, max(4, .35*len(statuses)+1.5)))
            cmap = ListedColormap(['#d8dce2', '#78baa3', '#efc56c', '#d97979'])
            ax.imshow(matrix, cmap=cmap, norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5], cmap.N), aspect='auto')
            ax.set_xticks(range(len(columns)), labels_x); ax.set_yticks(range(len(statuses)), [r['table'] for r in statuses])
            ax.axvline(len(GATE_DIMENSIONS)+.5, color='black', linewidth=1.5)
            for i in range(len(statuses)):
                for j, d in enumerate(columns): ax.text(j, i, statuses[i][d], ha='center', va='center', fontsize=8)
            ax.set_title('Calibrated status. The gate uses fidelity, utility and privacy;\n'
                         'diagnostics (right of the line) are reported only. NA never counts as Pass.')
            figures['status_heatmap'] = self._save(fig, folder, 'status_heatmap'); plt.close(fig)
        else:
            figures['status_heatmap'] = {'status': 'Unavailable', 'reason': 'No planned clean/control status rows.'}
        response = pd.DataFrame(tables['injection_responses'])
        if not response.empty:
            grouped = []
            for keys, group in response.groupby(['baseline_kind', 'family', 'dose', 'dimension'], dropna=False):
                values = group.response.tolist()
                complete = all(available_number(v) for v in values)
                grouped.append(dict(zip(['baseline_kind','family','dose','dimension'], keys)) | {'response': float(np.mean(values)) if complete else None})
            g = pd.DataFrame(grouped)
            g['label'] = g.apply(lambda r: f'{r.baseline_kind} / {r.family} / {r.dose:g}', axis=1)
            labels = list(dict.fromkeys(g.label))
            matrix = np.array([[next(r['response'] for r in grouped if f"{r['baseline_kind']} / {r['family']} / {r['dose']:g}" == label and r['dimension'] == d)
                                for d in GATE_DIMENSIONS] for label in labels], dtype=float)
            fig, ax = plt.subplots(figsize=(8, max(4, .35*len(labels)+1.7)))
            cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#d8dce2')
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(len(GATE_DIMENSIONS)), GATE_DIMENSIONS); ax.set_yticks(range(len(labels)), labels)
            for i, label in enumerate(labels):
                family = label.split(' / ')[1]
                for j, d in enumerate(GATE_DIMENSIONS):
                    ax.text(j, i, f'{matrix[i,j]:.2f}' if np.isfinite(matrix[i,j]) else 'NA', ha='center', va='center', fontsize=8)
                    if d in dict(self.cfg.injections.intended)[family]:
                        ax.add_patch(Rectangle((j-.49,i-.49), .98,.98, fill=False, edgecolor='black', linewidth=1.5))
            ax.set_title('Injection response: baseline score − injected score\nIntended dimensions outlined; missing comparisons gray')
            fig.colorbar(im, ax=ax, label='Score decrease (higher = more harm)')
            figures['injection_selectivity_heatmap'] = self._save(fig, folder, 'injection_selectivity_heatmap'); plt.close(fig)
        else:
            figures['injection_selectivity_heatmap'] = {'status': 'Unavailable', 'reason': 'No injection evidence.'}
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        alphas = list(self.cfg.calibration.target_rates)
        colors = ('#317a95', '#b07126', '#884c8c')
        pick = lambda rows, judge, alpha, prefix: next((r for r in rows if r['scheme'] == 'matched' and r['judge'] == judge
                                                        and r['alpha'] == alpha and r['set'].startswith(prefix)), None)
        panels = [(tables['judge_acceptance'], 'primary', 'Known-bad accepted (real data, one serious flaw)\nlower is better'),
                  (tables['honest_data_acceptance'], 'held-out', 'Honest data accepted (held-out audit runs)\nhigher is better')]
        for ax, (rows, prefix, title) in zip(axes[:2], panels):
            for j, alpha in enumerate(alphas):
                xs = np.arange(len(JUDGE_NAMES)) + (j-1)*.24
                for x, judge in zip(xs, JUDGE_NAMES):
                    row = pick(rows, judge, alpha, prefix)
                    value = None if row is None else row['acceptance_rate']
                    if available_number(value):
                        ax.bar(x, value, width=.22, color=colors[j])
                    else:
                        ax.text(x, .03, 'NA', rotation=90, ha='center', fontsize=8, color=colors[j])
            ax.set_xticks(range(len(JUDGE_NAMES)), JUDGE_NAMES); ax.set_ylim(0, 1); ax.set_title(title, fontsize=10)
        axes[0].set_ylabel('Fraction accepted')
        axes[1].legend(handles=[Patch(facecolor=colors[j], label=f'matched alpha {a:g}') for j, a in enumerate(alphas)], fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3, frameon=False)
        markers = dict(zip(JUDGE_NAMES, ('o', 's', '^', 'v', 'D')))
        for judge in JUDGE_NAMES:
            points = []
            for alpha in alphas:
                bad, ok = pick(tables['judge_acceptance'], judge, alpha, 'primary'), pick(tables['honest_data_acceptance'], judge, alpha, 'held-out')
                if bad and ok and available_number(bad['acceptance_rate']) and available_number(ok['acceptance_rate']):
                    points.append((1-ok['acceptance_rate'], bad['acceptance_rate']))
            if points:
                axes[2].plot(*zip(*points), marker=markers[judge], label=judge)
        axes[2].set_xlabel('Honest data rejected (audit)'); axes[2].set_ylabel('Known-bad accepted (primary set)')
        axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8)
        axes[2].set_title('Error trade-off across matched alphas\n(bottom-left is best)', fontsize=10)
        figures['judge_comparison'] = self._save(fig, folder, 'judge_comparison'); plt.close(fig)
        lines = [f'# SynTrustBench {PROTOCOL} campaign report', '',
                 '**Smoke execution only; not paper evidence.**' if self.cfg.execution.smoke_mode else 'Full campaign evidence; scientific acceptance requires reviewing the controls and audit.', '']
        lines += self.headline(tables)
        lines += ['## Tables', '']
        lines += [f'- [{name}]({path})' for name, path in analysis['tables'].items()]
        lines += ['', '## Figures', '']
        for name, value in figures.items():
            lines.append(f'![{name}]({value["png"]})' if value['status'] == 'Available' else f'{name}: Unavailable — {value["reason"]}')
        lines += ['', '## Interpretation boundaries', '',
                  'The gate decides on fidelity, utility and privacy only. Equity and robustness are computed and reported as diagnostics; they never change the verdict.',
                  'Frozen v0.3 is reported unmodified as the uncalibrated five-dimension baseline; its exact-row refusals are recorded as NotEvaluated.',
                  'Generator epsilon/delta accounting is in generator_formal_privacy. It is separate from empirical Privacy verdicts and does not cover this benchmark report or the joint release of multiple fits.',
                  'No clinical certification or formal privacy guarantee. Utility is the declared readmission prediction task on one public cohort.',
                  'Reference detection CIs omit refit variance. Native categorical encoders pool rare levels beyond 250.',
                  'Null repetitions reuse development patients and use smaller roles than the main campaign; the calibration guarantee assumes exchangeability, which the held-out audit tests.',
                  'Compensatory score normalization and positive-control thresholds are prespecified design choices, not externally validated utility functions.',
                  'No result is inferred from execution coverage. Missing values remain unavailable; inspect denominators and reasons.']
        (folder / 'report.md').write_text('\n'.join(lines) + '\n')
        return {'report': str(folder / 'report.md'), 'figures': figures, 'tables': analysis['tables']}

    @staticmethod
    def _save(fig, folder, name) -> dict:
        fig.tight_layout()
        paths = {ext: str(folder / f'{name}.{ext}') for ext in ('png', 'pdf')}
        for ext, path in paths.items(): fig.savefig(path, dpi=160, bbox_inches='tight')
        return {'status': 'Available', **paths}


def self_test() -> None:
    low, high = wilson(0, 20)
    assert low <= 1e-12 and 0 < high < 1
    assert known_bad_group({'kind': 'control', 'control': 'positive'}) is None
    assert known_bad_group({'kind': 'control', 'control': 'copy'}) == 'copy_control'
    assert known_bad_group({'kind': 'injection', 'dose': .1, 'baseline_kind': 'positive'}) == 'real_with_one_injected_flaw'
    assert known_bad_group({'kind': 'injection', 'dose': 0., 'baseline_kind': 'positive'}) is None
    assert known_bad_group({'kind': 'null'}) is None
