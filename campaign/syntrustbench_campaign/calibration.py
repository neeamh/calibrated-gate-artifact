"""Direction-aware full-conformal maxT calibration, with explicit evidence gaps.

Two families are calibrated separately: the gate family (fidelity, utility,
privacy), whose decision is the verdict, and the diagnostic family (equity,
robustness), which is reported but never decides the verdict.

Decision rule (full conformal, per family). Each signal is oriented so that
larger is worse. A new table is pooled with the n fit runs; every signal is
standardized by the pooled median and SD; each run's statistic is its maximum
standardized signal. With r = floor(alpha (n + 1)), the table fails when its
statistic exceeds the r-th largest fit-run statistic. Because the rule is a
symmetric function of the n + 1 pooled runs, an honest table exchangeable with
the fit runs is rejected with probability at most r / (n + 1) <= alpha, for any
signal distribution (Vovk et al. 2005; Lei et al. 2018). A dimension fails when
its own maximum exceeds that same family threshold, so the gate fails exactly
when any gating dimension fails.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import numpy as np

from .config import CalibrationConfig, DIMENSIONS, GATE_DIMENSIONS, DIAGNOSTIC_DIMENSIONS
from .core import Gate, UnavailableAnalysis, available_number

FAMILIES = {'gate': GATE_DIMENSIONS, 'diagnostic': DIAGNOSTIC_DIMENSIONS}


def family_of(dimension: str) -> str:
    return 'gate' if dimension in GATE_DIMENSIONS else 'diagnostic'


def rejection_rank(n: int, alpha: float) -> int:
    """r = floor(alpha (n+1)); r = 0 means no table can be rejected at this alpha."""
    return math.floor(alpha * (n + 1) + 1e-12)


def standardize(matrix: np.ndarray, floor: float) -> np.ndarray:
    center = np.median(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0, ddof=1), floor)
    return (matrix - center) / scale


def conformal_decision(fit: np.ndarray, candidate: np.ndarray, alpha: float, floor: float) -> dict:
    """Full-conformal max-statistic decision for one candidate row of oriented signals."""
    n = fit.shape[0]
    z = standardize(np.vstack([fit, candidate]), floor)
    maxima = z.max(axis=1)
    r = rejection_rank(n, alpha)
    threshold = float(np.sort(maxima[:n])[::-1][r - 1]) if r >= 1 else None
    pvalue = float((1 + np.sum(maxima[:n] >= maxima[-1])) / (n + 1))
    return {'candidate_z': z[-1].tolist(), 'threshold': threshold, 'candidate_max': float(maxima[-1]),
            'pvalue': pvalue, 'rejection_rank': r, 'nominal_level': r / (n + 1),
            'finite_sample_valid': r >= 1}


def binomial_admission(failures: int, runs: int, target: float, level: float) -> tuple[float, bool]:
    """One-sided exact test of excess honest-data non-acceptance over target."""
    from scipy.stats import binom
    pvalue = float(binom.sf(failures - 1, runs, target)) if failures > 0 else 1.
    return pvalue, pvalue >= level


def utility_signals(metrics: dict) -> dict:
    """Transform existing estimates only; no refits or new bootstrap calculations."""
    output = {}
    for probe in ('random_forest', 'logistic_regression', 'gradient_boosting'):
        key = f'synthetic_retention_{probe}'
        value = metrics[key]
        output[key] = {'estimate': value['ci_high'] if value['status'] != 'NotEvaluated' else None,
                       'higher_is_worse': False}
    for key in ('relative_calibration_in_the_large', 'relative_calibration_slope'):
        output[key+'_high'] = {'estimate': metrics[key]['estimate'], 'higher_is_worse': True}
        output[key+'_low'] = {'estimate': metrics[key]['estimate'], 'higher_is_worse': False}
    return output


@dataclass
class NullCalibrator:
    cfg: CalibrationConfig

    def counts(self, smoke: bool = False) -> tuple[int, int]:
        return ((self.cfg.smoke_repetitions, self.cfg.smoke_fit_runs) if smoke
                else (self.cfg.repetitions, self.cfg.fit_runs))

    def assignment(self, smoke: bool = False) -> tuple[list[int], list[int]]:
        count, fit = self.counts(smoke)
        order = np.random.default_rng(self.cfg.assignment_seed).permutation(count).tolist()
        return order[:fit], order[fit:]

    def fit(self, profiles: dict[int, dict], smoke: bool = False, sdmetrics_scores: dict | None = None) -> dict:
        """Freeze the oriented fit-run signals; audit runs are never read."""
        fit_ids, audit_ids = self.assignment(smoke)
        missing = set(fit_ids)-set(profiles)
        if missing:
            raise UnavailableAnalysis(f'Missing fit null profiles: {sorted(missing)}')
        # Menu comes from the frozen evaluator definitions, not generator outcomes.
        menu = profiles[fit_ids[0]]['signals']
        columns, excluded, blocked = [], [], set()
        for dimension in DIMENSIONS:
            for name in menu[dimension]:
                items = [profiles[i]['signals'][dimension][name] for i in fit_ids]
                values = [v['estimate'] for v in items]
                directions = [bool(v['higher_is_worse']) for v in items]
                if len(set(directions)) != 1:
                    raise ValueError('Metric direction changed between null repetitions.')
                if not all(available_number(v) for v in values):
                    excluded.append({'dimension': dimension, 'metric': name, 'family': family_of(dimension),
                                     'missing_run_ids': [i for i, v in zip(fit_ids, values) if not available_number(v)],
                                     'reason': 'Required signal missing; calibrated dimension cannot Pass.'})
                    blocked.add(dimension); continue
                direction = 1 if directions[0] else -1
                harm = direction * np.asarray(values, dtype=float)
                sd = float(np.std(harm, ddof=1))
                columns.append({'dimension': dimension, 'family': family_of(dimension), 'metric': name,
                                'direction': direction, 'fit_harm': harm.tolist(),
                                # Fixed fit-run location/scale: used only for the frozen
                                # compensatory score mapping and descriptive thresholds.
                                'center': float(np.median(harm)), 'scale': max(sd, self.cfg.scale_floor),
                                'observed_sd': sd, 'degenerate': sd <= self.cfg.scale_floor,
                                'degenerate_note': ('Constant on honest fit runs; any move in the harmful '
                                                    'direction ranks above every fit run.' if sd <= self.cfg.scale_floor else '')})
        if not any(c['family'] == 'gate' for c in columns):
            raise UnavailableAnalysis('No calibratable gate signals; nothing can be frozen.')
        n = len(fit_ids)
        levels = {str(a): {'rejection_rank': rejection_rank(n, a), 'nominal_level': rejection_rank(n, a)/(n+1),
                           'finite_sample_valid': rejection_rank(n, a) >= 1} for a in self.cfg.target_rates}
        # Descriptive per-signal thresholds: the signal value that alone would reach the
        # family threshold computed among fit runs. Decisions never use these numbers.
        descriptive = []
        for family in FAMILIES:
            cols = [c for c in columns if c['family'] == family]
            if not cols: continue
            z = standardize(np.array([c['fit_harm'] for c in cols]).T, self.cfg.scale_floor)
            maxima = np.sort(z.max(axis=1))[::-1]
            for a, level in levels.items():
                r = level['rejection_rank']
                cut = float(maxima[r-1]) if r >= 1 else None
                descriptive += [{**{k: c[k] for k in ('dimension', 'family', 'metric', 'direction', 'center', 'scale')},
                                 'alpha': float(a), 'family_threshold_z_among_fit_runs': cut,
                                 'raw_cutoff_descriptive': None if cut is None else (c['center'] + cut*c['scale'])/c['direction'],
                                 'comparison': '>' if c['direction'] == 1 else '<',
                                 'note': 'Descriptive only; decisions use full-conformal ranking with the candidate pooled.'}
                                for c in cols]
        return {'method': ('full-conformal studentized global max statistic per family (gate; diagnostic); '
                           'candidate pooled with fit runs; reject when the candidate statistic exceeds the '
                           'floor(alpha(n+1))-th largest fit-run statistic; strict exceedance'),
                'families': {f: list(d) for f, d in FAMILIES.items()}, 'scale_floor': self.cfg.scale_floor,
                'fit_ids': fit_ids, 'audit_ids': audit_ids, 'columns': columns, 'levels': levels,
                'raw_cutoffs': descriptive, 'excluded': excluded,
                'sdmetrics_fit_scores': [(sdmetrics_scores or {}).get(i) for i in fit_ids],
                'blocked_dimensions': sorted(blocked), 'smoke_only': smoke,
                'limitations': [f'Exact marginal level r/(n+1) <= alpha if honest tables are exchangeable with the {n} fit runs; '
                                'the held-out audit tests that assumption, and main-campaign roles are larger than null roles',
                                'audit run IDs held out; development patients reused across runs',
                                'with few fit runs some alpha levels cannot reject (finite_sample_valid false)']}


class CalibratedEvaluator:
    """Frozen-evidence wrapper: raw ReferenceFidelity/Utility dictionaries are untouched."""
    def __init__(self, model: dict):
        self.model = model

    def z_values(self, dimension: str, signals: dict) -> tuple[list[float], list[str]]:
        """Fixed fit-run standardization, used only by the frozen compensatory score mapping."""
        zs, unavailable = [], []
        if dimension in self.model['blocked_dimensions']:
            unavailable.append('Calibration lacks one or more required signals.')
        for c in self.model['columns']:
            if c['dimension'] != dimension:
                continue
            value = signals.get(c['metric'], {}).get('estimate')
            if not available_number(value):
                unavailable.append(c['metric']); continue
            zs.append((c['direction']*value-c['center'])/c['scale'])
        if not zs:
            unavailable.append('No standardized evidence.')
        return zs, unavailable

    def family_decision(self, family: str, all_signals: dict, alpha: float) -> dict:
        """Pool the candidate with the fit runs over every available signal of the family."""
        cols = [c for c in self.model['columns'] if c['family'] == family
                and available_number(all_signals.get(c['dimension'], {}).get(c['metric'], {}).get('estimate'))]
        if not cols:
            return {'threshold': None, 'columns': [], 'finite_sample_valid': False, 'candidate_z': [],
                    'pvalue': None, 'candidate_max': None}
        fit = np.array([c['fit_harm'] for c in cols]).T
        candidate = np.array([c['direction']*all_signals[c['dimension']][c['metric']]['estimate'] for c in cols])
        result = conformal_decision(fit, candidate, alpha, self.model['scale_floor'])
        result['columns'] = [(c['dimension'], c['metric']) for c in cols]
        return result

    def status(self, dimension: str, signals: dict, alpha: float, *, missing=False, hard_fail=False,
               all_signals: dict | None = None, decision: dict | None = None) -> tuple[str, dict]:
        """Dimension verdict under its family's conformal threshold."""
        _, unavailable = self.z_values(dimension, signals)
        pooled = dict(all_signals or {}); pooled[dimension] = signals
        decision = decision or self.family_decision(family_of(dimension), pooled, alpha)
        own = [z for (d, _), z in zip(decision['columns'], decision['candidate_z']) if d == dimension]
        threshold = decision['threshold']
        if not decision['finite_sample_valid']:
            unavailable.append('Too few fit runs to reject at this alpha.')
        exceeds = bool(own and threshold is not None and max(own) > threshold)
        status = Gate.aggregate(['Fail' if hard_fail or exceeds else 'Pass',
                                 'NotEvaluated' if missing or unavailable else 'Pass'])
        return status, {'max_z_pooled': max(own) if own else None, 'family_threshold_z': threshold,
                        'family': family_of(dimension), 'family_pvalue': decision['pvalue'], 'alpha': alpha,
                        'unavailable': unavailable, 'strict_exceedance': exceeds}

    def apply(self, evidence: dict, alpha: float) -> dict:
        signals_all = evidence['signals']
        decisions = {f: self.family_decision(f, signals_all, alpha) for f in FAMILIES}
        dimensions = {}
        for d in DIMENSIONS:
            source = evidence['dimensions'][d]
            hard = evidence['hard_fail'][d]
            missing = source['status'] == 'NotEvaluated' or bool(source.get('missing_required_evidence'))
            # Diagnostic robustness: any utility verdict flip under this variant's utility rule.
            if d == 'robustness' and source.get('stressors'):
                baseline = dimensions['utility']['status']
                stress = []
                for row in source['stressors']:
                    value = row['utility']
                    collapsed = any('collapsed' in s for s in value.get('weaknesses', []))
                    stressed = {**signals_all, 'utility': utility_signals(value['metrics'])}
                    status, _ = self.status('utility', stressed['utility'], alpha,
                                            missing=value['status'] == 'NotEvaluated', hard_fail=collapsed,
                                            all_signals=stressed)
                    stress.append(status)
                hard = any(s != baseline and 'NotEvaluated' not in (s, baseline) for s in stress)
                missing = missing or baseline == 'NotEvaluated' or 'NotEvaluated' in stress
            decision = decisions[family_of(d)]
            status, audit = self.status(d, signals_all[d], alpha, missing=missing, hard_fail=hard,
                                        all_signals=signals_all, decision=decision)
            if d == 'robustness' and (dimensions['utility']['status'] == 'NotEvaluated'
                                       or source.get('constant_predictor', False)):
                status, hard = 'NotEvaluated', False
                audit['eligibility_reason'] = 'Utility NotEvaluated or constant primary predictor.'
            calibrated_signals = {}
            for (dim, metric), z in zip(decision['columns'], decision['candidate_z']):
                if dim != d: continue
                flag = bool(decision['threshold'] is not None and z > decision['threshold'])
                calibrated_signals[metric] = {'estimate': signals_all[d][metric]['estimate'], 'z_pooled': z,
                                              'family_threshold_z': decision['threshold'], 'flag': flag,
                                              'status': 'Fail' if flag else 'Pass'}
            for c in self.model['columns']:
                if c['dimension'] == d and c['metric'] not in calibrated_signals:
                    calibrated_signals[c['metric']] = {'estimate': None, 'status': 'NotEvaluated', 'flag': False}
            dimensions[d] = {'dimension': d, 'status': status, 'calibration': audit,
                             'role': 'gate' if d in GATE_DIMENSIONS else 'diagnostic',
                             'hard_failure': hard, 'metrics': copy.deepcopy(source['metrics']),
                             'metric_rule_scope': 'Original metric rules retained for provenance; calibrated decisions use calibrated_signals.',
                             'calibrated_signals': calibrated_signals}
        return {'dimensions': dimensions, 'gate_dimensions': list(GATE_DIMENSIONS),
                'gate_family_pvalue': decisions['gate']['pvalue'],
                'gate': Gate.aggregate(dimensions[d]['status'] for d in GATE_DIMENSIONS)}


def self_test() -> None:
    fit, audit = NullCalibrator(CalibrationConfig()).assignment()
    assert len(fit) == len(audit) == 50 and not set(fit) & set(audit)
    assert rejection_rank(50, .05) == 2 and rejection_rank(50, .025) == 1 and rejection_rank(50, .10) == 5
    assert rejection_rank(3, .05) == 0
    assert binomial_admission(5, 50, .05, .05)[1] and not binomial_admission(6, 50, .05, .05)[1]
    assert binomial_admission(0, 50, .05, .05) == (1., True)
    rng = np.random.default_rng(1)
    fit_rows = rng.normal(size=(50, 3))
    assert not conformal_decision(fit_rows, np.zeros(3), .05, 1e-8)['candidate_max'] > 10
    assert conformal_decision(fit_rows, np.array([0, 0, 50.]), .05, 1e-8)['pvalue'] == 1/51
