"""Additive v0.4 fixes; supplied reference statistics remain unchanged."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from syntrustbench.privacy import _tpr_at_fpr
from syntrustbench.stats import categorical_values, metric, percentile_interval, safe_auroc
from syntrustbench.validation import canonical_row_keys, _coerce_frame, _map_target
from syntrustbench.config import load_config
from syntrustbench.models import Submission

from .config import ExperimentConfig, DIMENSIONS, GATE_DIMENSIONS, seed_for
from .core import Gate, available_number, read_json, write_json
from .data import read_analytic
from .reference import ReferenceContext, ReferenceFidelity, ReferenceUtility, _retention, _retention_verdict
from .results import DimensionEvaluator, DimensionResult
from .cached_privacy import evaluate_privacy, build_real_cache


def result(dimension: str, metrics: dict, statuses: list[str], **details) -> DimensionResult:
    return DimensionResult.from_dict({'dimension': dimension, 'status': Gate.aggregate(statuses),
                                     'metrics': metrics, 'weaknesses': [], 'warnings': [], **details})


def signal(value, *, higher: bool = True, reason: str = '') -> dict:
    return {'estimate': float(value) if available_number(value) else None,
            'higher_is_worse': higher, 'status': 'Available' if available_number(value) else 'NotEvaluated',
            'note': reason if not available_number(value) else ''}


@dataclass
class PrivacyV04(DimensionEvaluator):
    cfg: ExperimentConfig
    train: pd.DataFrame
    test: pd.DataFrame
    seed: int
    smoke: bool = False
    real_cache: dict | None = None

    def evaluate(self, synthetic: pd.DataFrame) -> DimensionResult:
        config = load_config(self.cfg.benchmark(smoke=self.smoke))
        frames, warnings = [], []
        for label, frame, require_two in [('real_train', self.train, True), ('real_test', self.test, True), ('synthetic', synthetic, False)]:
            value, notes = _coerce_frame(frame, config, label)
            value, targets = _map_target(value, config, label, require_two_classes=require_two)
            frames.append(value); warnings.extend(notes + targets)
        # RoleAssigner already proved patient disjointness. Analytic-row coincidences
        # must remain observable for this test, unlike frozen v0.3's overlap check.
        submission = Submission(*frames, config=config, input_hashes={}, warnings=warnings)
        cache = self.real_cache if self.real_cache is not None else build_real_cache(submission.real_train, config, seed_for(self.seed, 'real_floor'))
        raw = evaluate_privacy(submission, np.random.default_rng(self.seed), cache)
        internal = raw.pop('_internal')
        metrics = raw['metrics']
        keys = set(canonical_row_keys(submission.real_train, submission.config))
        coincidences = np.isin(canonical_row_keys(submission.real_test, submission.config), list(keys))
        copies = internal['duplicate_mask']
        p0 = float(coincidences.mean())
        pvalue = float(binomtest(int(copies.sum()), len(copies), p=p0, alternative='greater').pvalue)
        fail = pvalue < self.cfg.privacy_binomial_alpha
        duplicate = metrics['exact_training_duplicate_rate']
        duplicate.update(flag=fail, status='Fail' if fail else 'Pass',
                         threshold=f'one-sided binomial p < {self.cfg.privacy_binomial_alpha}',
                         note=f'Exact-match count against plug-in E-in-T coincidence rate {p0:.8g}; all T rows used.')
        metrics['exact_match_binomial_pvalue'] = metric(pvalue, higher_is_worse=False,
                                                       flag=fail, note='Plug-in null; uncertainty in p0 is not integrated.')
        metrics['real_real_coincidence_rate'] = metric(p0, higher_is_worse=True,
                                                      note=f'{int(coincidences.sum())}/{len(coincidences)} held-out rows found in T.')
        metrics['exact_match_excess'] = metric(float(copies.mean())-p0, higher_is_worse=True)
        labels, scores = internal['attack_labels'], internal['attack_scores']
        for rate, key, minimum in [(.001, 'distance_membership_attack_tpr_at_0_1pct_fpr', 1000),
                                    (.01, 'distance_membership_attack_tpr_at_1pct_fpr', 100)]:
            usable = labels is not None and scores is not None and int(np.sum(labels == 0)) >= minimum
            value = _tpr_at_fpr(labels, scores, rate) if usable else None
            metrics[key] = metric(value, higher_is_worse=True, status='Available' if usable else 'NotEvaluated',
                                  note=f'Exploratory weak lower-bound attack; requires >= {minimum} nonmembers for FPR resolution; no low-FPR CI.')
            metrics[key]['required'] = False
        raw['weaknesses'] = [s for s in raw['weaknesses'] if 'copied training rows' not in s]
        if fail:
            raw['weaknesses'].append('Exact matches exceed the empirical real–real coincidence null.')
        required = ('exact_training_duplicate_rate', 'share_closer_than_real_real_floor', 'distance_membership_attack_auc')
        raw['status'] = Gate.aggregate([metrics[k]['status'] for k in required])
        raw['failure_flags'] = [f'privacy:{k}' for k, v in metrics.items() if v.get('flag')]
        raw['signals'] = {k: signal(metrics[k]['estimate'], reason=metrics[k].get('note', '')) for k in
                          ('exact_match_excess', 'share_closer_than_real_real_floor', 'distance_membership_attack_auc')}
        return DimensionResult.from_dict(raw)


@dataclass
class EquityV04(DimensionEvaluator):
    cfg: ExperimentConfig
    context: ReferenceContext
    seed: int

    def evaluate(self, synthetic: pd.DataFrame) -> DimensionResult:
        ctx, rng = self.context, np.random.default_rng(self.seed)
        pred = ctx._predict('random_forest', synthetic)
        base, y = ctx.trtr['random_forest'], ctx.y_test
        groups, statuses, retentions, gaps = [], [], [], []
        for attribute in self.cfg.data.protected:
            test_values = categorical_values(ctx.test[attribute])
            ref_mass = categorical_values(ctx.reference[attribute]).value_counts(normalize=True)
            syn_mass = categorical_values(synthetic[attribute]).value_counts(normalize=True)
            levels = sorted(set(ref_mass.index) | set(syn_mass.index))
            gaps.extend(abs(float(ref_mass.get(level, 0))-float(syn_mass.get(level, 0))) for level in levels)
            for level in sorted(test_values.unique()):
                rows = np.flatnonzero((test_values == level).to_numpy())
                positives, negatives = int(y[rows].sum()), int(len(rows)-y[rows].sum())
                eligible = len(rows) >= self.cfg.min_subgroup_n and min(positives, negatives) >= self.cfg.min_subgroup_outcomes
                record = {'attribute': attribute, 'level': level, 'n': len(rows), 'positive': positives,
                          'negative': negatives, 'eligible': eligible, 'status': 'NotEvaluated',
                          'estimate': None, 'ci_low': None, 'ci_high': None,
                          'note': 'Below prespecified subgroup support; excluded from decision-eligible group set.'}
                if eligible:
                    point = _retention(y[rows], base[rows], pred[rows])
                    base_ci = percentile_interval(safe_auroc(y[rows], base[rows]),
                        lambda r: safe_auroc(y[rows][r], base[rows][r]), len(rows), rng, ctx.iterations, .95)
                    low, high = percentile_interval(point,
                        lambda r: _retention(y[rows][r], base[rows][r], pred[rows][r]), len(rows), rng, ctx.iterations, .95)
                    verdict = _retention_verdict(point, low, high, base_ci[0])
                    record.update(estimate=point, ci_low=low, ci_high=high, status=verdict,
                                  note='Paired subgroup E-row bootstrap; chance-corrected against subgroup TRTR.',
                                  subgroup_trtr_auc=safe_auroc(y[rows], base[rows]), subgroup_trtr_ci=base_ci)
                    statuses.append(verdict)
                    retentions.append(point if verdict != 'NotEvaluated' else None)
                groups.append(record)
        gap = max(gaps) if gaps else None
        worst = min(retentions) if retentions and all(available_number(v) for v in retentions) else None
        metrics = {
            'maximum_representation_gap': metric(gap, higher_is_worse=True, flag=available_number(gap) and gap > .1,
                                                  threshold='> 0.10 -> Conditional'),
            'worst_group_retention': metric(worst, higher_is_worse=False,
                                            status='Available' if worst is not None else 'NotEvaluated',
                                            note='Minimum over decision-eligible groups; unavailable if any eligible group has weak TRTR.'),
        }
        if not statuses:
            statuses = ['NotEvaluated']
        statuses.append('NotEvaluated' if gap is None else ('Conditional' if gap > .1 else 'Pass'))
        return result('equity', metrics, statuses, subgroup_results=groups,
                      eligible_groups=sum(g['eligible'] for g in groups), total_groups=len(groups),
                      signals={'maximum_representation_gap': signal(gap),
                               'worst_group_retention': signal(worst, higher=False, reason='No fully evaluable eligible subgroup set.')})


@dataclass
class RobustnessV04(DimensionEvaluator):
    cfg: ExperimentConfig
    context: ReferenceContext
    utility: dict
    utility_rng_state: dict
    seed: int
    smoke: bool = False

    @staticmethod
    def flips(baseline: str, stressed: str) -> bool:
        return baseline != stressed and baseline != 'NotEvaluated' and stressed != 'NotEvaluated'

    def _utility(self, table: pd.DataFrame, model_seed: int) -> tuple[dict, float]:
        ctx = copy.copy(self.context)
        ctx.rng = np.random.default_rng(); ctx.rng.bit_generator.state = copy.deepcopy(self.utility_rng_state)
        ctx.seed = model_seed
        value = ReferenceUtility(ctx).evaluate(table).to_dict()
        from .calibration import utility_signals
        missing = [k for k, v in utility_signals(value['metrics']).items() if not available_number(v['estimate'])]
        value['raw_status'] = value['status']
        value['missing_required_evidence'] = missing
        if missing:
            value['status'] = Gate.aggregate([value['status'], 'NotEvaluated'])
        auc = safe_auroc(ctx.y_test, ctx._predict('random_forest', table))
        return value, auc

    def evaluate(self, synthetic: pd.DataFrame) -> DimensionResult:
        ctx, baseline = self.context, self.utility['status']
        pred = ctx._predict('random_forest', synthetic)
        if baseline == 'NotEvaluated' or np.allclose(pred, pred[0]):
            reason = 'Utility NotEvaluated or constant TSTR random-forest predictor.'
            names = ('downstream_model_seed_auroc_cv', 'tstr_bootstrap_relative_ci_width',
                     'tstr_auroc_drop_at_maximum_missingness', 'training_size_auroc_spread')
            return result('robustness', {k: metric(None, higher_is_worse=True, status='NotEvaluated', note=reason) for k in names},
                          ['NotEvaluated'], stressors=[], constant_predictor=bool(np.allclose(pred, pred[0])),
                          signals={k: signal(None, reason=reason) for k in names}, verdict_flip=False)
        small = self.smoke or self.cfg.execution.smoke_mode
        seeds = self.cfg.model_seeds[:2] if small else self.cfg.model_seeds
        records, seed_aucs, missing_aucs, size_aucs = [], [], {}, []

        def score(label, frame, model_seed):
            value, auc = self._utility(frame, model_seed)
            records.append({'stressor': label, 'utility_status': value['status'], 'tstr_auroc': auc,
                            'verdict_flip': self.flips(baseline, value['status']), 'utility': value})
            return auc

        for model_seed in seeds:
            seed_aucs.append(score(f'model_seed/{model_seed}', synthetic, model_seed))
        repeats = 2 if small else self.cfg.missingness_repeats
        for rate in self.cfg.missingness_rates:
            missing_aucs[rate] = []
            for repeat in range(repeats):
                frame = synthetic.copy(deep=True)
                for column in ctx.config.feature_columns:
                    rng = np.random.default_rng(seed_for(self.seed, 'stress_missing', rate, repeat, column))
                    frame.loc[rng.random(len(frame)) < rate, column] = np.nan
                missing_aucs[rate].append(score(f'missingness/{rate}/{repeat}', frame, ctx.seed))
        for fraction in self.cfg.training_fractions:
            frame = synthetic if fraction == 1 else synthetic.sample(frac=fraction, random_state=seed_for(self.seed, 'training_size'))
            size_aucs.append(score(f'training_size/{fraction}', frame, ctx.seed))
        auc = safe_auroc(ctx.y_test, pred)
        ci = percentile_interval(auc, lambda r: safe_auroc(ctx.y_test[r], pred[r]), len(pred),
                                 np.random.default_rng(seed_for(self.seed, 'auc_ci')), ctx.iterations, .95)
        finite = lambda values: all(available_number(v) for v in values)
        cv = float(np.std(seed_aucs, ddof=1)/abs(np.mean(seed_aucs))) if finite(seed_aucs) and np.mean(seed_aucs) != 0 else None
        width = (ci[1]-ci[0])/auc if all(available_number(v) for v in ci) and auc > 0 else None
        maximum = missing_aucs[max(missing_aucs)]
        drop = auc-float(np.mean(maximum)) if finite(maximum) else None
        spread = max(size_aucs)-min(size_aucs) if finite(size_aucs) else None
        values = {'downstream_model_seed_auroc_cv': (cv, .05), 'tstr_bootstrap_relative_ci_width': (width, .20),
                  'tstr_auroc_drop_at_maximum_missingness': (drop, .10), 'training_size_auroc_spread': (spread, .10)}
        metrics, statuses = {}, []
        for name, (v, threshold) in values.items():
            status = 'NotEvaluated' if v is None else ('Conditional' if v > threshold else 'Pass')
            metrics[name] = metric(v, higher_is_worse=True, status=status, flag=status == 'Conditional', threshold=f'> {threshold} -> Conditional')
            statuses.append(status)
        flip = any(r['verdict_flip'] for r in records)
        statuses.extend('NotEvaluated' for r in records if r['utility_status'] == 'NotEvaluated')
        statuses.append('Fail' if flip else 'Pass')
        metrics['any_utility_verdict_flip'] = metric(int(flip), higher_is_worse=True, flag=flip,
                                                    threshold='Any evaluable verdict change in either direction -> Fail')
        return result('robustness', metrics, statuses, stressors=records, verdict_flip=flip, constant_predictor=False,
                      baseline_utility_status=baseline, signals={k: signal(v[0], reason='Stressor evidence unavailable.') for k, v in values.items()})


class OfficialProfiler:
    CANONICAL = ('summary.json', 'metrics.csv', 'subgroup_results.csv', 'benchmark_card.yaml',
                 'report.md', 'run_manifest.json', 'execution_log.txt', 'details.json')

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def evaluate(self, roles: dict, synthetic: pd.DataFrame, folder: Path, smoke=False) -> dict:
        import syntrustbench
        # Same n-sized floor and candidate as reference utility; full T remains the v0.4 privacy member set.
        train, test = [read_analytic(roles['paths'][k], self.cfg) for k in ('floor', 'test')]
        try:
            syntrustbench.evaluate(train, test, synthetic, self.cfg.benchmark(smoke=smoke), folder)
        except ValueError as exc:
            # Frozen v0.3 refuses to evaluate when one held-out real row exactly matches a
            # training row. Roles here are patient-disjoint by hashed ID, so this is a
            # real-real coincidence. Record v0.3's refusal as an explicit NotEvaluated
            # outcome (v0.3 itself is not modified); every other error is re-raised.
            if not str(exc).startswith('Real train/test overlap detected'):
                raise
            folder.mkdir(parents=True, exist_ok=True)
            stub = {'version': '0.3.0', 'benchmark_gate': 'NotEvaluated',
                    'dimension_status': {d: 'NotEvaluated' for d in DIMENSIONS},
                    'dimensions': {d: {'status': 'NotEvaluated', 'metrics': {}} for d in DIMENSIONS},
                    'official_stop': str(exc),
                    'official_stop_policy': 'v0.3 exact-row stop on patient-disjoint roles; recorded, not overridden.'}
            write_json(folder / 'official_stop.json', stub)
            return stub
        if not all((folder / p).is_file() for p in self.CANONICAL):
            raise RuntimeError('Official v0.3 did not write all eight canonical files.')
        summary = read_json(folder / 'summary.json')
        if summary['version'] != '0.3.0':
            raise RuntimeError('Unexpected official protocol version.')
        return summary


def build_context(cfg: ExperimentConfig, roles: dict, folder: Path, smoke=False) -> dict:
    frames = [read_analytic(roles['paths'][k], cfg) for k in ('floor', 'reference', 'test')]
    ctx = ReferenceContext(*frames, cfg.benchmark(smoke=smoke))
    members = read_analytic(roles['paths']['train'], cfg)
    members, _ = _coerce_frame(members, ctx.config, 'real_train')
    members, _ = _map_target(members, ctx.config, 'real_train', require_two_classes=True)
    ctx.privacy_real_cache = build_real_cache(members, ctx.config, seed_for(roles['key'], 'privacy_real_floor'))
    path = folder / 'reference_context.pkl'
    with path.open('wb') as handle:
        pickle.dump(ctx, handle, protocol=5)
    write_json(folder / 'reference_context_summary.json', {
        'floor': {k: v for k, v in ctx.floor.items() if k != 'coverage_rows'},
        'reference_calibration': ctx.reference_calibration, 'trtr_auroc_ci': ctx.trtr_auroc_ci,
        'rng_policy': 'Private copy of frozen post-construction state per table; paired utility state for stressors.',
        'rare_categories': 'Top 250 levels per column, others pooled as __OTHER__',
        'confidence_limit': 'Detection bootstrap omits classifier-refit variance.',
        'exact_row_coincidences_with_floor': ctx.exact_row_coincidences,
        'exact_row_policy': 'Recorded, not stopped: roles are patient-disjoint by hashed ID.'})
    return {'context_path': str(path), 'roles': roles}


def reference_signals(f: dict, u: dict, ctx: ReferenceContext) -> dict:
    fm, um = f['metrics'], u['metrics']
    get = lambda m, k, field='estimate': m[k].get(field)
    fs = {k: signal(get(fm, k)) for k in ('mass_weighted_coverage_deficit',
        'mean_normalized_wasserstein_excess', 'mean_jensen_shannon_divergence_excess',
        'association_matrix_error_excess', 'clinical_constraint_violation_rate')}
    fs['detection_auc_excess'] = signal(get(fm, 'detection_auc')-ctx.floor['detection_auc'])
    us = {}
    for probe in ctx.PROBES:
        key = f'synthetic_retention_{probe}'
        m = um[key]
        us[key] = signal(m['ci_high'] if m['status'] != 'NotEvaluated' else None, higher=False,
                         reason='Retention upper CI unavailable or weak real-trained baseline.')
    for key in ('relative_calibration_in_the_large', 'relative_calibration_slope'):
        us[key + '_high'] = signal(get(um, key))
        us[key + '_low'] = signal(get(um, key), higher=False)
    return {'fidelity': fs, 'utility': us}


def evaluate_v04(cfg: ExperimentConfig, context_path: str, roles: dict,
                 synthetic: pd.DataFrame, key: str, smoke=False) -> dict:
    # Only trusted, ledger-hash-verified local context artifacts are loaded.
    with open(context_path, 'rb') as handle:
        ctx = pickle.load(handle)
    f = ReferenceFidelity(ctx).evaluate(synthetic).to_dict()
    utility_rng = copy.deepcopy(ctx.rng.bit_generator.state)
    u = ReferenceUtility(ctx).evaluate(synthetic).to_dict()
    raw_reference = {'fidelity': copy.deepcopy(f), 'utility': copy.deepcopy(u)}
    signals = reference_signals(f, u, ctx)
    # Guard missing evidence without changing any statistic in the supplied evaluators.
    for dimension, value in [('fidelity', f), ('utility', u)]:
        unavailable = [k for k, v in signals[dimension].items() if v['status'] == 'NotEvaluated']
        if dimension == 'fidelity':
            unavailable += [f'column:{c}' for c in cfg.data.continuous
                            if any(not pd.to_numeric(frame[c], errors='coerce').notna().any()
                                   for frame in (synthetic, ctx.reference, ctx.train))]
        if unavailable:
            value['status'] = Gate.aggregate([value['status'], 'NotEvaluated'])
        value['missing_required_evidence'] = unavailable
        for name, m in value['metrics'].items():
            estimate = m.get('estimate')
            if estimate is None or isinstance(estimate, (int,float,np.number)) and not available_number(estimate):
                m['status'] = 'NotEvaluated'
                m['availability_reason'] = 'The preserved reference calculation returned no finite estimate.'
    members, test = [read_analytic(roles['paths'][k], cfg) for k in ('train', 'test')]
    p = PrivacyV04(cfg, members, test, seed_for(key, 'privacy'), smoke,
                   getattr(ctx, 'privacy_real_cache', None)).evaluate(synthetic).to_dict()
    e = EquityV04(cfg, ctx, seed_for(key, 'equity')).evaluate(synthetic).to_dict()
    r = RobustnessV04(cfg, ctx, u, utility_rng, seed_for(key, 'robustness'), smoke).evaluate(synthetic).to_dict()
    dimensions = dict(zip(DIMENSIONS, (f, u, p, e, r)))
    for dimension in ('privacy', 'equity', 'robustness'):
        signals[dimension] = dimensions[dimension].pop('signals')
    hard_fail = {'fidelity': False, 'utility': synthetic[cfg.data.target].nunique() < 2,
                 'privacy': False, 'equity': False, 'robustness': r.get('verdict_flip', False)}
    return {'dimensions': dimensions, 'raw_reference': raw_reference, 'signals': signals,
            'hard_fail': hard_fail, 'gate_dimensions': list(GATE_DIMENSIONS),
            'gate': Gate.aggregate(dimensions[d]['status'] for d in GATE_DIMENSIONS),
            'scope': ('provisional v0.4.1; gate over fidelity, utility and privacy; equity and robustness '
                      'are non-gating diagnostics; calibration is a separate saved-evidence wrapper')}


def self_test() -> None:
    assert RobustnessV04.flips('Pass', 'Fail') and RobustnessV04.flips('Fail', 'Pass')
    assert RobustnessV04.flips('Pass', 'Conditional')
    assert not RobustnessV04.flips('Pass', 'NotEvaluated')
