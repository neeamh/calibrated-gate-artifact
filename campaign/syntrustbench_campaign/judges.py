"""Non-compensatory and compensatory judges on identical saved evidence."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import math
import numpy as np
import pandas as pd

from .calibration import CalibratedEvaluator, rejection_rank, standardize
from .config import GATE_DIMENSIONS, ExperimentConfig, JudgeConfig
from .core import available_number


COMPENSATORY = ('Mean', 'Geometric', 'Harmonic', 'SDMetrics')
JUDGE_NAMES = ('Gate',) + COMPENSATORY
# What each judge stands for in published practice; cited in the paper's judge table.
JUDGE_PROVENANCE = {
    'Gate': 'Non-compensatory: Pass only if fidelity, utility and privacy each pass (this work; '
            'a fixed-threshold version appears in prior work cited in the paper).',
    'Mean': 'Equal-weight average of the three dimension scores: TabSynDex (Chundawat et al. 2022) and the '
            'default equal weights of the weighted rank sum in Yan et al. 2022.',
    'Geometric': 'Geometric mean, argued to act as an "and" operation: trust index of Belgodere et al. '
                 '(equal weights).',
    'Harmonic': 'Harmonic mean: the most low-score-sensitive of the classical means; strongest compensatory rival.',
    'SDMetrics': 'Native SDMetrics QualityReport overall score (average of property scores), run on each table; '
                 'fidelity-only, as shipped.'}


@dataclass(frozen=True)
class JudgeScore:
    judge: str
    status: str
    score: float | None
    reason: str = ''


class Judge(ABC):
    @abstractmethod
    def score(self, evidence: dict) -> JudgeScore:
        raise NotImplementedError


class GateJudge(Judge):
    def score(self, evidence):
        status = evidence['gate']
        return JudgeScore('Gate', status, None, 'Pass only when every required dimension passes; no ranking score.')


@dataclass
class MeanJudge(Judge):
    name: str = 'Mean'

    def combine(self, values):
        return float(np.mean(values))

    def score(self, evidence):
        values = [evidence['dimension_scores'].get(d) for d in GATE_DIMENSIONS]
        if not all(available_number(v) for v in values):
            return JudgeScore(self.name, 'NotEvaluated', None, 'All three gating dimension scores are required; no missing-value omission.')
        return JudgeScore(self.name, 'Available', self.combine(values))


class GeometricJudge(MeanJudge):
    def __init__(self): super().__init__('Geometric')
    def combine(self, values):
        return 0. if min(values) == 0 else float(math.exp(sum(math.log(v) for v in values)/len(values)))


class HarmonicJudge(MeanJudge):
    def __init__(self): super().__init__('Harmonic')
    def combine(self, values):
        return 0. if min(values) == 0 else float(len(values)/sum(1/v for v in values))


class SDMetricsJudge(Judge):
    def score(self, evidence):
        item = evidence['sdmetrics']
        if not available_number(item.get('score')):
            return JudgeScore('SDMetrics', 'NotEvaluated', None, item.get('reason', 'Quality Report unavailable.'))
        return JudgeScore('SDMetrics', 'Available', item['score'], 'External fidelity-only headline, as shipped; not a multi-dimension safety score.')


def quality_report(reference: pd.DataFrame, synthetic: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    from sdmetrics.reports.single_table import QualityReport
    metadata = {'columns': {c: {'sdtype': 'numerical' if c in cfg.data.continuous else 'categorical'}
                            for c in cfg.data.columns}}
    report = QualityReport()
    report.generate(reference, synthetic, metadata, verbose=False)
    properties = report.get_properties().to_dict(orient='records')
    omitted = []
    for item in properties:
        valid = available_number(item.get('Score'))
        item['status'] = 'Available' if valid else 'NotEvaluated'
        if not valid:
            item['reason'] = 'Native SDMetrics property is nonfinite; overall headline may omit it.'
            omitted.append(item.get('Property', 'unnamed property'))
    score = float(report.get_score())
    return {'score': score if np.isfinite(score) else None, 'properties': properties, 'omitted_properties': omitted,
            'reason': '' if np.isfinite(score) else 'SDMetrics returned a nonfinite score.',
            'native_missingness_policy': 'Native SDMetrics headline retained; omitted property scores recorded explicitly.'}


def dimension_scores(evidence: dict, model: dict, cfg: JudgeConfig) -> tuple[dict, dict]:
    calibrated = CalibratedEvaluator(model)
    scores, notes = {}, {}
    for d in GATE_DIMENSIONS:
        zs, missing = calibrated.z_values(d, evidence['signals'][d])
        if evidence['dimensions'][d]['status'] == 'NotEvaluated':
            missing.append('Required dimension evidence unavailable.')
        if evidence['dimensions'][d].get('missing_required_evidence'):
            missing.extend(evidence['dimensions'][d]['missing_required_evidence'])
        scores[d] = None if missing else float(np.exp(-max(0., max(zs))/cfg.temperature))
        notes[d] = missing
    return scores, notes


def _gate_family(model: dict, evidence: dict):
    """Fit-run and candidate oriented gate signals, or None if any gate signal is missing."""
    cols = [c for c in model['columns'] if c['family'] == 'gate']
    values = [evidence['signals'][c['dimension']].get(c['metric'], {}).get('estimate') for c in cols]
    if not cols or not all(available_number(v) for v in values) or set(model['blocked_dimensions']) & set(GATE_DIMENSIONS):
        return None
    fit = np.array([c['fit_harm'] for c in cols]).T
    return cols, fit, np.array([c['direction']*v for c, v in zip(cols, values)])


def matched_acceptance(judge: Judge, evidence: dict, sdmetrics: dict, model: dict, cfg: ExperimentConfig,
                       alpha: float) -> tuple[bool | None, str]:
    """Accept/reject a table with a compensatory judge calibrated to the gate's honest-data rate.

    Same 50 fit runs, same full-conformal pooling: each run's three dimension scores and
    combined score are computed after pooling the candidate with the fit runs; the table is
    rejected when its combined score is below the floor(alpha(n+1))-th lowest fit-run score.
    SDMetrics scores do not depend on the fit runs, so its threshold is the same order
    statistic of the fit runs' native SDMetrics scores (split conformal).
    """
    if isinstance(judge, GateJudge):
        return None, 'Gate decisions come from the calibrated evaluator.'
    if isinstance(judge, SDMetricsJudge):
        fit = model.get('sdmetrics_fit_scores') or []
        candidate = sdmetrics.get('score')
        if not fit or not all(available_number(v) for v in fit) or not available_number(candidate):
            return None, 'SDMetrics fit-run or candidate score unavailable.'
        fit_scores, cand = np.asarray(fit, dtype=float), float(candidate)
    else:
        if any(evidence['dimensions'][d]['status'] == 'NotEvaluated' or evidence['dimensions'][d].get('missing_required_evidence')
               for d in GATE_DIMENSIONS):
            return None, 'A gating dimension is NotEvaluated.'
        family = _gate_family(model, evidence)
        if family is None:
            return None, 'A gate-family signal is unavailable.'
        cols, fit, candidate = family
        z = standardize(np.vstack([fit, candidate]), model['scale_floor'])
        per_run = np.column_stack([np.exp(-np.maximum(0., z[:, [j for j, c in enumerate(cols) if c['dimension'] == d]].max(axis=1))
                                          / cfg.judges.temperature) for d in GATE_DIMENSIONS])
        combined = np.array([judge.combine(list(row)) for row in per_run])
        fit_scores, cand = combined[:-1], float(combined[-1])
    r = rejection_rank(len(fit_scores), alpha)
    if r < 1:
        return None, 'Too few fit runs to reject at this alpha.'
    threshold = float(np.sort(fit_scores)[r - 1])
    return bool(cand >= threshold), f'matched threshold {threshold:.6g} (rank {r} of {len(fit_scores)})'


def all_judges(cfg: JudgeConfig) -> tuple[Judge, ...]:
    judges = GateJudge(), MeanJudge(), GeometricJudge(), HarmonicJudge(), SDMetricsJudge()
    assert tuple(j.name if hasattr(j, 'name') else type(j).__name__.replace('Judge', '') for j in judges) == JUDGE_NAMES
    return judges


def score_table(profile: dict, model: dict, cfg: ExperimentConfig) -> list[dict]:
    evidence = profile['v04']
    scores, missing = dimension_scores(evidence, model, cfg.judges)
    rows = []
    for alpha in cfg.calibration.target_rates:
        calibrated = CalibratedEvaluator(model).apply(evidence, alpha)
        for variant, evaluated in [('provisional', evidence), ('calibrated', calibrated)]:
            shared = {'gate': evaluated['gate'], 'dimension_scores': scores, 'sdmetrics': profile['sdmetrics']}
            for judge in all_judges(cfg.judges):
                result = asdict(judge.score(shared))
                if variant == 'calibrated':
                    if result['judge'] == 'Gate':
                        matched, why = (None, 'Gate NotEvaluated') if result['status'] == 'NotEvaluated' else (result['status'] == 'Pass', 'calibrated gate')
                    else:
                        matched, why = matched_acceptance(judge, evidence, profile['sdmetrics'], model, cfg, alpha)
                else:
                    matched, why = None, 'Matched-rate decisions are defined for the calibrated variant only.'
                rows.append({**profile['labels'], 'alpha': alpha, 'variant': variant, **result,
                             'accepted_matched': matched, 'matched_note': why,
                             'dimension_scores': scores, 'score_missing': missing})
    return rows


def self_test() -> None:
    e = {'dimension_scores': dict(zip(GATE_DIMENSIONS, [1, 1, 0]))}
    assert abs(MeanJudge().score(e).score - 2/3) < 1e-12
    assert HarmonicJudge().score(e).score == GeometricJudge().score(e).score == 0
    e['dimension_scores'] = dict(zip(GATE_DIMENSIONS, [1, .5, 1]))
    assert MeanJudge().score(e).score > GeometricJudge().score(e).score > HarmonicJudge().score(e).score > .5
    e['dimension_scores']['privacy'] = None
    assert MeanJudge().score(e).status == 'NotEvaluated'
    assert len(all_judges(JudgeConfig())) == len(JUDGE_NAMES) == len(JUDGE_PROVENANCE)
