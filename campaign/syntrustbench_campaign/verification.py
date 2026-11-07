"""Small assertion-based instrument and engineering checks; no paper results."""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import ExperimentConfig, DIMENSIONS, GATE_DIMENSIONS, DIAGNOSTIC_DIMENSIONS, seed_for
from .core import Gate, JobLedger, JobRunner, UnavailableAnalysis, write_json
from .reference import ReferenceContext, ReferenceFidelity, ReferenceUtility
from .evaluation import EquityV04, PrivacyV04, RobustnessV04
from .perturbations import strategy, changed_cells
from .calibration import CalibratedEvaluator, NullCalibrator, conformal_decision, rejection_rank
from .judges import MeanJudge, HarmonicJudge, GeometricJudge, SDMetricsJudge, matched_acceptance


def toy_tables(n=160) -> tuple:
    """Balanced, strongly predictive artificial population; role rows do not match."""
    cfg = ExperimentConfig(bootstrap_iterations=20)
    frames = []
    for split in range(4):
        rng = np.random.default_rng(seed_for('unit_toy', split))
        y = np.tile([0, 1], n//2)
        values = {c: 2 + rng.uniform(0, .01, n) for c in cfg.data.continuous}
        values['number_inpatient'] = 1 + 6*y + rng.uniform(0, .001, n)
        frame = pd.DataFrame(values)
        for c in cfg.data.categorical:
            frame[c] = np.tile(['A','A','B','B'], n//4)
        frame['race'] = np.tile(['AfricanAmerican','AfricanAmerican','Other','Other'], n//4)
        frame['gender'] = np.tile(['Female','Male','Male','Female'], n//4)
        frame[cfg.data.target] = y
        frames.append(frame.sample(frac=1, random_state=split).reset_index(drop=True))
    return cfg, frames


def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): assert_same(x,y)
    elif isinstance(a, (np.ndarray, float, np.floating)):
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,equal_nan=True)
    else: assert a == b


def run_checks(folder: Path) -> dict:
    cfg, frames = toy_tables()
    train, reference, test, positive = frames
    checks = []
    with threadpool_limits(limits=1):
        ctx = ReferenceContext(train, reference, test, cfg.benchmark())
        initial = copy.deepcopy(ctx)
        f = ReferenceFidelity(ctx).evaluate(positive).to_dict()
        utility_state = copy.deepcopy(ctx.rng.bit_generator.state)
        u = ReferenceUtility(ctx).evaluate(positive).to_dict()
        assert f['status'] == 'Pass', f['status']
        assert u['status'] == 'Pass', u['status']
        checks.append('independent artificial positive control passes reference fidelity and utility')
        collapse = positive.assign(**{cfg.data.target: 0})
        collapsed = ReferenceUtility(copy.deepcopy(initial)).evaluate(collapse).to_dict()
        assert collapsed['status'] == 'Fail'
        robust = RobustnessV04(cfg, initial, collapsed, utility_state, 7).evaluate(collapse)
        assert robust.status == 'NotEvaluated'
        checks.append('collapsed synthetic outcome fails utility; constant predictor robustness NotEvaluated')
        assert RobustnessV04.flips('Fail','Pass') and RobustnessV04.flips('Conditional','Fail')
        assert not RobustnessV04.flips('NotEvaluated','Pass')
        checks.append('robustness flips detected in both directions; unavailable is not a flip')
        small_group_cfg = replace(cfg, min_subgroup_n=10000)
        assert EquityV04(small_group_cfg, initial, 9).evaluate(positive).status == 'NotEvaluated'
        checks.append('no eligible subgroups yields equity NotEvaluated')
        # Nonzero real–real coincidences must be accepted for the new privacy null.
        coincident_test = pd.concat([train.iloc[:40], test.iloc[40:]], ignore_index=True)
        privacy = PrivacyV04(cfg, train, coincident_test, 10).evaluate(train.copy()).to_dict()
        assert privacy['metrics']['real_real_coincidence_rate']['estimate'] == .25
        assert privacy['metrics']['exact_match_binomial_pvalue']['estimate'] < .05
        assert privacy['status'] == 'Fail'
        checks.append('binomial privacy correction handles nonzero coincidence floor and detects copied training data')
        source = Path(__file__).with_name('original_reference.py.txt').read_text()
        if hashlib.sha256(source.encode()).hexdigest() != '3ebc21f477459ac08f9d8c6d0dfdc4ae72e40530b9d62e9311e177a65cf6b33f':
            raise RuntimeError('Supplied reference regression fixture changed.')
        module = types.ModuleType('syntrustbench._v04_original_test')
        module.__package__ = 'syntrustbench'; sys.modules[module.__name__] = module
        exec(compile(source, '<supplied_reference>', 'exec'), module.__dict__)
        old = module.ReferenceContext(train, reference, test, cfg.benchmark())
        assert_same(old.floor, initial.floor)
        assert_same(old.fidelity(positive), f)
        assert_same(old.utility(positive), u)
        assert_same(old.rng.bit_generator.state, ctx.rng.bit_generator.state)
        checks.append('original reference estimates, intervals, flags, notes and RNG state match (1e-12 absolute tolerance)')
    for family, doses in cfg.injections.doses:
        transform = strategy(family, cfg, train)
        before = set()
        for dose in (0.,)+doses:
            candidate, audit = transform.apply(positive, dose, seed_for('unit',family))
            changed = changed_cells(positive, candidate)
            assert before <= changed
            before = changed
            if dose == 0:
                assert positive.to_csv(index=False) == candidate.to_csv(index=False)
            if family == 'feature_shuffle':
                pd.testing.assert_series_equal(positive[cfg.injections.shuffle_feature].value_counts(dropna=False).sort_index(),
                                               candidate[cfg.injections.shuffle_feature].value_counts(dropna=False).sort_index())
        checks.append(f'{family}: dose-zero identity and nested changed cells')
    assert Gate.aggregate(['Pass','NotEvaluated','Fail']) == 'Fail'
    assert Gate.aggregate(['Pass','Conditional','NotEvaluated']) == 'NotEvaluated'
    evidence = {'dimension_scores': dict.fromkeys(GATE_DIMENSIONS, 1.)}
    evidence['dimension_scores']['privacy'] = None
    assert MeanJudge().score(evidence).score is None
    evidence['dimension_scores']['privacy'] = 0.
    assert HarmonicJudge().score(evidence).score == GeometricJudge().score(evidence).score == 0.
    checks.append('gate precedence and missing-value propagation in compensatory judges')
    calibrator = NullCalibrator(cfg.calibration)
    synthetic_nulls = {i: {'signals': {d: {'x': {'estimate': float(i), 'higher_is_worse': True}} for d in DIMENSIONS}}
                       for i in range(calibrator.counts(smoke=True)[0])}
    fit, audit = calibrator.assignment(smoke=True)
    m1 = calibrator.fit(synthetic_nulls, smoke=True)
    for i in audit:
        for d in DIMENSIONS: synthetic_nulls[i]['signals'][d]['x']['estimate'] = 1e9
    assert m1 == calibrator.fit(synthetic_nulls, smoke=True)
    checks.append('held-out audit values cannot affect fitted calibration thresholds')
    fixture = {'dimensions':{d:{'status':'Pass','metrics':{}} for d in DIMENSIONS},
               'signals':copy.deepcopy(synthetic_nulls[fit[0]]['signals']),
               'hard_fail':dict.fromkeys(DIMENSIONS,False)}
    fixture['hard_fail']['utility'] = True
    fixture['signals']['fidelity']['x']['estimate'] = None
    wrapped = CalibratedEvaluator(m1).apply(fixture,.05)
    assert wrapped['dimensions']['utility']['status'] == 'Fail'
    assert wrapped['dimensions']['fidelity']['status'] == 'NotEvaluated'
    checks.append('calibration preserves structural failures and never promotes missing evidence to Pass')
    fixture['hard_fail']['utility'] = False
    fixture['signals']['utility']['x']['estimate'] = None
    fixture['signals']['robustness']['x']['estimate'] = 1e9
    assert CalibratedEvaluator(m1).apply(fixture,.05)['dimensions']['robustness']['status'] == 'NotEvaluated'
    fixture['signals']['utility']['x']['estimate'] = 0.
    fixture['dimensions']['robustness']['constant_predictor'] = True
    assert CalibratedEvaluator(m1).apply(fixture,.05)['dimensions']['robustness']['status'] == 'NotEvaluated'
    checks.append('calibrated robustness stays NotEvaluated for unavailable utility or constant predictors')
    checks.extend(gate_scope_checks(cfg))
    checks.append(conformal_size_check())
    checks.append(matched_judge_size_check(cfg))
    ledger = JobLedger(folder / 'ledger_test', 'fixture')
    runner = JobRunner(ledger)
    def artifact(where):
        path = write_json(where/'artifact.json', {'answer':1})
        return {'path':path}
    first = runner.run('test/artifact', artifact)
    assert runner.run('test/artifact', artifact) == first
    Path(first['path']).write_text('tampered')
    caught = False
    try: ledger.result('test/artifact')
    except RuntimeError: caught = True
    assert caught
    checks.append('resumption verifies artifact hashes and rejects tampering')
    def unavailable(where): raise UnavailableAnalysis('declared fixture unavailability')
    assert runner.run('test/unavailable', unavailable) is None
    assert ledger.state('test/unavailable')['state'] == 'unavailable'
    saved_unavailable = ledger.state('test/unavailable')
    assert runner.run('test/unavailable', unavailable) is None
    assert ledger.state('test/unavailable') == saved_unavailable
    def failure(where): raise ArithmeticError('deliberate failure-ledger fixture')
    caught = False
    try: runner.run('test/failure', failure)
    except ArithmeticError: caught = True
    assert caught and ledger.state('test/failure')['state'] == 'failed'
    assert 'ArithmeticError' in (folder/'ledger_test/failures.jsonl').read_text()
    checks.append('declared unavailability is explicit; unexpected failures are logged and re-raised')
    from .dp_generators import DPAIMGenerator, DPMSTGenerator, PublicDiscretizer
    from .tabddpm import TabDDPMEnvironment
    dp_cfg = replace(cfg, execution=replace(cfg.execution, smoke_mode=True))
    public = PublicDiscretizer(dp_cfg)
    before = public.manifest()
    extremes = train.copy()
    for col in dp_cfg.data.continuous:
        extremes[col] = np.linspace(-1e8,1e9,len(extremes))
    extremes['race'] = 'never_observed_publicly'
    public.encode(train)
    public.encode(extremes)
    assert before == public.manifest() == PublicDiscretizer(dp_cfg).manifest()
    assert public.transformer().cardinality == public.cards
    checks.append('DP bins and full categorical domains remain identical when private extremes/levels change')
    dp = DPAIMGenerator(dp_cfg,17).fit(train.iloc[:64])
    sample = dp.sample(64,1234)
    assert list(sample.columns) == list(train.columns) and len(sample) == 64
    assert all(pd.api.types.is_integer_dtype(sample[c]) and sample[c].ge(0).all() for c in dp_cfg.data.continuous)
    assert sample['time_in_hospital'].between(1,14).all()
    assert sample[dp_cfg.data.target].isin([0,1]).all()
    pd.testing.assert_frame_equal(sample,dp.sample(64,1234))
    assert dp.provenance['formal_privacy']['delta'] == 1/(10*64)
    assert dp.provenance['public_preprocessing']['preprocessing_epsilon'] == 0
    checks.append('DPAIM actual fit/sample: schema, row count, integer constraints, default delta and deterministic same-fit sampling')
    dp.save(folder/'dp_test_model.pkl')
    import cloudpickle
    with (folder/'dp_test_model.pkl').open('rb') as handle:
        dp.model = cloudpickle.load(handle)['model']
    pd.testing.assert_frame_equal(sample,dp.sample(64,1234))
    checks.append('DPAIM model persistence preserves deterministic sampling without replaying private fit noise')
    if cfg.generators.enable_dpmst:
        mst = DPMSTGenerator(dp_cfg,17).fit(train.iloc[:64])
        out = mst.sample(64,1234)
        assert len(out) == 64 and list(out.columns) == list(train.columns)
        pd.testing.assert_frame_equal(out,mst.sample(64,1234))
        checks.append('Optional DPMST actual fit/sample preserves schema and same-fit sampling seeds')
    absent_cfg = replace(cfg, generators=replace(cfg.generators,tabddpm_python='__missing_python310_test__'),
                         execution=replace(cfg.execution,output_root=str(folder/'optional_environment_test')))
    caught = False
    try:
        TabDDPMEnvironment(absent_cfg).ensure()
    except UnavailableAnalysis as exc:
        caught = 'FileNotFoundError' in str(exc)
    assert caught
    checks.append('Missing optional TabDDPM Python environment yields explicit Unavailable')
    return {'passed':len(checks), 'checks':checks, 'scope':'artificial test data only'}


def honest_signals(rng, n: int) -> np.ndarray:
    """Oriented honest signals: one spiky zero-inflated column, like coverage deficit."""
    spiky = np.where(rng.uniform(size=n) < .7, 0., rng.lognormal(-7, 1.2, n))
    return np.column_stack([spiky, rng.normal(size=n), rng.normal(size=n), rng.standard_t(3, n)])


def conformal_size_check() -> str:
    """Marginal honest-data rejection of the full-conformal rule stays at r/(n+1)."""
    rng = np.random.default_rng(seed_for('conformal_size_check'))
    rejections, trials = 0, 0
    for _ in range(40):
        fit = honest_signals(rng, 50)
        for row in honest_signals(rng, 100):
            decision = conformal_decision(fit, row, .05, 1e-8)
            rejections += decision['threshold'] is not None and decision['candidate_max'] > decision['threshold']
            trials += 1
    rate, nominal = rejections/trials, rejection_rank(50, .05)/51
    assert abs(rate - nominal) < .015, (rate, nominal)
    return f'full-conformal honest rejection {rate:.3f} vs nominal {nominal:.3f} with a spiky zero-inflated signal'


def matched_judge_size_check(cfg) -> str:
    """Compensatory judges calibrated by the matched rule reject honest tables at r/(n+1)."""
    rng = np.random.default_rng(seed_for('matched_judge_size_check'))
    calibrator = NullCalibrator(cfg.calibration)
    count, _ = calibrator.counts()
    def draw():
        return {'signals': {d: {f'{d}_{k}': {'estimate': float(rng.standard_t(4)), 'higher_is_worse': True} for k in range(2)}
                            for d in DIMENSIONS},
                'dimensions': {d: {'status': 'Pass', 'metrics': {}} for d in DIMENSIONS}}
    rates = {}
    for judge in (MeanJudge(), GeometricJudge(), HarmonicJudge(), SDMetricsJudge()):
        rejected = trials = 0
        for _ in range(12):
            nulls = {i: draw() for i in range(count)}
            sdm = {i: float(rng.uniform()) for i in range(count)}
            model = calibrator.fit(nulls, sdmetrics_scores=sdm)
            for _ in range(60):
                accepted, _ = matched_acceptance(judge, draw(), {'score': float(rng.uniform())}, model, cfg, .05)
                assert accepted is not None
                rejected += not accepted; trials += 1
        rates[type(judge).__name__] = rejected/trials
    nominal = rejection_rank(cfg.calibration.fit_runs, .05)/(cfg.calibration.fit_runs+1)
    assert all(abs(v - nominal) < .03 for v in rates.values()), (rates, nominal)
    return 'matched compensatory judges reject honest tables at the nominal rate: ' + ', '.join(f'{k} {v:.3f}' for k, v in rates.items())


def gate_scope_checks(cfg) -> list[str]:
    """The gate decides on the gating dimensions only; diagnostics never change it."""
    rng = np.random.default_rng(seed_for('gate_scope_checks'))
    calibrator = NullCalibrator(cfg.calibration)
    count, _ = calibrator.counts()
    nulls = {i: {'signals': {d: {f'{d}_signal': {'estimate': float(rng.normal()), 'higher_is_worse': True}}
                             for d in DIMENSIONS}} for i in range(count)}
    model = calibrator.fit(nulls)
    def table(**overrides):
        signals = {d: {f'{d}_signal': {'estimate': overrides.get(d, 0.), 'higher_is_worse': True}} for d in DIMENSIONS}
        return {'dimensions': {d: {'status': 'Pass', 'metrics': {}} for d in DIMENSIONS},
                'signals': signals, 'hard_fail': dict.fromkeys(DIMENSIONS, False)}
    evaluator = CalibratedEvaluator(model)
    honest = evaluator.apply(table(), .05)
    assert honest['gate'] == 'Pass' and all(honest['dimensions'][d]['status'] == 'Pass' for d in DIMENSIONS)
    for d in DIAGNOSTIC_DIMENSIONS:
        out = evaluator.apply(table(**{d: 1e6}), .05)
        assert out['dimensions'][d]['status'] == 'Fail' and out['gate'] == 'Pass', d
    for d in GATE_DIMENSIONS:
        out = evaluator.apply(table(**{d: 1e6}), .05)
        assert out['dimensions'][d]['status'] == 'Fail' and out['gate'] == 'Fail', d
        others = [x for x in GATE_DIMENSIONS if x != d]
        assert all(out['dimensions'][x]['status'] == 'Pass' for x in others), d
    missing = table(); missing['signals']['privacy']['privacy_signal']['estimate'] = None
    assert evaluator.apply(missing, .05)['gate'] == 'NotEvaluated'
    assert model['levels']['0.05']['rejection_rank'] == 2
    return ['diagnostic failures never change the gate; any gating failure fails it alone',
            'a missing gating signal makes the gate NotEvaluated, never Pass']


def self_test() -> None:
    assert Gate.aggregate(['Pass','Conditional']) == 'Conditional'
    print(conformal_size_check())
