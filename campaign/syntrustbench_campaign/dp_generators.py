"""SmartNoise AIM/MST adapters with entirely public discrete domains.

Fit noise is fresh and private. Public seeds control postprocessing only.
The model's nominal privacy accounting does not cover campaign diagnostics.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .generators import Generator


class PublicDiscretizer:
    """Data-independent bins; integer intervals inclusive at both ends."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.bins = {}
        for col, lower, cap, width, overflow in cfg.generators.dp_domains:
            intervals = [(lo, min(lo + width - 1, cap)) for lo in range(lower, cap + 1, width)]
            if overflow > cap:
                intervals.append((cap + 1, overflow))
            self.bins[col] = intervals
        self.categories = {c: levels + (('__MISSING__', '__OTHER__') if c != cfg.data.target else ())
                           for c, levels in cfg.generators.dp_categories}
        if set(self.bins) != set(cfg.data.continuous) or set(self.categories) != set(cfg.data.categorical + (cfg.data.target,)):
            raise ValueError('Every analytic column needs an explicit public DP domain.')
        if self.categories[cfg.data.target] != ('0','1'):
            raise ValueError('The public DP outcome domain must be exactly 0/1.')
        if self.bins['time_in_hospital'][0][0] != 1 or self.bins['time_in_hospital'][-1][1] != 14:
            raise ValueError('The public hospital-stay domain must enforce 1–14 days.')
        self.cards = [len(self.bins[c]) + 1 if c in self.bins else len(self.categories[c]) for c in cfg.data.columns]

    def manifest(self) -> dict:
        value = {'integer_intervals_inclusive': self.bins, 'categories': self.categories,
                 'cardinalities': dict(zip(self.cfg.data.columns, self.cards)),
                 'missing_numeric_bin': 'extra final bin; decoded to fixed public lower bound',
                 'overflow': 'above cap uses final interval; above overflow upper also uses that interval',
                 'diagnosis': 'public ICD-9 prefix before decimal; numeric leading zeroes normalized',
                 'source': 'UCI data dictionary and analyst-declared public count caps; no training statistics',
                 'preprocessing_epsilon': 0.0}
        value['schema_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        if list(frame.columns) != list(self.cfg.data.columns):
            raise ValueError('DP input schema/order differs from configured analytic columns.')
        result = {}
        for c in self.cfg.data.columns:
            if c in self.bins:
                x = pd.to_numeric(frame[c], errors='raise').to_numpy(dtype=float)
                highs = np.array([hi for _, hi in self.bins[c]])
                codes = np.searchsorted(highs, np.rint(x), side='left').clip(0, len(highs)-1)
                codes[~np.isfinite(x)] = len(highs)
                result[c] = codes.astype(int)
            else:
                vals = frame[c].map(lambda x: '__MISSING__' if pd.isna(x) or str(x) in ('', '?') else str(x))
                if c == 'diag_1':
                    vals = vals.map(lambda s: str(int(s.split('.')[0])) if s.split('.')[0].isdigit()
                                    else s.split('.')[0].upper())
                if c == self.cfg.data.target:
                    vals = vals.replace({'0.0':'0', '1.0':'1'})
                    if not vals.isin(self.categories[c]).all():
                        raise ValueError('DP outcome must be binary and nonmissing.')
                lookup = {v:i for i,v in enumerate(self.categories[c])}
                result[c] = vals.map(lambda v: lookup.get(v, lookup.get('__OTHER__'))).astype(int)
        return pd.DataFrame(result, columns=self.cfg.data.columns)

    def decode(self, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        result = {}
        for c, card in zip(self.cfg.data.columns, self.cards):
            codes = frame[c].to_numpy(dtype=int)
            if ((codes < 0) | (codes >= card)).any():
                raise ValueError('DP synthesizer returned an out-of-domain category.')
            if c in self.bins:
                intervals = self.bins[c] + [(self.bins[c][0][0], self.bins[c][0][0])]
                lo = np.array([intervals[i][0] for i in codes])
                hi = np.array([intervals[i][1] for i in codes])
                result[c] = rng.integers(lo, hi + 1)
            elif c == self.cfg.data.target:
                result[c] = np.array([int(self.categories[c][i]) for i in codes])
            else:
                result[c] = [np.nan if self.categories[c][i] == '__MISSING__' else self.categories[c][i] for i in codes]
        return pd.DataFrame(result, columns=self.cfg.data.columns)

    def transformer(self):
        # Fit labels ONLY on the complete public integer domain, never private rows.
        from snsynth.transform import LabelTransformer, TableTransformer
        transforms = []
        for card in self.cards:
            t = LabelTransformer(nullable=False)
            t.fit(list(range(card)))
            transforms.append(t)
        table = TableTransformer(transforms)
        assert table.fit_complete and not table.needs_epsilon and table.cardinality == self.cards
        return table


@contextlib.contextmanager
def private_fit_randomness():
    """Do not expose the upstream selector's random tape through public job seeds."""
    state = np.random.get_state()
    np.random.seed(np.frombuffer(os.urandom(128), dtype=np.uint32))
    try:
        yield
    finally:
        np.random.set_state(state)


class DPAIMGenerator(Generator):
    mechanism = 'AIM'

    def synthesizer_class(self):
        from snsynth.aim import AIMSynthesizer
        return AIMSynthesizer

    def fit(self, train: pd.DataFrame) -> DPAIMGenerator:
        import jax
        jax.config.update('jax_enable_x64', True)
        if importlib.metadata.version('smartnoise-synth') != '1.0.8':
            raise RuntimeError('SmartNoise version differs from the pinned adapter.')
        start = time.perf_counter()
        self.discretizer = PublicDiscretizer(self.cfg)
        discrete = self.discretizer.encode(train)
        public = self.discretizer.manifest()
        n = len(train)
        if n == 0:
            raise ValueError('Cannot fit DP generator to zero rows.')
        delta = self.cfg.generators.delta if self.cfg.generators.delta is not None else 1/(10*n)
        options = {'epsilon': self.cfg.generators.epsilon, 'delta': delta, 'verbose': False}
        if self.mechanism == 'AIM':
            options.update(rounds=self.cfg.generators.dp_smoke_rounds if (self.smoke or self.cfg.execution.smoke_mode)
                           else self.cfg.generators.dp_aim_rounds,
                           max_model_size=self.cfg.generators.dp_max_model_size_mb,
                           max_cells=self.cfg.generators.dp_max_cells, degree=2)
        self.model = self.synthesizer_class()(**options)
        with private_fit_randomness():
            self.model.fit(discrete, transformer=self.discretizer.transformer(), preprocessor_eps=0.0)
        if self.model._transformer.cardinality != self.discretizer.cards:
            raise RuntimeError('Private data altered the public domain cardinalities.')
        self.schema = list(train.columns)
        self.timing.update(fit_seconds=time.perf_counter()-start, device='CPU', parameters=options)
        self.provenance = {'public_preprocessing': public,
            'formal_privacy': {'status': 'NominalMechanismGuarantee', 'mechanism': self.mechanism,
                'epsilon': options['epsilon'], 'delta': delta, 'n_train_public': n,
                'delta_rule': 'configured' if self.cfg.generators.delta is not None else '1/(10*n_train)',
                'preprocessing_epsilon': 0.0, 'library': 'smartnoise-synth==1.0.8',
                'scope': 'One fitted synthesizer and its postprocessed samples; upstream unit-row accounting with public domain and sample size.',
                'randomness': 'Fresh unrecorded fit randomness and OpenDP Gaussian noise; public seed controls sampling only.',
                'exclusions': 'Raw inputs, hashes, exact cohort counts, all evaluator reports, other generators and joint multi-fit releases are not covered.',
                'assumptions': 'Public n and categorical schema; one record per patient; upstream implementation/accountant correctness. Not an independent DP certification.',
                'empirical_privacy_is_separate': True}}
        return self

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError('Fit before sampling.')
        start = time.perf_counter()
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            discrete = pd.DataFrame(self.model.sample(n), columns=self.schema)
        finally:
            np.random.set_state(state)
        frame = self.discretizer.decode(discrete, seed)
        if len(frame) != n or list(frame.columns) != self.schema:
            raise ValueError('DP output schema or row count changed.')
        self.timing.update(sample_seconds=time.perf_counter()-start, sample_seed=seed)
        return frame

    def save(self, path: Path) -> None:
        import cloudpickle
        with Path(path).open('wb') as handle:
            cloudpickle.dump({'model':self.model, 'public_preprocessing':self.discretizer.manifest()}, handle)


class DPMSTGenerator(DPAIMGenerator):
    mechanism = 'MST'

    def synthesizer_class(self):
        from snsynth.mst import MSTSynthesizer
        return MSTSynthesizer


def self_test() -> None:
    schema = PublicDiscretizer(ExperimentConfig())
    assert schema.bins['time_in_hospital'] == [(i,i) for i in range(1,15)]
    assert schema.manifest()['preprocessing_epsilon'] == 0
