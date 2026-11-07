"""Pinned SDV adapters with explicit fit/sample seeds and CPU-safe device choice."""
from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import random
import time
import numpy as np
import pandas as pd

from .config import ExperimentConfig


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Generator(ABC):
    def __init__(self, cfg: ExperimentConfig, seed: int, smoke: bool = False):
        self.cfg, self.seed, self.smoke = cfg, seed, smoke
        self.model = None
        self.timing = {}
        self.provenance = {'formal_privacy': {'status': 'NotApplicable', 'reason': 'Not a DP generator.'}}
        self.artifact_dir = None

    def save(self, path) -> None:
        self.model.save(str(path))

    @abstractmethod
    def synthesizer_class(self):
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> Generator:
        from sdv.metadata import SingleTableMetadata
        import torch
        seed_all(self.seed)
        metadata = SingleTableMetadata()
        for c in self.cfg.data.columns:
            metadata.add_column(column_name=c, sdtype='numerical' if c in self.cfg.data.continuous else 'categorical')
        cls = self.synthesizer_class()
        options = {'enforce_min_max_values': self.cfg.generators.enforce_min_max_values,
                   'enforce_rounding': self.cfg.generators.enforce_rounding}
        gpu = self.cfg.generators.allow_cuda and torch.cuda.is_available()
        if cls.__name__ != 'GaussianCopulaSynthesizer':
            gpu_parameter = 'enable_gpu' if 'enable_gpu' in inspect.signature(cls).parameters else 'cuda'
            options[gpu_parameter] = bool(gpu)  # Apple Silicon stays CPU by default.
            options['epochs'] = self.cfg.execution.smoke_epochs if (self.smoke or self.cfg.execution.smoke_mode) else self.cfg.generators.epochs
        self.model = cls(metadata, **options)
        start = time.perf_counter(); self.model.fit(train)
        self.timing.update(fit_seconds=time.perf_counter()-start, device='CUDA' if gpu else 'CPU', parameters=options)
        self.schema = list(train.columns)
        return self

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError('Fit before sampling.')
        seed_all(seed); self.model.reset_sampling()
        if not hasattr(self.model, '_set_random_state'):
            raise RuntimeError('Pinned SDV sampling API changed.')
        self.model._set_random_state(seed)
        start = time.perf_counter()
        frame = self.model.sample(num_rows=n)[self.schema]
        self.timing.update(sample_seconds=time.perf_counter()-start, sample_seed=seed)
        if len(frame) != n:
            raise ValueError('Generator returned the wrong row count.')
        return frame


class GaussianCopulaGenerator(Generator):
    def synthesizer_class(self):
        from sdv.single_table import GaussianCopulaSynthesizer
        return GaussianCopulaSynthesizer


class CTGANGenerator(Generator):
    def synthesizer_class(self):
        from sdv.single_table import CTGANSynthesizer
        return CTGANSynthesizer


class TVAEGenerator(Generator):
    def synthesizer_class(self):
        from sdv.single_table import TVAESynthesizer
        return TVAESynthesizer


def generator(name: str, cfg: ExperimentConfig, seed: int, smoke: bool = False) -> Generator:
    from .dp_generators import DPAIMGenerator, DPMSTGenerator
    from .tabddpm import TabDDPMGenerator
    return {'GaussianCopula': GaussianCopulaGenerator, 'CTGAN': CTGANGenerator,
            'TVAE': TVAEGenerator, 'DPAIM': DPAIMGenerator, 'DPMST': DPMSTGenerator,
            'TabDDPM': TabDDPMGenerator}[name](cfg, seed, smoke)


def self_test() -> None:
    assert len({GaussianCopulaGenerator, CTGANGenerator, TVAEGenerator}) == 3
