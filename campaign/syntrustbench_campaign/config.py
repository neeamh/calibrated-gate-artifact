"""Frozen scientific configuration and semantic random seeds."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

DIMENSIONS = ('fidelity', 'utility', 'privacy', 'equity', 'robustness')
# v0.4.1: only dimensions with established, real-referenced tests decide the verdict.
# Equity and robustness are still computed and reported as non-gating diagnostics.
GATE_DIMENSIONS = ('fidelity', 'utility', 'privacy')
DIAGNOSTIC_DIMENSIONS = ('equity', 'robustness')
PROTOCOL = 'v0.4.1-gate'
STAGES = ('prepare', 'smoke', 'null', 'calibrate', 'clean', 'controls',
          'injections', 'judges', 'analysis', 'report')
COMMIT = '3dd0f7ce088994d84f4925ca7e0949d7e0ecd3d8'
CONTINUOUS = ('time_in_hospital', 'num_lab_procedures', 'num_procedures',
              'num_medications', 'number_outpatient', 'number_emergency',
              'number_inpatient', 'number_diagnoses')
CATEGORICAL = ('race', 'gender', 'age', 'admission_type_id',
               'discharge_disposition_id', 'admission_source_id', 'diag_1',
               'max_glu_serum', 'A1Cresult', 'insulin', 'change', 'diabetesMed')
TARGET = 'readmitted_30d'


def seed_for(*parts: object) -> int:
    """A stable JSON tuple SHA-256, interpreted as an unsigned 32-bit seed."""
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':'),
                          ensure_ascii=True).encode()).digest()[:4], 'big')


@dataclass(frozen=True)
class DataConfig:
    raw_csv: str = 'data/diabetic_data.csv'
    frozen_dir: str | None = None
    frozen_manifest: str | None = None
    continuous: tuple[str, ...] = CONTINUOUS
    categorical: tuple[str, ...] = CATEGORICAL
    target: str = TARGET
    protected: tuple[str, ...] = ('race', 'gender')
    split_seed: int = 49173
    split_fractions: tuple[float, ...] = (.4, .4, .2)

    def __post_init__(self):
        if self.split_fractions != (.4, .4, .2):
            raise ValueError('This protocol fixes the patient split at 40/40/20.')
        if bool(self.frozen_dir) != bool(self.frozen_manifest):
            raise ValueError('Frozen splits require a manifest with hashes and patient audit.')
        if len(set(self.columns)) != len(self.columns):
            raise ValueError('Analytic columns must be unique.')

    @property
    def columns(self) -> tuple[str, ...]:
        return self.continuous + self.categorical + (self.target,)


@dataclass(frozen=True)
class GeneratorConfig:
    names: tuple[str, ...] = ('GaussianCopula', 'CTGAN', 'TVAE')
    seeds: tuple[int, ...] = (17, 29, 43)
    epochs: int = 300
    enforce_min_max_values: bool = True
    enforce_rounding: bool = True
    allow_cuda: bool = False
    enable_dpmst: bool = False
    enable_tabddpm: bool = False
    epsilon: float = 1.0
    delta: float | None = None
    # Public, analyst-declared integer domains: column, lower, cap, width, overflow upper.
    dp_domains: tuple[tuple[str, int, int, int, int], ...] = (
        ('time_in_hospital', 1, 14, 1, 14),
        ('num_lab_procedures', 0, 200, 10, 1000),
        ('num_procedures', 0, 20, 1, 100),
        ('num_medications', 0, 100, 5, 500),
        ('number_outpatient', 0, 50, 5, 250),
        ('number_emergency', 0, 50, 5, 250),
        ('number_inpatient', 0, 50, 5, 250),
        ('number_diagnoses', 0, 30, 2, 100))
    dp_categories: tuple[tuple[str, tuple[str, ...]], ...] = (
        ('race', ('Caucasian', 'Asian', 'AfricanAmerican', 'Hispanic', 'Other')),
        ('gender', ('Female', 'Male', 'Unknown/Invalid')),
        ('age', tuple(f'[{i}-{i+10})' for i in range(0, 100, 10))),
        ('admission_type_id', tuple(str(i) for i in range(1, 10))),
        ('discharge_disposition_id', tuple(str(i) for i in range(1, 31))),
        ('admission_source_id', tuple(str(i) for i in range(1, 27))),
        ('diag_1', tuple(str(i) for i in range(1000))
                   + tuple(f'V{i:02d}' for i in range(100))
                   + tuple(f'E{i:03d}' for i in range(1000))),
        ('max_glu_serum', ('None', 'Norm', '>200', '>300')),
        ('A1Cresult', ('None', 'Norm', '>7', '>8')),
        ('insulin', ('No', 'Steady', 'Up', 'Down')),
        ('change', ('No', 'Ch')), ('diabetesMed', ('No', 'Yes')),
        ('readmitted_30d', ('0', '1')))
    dp_aim_rounds: int | None = None
    dp_smoke_rounds: int = 32
    dp_max_model_size_mb: float = 80.
    dp_max_cells: int = 10000
    tabddpm_python: str = 'python3.10'
    tabddpm_build_environment: bool = True
    tabddpm_epochs: int = 1000
    tabddpm_timesteps: int = 1000
    tabddpm_timeout_seconds: int = 7200

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return self.names + ('DPAIM',) + (('DPMST',) if self.enable_dpmst else ()) + (('TabDDPM',) if self.enable_tabddpm else ())

    def __post_init__(self):
        if self.epochs < 1 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Positive epochs and unique seeds are required.')
        if set(self.names) != {'GaussianCopula', 'CTGAN', 'TVAE'}:
            raise ValueError('Only the three prespecified generators are supported.')
        if not math.isfinite(self.epsilon) or self.epsilon <= 0 or (self.delta is not None and not 0 < self.delta < 1):
            raise ValueError('DP requires positive epsilon and 0 < delta < 1.')
        if len(dict(self.dp_categories)) != len(self.dp_categories):
            raise ValueError('Duplicate public categorical domains.')
        for _, levels in self.dp_categories:
            if not levels or len(set(levels)) != len(levels) or {'__OTHER__', '__MISSING__'} & set(levels):
                raise ValueError('Public categories must be unique; reserved levels are appended automatically.')
        for _, lower, cap, width, overflow in self.dp_domains:
            if lower < 0 or cap < lower or width < 1 or overflow < cap:
                raise ValueError('Invalid public integer domain.')
        if self.dp_smoke_rounds < 32 or (self.dp_aim_rounds is not None and self.dp_aim_rounds < 32):
            raise ValueError('AIM rounds must cover its initial marginal privacy allocation (>=32).')
        if min(self.dp_max_model_size_mb, self.dp_max_cells, self.tabddpm_epochs,
               self.tabddpm_timesteps, self.tabddpm_timeout_seconds) <= 0:
            raise ValueError('Generator compute limits must be positive.')


@dataclass(frozen=True)
class InjectionConfig:
    doses: tuple[tuple[str, tuple[float, ...]], ...] = (
        ('exact_copy', (.001, .01, .05)),
        ('subgroup_suppression', (.10, .50, .90)),
        ('feature_shuffle', (.10, .50, .90)), ('mcar', (.05, .15, .30)))
    # Intended gating dimensions only: the gate is what is being tested.
    intended: tuple[tuple[str, tuple[str, ...]], ...] = (
        ('exact_copy', ('privacy',)), ('subgroup_suppression', ('fidelity',)),
        ('feature_shuffle', ('fidelity', 'utility')),
        ('mcar', ('fidelity', 'utility')))
    subgroup_column: str = 'race'
    subgroup_value: str = 'AfricanAmerican'
    shuffle_feature: str = 'number_inpatient'
    missing_features: tuple[str, ...] = (
        'time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient')

    def __post_init__(self):
        if set(dict(self.doses)) != set(dict(self.intended)):
            raise ValueError('Every injection needs an intended dimension.')
        if any(not dims or not set(dims) <= set(GATE_DIMENSIONS) for _, dims in self.intended):
            raise ValueError('Intended dimensions must be gating dimensions.')
        for _, doses in self.doses:
            if tuple(sorted(set(doses))) != doses or any(not 0 < d <= 1 for d in doses):
                raise ValueError('Doses must be unique, increasing and in (0, 1].')


@dataclass(frozen=True)
class CalibrationConfig:
    repetitions: int = 100
    fit_runs: int = 50
    assignment_seed: int = 81631
    target_rates: tuple[float, ...] = (.025, .05, .10)
    primary_rate: float = .05
    scale_floor: float = 1e-8
    design: str = 'development_scaled'
    # Full conformal: the candidate is pooled with the fit runs for standardization and
    # ranked against them; exact marginal level r/(n+1) under exchangeability.
    studentization: str = 'pooled_with_candidate'
    quantile: str = 'full_conformal'
    # Prespecified admission rule: a dimension may gate only if every audit run is
    # evaluable and an exact one-sided binomial test finds no excess of honest-data
    # non-acceptance over primary_rate at admission_test_alpha.
    admission_test_alpha: float = .05
    null_seed_salt: str = 'v0.4.1'
    smoke_repetitions: int = 5
    smoke_fit_runs: int = 3

    def __post_init__(self):
        if not 3 <= self.fit_runs < self.repetitions or self.scale_floor <= 0:
            raise ValueError('Need at least three fit runs, held-out audit runs and positive scale.')
        if not 3 <= self.smoke_fit_runs < self.smoke_repetitions:
            raise ValueError('Smoke calibration needs at least three fit runs and one audit run.')
        if self.primary_rate not in self.target_rates or any(not 0 < a < 1 for a in self.target_rates):
            raise ValueError('Invalid target rates.')
        if self.design != 'development_scaled':
            raise ValueError('Only the specified development-only null design is implemented.')
        if (self.studentization, self.quantile) != ('pooled_with_candidate', 'full_conformal'):
            raise ValueError('Only the full-conformal max-statistic rule is implemented.')
        if not self.null_seed_salt:
            raise ValueError('A null seed salt is required.')
        if not 0 < self.admission_test_alpha < 1:
            raise ValueError('Invalid admission test level.')


@dataclass(frozen=True)
class JudgeConfig:
    # No weighted judge: with the equal default weights used in published practice a
    # weighted sum equals the mean, and any unequal weights would be an unprespecified choice.
    threshold_offsets: tuple[float, ...] = (0., -.05, -.10)
    score_mapping: str = 'exp(-max(0,max_directional_z)/temperature)'
    temperature: float = 3.

    def __post_init__(self):
        if self.temperature <= 0 or self.score_mapping != 'exp(-max(0,max_directional_z)/temperature)':
            raise ValueError('Unknown or invalid frozen score mapping.')
        if self.threshold_offsets[0] != 0 or any(o > 0 for o in self.threshold_offsets):
            raise ValueError('Offsets start at zero and only relax the threshold.')


@dataclass(frozen=True)
class ExecutionConfig:
    output_root: str = 'syntrustbench_v04_runs'
    checkout_dir: str = 'syntrustbench_v04_dependencies/frozen_v030'
    stages: tuple[str, ...] = STAGES
    dry_run: bool = True
    workers: int = 1
    threads_per_worker: int = 1
    retry_failed: bool = False
    install_dependencies: bool = False
    smoke_mode: bool = False
    smoke_rows: int = 200
    smoke_bootstrap: int = 20
    smoke_epochs: int = 1

    def __post_init__(self):
        if not set(self.stages) <= set(STAGES) or self.workers < 1 or self.threads_per_worker < 1:
            raise ValueError('Invalid stage or worker configuration.')
        if not 50 <= self.smoke_rows <= 500 or self.smoke_bootstrap < 20:
            raise ValueError('Smoke uses 50–500 rows and at least 20 bootstrap iterations.')


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    generators: GeneratorConfig = field(default_factory=GeneratorConfig)
    injections: InjectionConfig = field(default_factory=InjectionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    judges: JudgeConfig = field(default_factory=JudgeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    bootstrap_iterations: int = 200
    confidence_level: float = .95
    evaluation_seed: int = 7103
    min_subgroup_n: int = 30
    min_subgroup_outcomes: int = 10
    privacy_max_rows: int = 10000
    privacy_binomial_alpha: float = .05
    model_seeds: tuple[int, ...] = (7103, 7104, 7105, 7106, 7107)
    missingness_rates: tuple[float, ...] = (.05, .10, .20)
    training_fractions: tuple[float, ...] = (.25, .50, .75, 1.)
    missingness_repeats: int = 3
    positive_replicates: tuple[int, ...] = (17, 29, 43)

    def __post_init__(self):
        if self.bootstrap_iterations < 20 or self.confidence_level != .95:
            raise ValueError('Reference implementation uses fixed 95% CIs; need >=20 resamples.')
        if self.privacy_max_rows < 10 or self.min_subgroup_n < 2 or self.min_subgroup_outcomes < 1:
            raise ValueError('Invalid row limits.')
        if self.missingness_repeats < 2 or len(self.model_seeds) < 2:
            raise ValueError('Robustness needs repeated stressors.')

    def scientific(self) -> dict:
        value = asdict(self)
        value['protocol'] = PROTOCOL
        value['gate_dimensions'] = list(GATE_DIMENSIONS)
        execution = value.pop('execution')
        value['smoke_settings'] = {k: execution[k] for k in
                                  ('smoke_mode', 'smoke_rows', 'smoke_bootstrap', 'smoke_epochs')}
        for path in ('raw_csv', 'frozen_dir', 'frozen_manifest'):
            value['data'].pop(path)
        return value

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.scientific(), sort_keys=True).encode()).hexdigest()

    @property
    def root(self) -> Path:
        label = 'smoke' if self.execution.smoke_mode else 'full'
        return Path(self.execution.output_root).resolve() / f'{label}_{self.fingerprint[:16]}'

    def benchmark(self, *, smoke: bool = False) -> dict:
        small = smoke or self.execution.smoke_mode
        d = self.data
        return {
            'dataset': {'name': 'UCI Diabetes earliest encounter', 'modality': 'tabular', 'outcome': d.target},
            'columns': {'continuous': list(d.continuous), 'categorical': list(d.categorical), 'binary': [d.target]},
            'utility_task': {'target': d.target, 'type': 'binary_classification', 'positive_label': '1',
                             'features': list(d.continuous + d.categorical)},
            'protected_attributes': list(d.protected),
            'clinical_constraints': [f'{c} >= 0' for c in d.continuous] + ['time_in_hospital >= 1', 'time_in_hospital <= 14'],
            'generator': {'name': 'v04_campaign', 'seed': self.evaluation_seed},
            'uncertainty': {'bootstrap_iterations': self.execution.smoke_bootstrap if small else self.bootstrap_iterations,
                            'confidence_level': .95, 'random_seed': self.evaluation_seed},
            'subgroup_analysis': {'minimum_n': self.min_subgroup_n,
                                  'minimum_outcomes_per_class': self.min_subgroup_outcomes},
            'privacy': {'distance_max_rows': min(500, self.privacy_max_rows) if small else self.privacy_max_rows,
                        'attack_max_rows': min(500, self.privacy_max_rows) if small else self.privacy_max_rows},
            'robustness': {'downstream_model_seeds': list(self.model_seeds[:2] if small else self.model_seeds),
                           'missingness_rates': list(self.missingness_rates),
                           'training_size_fractions': list(self.training_fractions)},
        }


def self_test() -> None:
    assert seed_for('a', 1) == seed_for('a', 1) != seed_for('a', 2)
    assert ExperimentConfig().benchmark()['uncertainty']['bootstrap_iterations'] == 200
    assert set(GATE_DIMENSIONS) | set(DIAGNOSTIC_DIMENSIONS) == set(DIMENSIONS)
    assert not set(GATE_DIMENSIONS) & set(DIAGNOSTIC_DIMENSIONS)
    c = CalibrationConfig()
    assert (c.repetitions, c.fit_runs) == (100, 50)


def from_mapping(settings: dict) -> ExperimentConfig:
    """Convert the single notebook configuration cell to frozen dataclasses."""
    def freeze(value):
        if isinstance(value, list): return tuple(freeze(v) for v in value)
        if isinstance(value, dict): return {k:freeze(v) for k,v in value.items()}
        return value
    values = freeze(settings)
    constructors = {'data':DataConfig, 'generators':GeneratorConfig, 'injections':InjectionConfig,
                    'calibration':CalibrationConfig, 'judges':JudgeConfig, 'execution':ExecutionConfig}
    return ExperimentConfig(**{k:constructors[k](**v) if k in constructors else v for k,v in values.items()})
