"""Earliest-encounter cohort and audited patient-disjoint real-data roles."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig, seed_for
from .core import UnavailableAnalysis, file_hash, read_json, save_frame, write_json


def read_analytic(path: str | Path, cfg: ExperimentConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, na_values=[''],
                        dtype={c: 'object' for c in cfg.data.categorical}, float_precision='round_trip')
    if list(frame.columns) != list(cfg.data.columns):
        raise ValueError(f'Unexpected analytic schema: {path}')
    frame[cfg.data.target] = pd.to_numeric(frame[cfg.data.target], errors='raise').astype(int)
    if not set(frame[cfg.data.target].unique()) <= {0, 1}:
        raise ValueError('Outcome must be integer binary data.')
    return frame


def stratified_indices(y: pd.Series, size: int, seed: int) -> np.ndarray:
    if size > len(y) or size < 1:
        raise ValueError('Invalid sample size.')
    if size == len(y):
        return np.random.default_rng(seed).permutation(len(y))
    rows, _ = train_test_split(np.arange(len(y)), train_size=size, stratify=y, random_state=seed)
    return np.asarray(rows)


def split_three(frame: pd.DataFrame, cfg: ExperimentConfig, seed: int) -> list[pd.DataFrame]:
    """Reuse the previous notebook's stratified 40/40/20 algorithm."""
    dev, test = train_test_split(frame, test_size=.2, stratify=frame[cfg.data.target], random_state=seed)
    train, pool = train_test_split(dev, test_size=.5, stratify=dev[cfg.data.target],
                                  random_state=seed_for(seed, 'half'))
    return [x.reset_index(drop=True) for x in (train, pool, test)]


@dataclass
class CohortBuilder:
    cfg: ExperimentConfig

    def _raw(self) -> tuple[dict, dict, list]:
        d = self.cfg.data
        raw = pd.read_csv(d.raw_csv, keep_default_na=False, na_values=['?', ''], dtype='object')
        required = set(d.continuous + d.categorical + ('patient_nbr', 'encounter_id', 'readmitted'))
        if not required <= set(raw):
            raise ValueError(f'Missing columns: {required - set(raw)}')
        if raw[['patient_nbr', 'encounter_id', 'readmitted']].isna().any().any():
            raise ValueError('Patient, encounter and outcome fields must be complete.')
        if not set(raw.readmitted) <= {'<30', '>30', 'NO'}:
            raise ValueError('Unknown readmission label.')
        raw['encounter_id'] = pd.to_numeric(raw.encounter_id, errors='raise')
        if not raw.encounter_id.is_unique:
            raise ValueError('Duplicate encounter IDs need source review.')
        cohort = raw.sort_values('encounter_id', kind='stable').drop_duplicates('patient_nbr').copy()
        cohort[d.target] = (cohort.readmitted == '<30').astype(int)
        for column in d.continuous:
            cohort[column] = pd.to_numeric(cohort[column], errors='raise')
        cohort['_patient_hash'] = cohort.patient_nbr.map(lambda s: hashlib.sha256(str(s).encode()).hexdigest())
        frames = dict(zip(('real_train', 'calibration_pool', 'real_test'), split_three(cohort, self.cfg, d.split_seed)))
        ids = {name: f['_patient_hash'].tolist() for name, f in frames.items()}
        flow = [{'section': 'flow', 'label': 'raw encounters', 'rows': len(raw)},
                {'section': 'flow', 'label': 'later encounters removed', 'rows': len(raw)-len(cohort)},
                {'section': 'flow', 'label': 'earliest encounters', 'rows': len(cohort)}]
        return {k: v[list(d.columns)].copy() for k, v in frames.items()}, ids, flow

    def _frozen(self) -> tuple[dict, dict, list]:
        d = self.cfg.data
        manifest = read_json(d.frozen_manifest)
        hashes = manifest.get('split_hashes', manifest.get('hashes', {}))
        frames = {}
        for name in ('real_train', 'calibration_pool', 'real_test'):
            path = Path(d.frozen_dir) / f'{name}.csv'
            if name not in hashes or file_hash(path) != hashes[name]:
                raise ValueError(f'Frozen split hash missing/mismatched: {name}')
            frames[name] = read_analytic(path, self.cfg)
        audit_path = Path(manifest.get('patient_audit', Path(d.frozen_dir).resolve() / 'patient_role_audit.csv'))
        if not audit_path.is_absolute():
            audit_path = Path(d.frozen_manifest).resolve().parent / audit_path
        if not audit_path.is_file():
            raise UnavailableAnalysis('Frozen splits need row-aligned hashed patient_role_audit.csv; no resplitting fallback.')
        if manifest.get('patient_audit_sha256') and file_hash(audit_path) != manifest['patient_audit_sha256']:
            raise ValueError('Patient audit hash mismatch.')
        audit = pd.read_csv(audit_path)
        ids = {}
        for name, frame in frames.items():
            rows = audit[audit.role == name].sort_values('analytic_row')
            if rows.analytic_row.tolist() != list(range(len(frame))):
                raise ValueError(f'Patient audit row alignment invalid: {name}')
            ids[name] = rows.patient_id_sha256.astype(str).tolist()
        flow = [{'section': 'flow', 'label': 'frozen earliest-encounter cohort',
                 'rows': sum(map(len, frames.values())), 'note': 'Upstream raw encounter flow not reconstructed.'}]
        return frames, ids, flow

    def build(self, folder: Path) -> dict:
        frames, ids, rows = self._frozen() if self.cfg.data.frozen_dir else self._raw()
        flattened = sum(ids.values(), [])
        if len(set(flattened)) != len(flattened):
            raise ValueError('Repeated patient hashes within or across frozen roles.')
        paths, patient_rows, missing, groups = {}, [], [], []
        for name, frame in frames.items():
            if self.cfg.execution.smoke_mode:
                indices = stratified_indices(frame[self.cfg.data.target], min(len(frame), self.cfg.execution.smoke_rows), seed_for('smoke', name))
                ids[name] = [ids[name][i] for i in indices]
                frame = frame.iloc[indices].reset_index(drop=True)
            paths[name] = save_frame(frame, folder / f'{name}.csv')
            patient_rows.extend({'role': name, 'analytic_row': i, 'patient_id_sha256': p} for i, p in enumerate(ids[name]))
            rows.append({'section': 'split', 'label': name, 'rows': len(frame),
                         'positive': int(frame[self.cfg.data.target].sum()),
                         'prevalence': float(frame[self.cfg.data.target].mean())})
            missing.extend({'role': name, 'column': c, 'missing': int(frame[c].isna().sum()),
                            'rows': len(frame), 'fraction': float(frame[c].isna().mean())} for c in frame)
            for attribute in self.cfg.data.protected:
                groups.extend({'role': name, 'attribute': attribute, 'level': str(level),
                               'rows': len(group), 'positive': int(group[self.cfg.data.target].sum())}
                              for level, group in frame.groupby(attribute, dropna=False))
        audit = save_frame(pd.DataFrame(patient_rows), folder / 'patient_role_audit.csv')
        for name, value in [('cohort_split_summary', rows), ('missingness', missing), ('subgroup_counts', groups)]:
            save_frame(pd.DataFrame(value), folder / f'{name}.csv')
        result = {'paths': paths, 'split_hashes': {k: file_hash(p) for k, p in paths.items()},
                  'patient_audit': audit, 'patient_audit_sha256': file_hash(audit), 'patient_overlap': 0,
                  'folder': str(folder), 'schema': list(self.cfg.data.columns),
                  'cohort_rule': 'minimum numeric encounter_id per patient; readmitted == <30',
                  'smoke_mode': self.cfg.execution.smoke_mode}
        write_json(folder / 'split_manifest.json', result)
        return result


@dataclass
class RoleAssigner:
    cfg: ExperimentConfig

    def assign(self, frames: list[pd.DataFrame], ids: list[list[str]], folder: Path, key: str) -> dict:
        train, pool, test = frames
        ti, pi, ei = ids
        a, b = train_test_split(np.arange(len(pool)), test_size=.5,
                                stratify=pool[self.cfg.data.target], random_state=seed_for('roles', key, 'reference'))
        n = min(len(a), len(b), len(train))
        a, b = a[:n], b[:n]
        floor = stratified_indices(train[self.cfg.data.target], n, seed_for('roles', key, 'floor'))
        tables = {'train': train, 'reference': pool.iloc[a], 'positive': pool.iloc[b],
                  'test': test, 'floor': train.iloc[floor]}
        roles = {'train': ti, 'reference': [pi[i] for i in a], 'positive': [pi[i] for i in b],
                 'test': ei, 'floor': [ti[i] for i in floor]}
        independent = [roles[k] for k in ('train', 'reference', 'positive', 'test')]
        if len(set(sum(independent, []))) != sum(map(len, independent)):
            raise ValueError('Patient overlap between independent roles.')
        paths = {k: save_frame(v.reset_index(drop=True), folder / f'{k}.csv') for k, v in tables.items()}
        audit = save_frame(pd.DataFrame([{'role': k, 'analytic_row': i, 'patient_id_sha256': p}
                            for k, values in roles.items() for i, p in enumerate(values)]), folder / 'role_patients.csv')
        result = {'paths': paths, 'patient_audit': audit, 'n': n, 'key': key,
                  'role_counts': {k: len(v) for k, v in tables.items()}, 'independent_role_overlap': 0,
                  'floor_subset_of_train': True, 'omitted_pool_rows': len(pool)-2*n,
                  'main_test_used': key == 'main'}
        write_json(folder / 'roles.json', result)
        return result

    def main(self, cohort: dict, folder: Path) -> dict:
        names = ('real_train', 'calibration_pool', 'real_test')
        audit = pd.read_csv(cohort['patient_audit'])
        frames = [read_analytic(cohort['paths'][n], self.cfg) for n in names]
        ids = [audit[audit.role == n].sort_values('analytic_row').patient_id_sha256.tolist() for n in names]
        return self.assign(frames, ids, folder, 'main')

    def null(self, cohort: dict, repeat: int | str, folder: Path) -> dict:
        # E is deliberately never read here.
        names = ('real_train', 'calibration_pool')
        audit = pd.read_csv(cohort['patient_audit'])
        development = pd.concat([read_analytic(cohort['paths'][n], self.cfg) for n in names], ignore_index=True)
        development['_patient_hash'] = sum([audit[audit.role == n].sort_values('analytic_row').patient_id_sha256.tolist()
                                           for n in names], [])
        # The salt gives v0.4.1 fresh null draws, disjoint in seed from the v0.4 runs whose
        # audit motivated the calibration change, so the new audit is not circular.
        salt = self.cfg.calibration.null_seed_salt
        parts = split_three(development, self.cfg, seed_for('null', self.cfg.calibration.assignment_seed, salt, repeat))
        ids = [p['_patient_hash'].tolist() for p in parts]
        frames = [p[list(self.cfg.data.columns)] for p in parts]
        result = self.assign(frames, ids, folder, f'null-{salt}-{repeat}')
        result.update(main_test_used=False, scope='development-scaled; development patients reused across runs')
        return result


def self_test() -> None:
    y = pd.Series([0, 1] * 30)
    assert len(set(stratified_indices(y, 20, 7))) == 20
