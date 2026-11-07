"""Nested perturbation strategies, adapted from the supplied v0.3 notebook."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import ExperimentConfig, seed_for
from .core import UnavailableAnalysis


def changed_cells(before: pd.DataFrame, after: pd.DataFrame) -> set:
    same = before.eq(after) | (before.isna() & after.isna())
    return set(map(tuple, np.argwhere(~same.to_numpy())))


@dataclass
class Perturbation(ABC):
    cfg: ExperimentConfig
    train: pd.DataFrame

    @abstractmethod
    def change(self, table: pd.DataFrame, dose: float, seed: int) -> tuple[pd.DataFrame, dict]:
        raise NotImplementedError

    def apply(self, table: pd.DataFrame, dose: float, seed: int) -> tuple[pd.DataFrame, dict]:
        if not 0 <= dose <= 1:
            raise ValueError('Dose must be in [0, 1].')
        if dose == 0:
            out, audit = table.copy(deep=True), {'realized_dose': 0., 'selected_rows': []}
        else:
            out, audit = self.change(table.copy(deep=True), dose, seed)
        if out.shape != table.shape or list(out) != list(table):
            raise ValueError('Perturbation changed the schema or row count.')
        cells = changed_cells(table, out)
        return out, {'requested_dose': dose, 'changed_cell_count': len(cells),
                     'changed_cells': sorted([list(p) for p in cells]), **audit}


class ExactCopy(Perturbation):
    def change(self, table, dose, seed):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(table))
        donors = rng.integers(0, len(self.train), len(table))
        rows = order[:int(np.ceil(dose * len(table)))]
        from syntrustbench.config import load_config
        from syntrustbench.validation import canonical_row_keys
        config = load_config(self.cfg.benchmark())
        keys = set(canonical_row_keys(self.train, config))
        before = np.isin(canonical_row_keys(table, config), list(keys))
        table.iloc[rows] = self.train.iloc[donors[:len(rows)]].to_numpy()
        after = np.isin(canonical_row_keys(table, config), list(keys))
        return table, {'realized_dose': float((after & ~before).mean()),
                       'total_copy_rate': float(after.mean()), 'baseline_copy_rate': float(before.mean()),
                       'selected_rows': rows.tolist(), 'donor_rows': donors[:len(rows)].tolist(),
                       'denominator': 'all rows; newly copied rows'}


class SubgroupSuppression(Perturbation):
    def change(self, table, dose, seed):
        s = self.cfg.injections
        mask = table[s.subgroup_column].eq(s.subgroup_value).to_numpy()
        members, others = np.flatnonzero(mask), np.flatnonzero(~mask)
        if not len(members) or not len(others):
            raise UnavailableAnalysis('Suppression needs target subgroup and replacement donors.')
        rng = np.random.default_rng(seed)
        order = rng.permutation(members)
        donors = rng.choice(others, len(members), replace=True)
        rows = order[:int(np.ceil(dose * len(members)))]
        table.iloc[rows] = table.iloc[donors[:len(rows)]].to_numpy()
        return table, {'realized_dose': len(rows)/len(members), 'selected_rows': rows.tolist(),
                       'donor_rows': donors[:len(rows)].tolist(), 'baseline_subgroup_n': len(members),
                       'remaining_subgroup_n': len(members)-len(rows), 'denominator': 'baseline target-group rows'}


class FeatureShuffle(Perturbation):
    def change(self, table, dose, seed):
        column = self.cfg.injections.shuffle_feature
        original = table[column].copy()
        order = np.random.default_rng(seed).permutation(len(table))
        previous = 0
        doses = dict(self.cfg.injections.doses)['feature_shuffle']
        if dose not in doses:
            raise ValueError('Feature-shuffle dose must be in the frozen nested grid.')
        for level in doses:
            end = int(np.ceil(level * len(table)))
            block = order[previous:end]
            if level <= dose and len(block) > 1:
                table.loc[table.index[block], column] = np.roll(original.iloc[block].to_numpy(), 1)
            previous = end
        same = original.eq(table[column]) | (original.isna() & table[column].isna())
        return table, {'realized_dose': float((~same).mean()),
                       'selected_rows': order[:int(np.ceil(dose*len(table)))].tolist(),
                       'denominator': 'all rows; fixed disjoint rotation blocks preserve marginals; ties remain'}


class MCARMissingness(Perturbation):
    def change(self, table, dose, seed):
        selected, total, count = {}, 0, 0
        for column in self.cfg.injections.missing_features:
            eligible = np.flatnonzero(table[column].notna().to_numpy())
            order = np.random.default_rng(seed_for('mcar', seed, column)).permutation(eligible)
            rows = order[:int(np.ceil(dose * len(eligible)))]
            table.loc[table.index[rows], column] = np.nan
            selected[column] = rows.tolist(); total += len(eligible); count += len(rows)
        if not total:
            raise UnavailableAnalysis('No observed cells available for MCAR injection.')
        return table, {'realized_dose': count/total, 'selected_cells_by_column': selected,
                       'denominator': 'initially observed cells in the four prespecified predictors'}


class TrainCopy(Perturbation):
    def change(self, table, dose, seed):
        if len(table) > len(self.train):
            raise UnavailableAnalysis('Training data smaller than comparator.')
        out = self.train.sample(n=len(table), random_state=seed).reset_index(drop=True)
        return out, {'realized_dose': 1., 'denominator': 'all rows copied from generator training set'}


class IndependentColumns(Perturbation):
    def change(self, table, dose, seed):
        out = pd.DataFrame({c: table[c].sample(frac=1, random_state=seed_for(seed, c)).to_numpy() for c in table})
        return out, {'realized_dose': 1., 'denominator': 'independently permuted columns; exact marginals retained'}


def strategy(name: str, cfg: ExperimentConfig, train: pd.DataFrame) -> Perturbation:
    classes = {'exact_copy': ExactCopy, 'subgroup_suppression': SubgroupSuppression,
               'feature_shuffle': FeatureShuffle, 'mcar': MCARMissingness,
               'copy': TrainCopy, 'negative': IndependentColumns}
    return classes[name](cfg, train)


def self_test() -> None:
    a = pd.DataFrame({'x': [1, np.nan]})
    assert not changed_cells(a, a.copy())
