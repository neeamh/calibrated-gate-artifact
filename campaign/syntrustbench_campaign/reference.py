"""Reference-calibrated Fidelity and Utility (experimental v0.4 add-on).

This module does not modify the frozen v0.3 dimensions. It adds three things
that v0.3 lacks, each tied to an established method:

1. A held-out real reference.  Fidelity is measured against real rows the
   generator never saw (as Synthcity's Benchmarks do with ``X.test()``), and
   every fidelity statistic is paired with the same statistic computed for an
   independent *real* sample of the same population (the "real-vs-real floor").
   A synthetic table is judged by its excess over what sampling alone produces.
2. Mass-weighted category coverage.  v0.3 counts category *levels*, so a level
   seen once weighs as much as a common one and a second real sample fails the
   0.95 rule.  Here coverage is the share of reference *patients* whose level
   appears in the synthetic table.
3. A joint-distribution test.  v0.3 checks marginals and pairwise associations
   only.  A cross-fitted classifier two-sample test (Lopez-Paz & Oquab, 2017;
   Synthcity ``detection_*``) and the propensity-score MSE (Snoke et al., 2018)
   are added, again relative to the real-vs-real floor.

Utility keeps the v0.3 primary rule (random-forest chance-corrected retention)
so official verdicts stay comparable, and adds: a real-reference retention
(positive control inside the dimension), a probe family (random forest,
logistic regression, gradient boosting, as in Synthcity's performance metrics),
and calibration (calibration-in-the-large and slope; Van Calster et al., 2019),
each relative to a model trained on real data.

All margins below are provisional, are recorded in the output, and are meant
to be replaced by empirically calibrated cut-offs from repeated real-vs-real
runs. Detection can raise Conditional but never Fail until that calibration
exists, because with tens of thousands of rows almost any difference is
statistically detectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from .results import DimensionEvaluator, DimensionResult

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from syntrustbench.constraints import constraint_violations
from syntrustbench.fidelity import _association_error, _jensen_shannon, _normalized_wasserstein
from syntrustbench.stats import (
    FeatureEncoder,
    aggregate_status,
    categorical_values,
    metric,
    percentile_interval,
    safe_auroc,
)
from syntrustbench.config import load_config
from syntrustbench.validation import _canonical_keys, _coerce_frame, _map_target

PROVISIONAL = MappingProxyType({
    # Fidelity
    "coverage_margin": 0.05,          # tolerated loss of reference patient mass beyond the floor
    "detection_margin": 0.05,         # AUC excess over the floor that raises Conditional
    "wasserstein_excess": 0.20,       # v0.3 thresholds, now applied to excess over floor
    "jsd_excess": 0.10,
    "association_excess": 0.15,
    "constraint_rate_fail": 0.01,     # unchanged from v0.3
    # Utility
    "retention_fail": 0.80,           # unchanged from v0.3
    "retention_conditional": 0.90,
    "calibration_ratio_band": 1.25,   # relative O/E and slope must stay within [1/1.25, 1.25]
})
MAX_TREE_LEVELS = 250  # HistGradientBoosting native categoricals allow at most 255 levels


# --------------------------------------------------------------------------- encoders
@dataclass
class _TreeEncoder:
    """Integer codes for native-categorical gradient boosting.

    Levels outside the ``MAX_TREE_LEVELS`` most frequent levels of the fitting
    table are pooled into one ``__OTHER__`` code (stated limitation: rare-level
    detail is invisible to this encoder; mass-weighted coverage covers it).
    """

    continuous: list[str]
    categorical: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "_TreeEncoder":
        self.levels = {
            c: categorical_values(frame[c]).value_counts().index[:MAX_TREE_LEVELS].tolist()
            for c in self.categorical
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[self.continuous].apply(pd.to_numeric, errors="coerce").copy()
        for c in self.categorical:
            values = categorical_values(frame[c])
            values = values.where(values.isin(self.levels[c]), "__OTHER__")
            out[c] = pd.Categorical(values, categories=self.levels[c] + ["__OTHER__"]).codes
        return out.reset_index(drop=True)

    @property
    def categorical_mask(self) -> list[bool]:
        return [False] * len(self.continuous) + [True] * len(self.categorical)


def _hgb(seed: int, mask: list[bool]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, categorical_features=mask, random_state=seed
    )


# --------------------------------------------------------------------------- fidelity pieces
def _mass_coverage(reference: pd.Series, candidate: pd.Series) -> np.ndarray:
    """Per-reference-row indicator: is this patient's level present in the candidate table?"""
    present = set(categorical_values(candidate))
    return categorical_values(reference).isin(present).to_numpy(dtype=float)


def _two_sample_detection(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    encoder: _TreeEncoder,
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted classifier two-sample test; returns labels and out-of-fold scores."""
    features = pd.concat(
        [encoder.transform(reference), encoder.transform(candidate)], ignore_index=True
    )
    labels = np.r_[np.zeros(len(reference)), np.ones(len(candidate))]
    scores = np.zeros(len(labels))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_rows, test_rows in splitter.split(features, labels):
        model = _hgb(seed, encoder.categorical_mask).fit(
            features.iloc[train_rows], labels[train_rows]
        )
        scores[test_rows] = model.predict_proba(features.iloc[test_rows])[:, 1]
    return labels, scores


def _pmse(labels: np.ndarray, scores: np.ndarray) -> float:
    share = float(labels.mean())
    return float(np.mean((scores - share) ** 2))


def _distance_block(reference, candidate, continuous, categorical) -> dict[str, float]:
    wass = [_normalized_wasserstein(reference[c], candidate[c]) for c in continuous]
    wass = [w for w in wass if np.isfinite(w)]
    jsd = [_jensen_shannon(reference[c], candidate[c])[0] for c in categorical]
    per_column_wass = {
        c: _normalized_wasserstein(reference[c], candidate[c]) for c in continuous
    }
    return {
        "mean_normalized_wasserstein": float(np.mean(wass)) if wass else float("nan"),
        "worst_column_normalized_wasserstein": float(np.nanmax(list(per_column_wass.values())))
        if per_column_wass else float("nan"),
        "worst_column": max(per_column_wass, key=lambda c: np.nan_to_num(per_column_wass[c], nan=-1))
        if per_column_wass else None,
        "mean_jensen_shannon_divergence": float(np.mean(jsd)) if jsd else float("nan"),
        "association_matrix_error": _association_error(
            reference, candidate, continuous, categorical
        ),
    }


# --------------------------------------------------------------------------- utility pieces
def _fit_predict(probe: str, train: pd.DataFrame, test: pd.DataFrame, target: str,
                 onehot: FeatureEncoder, tree: _TreeEncoder, seed: int) -> np.ndarray:
    y = train[target].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.full(len(test), float(y.mean()))
    if probe == "random_forest":  # identical to the v0.3 primary probe
        model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed
        )
        return model.fit(onehot.transform(train), y).predict_proba(onehot.transform(test))[:, 1]
    if probe == "logistic_regression":
        model = LogisticRegression(C=1.0, max_iter=3000)
        return model.fit(onehot.transform(train), y).predict_proba(onehot.transform(test))[:, 1]
    if probe == "gradient_boosting":
        model = _hgb(seed, tree.categorical_mask)
        return model.fit(tree.transform(train), y).predict_proba(tree.transform(test))[:, 1]
    raise ValueError(probe)


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration-in-the-large (observed/expected) and calibration slope."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2 or np.allclose(p, p[0]):
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=2000).fit(logit, y).coef_[0][0]
    return float(y.mean() / p.mean()), float(slope)


def _retention(y, base, cand):
    real, syn = safe_auroc(y, base), safe_auroc(y, cand)
    return (syn - 0.5) / (real - 0.5) if np.isfinite(real) and real > 0.5 else float("nan")


def _retention_verdict(point, low, high, base_low) -> str:
    if base_low is None or not np.isfinite(base_low) or base_low <= 0.5:
        return "NotEvaluated"
    if high is not None and high < PROVISIONAL["retention_fail"]:
        return "Fail"
    if low is None or high is None:
        return "NotEvaluated"
    return "Pass" if low >= PROVISIONAL["retention_conditional"] else "Conditional"


# --------------------------------------------------------------------------- main context
class ReferenceContext:
    """Everything that depends only on real data, computed once per cohort.

    ``real_train`` is the generator's training data, ``real_reference`` an
    independent real sample of the same population and size (e.g. the
    calibration pool), ``real_test`` the untouched utility test set.
    """

    PROBES = ("random_forest", "logistic_regression", "gradient_boosting")

    def __init__(self, real_train, real_reference, real_test, config, *,
                 iterations: int | None = None, seed: int | None = None):
        # Reuse frozen v0.3 validation helpers: declared columns, types, target mapping.
        # v0.3 stops on any exact row shared between two real tables, as a leakage proxy.
        # Here patient disjointness of every role is already asserted upstream by hashed
        # patient IDs (RoleAssigner), so an exact match is two different patients with
        # identical recorded values: an expected real-real coincidence (the quantity the
        # v0.4 privacy null models). Stopping, dropping or re-drawing such runs would
        # select the null sample, so the counts are recorded and evaluation continues.
        benchmark = load_config(config)

        def _real(frame, label):
            if not isinstance(frame, pd.DataFrame):
                frame = pd.read_csv(frame)
            frame, _ = _coerce_frame(frame.copy(), benchmark, label)
            frame, _ = _map_target(frame, benchmark, label, require_two_classes=True)
            return frame.reset_index(drop=True)

        self.config = benchmark
        self.train = _real(real_train, "real_train")
        self.test = _real(real_test, "real_test")
        self.reference = _real(real_reference, "real_reference")
        train_keys = set(_canonical_keys(self.train, benchmark).tolist())
        self.exact_row_coincidences = {
            name: int(_canonical_keys(frame, benchmark).isin(train_keys).sum())
            for name, frame in (("reference", self.reference), ("test", self.test))}
        self.iterations = iterations or self.config.bootstrap_iterations
        self.seed = self.config.random_seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        cfg = self.config
        self.continuous = cfg.continuous
        self.categorical = cfg.categorical + cfg.binary
        self.target = cfg.target
        self.tree_fidelity = _TreeEncoder(self.continuous, self.categorical).fit(self.reference)
        self.onehot = FeatureEncoder(cfg.continuous_features, cfg.categorical_features).fit(self.train)
        self.tree_utility = _TreeEncoder(cfg.continuous_features, cfg.categorical_features).fit(self.train)
        self.y_test = self.test[self.target].astype(int).to_numpy()

        # Fidelity floor: the generator's own training data versus the reference.
        self.floor = self._fidelity_statistics(self.train)
        # Utility baselines: train on real training data and on the real reference.
        self.trtr = {p: self._predict(p, self.train) for p in self.PROBES}
        self.reference_pred = {p: self._predict(p, self.reference) for p in self.PROBES}
        self.trtr_auroc_ci = {
            p: percentile_interval(
                safe_auroc(self.y_test, self.trtr[p]),
                lambda rows, p=p: safe_auroc(self.y_test[rows], self.trtr[p][rows]),
                len(self.y_test), self.rng, self.iterations, 0.95)
            for p in self.PROBES
        }
        lr = "logistic_regression"
        self.reference_calibration = _calibration(self.y_test, self.trtr[lr])

    # ---------------------------------------------------------------- helpers
    def _predict(self, probe, train_frame):
        return _fit_predict(probe, train_frame, self.test, self.target,
                            self.onehot, self.tree_utility, self.seed)

    def _fidelity_statistics(self, candidate: pd.DataFrame) -> dict:
        coverage_rows = {c: _mass_coverage(self.reference[c], candidate[c]) for c in self.categorical}
        coverage = {c: float(v.mean()) for c, v in coverage_rows.items()}
        labels, scores = _two_sample_detection(self.reference, candidate, self.tree_fidelity, self.seed)
        auc = safe_auroc(labels, scores)
        auc_ci = percentile_interval(
            auc, lambda rows: safe_auroc(labels[rows], scores[rows]),
            len(labels), self.rng, self.iterations, 0.95)
        stats = _distance_block(self.reference, candidate, self.continuous, self.categorical)
        stats.update({
            "coverage": coverage, "coverage_rows": coverage_rows,
            "detection_auc": auc, "detection_auc_ci": auc_ci,
            "pmse": _pmse(labels, scores),
        })
        return stats

    # Backward-compatible dict API; use the evaluators for typed results.
    def fidelity(self, synthetic: pd.DataFrame) -> dict:
        """Evaluate fidelity, consuming the shared RNG exactly as before."""
        return ReferenceFidelity(self).evaluate(synthetic).to_dict()

    def utility(self, synthetic: pd.DataFrame) -> dict:
        """Evaluate utility, consuming the shared RNG exactly as before."""
        return ReferenceUtility(self).evaluate(synthetic).to_dict()


@dataclass(frozen=True)
class ReferenceFidelity(DimensionEvaluator):
    """Original fidelity calculations over one shared real-data context.

    Metrics retain the original units and higher_is_worse metadata.
    Evaluator construction consumes no randomness and fits no models.
    Calls advance context.rng: preserve table and dimension ordering.
    """

    context: ReferenceContext

    def evaluate(self, synthetic: pd.DataFrame) -> DimensionResult:
        """Score a candidate without changing the supplied statistical rules."""
        ctx = self.context
        P = PROVISIONAL
        cand = ctx._fidelity_statistics(synthetic)
        floor = ctx.floor
        metrics: dict[str, dict] = {}
        weaknesses: list[str] = []

        # Mass-weighted coverage, judged as loss relative to the real floor.
        deficits = {}
        for c in ctx.categorical:
            rows_syn, rows_floor = cand["coverage_rows"][c], floor["coverage_rows"][c]
            point = float(rows_floor.mean() - rows_syn.mean())
            low, high = percentile_interval(
                point, lambda r, a=rows_floor, b=rows_syn: float(a[r].mean() - b[r].mean()),
                len(rows_syn), ctx.rng, ctx.iterations, 0.95)
            deficits[c] = (point, low, high)
        worst = max(deficits, key=lambda c: deficits[c][0])
        point, low, high = deficits[worst]
        coverage_fail = low is not None and low > P["coverage_margin"]
        metrics["mass_weighted_coverage_deficit"] = metric(
            point, ci_low=low, ci_high=high, interval_type="paired reference-row bootstrap (95%)",
            higher_is_worse=True, flag=coverage_fail,
            threshold=f"lower 95% bound > {P['coverage_margin']:.2f} -> Fail",
            note=(f"Worst column '{worst}': share of reference patients whose category is "
                  f"absent from the synthetic table, minus the same share for the real floor "
                  f"(synthetic {cand['coverage'][worst]:.4f}; floor {floor['coverage'][worst]:.4f})."))
        if coverage_fail:
            weaknesses.append(f"{worst}: synthetic omits categories held by "
                              f"{point:.1%} more reference patients than a real sample does")

        # Joint-distribution test (cannot Fail before null calibration).
        auc, (alow, ahigh) = cand["detection_auc"], cand["detection_auc_ci"]
        fhigh = floor["detection_auc_ci"][1]
        detected = (alow is not None and fhigh is not None
                    and alow > fhigh + P["detection_margin"])
        metrics["detection_auc"] = metric(
            auc, ci_low=alow, ci_high=ahigh,
            interval_type="bootstrap over cross-fitted predictions (95%)",
            higher_is_worse=True, flag=detected,
            threshold=(f"lower bound > floor upper bound + {P['detection_margin']:.2f} "
                       "-> Conditional (Fail requires null calibration)"),
            note=(f"Cross-fitted gradient-boosting classifier two-sample test against the "
                  f"held-out real reference; real-vs-real floor AUC {floor['detection_auc']:.3f}."))
        floor_pmse = floor["pmse"]
        metrics["pmse_ratio_to_real_floor"] = metric(
            cand["pmse"] / floor_pmse if floor_pmse > 0 else float("nan"), higher_is_worse=True,
            note="Propensity-score MSE divided by the real-vs-real pMSE (empirical null; 1 = like real).")

        # Low-order distances, now as excess over the floor (v0.3 thresholds).
        for name, key in [("mean_normalized_wasserstein", "wasserstein_excess"),
                          ("mean_jensen_shannon_divergence", "jsd_excess"),
                          ("association_matrix_error", "association_excess")]:
            excess = cand[name] - floor[name]
            metrics[f"{name}_excess"] = metric(
                excess, higher_is_worse=True, flag=bool(excess > P[key]),
                threshold=f"> {P[key]:.2f} -> Conditional",
                note=f"Synthetic {cand[name]:.4f} minus real floor {floor[name]:.4f}.")
        metrics["worst_column_normalized_wasserstein"] = metric(
            cand["worst_column_normalized_wasserstein"], higher_is_worse=True,
            note=(f"Worst continuous column '{cand['worst_column']}' (reported so a column "
                  "average cannot hide one broken variable); floor "
                  f"{floor['worst_column_normalized_wasserstein']:.4f}."))

        # Clinical constraints: unchanged v0.3 rule.
        rate, per_rule, _ = constraint_violations(synthetic, ctx.config.clinical_constraints)
        constraint_fail = rate is not None and rate > P["constraint_rate_fail"]
        metrics["clinical_constraint_violation_rate"] = metric(
            rate, higher_is_worse=True, flag=constraint_fail,
            status="Available" if rate is not None else "NotEvaluated",
            threshold=f"> {P['constraint_rate_fail']:.2f} -> Fail", note=str(per_rule))

        conditional = detected or any(
            metrics[f"{n}_excess"]["flag"] for n in
            ["mean_normalized_wasserstein", "mean_jensen_shannon_divergence", "association_matrix_error"])
        status = aggregate_status([
            "Fail" if (coverage_fail or constraint_fail) else "Pass",
            "NotEvaluated" if rate is None else "Pass",
            "Conditional" if conditional else "Pass",
        ])
        return DimensionResult.from_dict({"dimension": "fidelity_reference_calibrated", "status": status,
                "metrics": metrics, "weaknesses": weaknesses,
                "provisional_margins": {k: v for k, v in P.items() if not k.startswith(("retention", "calibration"))},
                "per_column_coverage": cand["coverage"],
                "floor_per_column_coverage": floor["coverage"]})


@dataclass(frozen=True)
class ReferenceUtility(DimensionEvaluator):
    """Original utility calculations over one shared real-data context.

    Metrics retain the original units and higher_is_worse metadata.
    Evaluator construction consumes no randomness and fits no models.
    Calls advance context.rng: preserve table and dimension ordering.
    """

    context: ReferenceContext

    def evaluate(self, synthetic: pd.DataFrame) -> DimensionResult:
        """Score a candidate without changing the supplied statistical rules."""
        ctx = self.context
        P = PROVISIONAL
        y = ctx.y_test
        metrics: dict[str, dict] = {}
        weaknesses: list[str] = []
        outcome_levels = synthetic[ctx.target].astype(int).nunique()
        verdicts = {}
        for probe in ctx.PROBES:
            tstr = ctx._predict(probe, synthetic)
            base_low = ctx.trtr_auroc_ci[probe][0]
            for label, cand in [("synthetic", tstr), ("real_reference", ctx.reference_pred[probe])]:
                point = _retention(y, ctx.trtr[probe], cand)
                low, high = (None, None)
                if np.isfinite(point):
                    low, high = percentile_interval(
                        point, lambda r, c=cand: _retention(y[r], ctx.trtr[probe][r], c[r]),
                        len(y), ctx.rng, ctx.iterations, 0.95)
                verdict = _retention_verdict(point, low, high, base_low)
                if label == "synthetic":
                    verdicts[probe] = verdict
                    if probe == "logistic_regression":
                        oe, slope = _calibration(y, cand)
                metrics[f"{label}_retention_{probe}"] = metric(
                    point if np.isfinite(point) else None, ci_low=low, ci_high=high,
                    interval_type="paired held-out real-test bootstrap (95%)",
                    higher_is_worse=False, status=verdict if label == "synthetic" else "Available",
                    note=("Chance-corrected AUROC retention versus the same probe trained on "
                          "real training data." + (" Positive control: an independent real "
                          "sample; should be near 1." if label == "real_reference" else "")))
        primary = verdicts["random_forest"]
        disagree = sorted(p for p, v in verdicts.items() if v != primary)
        # Only a confident failure elsewhere (upper bound below the Fail line) downgrades a
        # primary Pass; a merely wider interval does not (it would flag real data, as the
        # half-size positive control showed for logistic regression).
        probe_conditional = primary == "Pass" and any(verdicts[p] == "Fail" for p in disagree)
        metrics["probe_agreement"] = metric(
            ", ".join(f"{p}={v}" for p, v in verdicts.items()), higher_is_worse=False,
            flag=bool(disagree),
            note="Primary verdict is the v0.3 random forest; a primary Pass becomes "
                 "Conditional only if another probe confidently Fails (probe-dependent utility).")

        ref_oe, ref_slope = ctx.reference_calibration
        rel_oe = oe / ref_oe if np.isfinite(oe) and ref_oe else float("nan")
        rel_slope = slope / ref_slope if np.isfinite(slope) and ref_slope else float("nan")
        band = P["calibration_ratio_band"]
        outside = lambda v: bool(np.isfinite(v) and not (1 / band <= v <= band))
        miscalibrated = outside(rel_oe) or outside(rel_slope)
        metrics["relative_calibration_in_the_large"] = metric(
            rel_oe if np.isfinite(rel_oe) else None, higher_is_worse=False, flag=outside(rel_oe),
            threshold=f"outside [{1/band:.2f}, {band:.2f}] -> Conditional",
            note=f"Observed/expected events of the synthetic-trained logistic probe on real test data, "
                 f"divided by the real-trained value ({ref_oe:.3f}).")
        metrics["relative_calibration_slope"] = metric(
            rel_slope if np.isfinite(rel_slope) else None, higher_is_worse=False,
            flag=outside(rel_slope),
            threshold=f"outside [{1/band:.2f}, {band:.2f}] -> Conditional",
            note=f"Calibration slope of the synthetic-trained logistic probe divided by the "
                 f"real-trained slope ({ref_slope:.3f}).")
        if outcome_levels < 2:
            weaknesses.append("Synthetic outcome collapsed to a single class; no model can be trained.")
        status = aggregate_status([
            "Fail" if outcome_levels < 2 else "Pass",
            primary,
            "Conditional" if (probe_conditional or miscalibrated) else "Pass",
        ])
        return DimensionResult.from_dict({"dimension": "utility_reference_calibrated", "status": status,
                "metrics": metrics, "weaknesses": weaknesses,
                "probe_verdicts": verdicts,
                "provisional_margins": {k: v for k, v in P.items() if k.startswith(("retention", "calibration"))}})
