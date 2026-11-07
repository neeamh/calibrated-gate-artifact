# SynTrustBench v0.4.1 — one-page specification

## Claim (locked)

When synthetic clinical data is scored on several dimensions, averaging the scores lets a serious failure in one dimension hide behind good scores in the others. A gate that requires every dimension to pass catches those failures, but only if each dimension is first calibrated against real-versus-real comparisons, so the gate does not also reject real patients.

## Problem

Synthetic-data tools report per-dimension results and then combine them: SDMetrics and TabSynDex average, Yan et al. (2022) sum weighted ranks, and Belgodere et al. use a geometric mean argued to act as an "and." Gates have been proposed (in prior work cited in the paper) and used implicitly for privacy thresholds (SynQP), but no study has tested whether the combining rule matters. In clinical data it should: better accuracy cannot buy back a privacy leak or a distorted cohort. A gate also has an untested cost: it inherits the false alarms of its least specific dimension. Uncalibrated frozen v0.3 rejected all 35 honest real-vs-real comparisons it evaluated.

## Solution

A non-compensatory gate over **fidelity, utility and privacy**, the dimensions with established real-referenced tests. Each dimension is calibrated so honest real data passes at a known rate, and a table passes only if all three pass. **Equity and robustness are reported as diagnostics, not gated**: no benchmark offers a validated decision rule for them, and the v0.4 audit showed the obvious rules reject real data. The contribution is the first controlled test of combining rules on identical evidence, measuring both errors: known-bad tables accepted and honest data rejected.

## Methodology

**Data.** UCI Diabetes 130-US Hospitals, earliest encounter per patient, 30-day readmission. Frozen, hash-verified, patient-disjoint roles: generator training T, reference R, positive-control pool P, test E.

**Tables.** Gaussian Copula, CTGAN and TVAE (SDV 1.37.4, 300 epochs) and DP-AIM (ε = 1), three seeds each. Controls: positive (held-out real), copy (training rows) and negative (shuffled columns). Injections on the positive controls and the Gaussian Copula tables: copied rows (0.1–5%), subgroup suppression, feature shuffle (10–90%) and missingness (5–30%). All comparisons are size-matched.

**Dimensions.** Fidelity: coverage, classifier two-sample AUC, pMSE and low-order distances, each as excess over the real floor. Utility: chance-corrected train-on-synthetic, test-on-real retention for three probes, plus relative calibration. Privacy: exact-match excess over the real–real rate, distance membership attack, and the closer-than-real share.

**Calibration.** 100 fresh real-vs-real null runs (new seeds, never using E): 50 to fit, 50 held out for audit. Full conformal: a new table is pooled with the fit runs, standardized and ranked against them, giving exact honest rejection ⌊α(n+1)⌋/(n+1) (3.9% at α = 0.05) under exchangeability, however skewed the signals.

**Admission rule (prespecified).** A dimension may gate only if it is evaluable on all 50 audit runs and an exact one-sided binomial test finds no excess non-acceptance over 5%. The rule applies to all five dimensions and to the gate. A failure is reported, not re-tuned.

**Judges (matched cost).** The gate versus mean (TabSynDex; Yan et al. defaults), geometric (Belgodere et al.), harmonic and native SDMetrics. All are calibrated on the same fit runs to the same honest-data rate; the v0.4 threshold τ* is kept as a secondary analysis.

**Primary outcomes.** (1) The admission table. (2) For each judge at matched cost: acceptance of real data with one serious flaw (injections on positive controls, plus the copy control) and of held-out honest runs, by flaw and dose. (3) Ranking inversions. Frozen v0.3 is reported as the uncalibrated baseline.

**Prespecified expectation.** The gate is a max-type rule and averages are sum-type rules. At equal false-alarm rates, max-type rules catch one badly-off dimension better, and sum-type rules can win on small problems spread across dimensions. We expect the gate to beat the arithmetic mean and SDMetrics, with the harmonic mean (dominated by its lowest score) close to the gate.

**Limits.** One public cohort and one task; tabular data only. Null roles are smaller than the main roles; the audit tests exchangeability. Privacy results are empirical lower bounds. Not clinical certification.
