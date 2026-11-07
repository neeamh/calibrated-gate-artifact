# v0.4.2 follow-up: prespecified design and analysis

Written on 7 November 2026, before any follow-up table was scored. The run is `followup_v042.py`;
its SHA-256 is recorded in `runs/followup_v042/manifest.json` when the run starts. Nothing in this
follow-up changes the main_v5 campaign, its frozen calibration model, its evaluators or its judges.

## Why

Review identified two scientific gaps in the main_v5 comparison:

1. **Exchangeability.** The calibration guarantee assumes honest candidates are exchangeable with
   the fit runs. Main-experiment tables (14,303 rows) are larger than null-run tables (11,443 rows),
   so the guarantee did not cover them by construction.
2. **Replication.** The three positive controls contain the same patients, so the 39 flawed tables
   are 13 conditions on one base table with three injection seeds.

A third concern is depth: a maximum-type rule is expected to win against isolated (sparse) flaws,
and the main design contains mostly those.

## Design

**Bases.** The 50 held-out audit null runs of main_v5, in the frozen audit order recorded in the
calibration model. Each base's honest candidate is the pseudo-P table that the audit already scored.
Flaws are injected into that table and scored against the same base's pseudo-roles and saved
reference context. Honest and flawed tables therefore come from the fit runs' sampling procedure,
at the same role sizes and with the same code. If the run stops early, the completed bases are the
first k of this prespecified order, which is a random order.

**Single-flaw conditions (13 per base).** The main design's grid, unchanged: copied training rows at
0.1%, 1% and 5%; `number_inpatient` rotated in 10%, 50% and 90% of rows; values missing completely at
random at 5%, 15% and 30% in the four prespecified predictors; African-American patients replaced at
10%, 50% and 90%; and a copy control (all rows sampled from the base's pseudo-training set). Each flaw
family uses one seed stream per base, so higher doses contain the changes of lower doses.

**Dense conditions (6 per base).** Three mild components, chosen before any scoring from the main_v5
evidence and the code, each aimed at one gating dimension:

| Component | Change | Target | Why this dose |
|---|---|---|---|
| F | `number_inpatient` rotated in 50% of rows | fidelity | accepted by every rule in all three main_v5 replicates |
| U | outcome rotated in 10% of rows (prevalence unchanged) | utility | weakens the outcome's relation to predictors slightly |
| P | one training row copied in | privacy | in main_v5 a single copied row (AIM, seed 43) stayed below the calibrated exact-match threshold |

Conditions: U, P, FU, FP, UP, FUP. F alone is the single-flaw cell `feature_shuffle@0.5` (same seed
stream) and is not rerun. Components are applied in the order F, U, P, each with its own seed stream,
so every combination contains exactly the changes of its parts.

**Supplementary audit.** Fresh real-versus-real null runs with repeat ids 100 to 249, drawn by the
campaign's own `RoleAssigner.null` with the v0.4.1 seed salt, and scored only with the frozen model.
They never enter any fit.

**Scoring.** Every table gets the calibrated gate and the matched-cost decisions of the arithmetic,
geometric and harmonic means and SDMetrics at alpha 0.025, 0.05 and 0.10, exactly as in main_v5.
The frozen v0.3 profiler is not run. A table that a rule cannot score counts as not accepted by
that rule, and the number of such cases is reported.

## Primary analyses

1. **Replication of the matched comparison (alpha 0.05).** For each mean J and each base, the
   difference D_J = (single-flaw tables accepted by J) minus (accepted by the gate), over the 13
   single-flaw conditions. Report the mean of D_J with a 95% bootstrap interval over bases, the
   number of bases with D_J > 0, = 0 and < 0, and an exact two-sided sign test over bases with
   D_J different from 0. SDMetrics is reported separately as different evidence.
2. **Per-condition acceptance.** For every condition and rule, the share of the 50 bases accepted,
   with Wilson intervals.
3. **Dense map.** Acceptance of U, P, FU, FP, UP and FUP (and F, from the single grid) by each rule.
   The theory in the paper predicts that the gate's advantage shrinks, and may reverse, as the same
   amount of harm is spread over more dimensions. The result is reported whichever way it goes.
4. **Calibration validation.** (a) Reshuffled fit/audit splits of the 100 main_v5 null runs:
   mean audit rejection rate over 1,000 random splits against r/(n+1), computed from saved evidence.
   (b) Supplementary audit: gate and per-dimension non-acceptance among the 150 fresh runs, with exact
   one-sided 95% upper bounds, alone and pooled with the original 50 audit runs.

## Interpretation fixed in advance

- The main_v5 claim is **supported** if the mean of D_J is positive for the arithmetic mean with a
  95% interval excluding zero, and the gate is never worse on average than any mean.
- The comparison with the geometric and harmonic means is reported as an estimate with its
  interval; a difference whose interval includes zero is reported as no detectable difference.
- Calibration is **validated** if the reshuffled-split mean rejection rate is at or below the nominal
  level within Monte Carlo error, and the supplementary audit's one-sided upper bound is reported
  as the precision achieved, whatever its value.
- These follow-up results replace the main_v5 known-bad comparison as the paper's primary evidence;
  main_v5 results are kept and labelled as the original analysis.
