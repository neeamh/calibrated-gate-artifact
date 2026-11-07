# Implementation details

Full specification of the evidence behind the paper, summarised in Appendix E. Every value below is also fixed in `campaign/syntrustbench_campaign/` and `results/main_v5/configuration.json`.

## Cohort and features

We use the UCI Diabetes 130-US Hospitals data (encounters from 1999–2008). Values recorded as "?" are treated as missing. For each patient (`patient_nbr`) we keep the encounter with the smallest `encounter_id`. The outcome `readmitted_30d` is 1 when `readmitted` is "<30" and 0 otherwise. The eight count variables are `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient` and `number_diagnoses`. The twelve categorical variables are `race`, `gender`, `age`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `diag_1` (raw ICD-9 code, not grouped), `max_glu_serum`, `A1Cresult`, `insulin`, `change` and `diabetesMed`. No row is dropped for missing values. A missing category becomes its own level; where a model or distance needs a number, a missing count is replaced by the training median (gradient boosting handles missing values natively).

## Roles

Patients are split 40/40/20 into $T$, the calibration pool and $E$, stratified by outcome with a fixed seed. The pool is halved, again stratified, into $R$ and $P$, and the floor is an outcome-stratified sample of $T$ of the same size as $R$. Disjointness of every role is verified on hashed patient identifiers. Each null run applies the same procedure, with a fresh seed, to the 57,214 patients of $T$ and the pool, giving a pseudo-training set of 22,885 patients and pseudo-$R$, pseudo-$P$, pseudo-$E$ and floor of 11,443 each; $E$ is never read.

## Fidelity signals

Each is computed against $R$ and reported as the candidate’s value minus the floor’s value, except the constraint rate. *Coverage deficit*: for each categorical column (outcome included), the share of $R$’s patients whose level appears in the table; the signal is the largest deficit over columns. *Classifier two-sample AUC*: a gradient-boosting classifier (scikit-learn `HistGradientBoosting`, 200 iterations, native categorical splits on each column’s 250 most frequent levels, the rest pooled) is cross-fitted over five folds to separate $R$ from the table. *Wasserstein distance*: the 1-Wasserstein distance of each count variable, divided by the reference interquartile range (its standard deviation if the range is zero), averaged over variables. *Jensen–Shannon divergence*: base-2 divergence of each categorical column, averaged. *Association-matrix error*: the mean absolute difference between the upper triangles of the association matrices of $R$ and the table, using $|r|$ between counts, Cramér’s $V$ between categories and the correlation ratio between a category and a count. *Clinical constraints*: every count is non-negative and $1\le$ `time_in_hospital` $\le 14$; the signal is the largest per-rule violation rate among rows on which the rule can be evaluated.

## Utility signals

Each probe is trained on the table and tested on $E$, and compared with the same probe trained on the floor, so both models see the same number of rows. The probes are a random forest (200 trees, minimum leaf size 5), an $\ell_2$-regularised logistic regression ($C=1$) and the gradient-boosting model above; the first two use standardised counts and one-hot categories. *Retention* is $(\mathrm{AUC}_{\text{table}}-0.5)/(\mathrm{AUC}_{\text{real}}-0.5)$, and its signal is the upper limit of a paired bootstrap interval over the rows of $E$. Retention is not evaluable when the real-trained model’s own lower limit does not exceed 0.5. For the logistic probe, *calibration in the large* (observed over expected events) and the *calibration slope* (the coefficient of a logistic recalibration on the logit of predicted risk) are divided by the real-trained probe’s values. Both directions count as harm, so each contributes two signals. A table whose outcome has a single class fails utility outright.

## Privacy signals and threat model

The attacker holds the released table and the complete record of a patient, and wants to know whether that patient was in $T$; the attacker has no access to the generator. Disclosure means an exact training record in the release, or membership inference better than chance. Rows are encoded with counts standardised by $T$’s mean and standard deviation and one-hot categories (outcome included), and compared by Euclidean distance. *Exact-match excess*: the share of the table’s rows identical to some row of $T$ on all 21 columns, minus the share of $E$’s rows identical to some row of $T$, which estimates how often two different real patients coincide. *Closer-than-real share*: for up to 10,000 of the table’s rows, the share whose distance to the nearest of up to 10,000 rows of $T$ is below the 5th percentile of nearest-neighbour distances among those rows of $T$ themselves. *Membership-attack AUC*: up to 10,000 rows of $T$ (members) and of $E$ (non-members) are scored by their negative distance to the nearest released row; the signal is the AUC of that score. Row subsamples are drawn reproducibly; exact matches use all rows.

## Uncertainty

Intervals are 95% percentile bootstrap intervals with 200 resamples and fixed seeds. The detection-AUC interval resamples the cross-fitted predictions and omits the variance of refitting the classifier. Only retention enters the gate as an interval limit; every other signal is a point estimate.

## Missing and degenerate evidence

A signal missing on any fit run is excluded from calibration, and its dimension can then never pass. This happened only for equity, whose worst-group retention was missing on 36 of the 50 fit runs. A table missing a signal receives "not evaluated" for that dimension, which counts as non-acceptance; the verdict precedence is Fail, then not evaluated, then Pass. Each signal’s spread is floored at $10^{-8}$. One gating signal, the constraint rate, was zero on every fit run, so any violation ranks above all honest runs. The three means require all three dimension scores and do not score a table with an unevaluated dimension. The SDMetrics quality report (version 0.33.0) compares the table with $R$, treating counts as numerical and all other columns as categorical.

## Diagnostic dimensions

*Equity* considers each level of `race` and `gender` in $E$ eligible when it has at least 30 patients and at least 10 with each outcome. Its signals are the smallest within-group retention of the random-forest probe and the largest absolute difference in group share between the table and $R$. *Robustness* re-evaluates utility under five probe seeds, values deleted at random at 5%, 10% and 20% (three repeats each) and training fractions of 25%, 50%, 75% and 100%. Any change of the utility verdict counts as failure.

## Software

Python 3.12 with scikit-learn 1.7.2, SDV 1.37.4, SDMetrics 0.33.0 and smartnoise-synth 1.0.8 (AIM), with all versions pinned in the released environment. Every number in the paper is regenerated from the saved run outputs by the released scripts.
