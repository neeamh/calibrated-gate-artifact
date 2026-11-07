# Good Scores Can Hide Bad Data: research artifact

Code, saved results and analysis scripts for the paper *Good Scores Can Hide Bad Data: A Calibrated
Test of How Trust Evidence for Synthetic Clinical Data Should Be Combined* (submitted to ACM TRUST 2027).

The paper asks whether the rule that combines fidelity, utility and privacy evidence changes which
synthetic clinical tables pass, and whether a non-compensatory gate can reject flawed tables without
rejecting real patients. This repository lets a reader do two things:

1. **Regenerate every number, table and figure in the paper** from the saved results, on a laptop,
   without the source data. This is the main reproducibility check.
2. **Rerun the experiments** from the public source data: the main campaign and the v0.4.2 follow-up
   (several hours each on a CPU).

The paper's primary evidence is the **v0.4.2 follow-up**: flaws planted in the 50 held-out audit runs
of the main campaign (950 flawed tables), 150 fresh honest runs, reshuffled-split calibration checks
and sensitivity analyses on the 50 bases. The main campaign's 39-table comparison is reported in the
paper as the original analysis, and the follow-up reuses its frozen calibration model unchanged.

## 1. Regenerate the paper's numbers (recommended)

Requires Python 3.11 or later.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt
pip install --no-deps third_party/syntrustbench_v030
PYTHON=python ./reproduce_paper_numbers.sh
```

The script reruns every analysis on the saved evidence in `results/`, writes the LaTeX macros, tables
and figures to `paper/generated/`, and checks every regenerated file against the committed version:
`.tex` and `.csv` byte for byte, `.json` up to rounding in the last digits. It ends with
`All regenerated numbers match the paper.` The full run takes about 10 minutes, most of it the 1,000
reshuffled calibration splits; `QUICK=1` skips those and takes about 3 minutes.

| Paper item | Script | Output |
|---|---|---|
| **Primary results:** abstract and Sections 5.1--5.6 numbers, results figure, Table 4 (flawed tables accepted, 50 bases), Appendix C table (sensitivity on the 50 bases) | `extract_followup_tables.py`, `followup_reanalysis.py`, `make_followup.py` | `numbers_followup.tex`, `table_followup.tex`, `table_followup_sensitivity.tex`, `fig/fig1.pdf`; analysis files in `results/followup_v042/analysis/` |
| Calibration validation by 1,000 reshuffled splits (Section 5.1, Appendix D) | `resplit_validation.py` | `resplit_validation.json` |
| Table 3 (false alarms on honest data), Section 5.7 (original analysis), Appendix B | `make_numbers.py`, `extra_analyses.py`, `mechanism_status.py`, `make_extra_tables.py`, `reanalysis.py`, `make_reanalysis_tables.py` | `numbers.tex`, `numbers_extra.tex`, `numbers_reanalysis.tex`, `table_admission.tex` and others |
| Original 39-table figure and tables (superseded, kept for the record) | `make_figures.py`, `make_extra_tables.py` | `fig/fig1_original.pdf`, `table_injection.tex`, `table_sensitivity_all.tex` |
| Dimension icons in the design figure | `icons/make_*_icon.py` | SVG and PDF (run separately) |

The analysis scripts import the campaign's own calibration and judge code from `campaign/`, so every
decision is re-derived with the code that produced the results; nothing is refit.

## 2. Rerun the experiments

**Follow-up (v0.4.2).** `campaign/followup_v042/` holds the run script, its prespecified design
(`PREREGISTRATION_v0.4.2.md`, written before any follow-up table was scored), the run guide and the
analysis scripts. The run needs the original campaign's run folder, so it follows a full campaign
rerun. The judgement calls of the follow-up are appended to `campaign/JUDGEMENT_CALLS.md`. The saved
follow-up profiles (evidence and decisions of every table, no patient records) are in
`results/followup_v042/profiles/` and `supplementary/`.

**Main campaign.**

See `campaign/README.md`. In short: download the source data as described in `data/README.md`,
install `requirements-campaign.txt` and the frozen dependency in `third_party/`, open
`campaign/syntrustbench_v04_experiment.ipynb`, set `data.raw_csv`, and run smoke mode before the full
campaign. The notebook embeds the same library as `campaign/syntrustbench_campaign/`. The run reported
in the paper executed 552 jobs (100 real-versus-real null runs, 12 generator fits, controls and
injected flaws); `campaign/README.md` gives time estimates. The notebook's setup step checks out the
frozen dependency from its public repository at the pinned commit and verifies its source hashes; its
address is withheld during review (set `FROZEN_V030_REPOSITORY` once it is public), so a full rerun is
possible only after review, while step 1 works offline. Bitwise identity across hardware is not claimed; the decisions in the paper
are re-derived from the saved evidence in step 1.

## Layout

```
campaign/
  syntrustbench_campaign/        campaign library (calibration, judges, evaluation, perturbations, ...)
  syntrustbench_v04_experiment.ipynb   campaign notebook (outputs cleared)
  PROTOCOL_v0.4.1.md             one-page prespecified protocol for the confirmatory run
  JUDGEMENT_CALLS.md             every design choice and amendment, with its reason
  README.md                      how to run the campaign
third_party/syntrustbench_v030/  frozen v0.3 gate evaluated in the paper (pinned commit; see its README)
  followup_v042/                 v0.4.2 follow-up: run script, prespecified design, analysis scripts
results/followup_v042/
  profiles/baseXX/               evidence and decisions for the 19 flawed tables built on each of 50 bases
  supplementary/                 150 fresh honest real-versus-real runs scored with the frozen model
  analysis/                      summary.json, sensitivity.json and the CSV tables the paper reads
  manifest.json                  run manifest (bases, conditions, seeds, versions, script and model hashes)
results/main_v5/
  configuration.json             campaign configuration (read by the follow-up analysis)
  analysis_tables/               campaign summary tables used by the paper
  profiles/jobs/                 saved evidence for all 217 evaluated tables and the frozen calibration model
paper/
  scripts/                       analysis scripts that produce every reported number
  generated/                     their committed outputs (LaTeX macros, tables, figures, JSON audit files)
data/README.md                   where to obtain the public source data
```

## What the saved results contain

Each `profile.json` holds the evidence for one evaluated table: the signals and verdicts of every
dimension under the frozen v0.3 gate and the calibrated protocol, the native SDMetrics quality report,
and labels describing the table (null run, control, injected flaw and dose, or generator and seed).
They contain no patient records. The calibration model, including which null runs were used to fit
and which were held out as the audit, is in `results/main_v5/profiles/jobs/calibrate__model/`.

## Anonymization for review

For double-blind review, three non-numerical edits were made to the code that produced the results:
the provenance note of the gate in `campaign/syntrustbench_campaign/judges.py`, the bibliographic
citation in `campaign/PROTOCOL_v0.4.1.md`, and the clone address of the frozen v0.3 release in
`campaign/syntrustbench_campaign/environment.py` and the notebook (now read from the
`FROZEN_V030_REPOSITORY` environment variable). No calculation was changed; the reproduction check
in step 1 confirms that every number is unchanged. Local file paths in the saved evidence were
replaced by `<RUN_ROOT>`, the absolute paths in `results/followup_v042/manifest.json` by
repository-relative ones, and the metadata of `third_party/` was redacted.

## Data and licence

The source data are the public UCI Diabetes 130-US Hospitals dataset (CC BY 4.0), which is not
redistributed here. Code in this repository is released under the MIT licence (`LICENSE`); the
frozen dependency in `third_party/` keeps its own MIT licence.
