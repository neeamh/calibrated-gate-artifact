# Good Scores Can Hide Bad Data: research artifact

Code, saved results and analysis scripts for the paper *Good Scores Can Hide Bad Data: A Calibrated
Test of How Trust Evidence for Synthetic Clinical Data Should Be Combined* (submitted to ACM TRUST 2027).

The paper asks whether the rule that combines fidelity, utility and privacy evidence changes which
synthetic clinical tables pass, and whether a non-compensatory gate can reject flawed tables without
rejecting real patients. This repository lets a reader do two things:

1. **Regenerate every number, table and figure in the paper** from the saved campaign results, in under
   a minute on a laptop, without the source data. This is the main reproducibility check.
2. **Rerun the full campaign** from the public source data (several hours on a CPU).

## 1. Regenerate the paper's numbers (recommended)

Requires Python 3.11 or later.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt
pip install --no-deps third_party/syntrustbench_v030
PYTHON=python ./reproduce_paper_numbers.sh
```

The script reruns every analysis on the saved evidence in `results/main_v5/`, writes the LaTeX macros,
tables and figures to `paper/generated/`, and checks that each regenerated `.tex` and `.json` file is
byte-identical to the committed version. It ends with `All regenerated numbers match the paper.`
We checked this in a fresh environment on Python 3.13.

| Paper item | Script | Output in `paper/generated/` |
|---|---|---|
| All numbers in the text, Tables 3 and 4 | `make_numbers.py` | `numbers.tex`, `table_admission.tex`, `table_injection.tex`, `numbers.json` |
| Figure 1 | `make_figures.py` | `fig/fig1.pdf` |
| Table 5, Sec. 5.3 counts, temperature sensitivity | `extra_analyses.py`, `mechanism_status.py`, `make_extra_tables.py` | `table_mechanism.tex`, `numbers_extra.tex` |
| Appendix C (percentile mapping, no proximity signal), Appendix B (Gaussian Copula set), Appendix D (calibration diagnostics, Figure 2) | `reanalysis.py`, `make_reanalysis_tables.py` | `numbers_reanalysis.tex`, `table_sensitivity_all.tex`, `fig/fig_pvalues.pdf` |

The analysis scripts import the campaign's own calibration and judge code from `campaign/`, so every
decision is re-derived with the code that produced the results; nothing is refit.

## 2. Rerun the full campaign

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
results/main_v5/
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
replaced by `<RUN_ROOT>`, and the metadata of `third_party/` was redacted.

## Data and licence

The source data are the public UCI Diabetes 130-US Hospitals dataset (CC BY 4.0), which is not
redistributed here. Code in this repository is released under the MIT licence (`LICENSE`); the
frozen dependency in `third_party/` keeps its own MIT licence.
